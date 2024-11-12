import json
import logging
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Optional, Set, Tuple, Dict, Any, Union
from pydantic import BaseModel, Field
import networkx as nx
from fastnode2vec import Graph, Node2Vec as FastNode2Vec
import leidenalg
import igraph as ig
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from kg_api.llm_utils import LLMProcessor, logger, CustomHuggingFaceEndpoint, CustomHuggingFaceInferenceAPIEmbeddings, HF_INFERENCE_MODEL_NAME, HF_EMBEDDING_MODEL_NAME, OPENAI_API_KEY, HF_API_KEY
    from kg_api.data_utils import fetch_articles
except ImportError:
    from llm_utils import LLMProcessor, logger, CustomHuggingFaceEndpoint, CustomHuggingFaceInferenceAPIEmbeddings, HF_INFERENCE_MODEL_NAME, HF_EMBEDDING_MODEL_NAME, OPENAI_API_KEY, HF_API_KEY
    from data_utils import fetch_articles

class Article(BaseModel):
    title: str
    summary: str
    url: str

class KGNode(BaseModel):
    id: str
    type: str = ''
    detailed_type: str = ''
    summary: Optional[str] = None
    network_embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None
    community: Optional[int] = None
    articles: Set[str] = Field(default_factory=set)

    def dict(self, *args, **kwargs):
        d = super().model_dump(*args, **kwargs)
        d['articles'] = list(d['articles'])
        return d

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, KGNode):
            return self.id == other.id
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert KGNode to a dictionary format"""
        return {
            'id': self.id,
            'type': self.type,
            'detailed_type': self.detailed_type,
            'summary': self.summary,
            'network_embedding': self.network_embedding,
            'text_embedding': self.text_embedding,
            'community': self.community,
            'articles': list(self.articles)  # Convert set to list for JSON serialization
        }

class KGEdge(BaseModel):
    source: str
    target: str
    label: str
    count: int = Field(default=1, ge=1)

    def dict(self, *args, **kwargs):
        return {
            'source': self.source,
            'target': self.target,
            'label': self.label,
            'count': self.count
        }

    def __hash__(self):
        return hash((self.source, self.target, self.label))

    def __eq__(self, other):
        if isinstance(other, KGEdge):
            return (self.source, self.target, self.label) == (other.source, other.target, other.label)
        return False

    def to_dict(self):
        return {
            'source': self.source,
            'target': self.target,
            'label': self.label,
            'count': self.count
        }

class KnowledgeGraph(BaseModel):
    nodes: List[KGNode]
    edges: List[KGEdge]
    articles: List[Article] = Field(default_factory=list)
    summary: str

    def dict(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        for article in data['articles']:
            article['url'] = str(article['url'])
        for node in data['nodes']:
            node['articles'] = list(node['articles'])
        return data

    def to_networkx(self):
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.id, **node.model_dump(exclude={'articles'}))
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, label=edge.label)
        return G

class KGGenerator:
    def __init__(self):
        self.llm_processor = LLMProcessor()

    def generate_kg(self, text: str, article_data: Union[Dict[str, Any], Article]) -> KnowledgeGraph:
        prompt = f"""Generate a knowledge graph from the following news article, which should be similar to a mind map. 
        Identify key entities and their relationships. 
        For each knowledge graph, provide a summary of the main points in the news article.
        For each node, provide an id, type, detailed_type, and a brief one liner summary describing what the node is about.
        For each edge, provide a source, target, and label.
        Node ids should be informative text. Do not use numbers for the node id.
        The text is: {text}"""

        try:
            kg_data = self.llm_processor.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_model=KnowledgeGraph
            )
            
            if isinstance(article_data, dict):
                article = Article(
                    title=article_data.get('title', ''),
                    summary=article_data.get('news_text', '')[:200] + "...",
                    url=article_data.get('link', '')
                )
            elif isinstance(article_data, Article):
                article = article_data
            else:
                raise ValueError("article_data must be either a dictionary or an Article object")

            kg_data.articles.append(article)
            for node in kg_data.nodes:
                node.articles.add(article.url)
            return kg_data
        except Exception as e:
            logger.error(f"Error generating OpenAI KG: {e}")
            raise

class KGEnricher:
    def __init__(self, dimensions=64, walk_length=30, num_walks=200, workers=1, preprocess=True):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.preprocess = preprocess
        self.llm_processor = LLMProcessor()

    def process_and_enrich_kgs(self, kgs: List[KnowledgeGraph]) -> KnowledgeGraph:
        logger.info(f"Processing and enriching {len(kgs)} knowledge graphs")
        
        if self.preprocess:
            logger.info("Preprocessing: refining, cleaning, and merging KGs")
            refined_kgs = [self.refine_knowledge_graph(kg) for kg in kgs]
            cleaned_kgs = [self.clean_knowledge_graph(kg) for kg in refined_kgs]
            merged_kg = self.merge_knowledge_graphs(cleaned_kgs)
            logger.info(f"Merged KG: {len(merged_kg.nodes)} nodes, {len(merged_kg.edges)} edges")
        else:
            merged_kg = self.merge_knowledge_graphs(kgs)
        
        return self.enrich_kg(merged_kg)

    def enrich_kg(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        logger.info(f"Starting KG enrichment. Input KG has {len(kg.nodes)} nodes and {len(kg.edges)} edges.")
        
        if len(kg.nodes) == 0:
            logger.warning("Cannot enrich an empty knowledge graph.")
            return kg

        # Log initial node state
        logger.info("Initial node states:")
        for node in kg.nodes[:3]:  # Log first 3 nodes as sample
            logger.info(f"Node {node.id}: network_embedding={bool(node.network_embedding)}, "
                       f"text_embedding={bool(node.text_embedding)}, "
                       f"community={node.community}")

        G = nx.Graph()
        for edge in kg.edges:
            G.add_edge(edge.source, edge.target, label=edge.label)
        
        logger.info(f"Created networkx graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        # Convert NetworkX graph to igraph for Leidenalg
        logger.info("Converting NetworkX graph to igraph format...")
        node_mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
        reverse_mapping = dict(zip(range(len(G.nodes())), G.nodes()))
        
        edges_for_igraph = [(node_mapping[e[0]], node_mapping[e[1]]) for e in G.edges()]
        ig_graph = ig.Graph(edges=edges_for_igraph)
        
        # Convert to FastNode2Vec format for embeddings
        edges = list(G.edges())
        graph = Graph(edges, directed=False, weighted=False)
        logger.info(f"Converted to FastNode2Vec format with {len(edges)} edges")

        try:
            # Create and train model
            logger.info(f"Initializing FastNode2Vec with dimensions={self.dimensions}, "
                       f"walk_length={self.walk_length}")
            model = FastNode2Vec(
                graph,
                dim=self.dimensions,
                walk_length=self.walk_length,
                window=10,
                p=2.0,
                q=0.5,
                workers=self.workers,
                # epochs=10
            )
            
            # Train model
            logger.info("Starting FastNode2Vec training...")
            model.train(epochs=10)
            logger.info("Successfully trained FastNode2Vec model.")

            # Log model vocabulary size
            vocab_size = len(model.wv.index_to_key)
            logger.info(f"Model vocabulary size: {vocab_size}")

            # Run Leiden algorithm
            logger.info("Running Leiden community detection...")
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                seed=42,  # For reproducibility
                n_iterations=10
            )
            
            # Convert partition to dictionary mapping node IDs to communities
            communities = {
                reverse_mapping[node]: membership 
                for node, membership in enumerate(partition.membership)
            }
            logger.info(f"Found {len(set(partition.membership))} communities")
            
            enriched_nodes = []
            for node in kg.nodes:
                try:
                    logger.info(f"Processing node {node.id}")
                    
                    # Get network embedding
                    try:
                        network_embedding = model.wv[node.id].tolist()
                        logger.info(f"Generated network embedding for {node.id}: "
                                   f"length={len(network_embedding)}")
                    except KeyError:
                        logger.error(f"Node {node.id} not found in model vocabulary, "
                                     f"using zero embedding")
                        network_embedding = [0] * self.dimensions
                    
                    # Get text embedding
                    if node.summary:
                        try:
                            text_embedding = self.llm_processor.generate_text_embedding(node.summary)
                            logger.info(f"Generated text embedding for {node.id}: "
                                       f"length={len(text_embedding) if text_embedding else 0}")
                        except Exception as e:
                            logger.error(f"Error generating text embedding for {node.id}: {str(e)}")
                            text_embedding = None
                    else:
                        logger.info(f"No summary for node {node.id}, skipping text embedding")
                        text_embedding = None
                    
                    # Get community
                    community = communities.get(node.id, -1)
                    logger.info(f"Assigned community {community} to node {node.id}")
                    
                    enriched_node = KGNode(
                        id=node.id,
                        type=node.type,
                        detailed_type=node.detailed_type,
                        summary=node.summary,
                        network_embedding=network_embedding,
                        text_embedding=text_embedding,
                        community=community,
                        articles=node.articles
                    )
                    enriched_nodes.append(enriched_node)
                    
                except Exception as e:
                    logger.error(f"Error enriching node {node.id}: {str(e)}")
                    # Add the original node without enrichment
                    enriched_nodes.append(node)
            
            # Log final enrichment statistics
            successful_network_embeddings = sum(1 for n in enriched_nodes if n.network_embedding)
            successful_text_embeddings = sum(1 for n in enriched_nodes if n.text_embedding)
            successful_communities = sum(1 for n in enriched_nodes if n.community is not None)
            
            logger.info(f"Enrichment complete. Statistics:")
            logger.info(f"- Nodes with network embeddings: {successful_network_embeddings}/{len(enriched_nodes)}")
            logger.info(f"- Nodes with text embeddings: {successful_text_embeddings}/{len(enriched_nodes)}")
            logger.info(f"- Nodes with communities: {successful_communities}/{len(enriched_nodes)}")
            
            # Sample check of enriched nodes
            logger.info("Sample of enriched nodes:")
            for node in enriched_nodes[:3]:
                logger.info(f"Node {node.id}:")
                logger.info(f"- Network embedding: {bool(node.network_embedding)}")
                logger.info(f"- Text embedding: {bool(node.text_embedding)}")
                logger.info(f"- Community: {node.community}")
            
            return KnowledgeGraph(nodes=enriched_nodes, edges=kg.edges, summary=kg.summary, articles=kg.articles)
            
        except Exception as e:
            logger.error(f"Error training FastNode2Vec model: {str(e)}", exc_info=True)
            return kg

    def refine_knowledge_graph(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        logger.info(f"Refining knowledge graph: {kg}")
        
        refined_nodes = self.combine_nodes(kg.nodes)
        refined_edges = self.combine_edges(kg.edges)

        refined_kg = KnowledgeGraph(
            nodes=refined_nodes,
            edges=refined_edges,
            summary=kg.summary,
            articles=kg.articles
        )

        logger.info(f"Refined knowledge graph: {len(refined_kg.nodes)} nodes, {len(refined_kg.edges)} edges")
        return refined_kg

    def clean_knowledge_graph(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        logger.info("Cleaning Knowledge Graph")
        valid_edges = self.validate_edges(kg)
        combined_edges = self.combine_edges(valid_edges)
        connected_nodes = self.remove_isolated_nodes(kg.nodes, combined_edges)
        logger.info(f"Knowledge graph cleaned: {len(connected_nodes)} nodes, {len(combined_edges)} edges")
        return KnowledgeGraph(nodes=connected_nodes, edges=combined_edges, summary=kg.summary, articles=kg.articles)

    def merge_knowledge_graphs(self, kgs: List[KnowledgeGraph], use_openai: bool = False) -> KnowledgeGraph:
        logger.info(f"Merging {len(kgs)} knowledge graphs")
        all_nodes = []
        all_edges = []
        all_articles = []
        combined_summary = ""

        for kg in kgs:
            all_nodes.extend(kg.nodes)
            all_edges.extend(kg.edges)
            all_articles.extend(kg.articles)
            combined_summary += kg.summary + " "

        combined_nodes = self.combine_nodes(all_nodes)
        combined_edges = self.combine_edges(all_edges)

        # Create a simplified JSON representation of the merged graph
        simplified_graph = {
            "nodes": [{"id": node.id, "type": node.type, "summary": node.summary} for node in combined_nodes[:50]],  # Limit to first 50 nodes
            "edges": [{"source": edge.source, "target": edge.target, "label": edge.label} for edge in combined_edges[:50]]  # Limit to first 50 edges
        }
        graph_json = json.dumps(simplified_graph, indent=2)

        # Generate refined summary using HuggingFace model
        prompt = f"""Analyze and summarize the following information:

                Combined Summaries:
                {combined_summary}

                Networks (JSON):
                {graph_json}

                Please provide a concise, informative summary that:
                1. Highlights the main themes and topics across the summaries, leveraging on the networks to identify key entities and relationships.
                2. Explains the key entities and their relationships
                3. Notes any significant patterns, trends, or clusters in the networks
                4. Offers brief insights or analysis based on both the summaries and the networks

                Your summary should be coherent, well-structured, and capture the essence of the combined summaries and the networks, integrating information from both the textual summaries and the networks. There is no word limit. The summary should just go straight into the content. There is no need to talk about where the summaries came from. Begin by saying - Key takeaways are ....

                ##Summary##:

                """

        if use_openai:
            # Use OpenAI for summary generation
            openai_client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500  # adjust as needed
            )
            refined_summary = response.choices[0].message.content.strip()
        else:
            # Use HuggingFace for summary generation
            hf_endpoint = CustomHuggingFaceEndpoint(HF_INFERENCE_MODEL_NAME, HF_API_KEY)
            refined_summary = hf_endpoint.invoke(prompt)
            # Assuming `refined_summary` contains the full text returned by the CustomHuggingFaceEndpoint
            summary_start = refined_summary.find("##Summary##")  # Find the start of the summary section

            if summary_start != -1:
                # Extract the summary part, starting just after the "**Summary**" line
                extracted_summary = refined_summary[summary_start + len("##Summary##:"):].strip()
            else:
                extracted_summary = refined_summary  # Fallback to the full text if "**Summary**" is not found

            # Optionally, you can further clean up the extracted summary if needed
            # extracted_summary = extracted_summary.split("\n\n")  # Get the first paragraph if there are multiple

        merged_kg = KnowledgeGraph(
            nodes=combined_nodes, 
            edges=combined_edges, 
            summary=extracted_summary,
            articles=all_articles
        )
        logger.info(f"Merged knowledge graph: {len(merged_kg.nodes)} nodes, {len(merged_kg.edges)} edges, {len(merged_kg.articles)} articles")
        return merged_kg

    def combine_nodes(self, nodes: List[KGNode]) -> List[KGNode]:
        unique_nodes = {}
        article_sets = defaultdict(set)

        # First pass: collect all data
        for node in nodes:
            if node.id not in unique_nodes:
                unique_nodes[node.id] = node
                article_sets[node.id] = set(node.articles)
            else:
                existing_node = unique_nodes[node.id]
                existing_node.type = existing_node.type or node.type
                existing_node.detailed_type = existing_node.detailed_type or node.detailed_type
                existing_node.summary = existing_node.summary or node.summary
                existing_node.network_embedding = existing_node.network_embedding or node.network_embedding
                existing_node.text_embedding = existing_node.text_embedding or node.text_embedding
                existing_node.community = existing_node.community or node.community
                article_sets[node.id].update(node.articles)

        # Second pass: update articles
        for node_id, node in unique_nodes.items():
            node.articles = article_sets[node_id]

        return list(unique_nodes.values())

    def combine_edges(self, edges: List[KGEdge]) -> List[KGEdge]:
        edge_dict = defaultdict(lambda: defaultdict(set))

        for edge in edges:
            edge_dict[edge.source][edge.target].add(edge.label)

        combined_edges = []
        for source, targets in edge_dict.items():
            for target, labels in targets.items():
                combined_edges.append(KGEdge(source=source, target=target, label=", ".join(sorted(labels))))

        return combined_edges

    def validate_edges(self, kg: KnowledgeGraph) -> List[KGEdge]:
        logger.info("Validating edges")
        valid_node_ids = set(node.id for node in kg.nodes)
        valid_edges = [
            edge for edge in kg.edges
            if edge.source in valid_node_ids and edge.target in valid_node_ids
        ]
        invalid_edge_count = len(kg.edges) - len(valid_edges)
        if invalid_edge_count > 0:
            logger.warning(f"Removed {invalid_edge_count} invalid edges")
        return valid_edges

    def remove_isolated_nodes(self, nodes: List[KGNode], edges: List[KGEdge]) -> List[KGNode]:
        connected_nodes = set()
        for edge in edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        connected_node_list = [node for node in nodes if node.id in connected_nodes]
        
        removed_nodes = len(nodes) - len(connected_node_list)
        logger.info(f"Removed {removed_nodes} isolated nodes")
        
        return connected_node_list

class QueryProcessor:
    def __init__(self):
        self.llm_processor = LLMProcessor()

    def process_query(
        self, 
        query: str, 
        merged_kg: KnowledgeGraph, 
        selected_nodes: List[KGNode] = None,
        top_n: int = 5, 
        connected_hops: int = 1,
        find_text_similar: bool = True,
        find_entity_similar: bool = True,
        find_network_similar: bool = True
    ) -> Dict[str, List[Tuple[KGNode, float]]]:
        """
        Process a query against the knowledge graph with configurable similarity searches
        
        Args:
            query: User query string
            merged_kg: Knowledge graph to search
            selected_nodes: List of nodes selected in the frontend
            top_n: Number of similar entities to return for each similarity type
            connected_hops: Number of hops for finding connected entities
            find_text_similar: Whether to search for text-based similarities
            find_entity_similar: Whether to search for entity-based similarities
            find_network_similar: Whether to search for network-based similarities
            
        Returns:
            Dictionary containing similarity search results and connected entities
        """
        results = {}
        
        if find_text_similar:
            query_embedding = self.llm_processor.generate_text_embedding(query)
            results["similar_by_text"] = self.find_similar_entities_by_text(
                merged_kg, query_embedding, top_n
            )
        
        if find_entity_similar:
            extracted_entities = self.llm_processor.extract_entities(query)
            related_entities = self.llm_processor.infer_related_entities(extracted_entities)
            entity_embeddings = self.llm_processor.generate_entity_embeddings(
                extracted_entities + related_entities
            )
            results["similar_by_entities"] = self.find_similar_entities_by_entities(
                merged_kg, entity_embeddings, top_n
            )
        
        if find_network_similar and selected_nodes:
            results["similar_by_network"] = self.find_similar_entities_by_network(
                merged_kg, selected_nodes, top_n
            )
        
        # Only find connected entities if there are selected nodes
        if selected_nodes:
            connected_entities = self.get_connected_entities(
                merged_kg, selected_nodes, connected_hops
            )
            results["connected_entities"] = [(node, 0) for node in connected_entities]
        else:
            results["connected_entities"] = []
        
        return results

    def find_similar_entities_by_text(self, kg: KnowledgeGraph, query_embedding: List[float], top_n: int) -> List[Tuple[KGNode, float]]:
        similar_entities = []
        
        for node in kg.nodes:
            if node.text_embedding is None:
                continue
            
            similarity = self.compute_similarity(query_embedding, node.text_embedding)
            similar_entities.append((node, similarity))
        
        return sorted(similar_entities, key=lambda x: x[1], reverse=True)[:top_n]

    def find_similar_entities_by_entities(self, kg: KnowledgeGraph, entity_embeddings: List[List[float]], top_n: int) -> List[Tuple[KGNode, float]]:
        similar_entities = []
        
        for node in kg.nodes:
            if node.text_embedding is None:
                continue
            
            entity_similarities = [self.compute_similarity(entity_emb, node.text_embedding) 
                                   for entity_emb in entity_embeddings]
            max_entity_similarity = max(entity_similarities) if entity_similarities else 0
            
            similar_entities.append((node, max_entity_similarity))
        
        return sorted(similar_entities, key=lambda x: x[1], reverse=True)[:top_n]

    def find_similar_entities_by_network(
        self, 
        kg: KnowledgeGraph, 
        nodes: List[KGNode], 
        top_n: int
    ) -> List[Tuple[KGNode, float]]:
        """
        Find entities similar to a list of nodes based on network embeddings
        
        Args:
            kg: Knowledge graph containing all nodes
            nodes: List of nodes to find similar entities for
            top_n: Number of similar entities to return
            
        Returns:
            List of tuples containing (node, similarity_score) sorted by similarity
        """
        all_similar = []
        seen_nodes = set()
        
        for node in nodes:
            if not node.network_embedding:
                logger.warning(f"Node {node.id} has no network embedding")
                continue
            
            for other_node in kg.nodes:
                # Skip self, already seen nodes, and nodes without embeddings
                if (other_node.id == node.id or 
                    other_node.id in seen_nodes or 
                    not other_node.network_embedding):
                    continue
                
                try:
                    similarity = self.compute_similarity(
                        node.network_embedding,
                        other_node.network_embedding
                    )
                    all_similar.append((other_node, similarity))
                    seen_nodes.add(other_node.id)
                except Exception as e:
                    logger.error(
                        f"Error computing network similarity between {node.id} "
                        f"and {other_node.id}: {str(e)}"
                    )
                    continue
        
        # Sort by similarity score and return top_n unique results
        return sorted(all_similar, key=lambda x: x[1], reverse=True)[:top_n]

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def get_connected_entities(
        self, 
        kg: KnowledgeGraph, 
        selected_nodes: List[KGNode], 
        hops: int
    ) -> Set[KGNode]:
        """
        Find entities connected to the selected nodes within a specified number of hops
        
        Args:
            kg: Knowledge graph containing all nodes
            selected_nodes: List of nodes selected in the frontend
            hops: Maximum number of hops to traverse
            
        Returns:
            Set of nodes connected to the selected nodes
        """
        if not selected_nodes:
            return set()
        
        connected = set()
        queue = deque([(node, 0) for node in selected_nodes])
        visited = set(node.id for node in selected_nodes)  # Track IDs instead of nodes

        while queue:
            node, distance = queue.popleft()
            if distance > hops:
                break

            if node not in selected_nodes:  # Don't add selected nodes to connected set
                connected.add(node)

            # Find neighbors
            neighbors = self.get_neighbors(kg, node)
            for neighbor in neighbors:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor, distance + 1))

        return connected

    def get_neighbors(self, kg: KnowledgeGraph, node: KGNode) -> Set[KGNode]:
        neighbors = set()
        for edge in kg.edges:
            if edge.source == node.id:
                target_node = next((n for n in kg.nodes if n.id == edge.target), None)
                if target_node:
                    neighbors.add(target_node)
            elif edge.target == node.id:
                source_node = next((n for n in kg.nodes if n.id == edge.source), None)
                if source_node:
                    neighbors.add(source_node)
        return neighbors
    
    def answer_query(self, query: str, query_results: Dict[str, List[Tuple[KGNode, float]]], 
                    kg: KnowledgeGraph) -> Dict[str, Any]:
        """
        Generate a comprehensive answer based on query results
        
        Args:
            query: Original user query
            query_results: Results from process_query
            kg: Original knowledge graph
            
        Returns:
            Dict containing structured answer with evidence and metadata
        """
        try:
            # Build context from query results
            context = self._build_context(
                similar_text=query_results.get('similar_by_text', []),
                similar_entities=query_results.get('similar_by_entities', []),
                similar_network=query_results.get('similar_by_network', []),
                connected_entities=set(node for node, _ in query_results.get('connected_entities', [])),
                kg=kg
            )
            
            # Create prompt for answer generation
            prompt = f"""Based on the following information, answer the query: "{query}"

                Context Information:
                {context['text']}

                Relevant Entities:
                {context['entities']}

                Network Similar Entities:
                {context['network_similar']}

                Entity Relationships:
                {context['relationships']}

                Please provide a comprehensive response that includes:
                1. A direct answer to the query
                2. Supporting evidence from the provided context
                3. Relevant connections between entities
                4. References to source articles

                Format the response as JSON with the following structure:
                {{
                    "answer": "main answer text",
                    "evidence": ["list of supporting evidence"],
                    "connections": ["list of relevant relationships"],
                    "sources": ["list of source articles"],
                    "key_entities": ["list of most relevant entities"]
                }}

                Do not answer the query if you information is not found in the context. Suggest an alternate query relevant to the context instead.
                """

            # Generate answer using LLM
            response = self.llm_processor.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            answer_json = json.loads(response.choices[0].message.content)
            
            # Add metadata with optional similarity scores
            answer_json.update({
                "metadata": {
                    "confidence_scores": {
                        "text_similarity": self._calculate_confidence(query_results.get('similar_by_text', [])),
                        "entity_similarity": self._calculate_confidence(query_results.get('similar_by_entities', [])),
                        "network_similarity": self._calculate_confidence(query_results.get('similar_by_network', [])),
                        "connection_strength": len(query_results.get('connected_entities', [])) / len(kg.nodes) if kg.nodes else 0
                    },
                    "query_stats": {
                        "matched_nodes": len(query_results.get('similar_by_text', [])),
                        "related_entities": len(query_results.get('similar_by_entities', [])),
                        "network_similar": len(query_results.get('similar_by_network', [])),
                        "connected_nodes": len(query_results.get('connected_entities', []))
                    }
                }
            })
            
            return answer_json
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            return {
                "answer": "Unable to generate answer due to an error",
                "evidence": [],
                "connections": [],
                "sources": [],
                "key_entities": [],
                "metadata": {
                    "confidence_scores": {
                        "text_similarity": 0,
                        "entity_similarity": 0,
                        "network_similarity": 0,
                        "connection_strength": 0
                    },
                    "query_stats": {
                        "matched_nodes": 0,
                        "related_entities": 0,
                        "network_similar": 0,
                        "connected_nodes": 0
                    },
                    "error": str(e)
                }
            }

    def _build_context(self, similar_text: List[Tuple[KGNode, float]], 
                      similar_entities: List[Tuple[KGNode, float]], 
                      similar_network: List[Tuple[KGNode, float]],
                      connected_entities: Set[KGNode],
                      kg: KnowledgeGraph) -> Dict[str, str]:
        """Build context information for answer generation"""
        
        # Collect relevant text passages
        text_passages = [
            f"- {node.summary} (confidence: {similarity:.2f})"
            for node, similarity in similar_text
            if similarity > 0.5  # Filter by confidence threshold
        ]
        
        # Collect entity information
        entity_info = [
            f"- {node.id} ({node.type}): {node.summary} (relevance: {similarity:.2f})"
            for node, similarity in similar_entities
            if similarity > 0.5
        ]
        
        # Collect relationship information
        relationships = []
        relevant_node_ids = {node.id for node, _ in similar_entities}
        relevant_node_ids.update(node.id for node, _ in similar_text)
        
        for edge in kg.edges:
            if edge.source in relevant_node_ids or edge.target in relevant_node_ids:
                source_node = next((n for n in kg.nodes if n.id == edge.source), None)
                target_node = next((n for n in kg.nodes if n.id == edge.target), None)
                if source_node and target_node:
                    relationships.append(
                        f"- {source_node.id} ({source_node.type}) {edge.label} "
                        f"{target_node.id} ({target_node.type})"
                    )
        
        return {
            "text": "\n".join(text_passages) or "No relevant text found",
            "entities": "\n".join(entity_info) or "No relevant entities found",
            "relationships": "\n".join(relationships[:10]) or "No relevant relationships found",  # Limit to top 10
            "network_similar": "\n".join([f"- {node.id} ({node.type}): {node.summary} (similarity: {similarity:.2f})"
                                        for node, similarity in similar_network
                                        if similarity > 0.5]) or "No relevant network similar entities found",  # Limit to top 10
        }

    def _calculate_confidence(self, items: List[Tuple[KGNode, float]]) -> float:
        """Calculate overall confidence score from similarity scores"""
        if not items:
            return 0.0
        scores = [score for _, score in items]
        return sum(scores) / len(scores)



def save_to_json(data: Dict[str, Any], filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved output to {filename}")
    
# Example usage
if __name__ == "__main__":
    import time
    kg_generator = KGGenerator()
    enricher = KGEnricher(preprocess=True)
    query_processor = QueryProcessor()

    # Fetch and process articles
    tickers = ['AAPL', 'GOOGL']
    articles_df = fetch_articles(tickers, window=1, limit=2)
    logger.info(f"Fetched {len(articles_df)} articles for tickers: {tickers}")

    # Generate and process knowledge graphs
    all_kgs = []
    all_outputs = {}
    
    for index, row in articles_df.iterrows():
        article_id = f"article_{index}"
        logger.info(f"\nProcessing article {index + 1}/{len(articles_df)}: {row['title']}")
        
        start_time = time.time()
        article = Article(
            title=row['title'],
            summary=row['news_text'][:200] + "...",
            url=row['link']
        )
        logger.info(f"Article processed in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        kg = kg_generator.generate_kg(row['news_text'], article)
        logger.info(f"Initial KG generated: {len(kg.nodes)} nodes, {len(kg.edges)}")
        all_outputs[f"{article_id}_initial_kg"] = kg.dict()
        logger.info(f"KG processed in {time.time() - start_time:.2f} seconds")
        
        all_kgs.append(kg)

    # Process and enrich knowledge graphs
    if all_kgs:
        logger.info(f"\nProcessing and enriching {len(all_kgs)} knowledge graphs")
        start_time = time.time()
        final_kg = enricher.process_and_enrich_kgs(all_kgs)
        logger.info("\nFinal Knowledge Graph:")
        logger.info(f"Nodes: {len(final_kg.nodes)}")
        logger.info(f"Edges: {len(final_kg.edges)}")
        all_outputs["final_processed_enriched_kg"] = final_kg.dict()
        logger.info(f"Knowledge graphs processed in {time.time() - start_time:.2f} seconds")

        # Test different query scenarios
        test_scenarios = [
            {
                "name": "Basic text search",
                "query": "What are the latest developments in AI?",
                "selected_nodes": [],
                "find_text_similar": True,
                "find_entity_similar": False,
                "find_network_similar": False
            },
            {
                "name": "Entity-based search",
                "query": "What are the relationships between major tech companies?",
                "selected_nodes": [],
                "find_text_similar": False,
                "find_entity_similar": True,
                "find_network_similar": False
            },
            {
                "name": "Network-based search with selected nodes",
                "query": "How are these entities connected?",
                "selected_nodes": final_kg.nodes[:2],  # Select first two nodes
                "find_text_similar": False,
                "find_entity_similar": False,
                "find_network_similar": True
            },
            {
                "name": "Combined search",
                "query": "What are the main trends and connections in the tech industry?",
                "selected_nodes": final_kg.nodes[:2],
                "find_text_similar": True,
                "find_entity_similar": True,
                "find_network_similar": True
            }
        ]

        logger.info("\nTesting different query scenarios:")
        for scenario in test_scenarios:
            logger.info(f"\nScenario: {scenario['name']}")
            logger.info(f"Query: {scenario['query']}")
            logger.info(f"Selected nodes: {[node.id for node in scenario['selected_nodes']]}")
            
            try:
                # Process query with scenario parameters
                start_time = time.time()
                query_results = query_processor.process_query(
                    query=scenario['query'],
                    merged_kg=final_kg,
                    selected_nodes=scenario['selected_nodes'],
                    top_n=5,
                    connected_hops=2,
                    find_text_similar=scenario['find_text_similar'],
                    find_entity_similar=scenario['find_entity_similar'],
                    find_network_similar=scenario['find_network_similar']
                )
                logger.info(f"Query processed in {time.time() - start_time:.2f} seconds")
                
                # Log results for each similarity type
                for result_type, entities in query_results.items():
                    logger.info(f"\n{result_type.replace('_', ' ').title()}:")
                    for node, similarity in entities:
                        logger.info(f"- {node.id} (similarity: {similarity:.4f})")
                        logger.info(f"  Summary: {node.summary[:100]}...")
                
                # Generate and log answer
                start_time = time.time()
                answer = query_processor.answer_query(
                    scenario['query'],
                    query_results,
                    final_kg
                )
                logger.info(f"Answer generated in {time.time() - start_time:.2f} seconds")
                
                logger.info("\nGenerated Answer:")
                logger.info(f"Main Answer: {answer['answer']}")
                logger.info("\nEvidence:")
                for evidence in answer['evidence']:
                    logger.info(f"- {evidence}")
                logger.info("\nConfidence Scores:")
                for metric, score in answer['metadata']['confidence_scores'].items():
                    logger.info(f"- {metric}: {score:.4f}")
                
                # Save results
                all_outputs[f"query_scenario_{scenario['name']}"] = {
                    "scenario": scenario,
                    "results": {
                        result_type: [
                            {
                                "node_id": node.id,
                                "similarity": float(similarity),
                                "summary": node.summary
                            }
                            for node, similarity in entities
                        ]
                        for result_type, entities in query_results.items()
                    },
                    "answer": answer
                }
                
            except Exception as e:
                logger.error(f"Error in scenario {scenario['name']}: {str(e)}", exc_info=True)
                all_outputs[f"query_scenario_{scenario['name']}_error"] = str(e)
            
            logger.info("---")

    else:
        logger.warning("No knowledge graphs were generated.")

    # Save all outputs to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"kg_outputs_{timestamp}.json"
    save_to_json(all_outputs, output_filename)
