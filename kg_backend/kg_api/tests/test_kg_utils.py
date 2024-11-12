import pytest
from datetime import datetime
from typing import List, Set
from kg_api.kg_utils import (
    Article, KGNode, KGEdge, KnowledgeGraph,
    KGGenerator, KGEnricher, QueryProcessor
)

# Test Data
@pytest.fixture
def sample_article():
    return Article(
        title="Test Article",
        summary="This is a test article about AI and technology",
        url="https://example.com/article"
    )

@pytest.fixture
def sample_nodes():
    return [
        KGNode(
            id="node1",
            type="COMPANY",
            detailed_type="TECH_COMPANY",
            summary="Apple Inc.",
            articles={"article1", "article2"}
        ),
        KGNode(
            id="node2",
            type="TECHNOLOGY",
            detailed_type="AI",
            summary="Artificial Intelligence",
            articles={"article1"}
        ),
        KGNode(
            id="node3",
            type="COMPANY",
            detailed_type="TECH_COMPANY",
            summary="Google",
            articles={"article2"}
        )
    ]

@pytest.fixture
def sample_edges():
    return [
        KGEdge(source="node1", target="node2", label="DEVELOPS"),
        KGEdge(source="node3", target="node2", label="RESEARCHES"),
        KGEdge(source="node1", target="node3", label="COMPETES_WITH")
    ]

@pytest.fixture
def sample_kg(sample_nodes, sample_edges, sample_article):
    return KnowledgeGraph(
        nodes=sample_nodes,
        edges=sample_edges,
        articles=[sample_article],
        summary="Test knowledge graph about tech companies and AI"
    )

# Base Model Tests
class TestBaseModels:
    def test_article_creation(self, sample_article):
        assert sample_article.title == "Test Article"
        assert sample_article.url == "https://example.com/article"
        
    def test_kg_node_creation(self, sample_nodes):
        node = sample_nodes[0]
        assert node.id == "node1"
        assert node.type == "COMPANY"
        assert "article1" in node.articles
        
    def test_kg_node_serialization(self, sample_nodes):
        node = sample_nodes[0]
        node_dict = node.to_dict()
        assert isinstance(node_dict['articles'], list)
        assert set(node_dict['articles']) == node.articles
        
    def test_kg_edge_creation(self, sample_edges):
        edge = sample_edges[0]
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.label == "DEVELOPS"
        
    def test_kg_edge_serialization(self, sample_edges):
        edge = sample_edges[0]
        edge_dict = edge.to_dict()
        assert edge_dict['source'] == edge.source
        assert edge_dict['count'] == 1

# KGEnricher Tests
class TestKGEnricher:
    @pytest.fixture
    def enricher(self):
        return KGEnricher(preprocess=True)
    
    def test_combine_nodes(self, enricher, sample_nodes):
        # Create duplicate node with different articles
        duplicate_node = KGNode(
            id="node1",
            type="COMPANY",
            detailed_type="TECH_COMPANY",
            summary="Apple Inc.",
            articles={"article3"}
        )
        nodes = sample_nodes + [duplicate_node]
        
        combined = enricher.combine_nodes(nodes)
        assert len(combined) == 3  # Should combine duplicates
        apple_node = next(n for n in combined if n.id == "node1")
        assert len(apple_node.articles) == 3  # Should combine article sets
        
    def test_combine_edges(self, enricher, sample_edges):
        # Create duplicate edge with different label
        duplicate_edge = KGEdge(
            source="node1",
            target="node2",
            label="INVESTS_IN"
        )
        edges = sample_edges + [duplicate_edge]
        
        combined = enricher.combine_edges(edges)
        assert len(combined) == 3  # Should maintain unique source-target pairs
        dev_edge = next(e for e in combined if e.source == "node1" and e.target == "node2")
        assert "DEVELOPS" in dev_edge.label
        assert "INVESTS_IN" in dev_edge.label
        
    def test_validate_edges(self, enricher, sample_kg):
        # Add invalid edge
        invalid_edge = KGEdge(source="nonexistent", target="node1", label="TEST")
        kg_with_invalid = KnowledgeGraph(
            nodes=sample_kg.nodes,
            edges=sample_kg.edges + [invalid_edge],
            articles=sample_kg.articles,
            summary=sample_kg.summary
        )
        
        valid_edges = enricher.validate_edges(kg_with_invalid)
        assert len(valid_edges) == len(sample_kg.edges)
        assert all(e.source in [n.id for n in sample_kg.nodes] for e in valid_edges)

# QueryProcessor Tests
class TestQueryProcessor:
    @pytest.fixture
    def processor(self):
        return QueryProcessor()
    
    def test_get_neighbors(self, processor, sample_kg):
        node1 = sample_kg.nodes[0]
        neighbors = processor.get_neighbors(sample_kg, node1)
        assert len(neighbors) == 2  # Should find both connected nodes
        neighbor_ids = {n.id for n in neighbors}
        assert "node2" in neighbor_ids
        assert "node3" in neighbor_ids
        
    def test_get_connected_entities(self, processor, sample_kg):
        start_nodes = [sample_kg.nodes[0]]  # Start from node1
        connected = processor.get_connected_entities(sample_kg, start_nodes, hops=1)
        assert len(connected) == 3  # Should find all nodes within 1 hop
        
    def test_compute_similarity(self, processor):
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]
        
        assert processor.compute_similarity(emb1, emb2) == 1.0  # Same direction
        assert processor.compute_similarity(emb1, emb3) == 0.0  # Perpendicular
        
    def test_process_query(self, processor, sample_kg):
        query = "What companies are involved in AI?"
        results = processor.process_query(query, sample_kg)
        
        assert "similar_by_text" in results
        assert "similar_by_entities" in results
        assert "connected_entities" in results
        
    def test_answer_query(self, processor, sample_kg):
        query = "What companies are involved in AI?"
        query_results = processor.process_query(query, sample_kg)
        
        answer = processor.answer_query(query, query_results, sample_kg)
        
        assert "answer" in answer
        assert "evidence" in answer
        assert "connections" in answer
        assert "sources" in answer
        assert "metadata" in answer
        assert "confidence_scores" in answer["metadata"]

# Integration Tests
def test_full_workflow(sample_article):
    # Test the full workflow from article to query
    kg_generator = KGGenerator()
    enricher = KGEnricher(preprocess=True)
    processor = QueryProcessor()
    
    # Generate KG
    kg = kg_generator.generate_kg(sample_article.summary, sample_article)
    assert kg is not None
    assert len(kg.nodes) > 0
    
    # Enrich KG
    enriched_kg = enricher.process_and_enrich_kgs([kg])
    assert len(enriched_kg.nodes) > 0
    
    # Query KG
    query = "What is the main topic?"
    results = processor.process_query(query, enriched_kg)
    assert len(results["similar_by_text"]) > 0
    
    # Generate answer
    answer = processor.answer_query(query, results, enriched_kg)
    assert answer["answer"] is not None
    assert len(answer["evidence"]) > 0
