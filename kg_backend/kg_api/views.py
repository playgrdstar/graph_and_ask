from typing import List, Dict, Any, Optional
from ninja import NinjaAPI, Schema, Field
from django.http import HttpRequest, StreamingHttpResponse, HttpResponse
from django.utils.html import escape
from urllib.parse import urlparse
from datetime import datetime
import uuid
from django.contrib.sessions.backends.db import SessionStore
import json
import logging
from django.core.cache import cache
import hashlib
import time
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .kg_utils import (
    KGGenerator, 
    KGEnricher, 
    QueryProcessor,
    KnowledgeGraph,
    Article,
    KGNode,
    KGEdge
)

from .data_utils import fetch_articles

logger = logging.getLogger(__name__)

# Simple initialization without CORS parameters
api = NinjaAPI()

# Request/Response Models
class ArticleSchema(Schema):
    title: str
    summary: str
    url: str

class GenerateKGRequest(Schema):
    tickers: str = Field(
        ..., 
        description="Comma-separated list of stock tickers",
        pattern="^[A-Z,\s]+$",  # Only uppercase letters and commas
        min_length=1,
        max_length=100
    )
    window: int = Field(default=1, ge=1, description="Time window in days")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of articles to fetch")

class KGNodeSchema(Schema):
    id: str
    type: str = ''
    detailed_type: str = ''
    summary: Optional[str] = None
    network_embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None
    community: Optional[int] = None
    articles: List[str] = Field(default_factory=list)

class KGEdgeSchema(Schema):
    source: str
    target: str
    label: str
    count: int

class KnowledgeGraphSchema(Schema):
    nodes: List[KGNodeSchema]
    edges: List[KGEdgeSchema]
    articles: List[ArticleSchema]
    summary: str

class QueryRequest(Schema):
    kg_id: str = Field(..., description="ID of the knowledge graph to query")
    query: str = Field(..., description="Natural language query")
    top_n: int = Field(default=5, ge=1, description="Number of top results to return")
    connected_hops: int = Field(default=1, ge=1, description="Number of hops for connected entities")
    selected_node_ids: Optional[List[str]] = Field(default=None, description="List of selected node IDs")

class QueryAnswerSchema(Schema):
    answer: str
    evidence: List[str]
    connections: List[str]
    sources: List[str]
    key_entities: List[str]
    metadata: Dict[str, Any]

class QueryResultSchema(Schema):
    query_results: Dict[str, List[Dict[str, Any]]]
    answer: QueryAnswerSchema

class KnowledgeGraphResponseSchema(Schema):
    kg_id: str
    kg: KnowledgeGraphSchema

class EnrichKGRequest(Schema):
    kg_ids: List[str] = Field(..., description="List of knowledge graph IDs to enrich")

# Add session cleanup and size limits
MAX_SESSION_KGS = 10
MAX_KG_SIZE = 1024 * 1024  # 1MB

def cleanup_old_sessions(session):
    """Cleanup old sessions while preserving recent KGs"""
    kgs = session.get('kgs', {})
    if len(kgs) > MAX_SESSION_KGS:
        # Remove oldest KGs
        sorted_kgs = sorted(kgs.items(), key=lambda x: x[1].get('created_at', 0))
        for kg_id, _ in sorted_kgs[:-MAX_SESSION_KGS]:
            del kgs[kg_id]

def get_cached_kg(kg_id: str) -> Optional[Dict]:
    cache_key = f"kg_{kg_id}"
    return cache.get(cache_key)

def cache_kg(kg_id: str, kg_data: Dict):
    cache_key = f"kg_{kg_id}"
    cache.set(cache_key, kg_data, timeout=3600)  # 1-hour cache

def generate_kg_id(article_url: str) -> str:
    """Generate a unique KG ID based on the article URL."""
    return hashlib.sha256(article_url.encode()).hexdigest()

def validate_url(url: str) -> bool:
    """Validate that a URL is properly formatted and uses http(s)"""
    try:
        parsed = urlparse(url)
        return bool(parsed.netloc and parsed.scheme in ['http', 'https'])
    except Exception as e:
        logger.error(f"URL validation failed: {str(e)}")
        return False

def sanitize_input(text: str) -> str:
    """Sanitize text input to prevent XSS and clean whitespace"""
    return escape(text.strip()) if text else ""

# API Endpoints
@csrf_exempt
@require_http_methods(["OPTIONS"])
def handle_options(request):
    """Handle CORS preflight requests"""
    response = HttpResponse()
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response['Access-Control-Allow-Headers'] = '*'
    return response

@csrf_exempt
@api.get("/generate")
@api.post("/generate")
def generate_kg(request: HttpRequest, data: Optional[GenerateKGRequest] = None):
    """Stream knowledge graphs as they are generated from articles"""
    try:
        if request.method == "POST":
            if not request.session.session_key:
                request.session.create()
            
            # Sanitize ticker input
            tickers = [sanitize_input(ticker.strip()) for ticker in data.tickers.split(',')] if data.tickers else []
            window = data.window
            limit = data.limit
        elif request.method == "GET":
            # Handle GET request parameters
            tickers = [sanitize_input(ticker.strip()) for ticker in request.GET.get("tickers", "").split(',')] if request.GET.get("tickers") else []
            window = int(request.GET.get("window", 1))
            limit = int(request.GET.get("limit", 10))

        # Add processing flag here
        # processing = True  # Initialize processing flag

        def stream_response():
            try:
                # Send initial connection message
                yield 'data: {"type": "connection", "status": "established"}\n\n'
                
                articles_df = fetch_articles(tickers, window=window, limit=limit)
                logger.info(f"Fetched {len(articles_df)} articles for {tickers} in {window} days")
                kg_generator = KGGenerator()
                
                # Initialize kg_id here
                kg_id = None  # Initialize kg_id to None or an appropriate default value
                
                for idx, article in articles_df.iterrows():
                    logger.info(f"Processing article {idx + 1} of {len(articles_df)}")
                    try:
                        # Sanitize and validate article data
                        article_url = article.get('link', '')
                        if not validate_url(article_url):
                            logger.warning(f"Invalid article URL: {article_url}")
                            continue
                            
                        article_data = {
                            'title': sanitize_input(article.get('title', '')),
                            'news_text': sanitize_input(article.get('news_text', '')),
                            'link': article_url
                        }
                        
                        # Generate a unique kg_id based on the article URL
                        kg_id = generate_kg_id(article_data['link'])  # Ensure kg_id is assigned here
                        
                        # Check cache first
                        cached_kg = get_cached_kg(kg_id)
                        if cached_kg:
                            logger.info(f"Retrieved KG {kg_id} from cache")
                            kg_response = {
                                "type": "kg_update",
                                "kg_id": kg_id,
                                "data": cached_kg,
                                "progress": {
                                    "current": idx + 1,
                                    "total": len(articles_df)
                                }
                            }
                            response_data = f"data: {json.dumps(kg_response)}\n\n"
                            yield response_data
                            continue
                        
                        kg = kg_generator.generate_kg(article_data['news_text'], article_data)
                        logger.info(f"Generated KG: {kg.summary}")
                        if kg:
                            kg_response = {
                                "type": "kg_update",
                                "kg_id": kg_id,
                                "data": {
                                    "nodes": [node.to_dict() for node in kg.nodes],
                                    "edges": [edge.to_dict() for edge in kg.edges],
                                    "article": Article(
                                        title=article_data['title'],
                                        # summary=f"{article_data['news_text'][:200]}...",
                                        summary=kg.summary,
                                        url=article_data['link']
                                    ).model_dump()
                                },
                                "progress": {
                                    "current": idx + 1,
                                    "total": len(articles_df)
                                }
                            }
                            # time.sleep(0.1)  # Rate limit to 10 updates per second
                            yield f"data: {json.dumps(kg_response)}\n\n"
                            
                            # Cache the generated KG
                            cache_kg(kg_id, kg_response['data'])
                    
                    except Exception as e:
                        error_response = {
                            "type": "error",
                            "data": str(e)
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        continue
                
                # Final response after processing all articles
                final_response = {
                    "type": "complete",
                    "kg_id": kg_id,
                    "data": {
                        "message": "Knowledge graph generation complete"
                    }
                }
                yield f"data: {json.dumps(final_response)}\n\n"
            
            except Exception as e:
                logger.error(f"Error in stream_response: {str(e)}", exc_info=True)
                yield f'data: {{"type": "error", "message": "{str(e)}"}}\n\n'

        response = StreamingHttpResponse(
            streaming_content=stream_response(),
            content_type='text/event-stream',
            status=200,
        )
        
        # Add required headers for proper streaming
        response['Cache-Control'] = 'no-cache, no-transform'
        response['X-Accel-Buffering'] = 'no'
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = '*'
        
        # Ensure Django doesn't buffer the response
        response._iterator = iter(response.streaming_content)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate knowledge graphs: {str(e)}", exc_info=True)
        error_response = HttpResponse(
            content=json.dumps({"error": str(e)}),
            content_type='application/json',
            status=500
        )
        error_response['Access-Control-Allow-Origin'] = '*'
        error_response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        error_response['Access-Control-Allow-Headers'] = '*'
        return error_response

@csrf_exempt
@api.post("/enrich", response=KnowledgeGraphResponseSchema)
def enrich_kg(request: HttpRequest, data: EnrichKGRequest):
    """Enrich multiple knowledge graphs from cache and session storage"""
    try:
        enricher = KGEnricher(preprocess=True)
        kgs_to_enrich = []
        
        logger.info(f"Attempting to enrich KGs with IDs: {data.kg_ids}")
        
        for kg_id in data.kg_ids:
            # Try cache first
            cached_kg = get_cached_kg(kg_id)
            if cached_kg:
                logger.info(f"Retrieved KG {kg_id} from cache")
                # Debug the structure of cached data
                logger.info(f"Cached KG structure: {cached_kg.keys()}")
                logger.debug(f"Cached KG data: {json.dumps(cached_kg, indent=2)}")
                
                try:
                    # Check if 'data' wrapper exists
                    if 'data' in cached_kg:
                        cached_kg = cached_kg['data']
                        logger.debug("Unwrapped 'data' from cached KG")
                    
                    # Handle different article structures
                    articles_data = []
                    if 'articles' in cached_kg:
                        articles_data = cached_kg['articles']
                    elif 'article' in cached_kg:
                        articles_data = [cached_kg['article']]
                        logger.info(f"Using single article data: {cached_kg['article']}")
                    
                    logger.debug(f"Articles data: {articles_data}")
                    
                    kg = KnowledgeGraph(
                        nodes=[KGNode(**node) for node in cached_kg.get('nodes', [])],
                        edges=[KGEdge(**edge) for edge in cached_kg.get('edges', [])],
                        # articles=[Article(**article) for article in articles_data],
                        articles = [Article(**cached_kg['article'])],
                        summary=cached_kg.get('summary', '')
                    )
                    kgs_to_enrich.append(kg)
                    logger.info(f"Converted cached KG {kg_id}: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
                    continue
                except Exception as e:
                    logger.error(f"Error converting cached KG {kg_id}: {str(e)}")
                    logger.error(f"Error details:", exc_info=True)
            
            # Fallback to session if not in cache
            session_kgs = request.session.get('kgs', {})
            if kg_id in session_kgs:
                try:
                    kg_data = session_kgs[kg_id]
                    kg = KnowledgeGraph(
                        nodes=[KGNode(**node) for node in kg_data['nodes']],
                        edges=[KGEdge(**edge) for edge in kg_data['edges']],
                        articles=[Article(**article) for article in kg_data['article']],
                        summary=kg_data.get('summary', '')
                    )
                    kgs_to_enrich.append(kg)
                    logger.info(f"Retrieved KG {kg_id} from session: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
                    
                    # Update cache for future requests
                    cache_kg(kg_id, kg_data)
                    continue
                except Exception as e:
                    logger.error(f"Error converting session KG {kg_id}: {str(e)}")
            
            logger.warning(f"KG with ID {kg_id} not found in cache or session")
        
        if not kgs_to_enrich:
            return api.create_response(
                request,
                {"error": "No valid knowledge graphs found to enrich"},
                status=404
            )
        
        # Process and enrich the KGs
        enriched_kg = enricher.process_and_enrich_kgs(kgs_to_enrich)
        logger.info(f"Enriched KG created with {len(enriched_kg.nodes)} nodes and {len(enriched_kg.edges)} edges")
        
        # Generate new ID for enriched KG
        enriched_kg_id = str(uuid.uuid4())
        
        # Convert enriched KG to dictionary format
        enriched_kg_dict = {
            "nodes": [node.to_dict() for node in enriched_kg.nodes],
            "edges": [edge.to_dict() for edge in enriched_kg.edges],
            "articles": [article.model_dump() for article in enriched_kg.articles],
            "summary": enriched_kg.summary,
            "created_at": time.time()
        }
        
        # Store in both cache and session for reliability
        cache_kg(enriched_kg_id, enriched_kg_dict)
        request.session['kgs'] = request.session.get('kgs', {})
        request.session['kgs'][enriched_kg_id] = enriched_kg_dict
        request.session.modified = True
        
        logger.info(f"Stored enriched KG {enriched_kg_id} in cache and session")
        
        logger.info(f"Stored enriched KG with ID {enriched_kg_id} in session and cache")
        
        # Return enriched KG and its ID
        response = {
            "kg_id": enriched_kg_id,
            "kg": enriched_kg_dict
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to enrich knowledge graphs: {str(e)}", exc_info=True)
        return api.create_response(
            request,
            {"error": f"Failed to enrich knowledge graphs: {str(e)}"},
            status=500
        )

@csrf_exempt
@api.post("/query", response=QueryResultSchema)
def query_kg(request: HttpRequest, query_data: QueryRequest):
    """Query a knowledge graph with natural language"""
    try:
        logger.info(f"Received query request for KG {query_data.kg_id}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Query Data: {query_data.dict()}")  # Log incoming query data

        # Log topN and numHops values
        logger.info(f"Top N: {query_data.top_n}, Connected Hops: {query_data.connected_hops}")

        # Try cache first
        cached_kg = get_cached_kg(query_data.kg_id)
        if cached_kg:
            logger.info(f"Retrieved KG {query_data.kg_id} from cache")
            kg_data = cached_kg
        else:
            # Fallback to session
            session_kgs = request.session.get('kgs', {})
            if query_data.kg_id not in session_kgs:
                logger.warning(f"Knowledge graph with ID {query_data.kg_id} not found in cache or session")
                return api.create_response(
                    request,
                    {"error": f"Knowledge graph with ID {query_data.kg_id} not found in cache or session"},
                    status=404
                )
            kg_data = session_kgs[query_data.kg_id]
            logger.info(f"Retrieved KG {query_data.kg_id} from session")

            # Update cache for future requests
            cache_kg(query_data.kg_id, kg_data)

        # Initialize query processor
        query_processor = QueryProcessor()

        try:
            # Convert stored dict back to KnowledgeGraph object
            input_kg = KnowledgeGraph(
                nodes=[KGNode(**node) for node in kg_data['nodes']],
                edges=[KGEdge(**edge) for edge in kg_data['edges']],
                articles=[Article(**article) for article in kg_data['articles']],
                summary=kg_data['summary']
            )
            logger.info(f"Converted KG: {len(input_kg.nodes)} nodes, {len(input_kg.edges)} edges")

            # Process selected nodes
            selected_nodes = None
            if query_data.selected_node_ids:
                # Create a mapping of node IDs to actual nodes
                node_map = {node.id: node for node in input_kg.nodes}
                # Filter valid node IDs and get corresponding nodes
                selected_nodes = [
                    node_map[node_id] 
                    for node_id in query_data.selected_node_ids 
                    if node_id in node_map
                ]
                logger.info(f"Found {len(selected_nodes)} valid nodes out of {len(query_data.selected_node_ids)} selected IDs")

                if not selected_nodes:
                    logger.warning("None of the provided node IDs were found in the graph")

        except Exception as e:
            logger.error(f"Error converting KG data: {str(e)}")
            return api.create_response(
                request,
                {"error": "Invalid knowledge graph data structure"},
                status=400
            )

        try:
            # Process the query with updated parameters
            query_results = query_processor.process_query(
                query=query_data.query,
                merged_kg=input_kg,
                selected_nodes=selected_nodes,  # Pass the selected nodes
                top_n=query_data.top_n,
                connected_hops=query_data.connected_hops,
                find_text_similar=True,
                find_entity_similar=True,
                find_network_similar=bool(selected_nodes)  # Only enable if nodes are selected
            )

            logger.info(f"Query processed successfully with {len(query_results['similar_by_text'])} text matches, "
                        f"{len(query_results['similar_by_entities'])} entity matches, "
                        f"{len(query_results['similar_by_network'])} network matches")

            # Generate comprehensive answer
            answer = query_processor.answer_query(
                query=query_data.query,
                query_results=query_results,
                kg=input_kg
            )

            logger.info("Generated comprehensive answer")

            # Convert results to serializable format
            formatted_response = {
                "query_results": {
                    "similar_by_text": [
                        {"node": node.to_dict(), "similarity": float(similarity)}
                        for node, similarity in query_results.get("similar_by_text", [])
                    ],
                    "similar_by_entities": [
                        {"node": node.to_dict(), "similarity": float(similarity)}
                        for node, similarity in query_results.get("similar_by_entities", [])
                    ],
                    "similar_by_network": [
                        {"node": node.to_dict(), "similarity": float(similarity)}
                        for node, similarity in query_results.get("similar_by_network", [])
                    ],
                    "connected_entities": [
                        {"node": node.to_dict(), "similarity": float(similarity)}
                        for node, similarity in query_results.get("connected_entities", [])
                    ]
                },
                "answer": answer,
                "selected_nodes": [node.to_dict() for node in (selected_nodes or [])]  # Include selected nodes in response
            }

            # Log answer metrics
            logger.info(f"Answer generated with confidence scores: "
                        f"text_similarity={answer['metadata']['confidence_scores']['text_similarity']:.2f}, "
                        f"entity_similarity={answer['metadata']['confidence_scores']['entity_similarity']:.2f}, "
                        f"network_similarity={answer['metadata']['confidence_scores'].get('network_similarity', 0):.2f}")

            return formatted_response

        except Exception as e:
            logger.error(f"Error processing query or generating answer: {str(e)}")
            return api.create_response(
                request,
                {"error": f"Error processing query: {str(e)}"},
                status=500
            )

    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}", exc_info=True)
        return api.create_response(
            request,
            {"error": f"Failed to process query: {str(e)}"},
            status=500
        )
