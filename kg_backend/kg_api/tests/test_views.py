from django.test import TestCase, Client
from unittest.mock import patch
import pandas as pd
from kg_api.kg_utils import KnowledgeGraph, KGNode, KGEdge

class TestKGAPIViews(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Initialize Django Test Client
        cls.client = Client()

    @patch('kg_api.views.fetch_articles')
    @patch('kg_api.views.KGGenerator')
    def test_generate_kg(self, mock_kg_generator, mock_fetch_articles):
        """Test the generate KG endpoint with minimal valid data"""
        # Mock the fetch_articles to return a DataFrame
        mock_fetch_articles.return_value = pd.DataFrame({
            'title': ['Test Article'],
            'news_text': ['This is a test news article.'],
            'link': ['http://test.com/article']
        })
        
        # Mock KGGenerator to return a simple KnowledgeGraph instance
        mock_generator_instance = mock_kg_generator.return_value
        mock_generator_instance.generate_kg.return_value = KnowledgeGraph(
            nodes=[],
            edges=[],
            articles=[],
            summary="Test summary"
        )

        payload = {
            "tickers": "AAPL,GOOGL",
            "window": 1,
            "limit": 2
        }
        
        # Adjust the URL prefix based on your urls.py configuration
        response = self.client.post("/api/generate", data=payload, content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("kg_id", response_data)
        self.assertIn("kg", response_data)

    @patch('kg_api.views.KGEnricher')
    def test_enrich_kg(self, mock_enricher):
        """Test the enrich KG endpoint"""
        # Store a mock KG in the session
        test_kg_id = "test-kg-id"
        session = self.client.session
        session['kgs'] = {
            test_kg_id: {
                "nodes": [],
                "edges": [],
                "articles": [],
                "summary": "Test summary"
            }
        }
        session.save()

        # Mock the enricher to return a KnowledgeGraph instance
        mock_enricher_instance = mock_enricher.return_value
        mock_enricher_instance.process_and_enrich_kgs.return_value = KnowledgeGraph(
            nodes=[],
            edges=[],
            articles=[],
            summary="Enriched summary"
        )

        enrich_payload = {
            "kg_ids": [test_kg_id]
        }
        
        # Adjust the URL prefix based on your urls.py configuration
        response = self.client.post("/api/enrich", data=enrich_payload, content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("kg_id", response_data)
        self.assertIn("kg", response_data)

    @patch('kg_api.views.QueryProcessor')
    def test_query_kg(self, mock_query_processor):
        """Test the query KG endpoint"""
        # Store a mock KG in the session
        test_kg_id = "test-kg-id"
        session = self.client.session
        session['kgs'] = {
            test_kg_id: {
                "nodes": [],
                "edges": [],
                "articles": [],
                "summary": "Test summary"
            }
        }
        session.save()

        # Mock the query processor
        mock_processor_instance = mock_query_processor.return_value
        mock_processor_instance.process_query.return_value = {
            "similar_by_text": [],
            "similar_by_entities": [],
            "connected_entities": []
        }
        mock_processor_instance.answer_query.return_value = {
            "answer": "Test answer",
            "evidence": [],
            "connections": [],
            "sources": [],
            "key_entities": [],
            "metadata": {
                "confidence_scores": {
                    "text_similarity": 0.8,
                    "entity_similarity": 0.7,
                    "connection_strength": 0.5
                }
            }
        }

        query_payload = {
            "kg_id": test_kg_id,
            "query": "What are the main developments?",
            "top_n": 5,
            "connected_hops": 1
        }
        
        # Adjust the URL prefix based on your urls.py configuration
        response = self.client.post("/api/query", data=query_payload, content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("query_results", response_data)
        self.assertIn("answer", response_data)
