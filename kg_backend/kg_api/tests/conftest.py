import os
import django
import pytest
from django.conf import settings
from ninja.testing import TestClient
from kg_api.views import api

# Global API client instance
_api_client = None

# Set up Django settings for tests
def pytest_configure():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'kg_project.test_settings')
    django.setup()

@pytest.fixture(scope="session")
def api_client():
    global _api_client
    if _api_client is None:
        _api_client = TestClient(api)
    return _api_client

@pytest.fixture(scope="function")
def db_session(django_db_setup, django_db_blocker):
    with django_db_blocker.unblock():
        yield

@pytest.fixture
def mock_request(db_session):
    from django.test import RequestFactory
    from django.contrib.sessions.middleware import SessionMiddleware
    
    factory = RequestFactory()
    request = factory.post('/api/generate')
    middleware = SessionMiddleware(lambda x: None)
    middleware.process_request(request)
    request.session.save()
    return request

@pytest.fixture
def sample_kg_data():
    return {
        "nodes": [
            {
                "id": "node1",
                "type": "COMPANY",
                "detailed_type": "TECH_COMPANY",
                "summary": "Apple Inc.",
                "articles": ["article1"]
            }
        ],
        "edges": [
            {
                "source": "node1",
                "target": "node2",
                "label": "DEVELOPS",
                "count": 1
            }
        ],
        "articles": [
            {
                "title": "Test Article",
                "summary": "Test summary",
                "url": "https://example.com"
            }
        ],
        "summary": "Test knowledge graph"
    }
