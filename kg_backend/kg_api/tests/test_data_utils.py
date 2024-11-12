import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
import requests
from kg_api.data_utils import (
    get_json_data,
    fetch_articles,
    get_company_info
)

# Mock fixtures for reuse across tests
@pytest.fixture
def mock_article():
    """
    Mock article data that simulates a response from financial news APIs.
    Includes essential fields like title, text, date, and ticker symbol.
    """
    return {
        "title": "Test Article",
        "text": "This is a test article about AAPL",
        "publishedDate": "2024-01-01",
        "url": "https://example.com/article",
        "symbol": "AAPL"
    }

@pytest.fixture
def mock_company():
    """
    Mock company data that simulates a response from company info API.
    Includes basic company information fields.
    """
    return {
        "symbol": "AAPL",
        "companyName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide."  # Added description field
    }

class TestGetJsonData:
    """Tests for the basic JSON fetching functionality"""
    
    @patch('requests.get')
    def test_successful_fetch(self, mock_get):
        """
        Test successful API response parsing.
        Should return parsed JSON when API returns valid data.
        """
        # Setup mock response
        mock_get.return_value.json.return_value = {"data": "test"}
        mock_get.return_value.raise_for_status = Mock()
        
        # Execute and verify
        result = get_json_data("https://test.com/api")
        assert result == {"data": "test"}
        mock_get.assert_called_once_with("https://test.com/api")

    @patch('requests.get')
    def test_api_error(self, mock_get):
        """
        Test handling of API errors.
        Should return None when API request fails.
        """
        # Simulate API error
        mock_get.side_effect = requests.RequestException("API Error")
        
        # Execute and verify
        result = get_json_data("https://test.com/api")
        assert result is None

class TestFetchArticles:
    """Tests for article fetching functionality"""
    
    @patch('requests.get')
    def test_successful_fetch(self, mock_get, mock_article):
        """
        Test successful article fetching.
        Should return DataFrame with processed article data.
        """
        # Setup mock response
        mock_get.return_value.json.return_value = [mock_article]
        mock_get.return_value.raise_for_status = Mock()
        
        # Execute
        df = fetch_articles(["AAPL"], window=1, limit=1)
        
        # Verify
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1
        assert "title" in df.columns
        assert "news_text" in df.columns
        assert "date" in df.columns
        assert "ticker" in df.columns

    @patch('requests.get')
    def test_rate_limit_handling(self, mock_get):
        """
        Test handling of rate limiting.
        Should return empty DataFrame when rate limited.
        """
        # Simulate rate limit response
        mock_get.side_effect = requests.exceptions.HTTPError("429 Rate Limited")
        
        # Execute and verify
        df = fetch_articles(["AAPL"])
        assert len(df) == 0

class TestGetCompanyInfo:
    """Tests for company information fetching"""
    
    @patch('requests.get')
    def test_successful_fetch(self, mock_get, mock_company):
        """
        Test successful company info fetching.
        Should return DataFrame with company information.
        """
        # Setup mock response
        mock_get.return_value.json.return_value = [mock_company]
        mock_get.return_value.raise_for_status = Mock()
        
        # Execute
        df = get_company_info(["AAPL"])
        
        # Verify
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"
        assert "sector" in df.columns
        assert "industry" in df.columns

    @patch('requests.get')
    def test_invalid_ticker(self, mock_get):
        """
        Test handling of invalid ticker symbols.
        Should return empty DataFrame for invalid tickers.
        """
        # Setup mock empty response
        mock_get.return_value.json.return_value = []
        mock_get.return_value.raise_for_status = Mock()
        
        # Execute and verify
        df = get_company_info(["INVALID"])
        assert len(df) == 0

@pytest.mark.integration
class TestIntegration:
    """Integration tests for data fetching pipeline"""
    
    def test_full_pipeline(self):
        """
        Test the complete data fetching pipeline.
        Verifies that company info and articles can be fetched together.
        """
        # Test with real API calls
        tickers = ["AAPL"]
        
        # Fetch company info
        companies = get_company_info(tickers)
        assert len(companies) > 0
        
        # Fetch articles
        articles = fetch_articles(tickers, window=1, limit=5)
        assert len(articles) > 0
