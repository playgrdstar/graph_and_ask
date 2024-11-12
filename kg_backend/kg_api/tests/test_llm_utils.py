import pytest
from unittest.mock import Mock, patch
import numpy as np
import requests
from kg_api.llm_utils import (
    LLMProcessor,
    CustomHuggingFaceEndpoint,
    CustomHuggingFaceInferenceAPIEmbeddings,
    HF_INFERENCE_MODEL_NAME,
    HF_EMBEDDING_MODEL_NAME
)

# Mock responses for reuse across tests
@pytest.fixture
def mock_hf_text_response():
    """Mock response from Hugging Face text generation API"""
    return [{"generated_text": "This is a test response"}]

@pytest.fixture
def mock_hf_embedding_response():
    """Mock response from Hugging Face embedding API"""
    return [[0.1, 0.2, 0.3, 0.4, 0.5]]  # 5-dimensional embedding for testing

class TestCustomHuggingFaceEndpoint:
    """Tests for the Hugging Face text generation endpoint"""
    
    @pytest.fixture
    def hf_endpoint(self):
        """Create a test endpoint instance"""
        return CustomHuggingFaceEndpoint(
            repo_id=HF_INFERENCE_MODEL_NAME,
            api_key="test_key"
        )

    def test_init(self, hf_endpoint):
        """
        Test endpoint initialization.
        Verifies API URL, headers, and default parameters.
        """
        assert hf_endpoint.api_url == f"https://api-inference.huggingface.co/models/{HF_INFERENCE_MODEL_NAME}"
        assert hf_endpoint.headers["Authorization"] == "Bearer test_key"
        assert hf_endpoint.default_params["max_new_tokens"] == 512
        assert hf_endpoint.default_params["temperature"] == 0.1
        assert hf_endpoint.default_params["top_p"] == 0.95

    @patch('requests.post')
    def test_invoke_success(self, mock_post, hf_endpoint, mock_hf_text_response):
        """
        Test successful text generation.
        Should return generated text from the model.
        """
        mock_post.return_value.json.return_value = mock_hf_text_response
        mock_post.return_value.raise_for_status = Mock()

        response = hf_endpoint.invoke("Test prompt")
        assert response == mock_hf_text_response[0]["generated_text"]
        
        # Verify the request payload
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "inputs" in call_kwargs["json"]
        assert "parameters" in call_kwargs["json"]

class TestCustomHuggingFaceEmbeddings:
    """Tests for the Hugging Face embeddings API"""
    
    @pytest.fixture
    def embeddings(self):
        """Create a test embeddings instance"""
        return CustomHuggingFaceInferenceAPIEmbeddings(
            api_key="test_key",
            model_name=HF_EMBEDDING_MODEL_NAME
        )

    @patch('requests.post')
    def test_embed_text(self, mock_post, embeddings, mock_hf_embedding_response):
        """
        Test text embedding generation.
        Should return numpy array of correct dimensions.
        """
        mock_post.return_value.json.return_value = mock_hf_embedding_response
        mock_post.return_value.raise_for_status = Mock()

        embedding = embeddings.embed_text("test text")
        
        assert len(embedding) == 5
        assert all(isinstance(x, float) for x in embedding)
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_embed_documents(self, mock_post, embeddings, mock_hf_embedding_response):
        """
        Test batch document embedding generation.
        Should return list of embeddings.
        """
        mock_post.return_value.json.return_value = mock_hf_embedding_response
        mock_post.return_value.raise_for_status = Mock()

        embeddings_result = embeddings.embed_documents(["test text 1", "test text 2"])
        
        assert len(embeddings_result) == 1  # Based on mock response
        assert len(embeddings_result[0]) == 5
        mock_post.assert_called_once()

class TestLLMProcessor:
    """Tests for the main LLM processing class"""
    
    @pytest.fixture
    def llm_processor(self):
        """Create a test LLM processor instance"""
        with patch('openai.OpenAI'), patch('instructor.patch'):
            return LLMProcessor()

    @patch('kg_api.llm_utils.CustomHuggingFaceInferenceAPIEmbeddings.embed_text')
    def test_generate_text_embedding(self, mock_embed, llm_processor):
        """
        Test text embedding generation workflow.
        Should process text and return embeddings.
        """
        mock_embed.return_value = [0.1, 0.2, 0.3]
        
        embedding = llm_processor.generate_text_embedding("test text")
        
        assert len(embedding) == 3
        assert all(isinstance(x, float) for x in embedding)
        mock_embed.assert_called_once_with("test text")

    @patch('kg_api.llm_utils.CustomHuggingFaceEndpoint.invoke')
    def test_extract_entities(self, mock_invoke, llm_processor):
        """
        Test entity extraction from text.
        Should return list of extracted entities.
        """
        mock_invoke.return_value = "Entity1, Entity2, Entity3"
        
        entities = llm_processor.extract_entities("test text")
        
        assert len(entities) == 3
        assert all(isinstance(e, str) for e in entities)
        mock_invoke.assert_called_once()

    @patch('kg_api.llm_utils.CustomHuggingFaceEndpoint.invoke')
    def test_infer_related_entities(self, mock_invoke, llm_processor):
        """
        Test related entity inference.
        Should return list of related entities.
        """
        mock_invoke.return_value = "Related1, Related2, Related3"
        
        entities = llm_processor.infer_related_entities(["Entity1", "Entity2"])
        
        assert len(entities) == 3
        assert all(isinstance(e, str) for e in entities)
        mock_invoke.assert_called_once()

@pytest.mark.integration
class TestIntegration:
    """Integration tests for LLM processing pipeline"""
    
    def test_full_llm_pipeline(self):
        """
        Test the complete LLM processing pipeline.
        Requires actual API access.
        """
        processor = LLMProcessor()
        text = "Apple and Google are developing new AI technologies."
        
        # Test complete pipeline
        embedding = processor.generate_text_embedding(text)
        entities = processor.extract_entities(text)
        related = processor.infer_related_entities(entities)
        
        assert len(embedding) > 0
        assert len(entities) > 0
        assert len(related) > 0
