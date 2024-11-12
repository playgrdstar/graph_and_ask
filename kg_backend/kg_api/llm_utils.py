import os
import logging
import requests
from pathlib import Path
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import instructor

# Load environment variables
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_DIR = os.path.join(BASE_DIR, '.env')
load_dotenv(ENV_DIR)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_READ_API_KEY")

HF_INFERENCE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"
HF_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomHuggingFaceEndpoint:
    def __init__(self, repo_id: str, api_key: str, **kwargs):
        self.api_url = f"https://api-inference.huggingface.co/models/{repo_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.default_params = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.95),
        }

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=60))
    def invoke(self, prompt: str, **kwargs) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {**self.default_params, **kwargs}
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"]

class CustomHuggingFaceInferenceAPIEmbeddings:
    def __init__(self, api_key: str, model_name: str):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=60))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts})
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=60))
    def embed_text(self, text: str) -> List[float]:
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": [text]})
        response.raise_for_status()
        return response.json()[0]

class LLMProcessor:
    def __init__(self):
        self.openai_client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))
        self.hf_endpoint = CustomHuggingFaceEndpoint(HF_INFERENCE_MODEL_NAME, HF_API_KEY)
        self.text_embedder = CustomHuggingFaceInferenceAPIEmbeddings(HF_API_KEY, HF_EMBEDDING_MODEL_NAME)

    def generate_text_embedding(self, text: str) -> List[float]:
        return self.text_embedder.embed_text(text)

    def extract_entities(self, text: str) -> List[str]:
        prompt = f"Extract the main entities from the following text. Return them as a comma-separated list:\n\n{text}"
        response = self.hf_endpoint.invoke(prompt)
        return [entity.strip() for entity in response.split(',')]

    def infer_related_entities(self, entities: List[str]) -> List[str]:
        entities_str = ', '.join(entities)
        prompt = f"Given the following entities: {entities_str}\n\nInfer and list 5 related entities. Return them as a comma-separated list:"
        response = self.hf_endpoint.invoke(prompt)
        return [entity.strip() for entity in response.split(',')]

    def generate_entity_embeddings(self, entities: List[str]) -> List[List[float]]:
        return [self.generate_text_embedding(entity) for entity in entities]


