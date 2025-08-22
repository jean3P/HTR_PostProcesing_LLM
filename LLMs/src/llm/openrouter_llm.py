# repos/HTR_PostProcesing_LLM/LLMs/src/llm/openrouter_llm.py

import os
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from llm.base_llm import BaseLLM
import requests

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost:3000")
YOUR_APP_NAME = os.getenv("YOUR_APP_NAME", "HTR_PostProcessing_LLM")


class OpenRouterLLM(BaseLLM):
    """
    OpenRouter LLM implementation supporting multiple models:
    - Qwen 2.5 VL 72B
    - PHI-4
    - InternalVL3
    - Mistral
    - Gemini 2.5 Pro
    - GPT 4.1-mini
    - Claude Sonnet 4
    """

    # Model mapping to OpenRouter model IDs
    MODEL_MAPPING = {
        "qwen-2.5-vl-72b": "qwen/qwen-2.5-72b-instruct",
        "phi-4": "microsoft/phi-4",
        "internalvl3": "opengvlab/internvl3-14b",  # Adjust based on actual OpenRouter ID
        "mistral": "mistralai/mistral-7b-instruct",
        "gemini-2.5-pro": "google/gemini-2.5-pro",  # Adjust based on actual availability
        "gpt-4.1-mini": "openai/gpt-4.1-mini",
        "claude-sonnet-4": "anthropic/claude-sonnet-4"
    }

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_key = OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.stream = False

        # Map the model name to OpenRouter model ID
        self.openrouter_model_id = self.MODEL_MAPPING.get(
            model_name.lower(),
            model_name  # Use as-is if not in mapping
        )

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    def create_headers(self) -> Dict[str, str]:
        """Create headers for OpenRouter API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": YOUR_SITE_URL,
            "X-Title": YOUR_APP_NAME,
            "Content-Type": "application/json"
        }

    def make_request(
            self,
            prompt: str,
            max_tokens: int = 100,
            temperature: float = 0.0,
            top_p: float = 1.0,
            max_retries: int = 5,
            initial_retry_delay: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Make a request to OpenRouter API with retry logic

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            max_retries: Maximum number of retry attempts
            initial_retry_delay: Initial delay between retries (seconds)

        Returns:
            API response or None if failed
        """
        headers = self.create_headers()

        data = {
            "model": self.openrouter_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": self.stream
        }

        retry_delay = initial_retry_delay

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=60  # 60 second timeout
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2

            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2

        return None
