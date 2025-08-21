# vision_correction_handler.py
# Add this to your LLMs/src directory

import os
import base64
import logging
from typing import Dict, Any, Optional
import requests
from datetime import datetime

from llm.llm_factory import LLMFactory
from llm.openrouter_llm import OpenRouterLLM

logger = logging.getLogger(__name__)


class VisionCorrectionHandler:
    """
    Handler for vision-based OCR correction using OpenRouter LLMs.
    """

    # Vision-capable models on OpenRouter
    VISION_MODELS = {
        "qwen-2.5-vl-72b": "qwen/qwen-2-vl-72b-instruct",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gemini-2.5-pro": "google/gemini-pro-vision",
        "claude-sonnet-4": "anthropic/claude-3-sonnet-20240229",
        "internalvl3": "internlm/internlm-xcomposer2-vl-7b"
    }

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    def correct_with_vision(
            self,
            image_data: str,
            ocr_text: str,
            model_name: str,
            dataset: str,
            prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform vision-based OCR correction.

        Args:
            image_data: Base64 encoded image
            ocr_text: OCR output text
            model_name: Vision model to use
            dataset: Dataset name for context
            prompt: Custom prompt (optional)

        Returns:
            Dictionary with corrected text and metadata
        """
        try:
            # Get the OpenRouter model ID
            model_id = self.VISION_MODELS.get(model_name)
            if not model_id:
                raise ValueError(f"Unknown vision model: {model_name}")

            # Default prompt if not provided
            if not prompt:
                prompt = self._get_default_vision_prompt(ocr_text, dataset)

            # Make request to OpenRouter
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "HTR_PostProcessing_Vision"
            }

            # Prepare the message with image
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }]

            data = {
                "model": model_id,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.0,
                "top_p": 1.0
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                corrected_text = self._extract_corrected_text(
                    result['choices'][0]['message']['content'],
                    ocr_text
                )

                # Calculate confidence based on response
                confidence = self._calculate_confidence(ocr_text, corrected_text)

                return {
                    'correctedText': corrected_text,
                    'confidence': confidence,
                    'model': model_name,
                    'usage': result.get('usage', {}),
                    'processingTime': 0  # Would need to track actual time
                }
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return {
                    'correctedText': ocr_text,
                    'confidence': 0,
                    'error': f"API error: {response.status_code}"
                }

        except Exception as e:
            logger.error(f"Vision correction error: {str(e)}")
            return {
                'correctedText': ocr_text,
                'confidence': 0,
                'error': str(e)
            }

    def _get_default_vision_prompt(self, ocr_text: str, dataset: str) -> str:
        """
        Generate default vision prompt based on dataset.
        """
        century_map = {
            'bentham': '19th',
            'washington': '18th',
            'iam': '20th'
        }

        century = century_map.get(dataset, '19th')

        return f"""You are an expert in {century}-century handwriting recognition and OCR correction.

I'm providing you with:
1. An image of handwritten text
2. OCR output: "{ocr_text}"

Please carefully analyze the handwritten text in the image and correct any errors in the OCR output.
Consider:
- Letter shapes and writing style typical of {century}-century documents
- Context and word likelihood
- Punctuation and spacing

Guidelines:
- Preserve original punctuation and hyphenation
- Don't add content not visible in the image
- Focus on accuracy over interpretation

Return ONLY the corrected text, without any explanation or commentary."""

    def _extract_corrected_text(self, response: str, original: str) -> str:
        """
        Extract corrected text from model response.
        """
        # Remove common prefixes/suffixes
        response = response.strip()

        # If response is wrapped in quotes, extract it
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]

        # Remove any explanation markers
        if "Corrected text:" in response:
            response = response.split("Corrected text:")[-1].strip()

        # If response contains multiple lines, take the first substantial one
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 5 and not any(
                    phrase in line.lower() for phrase in
                    ['the corrected', 'here is', 'based on', 'analyzing']
            ):
                return line

        # If nothing found, return the full response or original
        return response if len(response) < 2 * len(original) else original

    def _calculate_confidence(self, original: str, corrected: str) -> float:
        """
        Calculate confidence score based on changes made.
        """
        if original == corrected:
            return 50.0  # No changes made

        # Calculate edit distance ratio
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, original, corrected).ratio()

        # High similarity but with changes suggests confident corrections
        if 0.7 < ratio < 0.95:
            return 85.0 + (ratio - 0.7) * 40  # 85-95%
        elif ratio >= 0.95:
            return 95.0
        else:
            return 70.0 + ratio * 20  # 70-84%


# Integration function for Flask endpoint
def process_vision_correction(
        image_data: str,
        ocr_text: str,
        llm_model: str,
        dataset: str,
        prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a vision correction request.

    This function can be called from your Flask endpoint.
    """
    handler = VisionCorrectionHandler()
    return handler.correct_with_vision(
        image_data=image_data,
        ocr_text=ocr_text,
        model_name=llm_model,
        dataset=dataset,
        prompt=prompt
    )
