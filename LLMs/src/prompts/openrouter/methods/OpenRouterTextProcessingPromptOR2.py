# repos/HTR_PostProcesing_LLM/LLMs/src/prompts/openrouter/methods/OpenRouterTextProcessingPromptOR2.py

import re
from prompts.openrouter.OpenRouterProcessingStrategy import OpenRouterProcessingStrategy
from prompts.openrouter.methods.openrouter_response_parser import OpenRouterResponseParser

from utils.aux_processing import get_time


class OpenRouterTextProcessingPromptOR2(OpenRouterProcessingStrategy):
    """
    PromptOR 2: Domain-specific guidance
    Act as an {century}-century document analyst specializing in OCR correction.
    """

    def __init__(self):
        self.suggestions_memory = {}
        self.parser = OpenRouterResponseParser()

    def get_name_method(self):
        return "promptOR_2"

    def check_and_correct_text_line(self, text_line, train_set_lines, llm_instance, name_dataset, logger):
        logger.info(
            f"Start processing text line with PromptOR 2: '{text_line}' -- from {get_time(name_dataset)}-century")

        corrected_text = self.correct_with_prompts(
            text_line,
            llm_instance,
            name_dataset,
            logger
        )

        logger.info(f"Text after correction: {corrected_text}")

        confidence, justification = self.evaluate_corrected_text(
            text_line,
            corrected_text,
            llm_instance,
            logger
        )

        if confidence and justification:
            logger.info(f"Confidence - {confidence}, Justification - {justification}")
        else:
            logger.info(f"Could not evaluate the corrected text for '{corrected_text}'")

        logger.info(f"Finished processing text line: {text_line} ===> {corrected_text}")
        return corrected_text, confidence, justification

    def correct_with_prompts(self, ocr_text, llm_instance, name_dataset, logger):
        # PromptOR 2: Domain-specific version
        system_prompt = (
            f"Act as an {get_time(name_dataset)}-century document analyst specializing in OCR correction. "
            f"Your task is to correct OCR errors in words or numbers.\n"
            f"Guidelines:\n"
            f"1. Ensure corrections accurately reflect the {get_time(name_dataset)}-century language and conventions.\n"
            f"\nBased on the guidelines, please analyze the following text line and provide "
            f"the corrected version: '{ocr_text}'. Then the corrected text line is:"
        )

        response = llm_instance.make_request(
            prompt=system_prompt,
            max_tokens=len(ocr_text.split()) * 2 + 50,
            temperature=0.0
        )

        if response is None:
            logger.error("Failed to retrieve a valid response from OpenRouter.")
            return ocr_text

        try:
            raw_response = response['choices'][0]['message']['content'].strip()

            # Use the parser to extract the corrected text
            corrected_text = self.parser.extract_corrected_text(raw_response, ocr_text, logger)

            # Additional cleaning using parser's clean_text_line method
            corrected_text = self.parser.clean_text_line(ocr_text, corrected_text)

            return corrected_text

        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response structure: {e}")
            return ocr_text

    def correct_with_suggestions(self, ocr_text, suggestions, llm_instance, name_dataset, logger):
        return self.correct_with_prompts(ocr_text, llm_instance, name_dataset, logger)

    def correct_duplicated_words(self, text_line, llm_instance, logger):
        return text_line

    def evaluate_corrected_text(self, original_text_line, corrected_text_line, llm_instance, logger):
        """Evaluate the corrected text and return confidence and justification."""
        logger.info(f"Evaluating the corrected text: '{corrected_text_line}' for the original: '{original_text_line}'")

        # Return default values to skip API call
        confidence = "85"  # Default confidence score as string
        justification = "Evaluation skipped - using default confidence"

        return confidence, justification
