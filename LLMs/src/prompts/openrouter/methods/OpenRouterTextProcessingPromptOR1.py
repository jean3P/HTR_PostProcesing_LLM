# repos/HTR_PostProcesing_LLM/LLMs/src/prompts/openrouter/methods/OpenRouterTextProcessingPromptOR1.py

import re
from prompts.openrouter.OpenRouterProcessingStrategy import OpenRouterProcessingStrategy
from prompts.openrouter.methods.openrouter_response_parser import OpenRouterResponseParser

from utils.aux_processing import get_time


class OpenRouterTextProcessingPromptOR1(OpenRouterProcessingStrategy):
    """
    PromptOR 1: Minimalist prompt
    Your task is to correct OCR errors. Please analyze the following text line
    and provide the corrected version: '{ocr_text}'. Then the corrected text line is:
    """

    def __init__(self):
        self.suggestions_memory = {}
        self.parser = OpenRouterResponseParser()

    def get_name_method(self):
        return "promptOR_1"

    def check_and_correct_text_line(self, text_line, train_set_lines, llm_instance, name_dataset, logger):

        logger.info(f"Start processing text line with PromptOR 1: '{text_line}'")

        # Direct correction without suggestions
        corrected_text = self.correct_with_prompts(
            text_line,
            llm_instance,
            name_dataset,
            logger
        )

        logger.info(f"Text after correction: {corrected_text}")

        # Evaluate the corrected text
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

    def correct_with_prompts(
            self,
            ocr_text,
            llm_instance,
            name_dataset,
            logger
    ):
        # PromptOR 1: Minimalist version
        system_prompt = (
            f"Your task is to correct OCR errors. Please analyze the following text line "
            f"and provide the corrected version: '{ocr_text}'. Then the corrected text line is:"
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
        # This method is not used in PromptOR 1
        return self.correct_with_prompts(ocr_text, llm_instance, name_dataset, logger)

    def correct_duplicated_words(self, text_line, llm_instance, logger):
        # PromptOR 1 doesn't include duplicate word correction
        return text_line

    def evaluate_corrected_text(self, original_text_line, corrected_text_line, llm_instance, logger):
        """Evaluate the corrected text and return confidence and justification."""
        logger.info(f"Evaluating the corrected text: '{corrected_text_line}' for the original: '{original_text_line}'")

        system_prompt = (
            f"Act as a text line evaluator. Your task is to analyze the original text line provided by an "
            f"OCR model and evaluate the corrected text line. Determine if the corrected text "
            f"line accurately fixes the OCR errors. Measure your confidence in the accuracy of the corrected text "
            f"line on a scale from 0 to 100 and provide a detailed justification for your assessment."
            f"\n\nProvide the confidence score and the justification as follows:\n"
            f"Confidence: <confidence_score>\nJustification: <justification>"
            f"\nGiven the original text line: '{original_text_line}' and the corrected text line: '{corrected_text_line}'"
            f"\n\nThe confidence and justification should be provided below:"
        )

        response = llm_instance.make_request(
            prompt=system_prompt,
            max_tokens=200,
            temperature=0.0
        )

        if response is None:
            logger.error("Failed to retrieve a valid response from OpenRouter.")
            return None, None

        try:
            raw_response = response['choices'][0]['message']['content'].strip()

            # Use the parser to extract confidence and justification
            confidence, justification = self.parser.extract_confidence_and_justification(raw_response, logger)

            return confidence, justification

        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return None, None
