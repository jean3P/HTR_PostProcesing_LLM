# src/prompts/text_processing_base.py

from abc import ABC, abstractmethod

from utils.aux_processing import count_tokens, calculate_pipe
from utils.logger import setup_logger

logger = setup_logger()


class TextProcessingStrategy(ABC):

    @abstractmethod
    def get_name_method(self):
        pass

    @abstractmethod
    def check_and_correct_text_line(self, text_line,  pipe, tokenizer, train_set_lines):
        pass

    @abstractmethod
    def correct_with_suggestions(self, ocr_text, suggestions, pipe, tokenizer):
        pass

    @abstractmethod
    def correct_duplicated_words(self, text_line, pipe, tokenizer):
        pass

    def evaluate_corrected_text(self, original_text_line, corrected_text_line, pipe, tokenizer):
        system_prompt = (
            f"[INST] Act as an 18th-century text line evaluator. Your task is to analyze the original text line by an "
            f"OCR model and evaluate the corrected text line provided by the LLM. Determine if the LLM's corrected text"
            f" line accurately fixes the OCR errors. Measure your confidence in the accuracy of the corrected text "
            f"line on a scale from 0 to 100 and provide a detailed justification for your assessment."
            f"\nProvide the confidence score and the justification as follows:\n"
            f"Confidence: <confidence_score>\nJustification: <justification>"
            f"\nGiven the original text line: '{original_text_line}' and the text line from the LLM: '{corrected_text_line}'"
            f"[/INST]\nThen the confidence and the justification is:"
        )

        tokens_prompt = count_tokens(system_prompt, tokenizer) + 100
        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']

        json_output_marker = "Then the confidence and the justification is:"
        corrected_text = raw_response.split(json_output_marker)[-1].strip()

        # Extract the confidence and justification
        confidence_marker = "Confidence:"
        justification_marker = "Justification:"

        # Extract confidence
        confidence_section = corrected_text.split(confidence_marker)[-1].strip()
        confidence = confidence_section.split('\n')[0].strip()

        # Extract justification
        justification_section = corrected_text.split(justification_marker)[-1].strip()
        justification_lines = justification_section.split('\n')
        justification = justification_lines[0].strip()  # Take the first line of the justification

        logger.info(f"confidence: {confidence}")
        logger.info(f"justification: {justification}")

        return confidence, justification
