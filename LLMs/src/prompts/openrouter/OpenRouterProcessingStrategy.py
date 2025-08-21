# repos/HTR_PostProcesing_LLM/LLMs/src/prompts/openrouter/OpenRouterProcessingStrategy.py

from prompts.gpt.GPTProcessingStrategy import TextProcessingStrategy


class OpenRouterProcessingStrategy(TextProcessingStrategy):
    """
    Base class for OpenRouter text processing strategies.
    Inherits from the GPT strategy as they share similar interfaces.
    """

    def evaluate_corrected_text(self, original_text_line, corrected_text_line, llm_instance, logger):
        """
        Default implementation of evaluate_corrected_text.
        Can be overridden by subclasses if needed.
        """
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

            confidence_marker = "Confidence:"
            justification_marker = "Justification:"

            confidence_section = raw_response.split(confidence_marker)[-1].split(justification_marker)[0].strip()
            confidence = confidence_section.split('\n')[0].strip()

            justification_section = raw_response.split(justification_marker)[-1].strip()
            justification_lines = justification_section.split('\n')
            justification = justification_lines[0].strip()

            return confidence, justification

        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return None, None
