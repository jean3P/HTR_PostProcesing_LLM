# src/prompts/methods/mistral_text_processing_m0.py

import re
from prompts.mistral.text_processing_base import TextProcessingStrategy
from utils.aux_processing import count_tokens, calculate_pipe_mistral, detect_immediate_repeated_words, \
    detect_close_repeated_word_sequences, suggest_corrections_for_ocr_text_m1, get_time


class MistralTextProcessingM0S2(TextProcessingStrategy):
    def correct_duplicated_words(self, text_line, pipe, tokenizer, logger):
        pass

    def __init__(self):
        self.suggestions_memory = {}

    def get_name_method(self):
        return "method_0_simple_2"

    def check_and_correct_text_line(self, text_line, pipe, tokenizer, train_set_line, name_dataset, logger):
        logger.info(f"Start processing text line: '{text_line}' -- from {get_time(name_dataset)}-century")

        # Apply suggestions to correct text
        corrected_text = self.correct_with_suggestions(text_line, pipe, tokenizer, name_dataset, logger)
        logger.info(f"Text after applying corrections for '{text_line}': {corrected_text}")

        confidence, justification = self.evaluate_corrected_text(
            text_line, corrected_text, pipe, tokenizer, logger
        )

        if confidence and justification:
            logger.info(
                f"Confidence - {confidence}, Justification - {justification}")
        else:
            logger.info(f"Could not evaluate the corrected text for '{corrected_text}'")

        logger.info(f"Finished processing text line: {text_line} ===> {corrected_text}")
        return corrected_text, confidence, justification

    def correct_with_suggestions(self, ocr_text, pipe, tokenizer, name_dataset, logger):

        system_prompt = (
            f"Correct the spelling and grammar of the following text:\n {ocr_text} \n CORRECTED TEXT:"
        )

        tokens_prompt = count_tokens(system_prompt, tokenizer) + 25
        response = calculate_pipe_mistral(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']
        json_output_marker = "CORRECTED TEXT:"

        if json_output_marker in raw_response:
            result = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()
        else:
            result = ocr_text

        # Remove any remaining '[INST]' or '[/INST]' tags manually and ensure text formatting
        corrected_text = result.replace('[INST]', '').replace('[/INST]', '').strip()
        corrected_text = corrected_text.replace('[INST', '').replace('[/INST', '').strip()
        corrected_text = corrected_text.replace('[COR]', '').replace('[/COR]', '').strip()
        corrected_text = corrected_text.replace('/C', '').strip()
        corrected_text = corrected_text.replace('ORRECT]', '').strip()
        corrected_text = corrected_text.replace('ORRECTED]', '').strip()
        corrected_text = corrected_text.replace(' [', '')
        corrected_text = re.sub(r'\s+', ' ', corrected_text)

        # Ensure the length is within reasonable bounds
        if len(corrected_text) > (len(ocr_text) * 1.2):
            corrected_text = ocr_text
        elif len(corrected_text) < (len(ocr_text) / 2):
            corrected_text = ocr_text

        return corrected_text



