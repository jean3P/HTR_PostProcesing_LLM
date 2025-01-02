import re

from prompts.llama.LlamaProcessingStrategy import TextProcessingStrategy
from utils.aux_processing import suggest_corrections_for_ocr_text_m1, get_time, count_tokens, calculate_pipe_llama, \
    clean_text


class LlamaTextProcessingM0S2(TextProcessingStrategy):
    def __init__(self):
        self.suggestions_memory = {}

    def check_and_correct_text_line(self, text_line, pipe, tokenizer, train_set_lines, name_dataset, terminators,
                                    logger):
        logger.info(f"Start processing text line: '{text_line}' -- from {get_time(name_dataset)}-century")

        # Suggest corrections based on OCR and training lines
        suggestions = suggest_corrections_for_ocr_text_m1(text_line, train_set_lines, self.suggestions_memory)

        # Apply suggestions to correct text
        corrected_text = self.correct_with_suggestions(text_line, suggestions, pipe, tokenizer, name_dataset,
                                                       terminators, logger)
        logger.info(f"Text after applying corrections for '{text_line}': {corrected_text}")

        # Evaluate the corrected text
        confidence, justification = self.evaluate_corrected_text(
            text_line, corrected_text, pipe, tokenizer, terminators, logger
        )

        if confidence and justification:
            logger.info(
                f"Confidence - {confidence}, Justification - {justification}")
        else:
            logger.info(f"Could not evaluate the corrected text for '{corrected_text}'")

        logger.info(f"Finished processing text line: {text_line} ===> {corrected_text}")
        return corrected_text, confidence, justification

    def correct_with_suggestions(self, ocr_text, suggestions, pipe, tokenizer, name_dataset, terminators, logger):

        system_prompt = (
            f"Correct the spelling and grammar of the following text:\n {ocr_text} \n CORRECTED TEXT:"
        )

        tokens_prompt = count_tokens(system_prompt, tokenizer) + 25
        response = calculate_pipe_llama(pipe, system_prompt, tokens_prompt, terminators, 0.1)
        raw_response = response[0]['generated_text']
        json_output_marker = "CORRECTED TEXT:"

        if json_output_marker in raw_response:
            result = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()
        else:
            result = ocr_text

        # Remove any remaining '[INST]' or '[/INST]' tags manually and ensure text formatting
        # corrected_text = result.replace('[INST]', '').replace('[/INST]', '').strip()
        # # corrected_text = corrected_text.replace('[INST', '').replace('[/INST', '').strip()
        # # corrected_text = corrected_text.replace('[COR]', '').replace('[/COR]', '').strip()
        # # corrected_text = corrected_text.replace('/C', '').strip()
        # corrected_text = corrected_text.replace('ORRECT]', '').strip()
        # corrected_text = corrected_text.replace('ORRECTED]', '').strip()
        # # Remove the specific ' [' character sequence
        # corrected_text = corrected_text.replace(' [', '')
        # corrected_text = re.sub(r'\s+', ' ', corrected_text)
        corrected_text = clean_text(ocr_text, result)

        # Ensure the length is within reasonable bounds
        if len(corrected_text) > (len(ocr_text) * 1.2):
            corrected_text = ocr_text
        elif len(corrected_text) < (len(ocr_text) / 2):
            corrected_text = ocr_text

        return corrected_text

    def get_name_method(self):
        return "method_0_simple_2"
