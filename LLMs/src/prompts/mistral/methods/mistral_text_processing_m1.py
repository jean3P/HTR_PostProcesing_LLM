# src/prompts/methods/mistral_text_processing_m1.py

import re
from prompts.mistral.text_processing_base import TextProcessingStrategy
from utils.aux_processing import count_tokens, calculate_pipe, detect_immediate_repeated_words, \
    detect_close_repeated_word_sequences, suggest_corrections_for_ocr_text_m1
from utils.logger import setup_logger

logger = setup_logger()


class MistralTextProcessingM1(TextProcessingStrategy):
    def __init__(self):
        self.suggestions_memory = {}

    def get_name_method(self):
        return "method_1"

    def check_and_correct_text_line(self, text_line, pipe, tokenizer, train_set_line):
        logger.info(f"Start processing text line: '{text_line}'")

        # Suggest corrections based on OCR and training lines
        suggestions = suggest_corrections_for_ocr_text_m1(text_line, train_set_line, self.suggestions_memory)

        # Apply suggestions to correct text
        corrected_text = self.correct_with_suggestions(text_line, suggestions, pipe, tokenizer)
        logger.info(f"Text after applying corrections for '{text_line}': {corrected_text}")

        # Correct duplicated words in the text
        corrected_text = self.correct_duplicated_words(corrected_text, pipe, tokenizer)
        logger.info(f"Final text after correcting duplicated words: '{corrected_text}'")

        # Evaluate the corrected text
        confidence, justification = self.evaluate_corrected_text(
            text_line, corrected_text, pipe, tokenizer
        )

        if confidence and justification:
            logger.info(
                f"Confidence - {confidence}, Justification - {justification}")
        else:
            logger.info(f"Could not evaluate the corrected text for '{corrected_text}'")

        logger.info(f"Finished processing text line: {text_line} ===> {corrected_text}")
        return corrected_text, confidence, justification

    def correct_with_suggestions(self, ocr_text, suggestions, pipe, tokenizer):
        suggestion_part = "\n".join(
            f"Original word from the text line: {ocr_word}, Suggestions for corrections: {', '.join(set(similar_words))}"
            for ocr_word, similar_words in suggestions if similar_words and set(similar_words) != {ocr_word}
        )

        if not suggestion_part:
            logger.info(f"No suggestions available for '{ocr_text}'")
            suggestion_part = "No suggestions available."
        else:
            logger.info(f"Generating correction suggestions for text: '{ocr_text}' ==> {suggestion_part}")

        system_prompt = (
            f"[INST] Act as an 18th-century document analyst specializing in OCR correction. "
            f"Your task is to correct OCR errors in words or numbers taking into account the suggestions of similar "
            f"words for corrections with a high priority in 18th-century documents."
            f"\n\n## Guidelines:"
            f"\n1. Ensure corrections accurately reflect the 18th-century language and conventions"
            f"\n2. Be sure to keep punctuation marks of the original text line and do not add new punctuation that is "
            f"not in the original line of text"
            f"\n3. Preserve original word splits or cuts (e.g., 'incomple-' should not be combined into 'incomplete')"
            f"\n4. If the original text line is hyphenated, you have to keep the hyphen "
            f"\n5. Don't delete words that are not duplicated"
            f"\n6. Words cut off at the end with a hyphen should not be completed"
            f"\n7. Do not modify the end of the text line by adding new content"
            f"\n\n## Suggestions of similar words of the training set:"
            f"\n{suggestion_part}"
            f"\nBased on the guidelines and suggestions correct the text line: {ocr_text} [/INST]"
            f"\nThen corrected text line is:"
        )
        tokens_prompt = count_tokens(system_prompt, tokenizer) + 25
        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']
        json_output_marker = "Then corrected text line is:"

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
        # Remove the specific ' [' character sequence
        corrected_text = corrected_text.replace(' [', '')
        corrected_text = re.sub(r'\s+', ' ', corrected_text)

        # Ensure the length is within reasonable bounds
        if len(corrected_text) > (len(ocr_text) * 1.2):
            corrected_text = ocr_text
        elif len(corrected_text) < (len(ocr_text) / 2):
            corrected_text = ocr_text

        return corrected_text

    def correct_duplicated_words(self, text_line, pipe, tokenizer):
        logger.info(f"Checking for duplicated words in: '{text_line}'")
        # First, use the find_immediate_repeated_words function to detect repeated words
        immediate_duplicated_words = detect_immediate_repeated_words(text_line)
        close_repeated_word_sets = detect_close_repeated_word_sequences(text_line)

        if not immediate_duplicated_words and not close_repeated_word_sets:
            # If no duplicated words are found, return the original text line
            logger.info(f"No duplicated words found for text: {text_line}")
            return text_line

        # Combine both sets of duplicated words for the prompt
        duplicated_words_part = "\n".join(
            f"Duplicated word: {word}" for word in immediate_duplicated_words
        )
        repeated_sets_part = "\n".join(
            f"Repeated set: {word_set}" for word_set in close_repeated_word_sets
        )

        logger.info(f"Duplicated words found: {immediate_duplicated_words}")
        logger.info(f"Repeated word sets found: {close_repeated_word_sets}")
        system_prompt = (
            f"[INST] Act as an 18th-century document analyst specializing in OCR correction. "
            f"Your task is to correct duplicated words in the given text line, ensuring the corrected line retains the "
            f"original meaning and style of 18th-century documents."
            f"\n\n## Guidelines:"
            f"\n1. Identify and correct any duplicated words in the text line."
            f"\n2. Maintain the original meaning and style of the text."
            f"\n3. Ensure corrections accurately reflect the 18th-century language and conventions."
            f"\n4. Just leave one occurrence of the duplicate word, don't delete everything."
            f"\n5. If it is the same word and one is capitalised and one is lowercase, delete one."
            f"\n\n## Duplicated words detected in the text line:"
            f"\n{duplicated_words_part}"
            f"\n\n## Repeated word sets detected in the text line:"
            f"\n{repeated_sets_part}"
            f"\nBased on the guidelines, please analyze the following text line and"
            f" provide the corrected version: {text_line} "
            f"[/INST]\nThen corrected text line is:"
        )

        tokens_prompt = count_tokens(system_prompt, tokenizer) + 25
        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']
        # print("Raw response:", raw_response)  # Debug print

        json_output_marker = "Then corrected text line is:"

        if json_output_marker in raw_response:
            corrected_text = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()
            corrected_text = re.sub(r'\[.*?\]', '', corrected_text)  # Remove any remaining [INST] or similar tags
            corrected_text = corrected_text.replace('[INST', '').replace('[/INST', '').strip()
            corrected_text = corrected_text.replace('[COR]', '').replace('[/COR]', '').strip()
            corrected_text = corrected_text.replace(' [', '')
            corrected_text = corrected_text.replace('/', '')
            corrected_text = re.sub(r'\s+', ' ', corrected_text)
            corrected_text = corrected_text.strip()  # Trim any surrounding whitespace
            if len(corrected_text) < (len(text_line) / 1.8):
                corrected_text = text_line
        else:
            corrected_text = text_line

        return corrected_text


