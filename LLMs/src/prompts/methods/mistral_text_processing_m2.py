# src/prompts/methods/mistral_text_processing_m2.py

import re
from prompts.text_processing_base import TextProcessingStrategy
from utils.aux_processing import calculate_pipe, count_tokens, detect_immediate_repeated_words, \
    detect_close_repeated_word_sequences, detect_similar_immediate_repeated_words, has_misplaced_punctuation, \
    check_missing_or_extra_words, suggest_corrections_for_ocr_text_m2
from utils.logger import setup_logger

logger = setup_logger()


class MistralTextProcessingM2(TextProcessingStrategy):

    def check_and_correct_text_line(self, text_line, train_set_lines, pipe, tokenizer):
        logger.debug(f"Checking and correcting text line: {text_line}")
        spelling_erros = self.check_spelling_in_text_line(text_line, pipe, tokenizer)
        corrected_text = text_line
        if spelling_erros == 'Yes':
            suggestions = suggest_corrections_for_ocr_text_m2(corrected_text, train_set_lines)
            corrected_text = self.correct_with_suggestions(corrected_text, suggestions, pipe, tokenizer)
        corrected_text = self.correct_duplicated_words(corrected_text, pipe, tokenizer)
        corrected_text = self.check_and_correct_punctuation(corrected_text, pipe, tokenizer)
        confidence, justification = self.evaluate_corrected_text(
            text_line, corrected_text, pipe, tokenizer
        )
        logger.info(f"Text after correcting duplicated words: '{corrected_text}'")

        return corrected_text, confidence, justification

    def correct_duplicated_words(self, text_line, pipe, tokenizer):
        # First, use the find_immediate_repeated_words function to detect repeated words
        immediate_duplicated_words = detect_immediate_repeated_words(text_line)
        close_repeated_word_sets = detect_close_repeated_word_sequences(text_line)
        similar_duplicated_words = detect_similar_immediate_repeated_words(text_line)
        # immediate_duplicated_punctuation = detect_immediate_repeated_punctuation(text_line)

        if (not immediate_duplicated_words and not close_repeated_word_sets and not
        similar_duplicated_words):
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

        similar_sets_pair = "\n".join(
            f"Duplicated word: {word_set}" for word_set in similar_duplicated_words
        )

        combined_result = "\n".join([duplicated_words_part, similar_sets_pair])
        logger.info(f"Duplicated words: {combined_result}")
        logger.info(f"Repeated sets: {close_repeated_word_sets}")

        system_prompt = (
            f"[INST] Act as an 18th-century document analyst specializing in OCR correction. "
            f"Your task is to correct duplicated words in the given text line, ensuring the corrected line retains the "
            f"original meaning and style of 18th-century documents."
            f"\n\n## Guidelines:"
            f"\n1. Identify and correct any duplicated words in the text line"
            f"\n2. Maintain the original meaning and style of the text"
            f"\n3. Ensure corrections accurately reflect the 18th-century language and conventions"
            f"\n4. Just leave one occurrence of the duplicate word, don't delete everything"
            f"\n5. If it is the same word and one is capitalised and one is lowercase, delete one"
            f"\n\n## Duplicated words detected in the text line:"
            f"\n{combined_result}"
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

    def correct_with_suggestions(self, ocr_text, suggestions, pipe, tokenizer):
        # Return the same text line if it ends with an underscore
        if (
                ocr_text.startswith('\"') and ocr_text.endswith('\"') or
                ocr_text.endswith('_') or
                ocr_text.endswith('=') or
                (ocr_text.count('(') == 1 and ocr_text.count(')') == 0) or
                (ocr_text.count(')') == 1 and ocr_text.count('(') == 0) or
                '&' in ocr_text
        ):
            return ocr_text
        suggestion_part = "\n".join(
            f"Original word from the text line: {ocr_word}, Suggestions for corrections: {', '.join(set(similar_words))}"
            for ocr_word, similar_words in suggestions if similar_words and set(similar_words) != {ocr_word}
        )

        if not suggestion_part:
            suggestion_part = "No suggestions available."

        system_prompt = (
            f"[INST] Act as an 18th-century document analyst specializing in OCR error correction. "
            f"Your task is to correct OCR errors in words or numbers taking into account the suggestions of similar "
            f"words for corrections with a high priority in 18th-century documents (without adding new content)."
            f"Correction does not consist of deleting words but of replacing them with words that are more correct "
            f"in the context of the of the text line"
            f"\n\n## Guidelines:"
            f"\n1. Ensure corrections accurately reflect the 18th-century language and conventions"
            f"\n2. Be sure to keep punctuation marks of the original text line and do not add new punctuation that is "
            f"not in the original line of text"
            f"\n3. Preserve original word splits or cuts (e.g., 'incomple-' should not be combined into 'incomplete')"
            f"\n4. If the original text line is hyphenated, you have to keep the hyphen "
            f"\n5. Don't delete words and dont add words at the end, keep the same number of words from the original"
            f" text line"
            f"\n6. Words cut off at the end with a hyphen must not be completed and do not add contenct at the end of"
            f" the text line"
            f"\n7. Do not modify the end of the text line by adding new content"
            f"\n8. Do not add punctuation mark at the end if the original text line does not have it"
            f"\n9. Do not change proper names even if they are not common"
            f"\n10. Do not delete words like: or, and"
            f"\n11. If the tex line contains something like this: ( k ) must be (k)"
            f"\n12. If the text line starts with ':', do not delete it"
            f"\n\n## Suggestions of similar words of the training set:"
            f"\n{suggestion_part}"
            f"\nYour task is to replace the words with OCR errors based on the guidelines and suggestions correct "
            f"the text line: {ocr_text} [/INST]"
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

        corrected_text = result.replace('[INST]', '').replace('[/INST]', '').strip()
        corrected_text = corrected_text.replace('[INST', '').replace('[/INST', '').strip()
        corrected_text = corrected_text.replace('[COR]', '').replace('[/COR]', '').strip()
        corrected_text = corrected_text.replace('/C', '').strip()
        corrected_text = corrected_text.replace('ORRECT]', '').strip()
        corrected_text = corrected_text.replace('ORRECTED]', '').strip()
        corrected_text = corrected_text.replace(' "', '').strip()
        corrected_text = corrected_text.replace(' \"', '').strip()
        # Remove the specific ' [' character sequence
        corrected_text = corrected_text.replace('ORR]', '')
        if not ']' in corrected_text:
            corrected_text = corrected_text.replace(' [', '')
        corrected_text = re.sub(r'\s+', ' ', corrected_text)
        # Remove only leading and trailing single quotation marks
        if corrected_text.startswith("'") and corrected_text.endswith("'"):
            corrected_text = corrected_text[1:-1].strip()

        # Ensure the length is within reasonable bounds
        if len(corrected_text) > (len(ocr_text) * 1.11):
            corrected_text = ocr_text
        elif len(corrected_text) < (len(ocr_text) / 1.5):
            corrected_text = ocr_text

        logger.info(f"Corrected text for '{ocr_text}': {corrected_text}")
        return corrected_text

    def check_and_correct_punctuation(self, text_line, pipe, mistral_tokenizer):
        """
        This function checks if a text line contains a punctuation mark and verifies if it is correctly placed.
        If the punctuation is not correctly placed, it uses the LLM to correct it.
        """
        # Check if the text line contains punctuation
        is_misplaced = has_misplaced_punctuation(text_line)
        logger.info(f"Is the punctuation mark misplaced?: {is_misplaced}")
        if is_misplaced:
            # Create the system prompt to check punctuation placement
            system_prompt = (
                f"[INST] Your task is to correct only the placement of punctuation marks in the given text line. "
                f"Ensure the punctuation marks are correctly placed close to the left word and with exactly one space "
                f"between the punctuation mark and the word to its right. Do not add any new punctuation or alter the "
                f"original meaning of the text"
                f"\n\n## Guidelines:"
                f"\n1. Punctuation marks should be immediately next to the word on their left"
                f"\n2. Ensure there is exactly one space between the punctuation mark and the word to its right"
                f"\n3. Do not remove any words from the text line "
                f"\n4. Just focus only on misplaced punctuation, if any punctuation is missing don't add it"
                f"\n5. Do not delete this character & from the text line"
                f"\n\nExamples of Corrected Text:"
                f"\n- text line: 'Hello , world' the corrected text line is: 'Hello, world'"
                f"\n- text line: 'Good morning ; everyone' the corrected text line is: 'Good morning; everyone'"
                f"\n- text line: 'Are you ready ? ' the corrected text line is: 'Are you ready? '"
                f"\n\nYour task is only correct the misplaced punctuation marks base on the guidelines and examples, "
                f"then given the text line: '{text_line}'"
                f"[/INST]\nThen the corrected text line is:"
            )

            tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 25
            response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
            raw_response = response[0]['generated_text']

            json_output_marker = "Then the corrected text line is:"

            if json_output_marker in raw_response:
                corrected_text = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()
                corrected_text = re.sub(r'\[.*?\]', '', corrected_text)  # Remove any remaining [INST] or similar tags
                corrected_text = corrected_text.replace('[INST', '').replace('[/INST', '').strip()
                corrected_text = corrected_text.replace('[COR]', '').replace('[/COR]', '').strip()
                corrected_text = corrected_text.replace('ORR]', '')
                if not ']' in corrected_text:
                    corrected_text = corrected_text.replace(' [', '')
                corrected_text = corrected_text.replace('/', '')
                corrected_text = re.sub(r'\s+', ' ', corrected_text)
                corrected_text = corrected_text.strip()  # Trim any surrounding whitespace

                if corrected_text.startswith("'") and (corrected_text.endswith("'") or corrected_text.endswith("'.")):
                    corrected_text = corrected_text[1:-1].strip()

                if check_missing_or_extra_words(text_line, corrected_text):
                    corrected_text = text_line
            else:
                corrected_text = text_line

            logger.info(f"Punctuation checked and corrected for '{text_line}': {corrected_text}")
            return corrected_text
        else:
            # If there are no punctuation marks, return the original text line
            logger.info(f"No punctuation marks found for text: {text_line}")
            return text_line

    def check_spelling_in_text_line(self, original_text_line, pipe, tokenizer):
        system_prompt = (

            f"[INST] Act as a spelling evaluator. Your task is to analyze the original text line and determine if any "
            f"word is spelled "
            f"incorrectly. Provide your answer as either Yes or No."
            f"\nRespond with either Yes or No only."
            f"\nGiven the original text line: '{original_text_line}'"
            f"[/INST]\nThen the response is:"
        )

        tokens_prompt = count_tokens(system_prompt, tokenizer) + 10
        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']

        json_output_marker = "Then the response is:"
        spelling_error = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()

        logger.info(f"The text line '{original_text_line}' contains spelling error?: {spelling_error}")

        return spelling_error