import re
import time

from prompts.gpt.GPTProcessingStrategy import TextProcessingStrategy
from utils.aux_processing import suggest_corrections_for_ocr_text_m1, count_tokens_gpt, calculate_pipe_openai, \
    detect_immediate_repeated_words, detect_close_repeated_word_sequences, clean_text, \
    suggest_corrections_for_ocr_text_m2, detect_similar_immediate_repeated_words, has_misplaced_punctuation, \
    check_missing_or_extra_words
from utils.logger import setup_logger

logger = setup_logger()


class GptTextProcessingM2(TextProcessingStrategy):

    def __init__(self):
        self.suggestions_memory = {}

    def get_name_method(self):
        return "method_2"

    def check_and_correct_text_line(self, text_line, train_set_lines, model_name, openai_token):
        logger.info(f"Start processing text line: '{text_line}'")

        # Check if there are spelling errors
        spelling_errors = self.check_spelling_in_text_line(text_line, model_name, openai_token)
        corrected_text = text_line
        if spelling_errors == 'Yes':
            # Suggest corrections based on OCR and training lines
            suggestions = suggest_corrections_for_ocr_text_m2(text_line, train_set_lines, self.suggestions_memory)
            # Apply suggestions to correct text
            corrected_text = self.correct_with_suggestions(text_line, suggestions, model_name, openai_token)
            logger.info(f"Text after applying corrections for '{text_line}': {corrected_text}")

        # Correct duplicated words in the text
        corrected_text = self.correct_duplicated_words(corrected_text, model_name, openai_token)
        logger.info(f"Text after correcting duplicated words: '{corrected_text}'")

        # Check if there are punctuation marks errors
        corrected_text = self.check_and_correct_punctuation(corrected_text, model_name, openai_token)
        # Evaluate the corrected text
        confidence, justification = self.evaluate_corrected_text(
            text_line, corrected_text, model_name, openai_token
        )
        if confidence and justification:
            logger.info(
                f"Confidence - {confidence}, Justification - {justification}")
        else:
            logger.info(f"Could not evaluate the corrected text for '{corrected_text}'")

        logger.info(f"Finished processing text line: {text_line} ===> {corrected_text}")
        return corrected_text, confidence, justification

    def correct_with_suggestions(self, ocr_text, suggestions, model_name, openai_token):
        # Return the same text line if it ends with an underscore
        if (
                ocr_text.startswith('\"') and ocr_text.endswith('\"') or
                ocr_text.endswith('_') or
                ocr_text.endswith('=') or
                (ocr_text.count('(') == 1 and ocr_text.count(')') == 0) or
                (ocr_text.count(')') == 1 and ocr_text.count('(') == 0) or
                '&' in ocr_text
        ):
            logger.info(f"Evaluating special conditions for text: '{ocr_text}'")

            if ocr_text.startswith('\"') and ocr_text.endswith('\"'):
                logger.info(f"Text starts and ends with quotation marks: '{ocr_text}'")

            if ocr_text.endswith('_'):
                logger.info(f"Text ends with an underscore: '{ocr_text}'")

            if ocr_text.endswith('='):
                logger.info(f"Text ends with an equals sign: '{ocr_text}'")

            if ocr_text.count('(') == 1 and ocr_text.count(')') == 0:
                logger.info(f"Text contains an unmatched opening parenthesis: '{ocr_text}'")

            if ocr_text.count(')') == 1 and ocr_text.count('(') == 0:
                logger.info(f"Text contains an unmatched closing parenthesis: '{ocr_text}'")

            if '&' in ocr_text:
                logger.info(f"Text contains an ampersand: '{ocr_text}'")
            return ocr_text

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
            f"Act as an 18th-century document analyst specializing in OCR correction. "
            f"Your task is to correct OCR errors in words or numbers while taking into account the suggestions of similar "
            f"words for corrections, with a focus on 18th-century documents."
            f"\n\n## Guidelines:"
            f"\n1. Ensure corrections accurately reflect the 18th-century language and conventions."
            f"\n2. Keep punctuation marks from the original text line and do not add new punctuation that is "
            f"not in the original line."
            f"\n3. Preserve original word splits or cuts (e.g., 'incomple-' should not be combined into 'incomplete')."
            f"\n4. If the original text line is hyphenated, retain the hyphen."
            f"\n5. Do not delete words that are not duplicated."
            f"\n6. Words cut off at the end with a hyphen should not be completed."
            f"\n7. Do not modify the end of the text line by adding new content."
            f"\n\n## Suggestions of similar words from the training set:"
            f"\n{suggestion_part}"
            f"\n\nBased on the guidelines and suggestions, please correct the following text line:"
            f"\n'{ocr_text}'"
            f"\n\nThe corrected text line should be:"
        )

        tokens_prompt = count_tokens_gpt(system_prompt) + 25
        # Retry logic for handling rate limiting (429 Too Many Requests)
        max_retries = 5
        retry_delay = 5  # Start with 5 seconds
        response = None

        for attempt in range(max_retries):
            response = calculate_pipe_openai(model_name, system_prompt, tokens_prompt, openai_token)

            if response is not None:
                # Break the loop if the response is valid
                break

            # If the response is None, wait and retry
            logger.info(f"Rate limit reached. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

            # Check if the response is still None after retries
        if response is None:
            logger.error("Failed to retrieve a valid response from OpenAI after multiple attempts.")
            return ocr_text  # Return original text if the correction fails

        try:
            # Access response data using the correct attributes
            raw_response = response.choices[0].message.content.strip()
        except (AttributeError, KeyError, IndexError):
            logger.error("Unexpected response structure or empty response.")
            return ocr_text  # Return original text if the response is not structured correctly

            # Process and clean the corrected text
        corrected_text = re.sub(r'\s+', ' ', raw_response).strip()
        corrected_text = clean_text(ocr_text, corrected_text)
        return corrected_text

    def correct_duplicated_words(self, text_line, model_name, openai_token):
        logger.info(f"Checking for duplicated words in: '{text_line}'")
        # First, use the find_immediate_repeated_words function to detect repeated words
        immediate_duplicated_words = detect_immediate_repeated_words(text_line)
        close_repeated_word_sets = detect_close_repeated_word_sequences(text_line)
        similar_duplicated_words = detect_similar_immediate_repeated_words(text_line)

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
        logger.info(f"Duplicated words found: {combined_result}")
        logger.info(f"Repeated word sets found: {close_repeated_word_sets}")

        # Construct the system prompt
        system_prompt = (
            f"Act as an 18th-century document analyst specializing in OCR correction. "
            f"Your task is to correct duplicated words in the given text line, ensuring the corrected line retains the "
            f"original meaning and style of 18th-century documents."
            f"\n\n## Guidelines:"
            f"\n1. Identify and correct any duplicated words in the text line."
            f"\n2. Maintain the original meaning and style of the text."
            f"\n3. Ensure corrections accurately reflect the 18th-century language and conventions."
            f"\n4. Just leave one occurrence of the duplicate word, don't delete everything."
            f"\n5. If it is the same word and one is capitalized and one is lowercase, delete one."
            f"\n\n## Duplicated words detected in the text line:"
            f"\n{combined_result}"
            f"\n\n## Repeated word sets detected in the text line:"
            f"\n{repeated_sets_part}"
            f"\nBased on the guidelines, please analyze the following text line and provide the corrected version: {text_line}."
            f"\n\nThe corrected text line should be:"
        )

        # Calculate the tokens and prepare for the API call
        tokens_prompt = count_tokens_gpt(system_prompt) + 25
        max_retries = 5
        retry_delay = 5  # Start with 5 seconds
        response = None

        for attempt in range(max_retries):
            response = calculate_pipe_openai(model_name, system_prompt, tokens_prompt, openai_token)

            if response is not None:
                break

            logger.info(f"Rate limit reached. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

        if response is None:
            logger.error("Failed to retrieve a valid response from OpenAI after multiple attempts.")
            return text_line  # Return original text if the correction fails

        try:
            # Parse the response to get the corrected text
            raw_response = response.choices[0].message.content.strip()

            # Define the marker to extract the corrected text
            corrected_text_marker = "Then corrected text line is:"

            # Extract the corrected text
            if corrected_text_marker in raw_response:
                corrected_text = raw_response.split(corrected_text_marker)[-1].split('\n')[0].strip()

                # Clean up the corrected text by removing unnecessary tags
                corrected_text = re.sub(r'\[.*?\]', '', corrected_text)
                corrected_text = re.sub(r'\s+', ' ', corrected_text).strip()

                # Ensure the corrected text is within reasonable length limits
                if len(corrected_text) < (len(text_line) / 1.8):
                    corrected_text = text_line

            else:
                corrected_text = text_line

        except (AttributeError, KeyError, IndexError):
            logger.error("Unexpected response structure or empty response.")
            return text_line  # Return original text if the response is not structured correctly
        return corrected_text

    def evaluate_corrected_text(self, original_text_line, corrected_text_line, model_name, openai_token):
        logger.info(f"Evaluating the corrected text: '{corrected_text_line}' for the original: '{original_text_line}'")
        # Construct the system prompt for evaluating the corrected text
        system_prompt = (
            f"Act as an 18th-century text line evaluator. Your task is to analyze the original text line provided by an "
            f"OCR model and evaluate the corrected text line provided by the LLM. Determine if the LLM's corrected text "
            f"line accurately fixes the OCR errors. Measure your confidence in the accuracy of the corrected text "
            f"line on a scale from 0 to 100 and provide a detailed justification for your assessment."
            f"\n\nProvide the confidence score and the justification as follows:\n"
            f"Confidence: <confidence_score>\nJustification: <justification>"
            f"\nGiven the original text line: '{original_text_line}' and the corrected text line: '{corrected_text_line}'"
            f"\n\nThe confidence and justification should be provided below:"
        )

        # Calculate tokens required for the prompt
        tokens_prompt = count_tokens_gpt(system_prompt) + 100
        max_retries = 5
        retry_delay = 5  # Start with 5 seconds
        response = None

        # Retry logic for handling rate limits (429 Too Many Requests)
        for attempt in range(max_retries):
            response = calculate_pipe_openai(model_name, system_prompt, tokens_prompt, openai_token)

            if response is not None:
                break  # Exit the loop if a valid response is obtained

            logger.info(f"Rate limit reached. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff for retry delays

        if response is None:
            logger.error("Failed to retrieve a valid response from OpenAI after multiple attempts.")
            return None, None  # Return None if the evaluation fails

        try:
            # Parse the response to get the confidence and justification
            raw_response = response.choices[0].message.content.strip()

            # Define the markers to extract confidence and justification
            confidence_marker = "Confidence:"
            justification_marker = "Justification:"

            # Extract confidence score
            confidence_section = raw_response.split(confidence_marker)[-1].split(justification_marker)[0].strip()
            confidence = confidence_section.split('\n')[0].strip()

            # Extract justification
            justification_section = raw_response.split(justification_marker)[-1].strip()
            justification_lines = justification_section.split('\n')
            justification = justification_lines[0].strip()  # Take the first line of the justification

        except (AttributeError, KeyError, IndexError):
            logger.error("Unexpected response structure or empty response.")
            return None, None  # Return None if the response is not structured correctly

        return confidence, justification

    def check_spelling_in_text_line(self, original_text_line, model_name, openai_token):
        # Create a system prompt for checking spelling
        system_prompt = (
            f"Act as a spelling evaluator. Your task is to analyze the original text line and determine if any word is "
            f"spelled incorrectly. Respond with either 'Yes' or 'No' only."
            f"\nGiven the original text line: '{original_text_line}'"
            f"\nThen the response is:"
        )

        # Calculate token count and add a buffer
        tokens_prompt = count_tokens_gpt(system_prompt) + 10
        max_retries = 5
        retry_delay = 5  # Start with 5 seconds
        response = None

        # Retry logic for handling rate limits (429 Too Many Requests)
        for attempt in range(max_retries):
            response = calculate_pipe_openai(model_name, system_prompt, tokens_prompt, openai_token)

            if response is not None:
                break  # Exit the loop if a valid response is obtained

            logger.info(f"Rate limit reached. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff for retry delays

        if response is None:
            logger.error("Failed to retrieve a valid response from OpenAI after multiple attempts.")
            return "Unknown"  # Return "Unknown" if the spelling check fails

        try:
            # Access response data using the correct attributes
            raw_response = response.choices[0].message.content.strip()

            # Extract the response ('Yes' or 'No')
            spelling_error = raw_response.split('\n')[0].strip()

            logger.info(f"The text line '{original_text_line}' contains spelling error?: {spelling_error}")

            return spelling_error

        except (AttributeError, KeyError, IndexError):
            logger.error("Unexpected response structure or empty response.")
            return "Unknown"  # Return "Unknown" if the response is not structured correctly


    def check_and_correct_punctuation(self, text_line, model_name, openai_token):
        """
        This function checks if a text line contains misplaced punctuation and uses GPT to correct it.
        """
        # Check if the text line contains punctuation
        is_misplaced = has_misplaced_punctuation(text_line)
        logger.info(f"Is the punctuation mark misplaced?: {is_misplaced}")

        if is_misplaced:
            # Create the system prompt to check punctuation placement
            system_prompt = (
                f"Your task is to correct only the placement of punctuation marks in the given text line. "
                f"Ensure the punctuation marks are correctly placed close to the left word and with exactly one space "
                f"between the punctuation mark and the word to its right. Do not add any new punctuation or alter the "
                f"original meaning of the text."
                f"\n\n## Guidelines:"
                f"\n1. Punctuation marks should be immediately next to the word on their left."
                f"\n2. Ensure there is exactly one space between the punctuation mark and the word to its right."
                f"\n3. Do not remove any words from the text line."
                f"\n4. Focus only on misplaced punctuation; if any punctuation is missing, do not add it."
                f"\n5. Do not delete this character '&' from the text line."
                f"\n\nExamples of Corrected Text:"
                f"\n- Text line: 'Hello , world' -> Corrected text: 'Hello, world'"
                f"\n- Text line: 'Good morning ; everyone' -> Corrected text: 'Good morning; everyone'"
                f"\n- Text line: 'Are you ready ? ' -> Corrected text: 'Are you ready?'"
                f"\n\nYour task is only to correct the misplaced punctuation marks based on the guidelines and examples."
                f"\nGiven the text line: {text_line}, the corrected text line is:"
            )

            # Calculate the number of tokens for the prompt
            tokens_prompt = count_tokens_gpt(system_prompt) + 25
            max_retries = 5
            retry_delay = 5  # Start with 5 seconds
            response = None

            # Retry logic for handling rate limits (429 Too Many Requests)
            for attempt in range(max_retries):
                response = calculate_pipe_openai(model_name, system_prompt, tokens_prompt, openai_token)

                if response is not None:
                    break  # Exit the loop if a valid response is obtained

                logger.info(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff for retry delays

            if response is None:
                logger.error("Failed to retrieve a valid response from OpenAI after multiple attempts.")
                return text_line  # Return original text if correction fails

            try:
                # Access response data using the correct attributes
                raw_response = response.choices[0].message.content.strip()

                # Define the marker to extract the corrected text
                json_output_marker = "The corrected text line should be:"

                # Extract the corrected text
                if json_output_marker in raw_response:
                    corrected_text = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()

                    # Clean up any unnecessary tags or whitespace
                    corrected_text = re.sub(r'\[.*?\]', '', corrected_text).strip()
                    corrected_text = re.sub(r'\s+', ' ', corrected_text).strip()

                    # Check for major errors (missing or extra words) and revert to original if necessary
                    if check_missing_or_extra_words(text_line, corrected_text):
                        corrected_text = text_line

                else:
                    corrected_text = text_line

                logger.info(f"Punctuation checked and corrected for '{text_line}': {corrected_text}")
                return corrected_text

            except (AttributeError, KeyError, IndexError):
                logger.error("Unexpected response structure or empty response.")
                return text_line  # Return original text if the response is not structured correctly

        else:
            # If no punctuation issues are detected, return the original text line
            logger.info(f"No misplaced punctuation found for text: {text_line}")
            return text_line
