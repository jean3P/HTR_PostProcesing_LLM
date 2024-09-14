# src/utils/aux_processing.py

from collections import defaultdict
from difflib import get_close_matches, SequenceMatcher
from utils.logger import setup_logger


suggestions_memory = {}
logger = setup_logger()


def calculate_pipe(pipe, prompt, nummer_length, top_k):
    return pipe(prompt, max_length=nummer_length, do_sample=True, top_k=top_k, num_return_sequences=1,
                pad_token_id=pipe.tokenizer.eos_token_id)


def count_tokens(prompt, mistral_tokenizer):
    input_ids = mistral_tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input_ids.shape[1]


def find_top_3_processed_similar_words_m1(word, train_set_lines, is_start_of_line=False):
    def split_suggestion(suggestion, original):
        if original.endswith('-') and not suggestion.endswith('-'):
            return suggestion[:len(original) - 1] + '-'
        if is_start_of_line and not suggestion.startswith(original):
            return original
        # Ensure colon preservation
        if ':' in original and ':' not in suggestion:
            return suggestion + ':'
        # Remove punctuation if the original doesn't have it
        if not re.search(r'[^\w\s]', original):
            suggestion = re.sub(r'[^\w\s]', '', suggestion)
        return suggestion

    # Check if we have previously saved suggestions
    if word in suggestions_memory:
        return suggestions_memory[word]

    all_words = [line.split() for line in train_set_lines]
    flat_list = [item for sublist in all_words for item in sublist]

    matches = get_close_matches(word, flat_list, n=3, cutoff=0.85)
    unique_matches = list(dict.fromkeys(matches))  # Remove duplicates while preserving order
    split_matches = [split_suggestion(match, word) for match in unique_matches]

    # Save suggestions in memory if they are different from the original word
    if split_matches and split_matches != [word]:
        suggestions_memory[word] = split_matches

    return split_matches if split_matches else [word]


def suggest_corrections_for_ocr_text_m1(ocr_text, train_set_lines):
    suggestions = []
    words = ocr_text.split()
    for idx, word in enumerate(words):
        is_start_of_line = (idx == 0)
        similar_words = find_top_3_processed_similar_words_m1(word, train_set_lines, is_start_of_line)
        suggestions.append((word, similar_words))
    return suggestions


def find_top_3_processed_similar_words_m2(word, train_set_lines, is_start_of_line=False):
    def split_suggestion(suggestion, original):
        # Preserve any ending characters from the original word
        if original and suggestion and not suggestion.endswith(original[-1]):
            suggestion += original[-1]

        if is_start_of_line and not suggestion.startswith(original):
            return original

        # Remove punctuation if the original doesn't have it
        if not re.search(r'[^\w\s]', original):
            suggestion = re.sub(r'[^\w\s]', '', suggestion)

        return suggestion

    # Check if we have previously saved suggestions
    if word in suggestions_memory:
        return suggestions_memory[word]

    all_words = [line.split() for line in train_set_lines]
    flat_list = [item for sublist in all_words for item in sublist]

    matches = get_close_matches(word, flat_list, n=3, cutoff=0.85)
    unique_matches = list(dict.fromkeys(matches))  # Remove duplicates while preserving order

    # Remove the original word from matches
    unique_matches = [match for match in unique_matches if match != word]

    split_matches = [split_suggestion(match, word) for match in unique_matches]

    # Save suggestions in memory if they are different from the original word
    if split_matches:
        suggestions_memory[word] = split_matches

    return split_matches if split_matches else [word]


# Detect OCR Errors and Suggest Corrections
def suggest_corrections_for_ocr_text_m2(ocr_text, train_set_lines):
    suggestions = []
    words = ocr_text.split()
    for idx, word in enumerate(words):
        is_start_of_line = (idx == 0)
        similar_words = find_top_3_processed_similar_words_m2(word, train_set_lines, is_start_of_line)
        suggestions.append((word, similar_words))
    logger.info(f"OCR text '{ocr_text}' suggestions: {suggestions}")
    return suggestions


def detect_immediate_repeated_words(text_line):
    """
    This function takes a text line as input and returns a list of immediately repeated words found in the text line.
    This includes cases where there is punctuation or whitespace between the repeated words, and insensitive.
    """
    # Match words that are repeated with or without punctuation in between, insensitive
    repeated_words = re.findall(r'\b(\w+)\b[\s\W]+\b\1\b', text_line, flags=re.IGNORECASE)
    # # Ensure the found duplicates are indeed duplicates
    # actual_repeats = [match for match in repeated_words if text_line.lower().count(match.lower()) > 1]
    return repeated_words


def detect_close_repeated_word_sequences(text_line):
    """
    This function takes a text line as input and returns a list of sets of closely repeated words found in the text line.
    It detects repeated sequences of words in the text line.
    """
    words = re.findall(r'\b\w+\b', text_line)
    repeated_word_sets = []

    # Check for repeated sequences
    for length in range(2, len(words) // 2 + 1):  # Sequence lengths from 2 to half the length of the words
        sequence_positions = defaultdict(list)

        # Record positions of each word sequence
        for i in range(len(words) - length + 1):
            sequence = ' '.join(words[i:i + length]).lower()
            sequence_positions[sequence].append(i)

        # Identify closely repeated word sets
        for positions in sequence_positions.values():
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    if positions[i + 1] - positions[i] <= length:
                        repeated_set = ' '.join(words[positions[i]:positions[i] + length])
                        if repeated_set not in repeated_word_sets:
                            repeated_word_sets.append(repeated_set)

    return repeated_word_sets


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def detect_similar_immediate_repeated_words(text_line, similarity_threshold=0.8):
    """
    This function takes a text line as input and returns a list of similar immediately repeated words found in the text line.
    The function detects words that are at least `similarity_threshold` similar and are immediately adjacent.
    """
    # Tokenize the line into words
    words = re.findall(r'\b\w+\b', text_line)

    # Store pairs of similar words
    similar_repeated_words = []

    # Compare each word with its next immediate word
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        if similar(word1, word2) >= similarity_threshold:
            similar_repeated_words.append((word1, word2))

    return similar_repeated_words


import re

def has_misplaced_punctuation(text_line):
    """
    This function checks if a text line contains any misplaced punctuation marks.
    A punctuation mark is considered misplaced if it is not immediately next to the word on its left
    or if there is not exactly one space between the punctuation mark and the word to its right.
    Additionally, it returns False if the text contains an ampersand (&) or unpaired quotation marks.

    Returns:
        bool: True if there is misplaced punctuation, otherwise False.
    """
    # Return True if the text contains both '(' and ')'
    if '(' in text_line and ')' in text_line and not '[ (' in text_line:
        return True
    elif '&' in text_line or '[' in text_line or '[ (' in text_line:
        return False
    elif text_line.startswith('-') and text_line[1:].strip() and '-' not in text_line[1:]:
        return False

    # Check for unpaired quotation marks
    if text_line.count('"') % 2 != 0:
        return False  # Unpaired quotes, so consider it as no misplaced punctuation

    # Define a pattern to match correct punctuation usage
    correct_punctuation_pattern = r'\w[.,;:?!-]\s\w|\w[.,;:?!-]$'

    # Split the text line into tokens
    tokens = text_line.split()

    for token in tokens:
        # Check if the token contains punctuation
        if re.search(r'[.,;:?!-]', token):
            # If the token doesn't match the correct punctuation pattern, it's misplaced
            if not re.search(correct_punctuation_pattern, token):
                return True

    # If no misplaced punctuation is found, return False
    return False


def check_missing_or_extra_words(original_text, corrected_text):
    """
    This function checks if any word from the original text line is missing in the corrected text line
    or if the corrected text line has extra words, ignoring punctuation marks.
    It maintains the punctuation marks of the corrected text line.

    Args:
        original_text (str): The original text line.
        corrected_text (str): The corrected text line.

    Returns:
        bool: True if any word is missing or if there are extra words in the corrected text line, otherwise False.
    """

    # Remove punctuation from the original text for comparison
    original_words = re.findall(r'\b\w+\b', original_text)

    # Remove punctuation from the corrected text for comparison
    corrected_words = re.findall(r'\b\w+\b', corrected_text)

    # Check if any word in the original text is missing in the corrected text
    for word in original_words:
        if word not in corrected_words:
            return True

    # Check if there are extra words in the corrected text
    for word in corrected_words:
        if word not in original_words:
            return True

    # If all words match and there are no extras, return False
    return False
