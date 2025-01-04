import json
import re
import glob
from datetime import datetime
import os


def correct_punctuation_spacing(text):
    """
    Correct the spacing of punctuation marks in the text.
    Ensures proper spacing around punctuation marks, parentheses, quotes, and brackets.
    This includes characters like . , ; : ! ? as well as (), "", '', {}, [], and |.
    """

    # Remove spaces before punctuation marks like . , ; : ! ?
    text = re.sub(r'\s([?.!,;:])', r'\1', text)

    # Ensure there is one space after punctuation marks if needed (except at the end of the sentence)
    text = re.sub(r'([?.!,;:])(\S)', r'\1 \2', text)

    # Handle parentheses: no space after opening '(' or before closing ')'
    text = re.sub(r'\(\s*', '(', text)  # No spaces after '('
    text = re.sub(r'\s*\)', ')', text)  # No spaces before ')'

    # Handle curly braces: no space after opening '{' or before closing '}'
    text = re.sub(r'\{\s*', '{', text)  # No spaces after '{'
    text = re.sub(r'\s*\}', '}', text)  # No spaces before '}'

    # Handle square brackets: no space after opening '[' or before closing ']'
    text = re.sub(r'\[\s*', '[', text)  # No spaces after '['
    text = re.sub(r'\s*\]', ']', text)  # No spaces before ']'

    # Handle quotes (both single and double quotes): no spaces inside quotes
    text = re.sub(r'\s*["\']\s*', lambda match: match.group(0).strip(), text)

    # Handle vertical bar (pipe) symbol: no spaces around '|'
    text = re.sub(r'\s*\|\s*', '|', text)

    # Remove double spaces, if they appear after correction
    text = re.sub(r'\s{2,}', ' ', text)

    return text


# print(correct_punctuation_spacing("listening in the night . Listening ( in ) vain . For the"))

def find_latest_json_files(directory):
    """
    Finds the latest 'results_empty_*.json' and 'results_*dataset*.json' files in a directory.
    Returns a dictionary with keys 'empty' and 'dataset', mapping to the latest file paths.
    """
    files = glob.glob(os.path.join(directory, 'results_*.json'))
    if not files:
        return {}

    files_with_dates = {'empty': [], 'dataset': []}
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')

    for f in files:
        basename = os.path.basename(f)
        # Use regex to extract the date string from the filename
        date_match = date_pattern.search(basename)
        if date_match:
            date_str = date_match.group(1)
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d_%H-%M-%S')
                if 'empty' in basename:
                    files_with_dates['empty'].append((f, date))
                else:
                    files_with_dates['dataset'].append((f, date))
            except ValueError:
                continue  # Skip files with invalid date formats
        else:
            continue  # Skip files without a valid date in the filename

    latest_files = {}
    for key in ['empty', 'dataset']:
        if files_with_dates[key]:
            # Get the file with the latest date
            latest_file = max(files_with_dates[key], key=lambda x: x[1])[0]
            latest_files[key] = latest_file
    return latest_files


import json
from collections import Counter


def load_llm_corrected_labels(json_file_path, mode='llm'):
    """
    Loads the labels from a JSON file based on the selected mode and returns a mapping from file names to labels.

    Modes:
        - 'llm': Use the predicted_label from the 'Prompt correcting' field.
        - 'htr': Use the predicted_label from the 'OCR' field.
        - 'best': Use the label with the best CER value (minimum CER), and calculate the percentage of labels from OCR and LLM.

    Returns:
        label_mapping (dict): A dictionary mapping file names to labels.
    """
    if mode not in {'llm', 'htr', 'best'}:
        raise ValueError("Invalid mode. Choose from 'llm', 'htr', or 'best'.")

    label_mapping = {}
    source_count = Counter()  # To count sources for the 'best' mode.

    if json_file_path:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        for item in data:
            file_name = item['file_name']

            if mode == 'llm':
                label_mapping[file_name] = item.get('Prompt correcting', {}).get('predicted_label', '')

            elif mode == 'htr':
                label_mapping[file_name] = item.get('OCR', {}).get('predicted_label', '')

            elif mode == 'best':
                # Get CER values and corresponding labels
                ocr_label = item.get('OCR', {}).get('predicted_label', '')
                ocr_cer = item.get('OCR', {}).get('cer', float('inf'))
                llm_label = item.get('Prompt correcting', {}).get('predicted_label', '')
                llm_cer = item.get('Prompt correcting', {}).get('cer', float('inf'))

                # Choose the label with the lowest CER value
                if ocr_cer <= llm_cer:
                    label_mapping[file_name] = ocr_label
                    source_count['OCR'] += 1
                else:
                    label_mapping[file_name] = llm_label
                    source_count['LLM'] += 1

    # Print statistics for 'best' mode
    if mode == 'best' and source_count:
        total_labels = sum(source_count.values())
        ocr_percentage = (source_count['OCR'] / total_labels) * 100
        llm_percentage = (source_count['LLM'] / total_labels) * 100
        print(f"Best Mode Statistics: {ocr_percentage:.2f}% labels from OCR, {llm_percentage:.2f}% labels from LLM.")

    return label_mapping

