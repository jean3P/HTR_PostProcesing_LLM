import os
import json

from constants import results_from_Flor_path, results_llm
from evaluations.metrics_evaluation import wer_only


def process_json_files_in_subdirectories(base_dir, norm_accentuation=False, norm_punctuation=False):
    """
    Traverse subdirectories (train_25, train_50, etc.) inside base_dir, process each JSON file,
    calculate WER and update the file.
    """
    # List subdirectories (train_25, train_50, etc.)
    subdirectories = ['train_25', 'train_50', 'train_75', 'train_100']

    for subdirectory in subdirectories:
        sub_dir_path = os.path.join(base_dir, subdirectory)
        if os.path.exists(sub_dir_path):
            for root, dirs, files in os.walk(sub_dir_path):
                for file in files:
                    if file.endswith(".json"):  # Only process JSON files
                        json_file_path = os.path.join(root, file)
                        update_json_with_wer(json_file_path)


def update_json_with_wer(file_path):
    """Reads a JSON file, calculates WER for each entry, and saves the updated file"""
    # Read the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Process each entry in the JSON file
    for item in data:
        # Add 'run_id' field if it doesn't exist
        if 'run_id' not in item:
            item['run_id'] = ""
        ground_truth = item['ground_truth_label']

        # Calculate WER for the OCR section
        ocr_predicted = item['OCR']['predicted_label']
        ocr_wer = wer_only([ocr_predicted], [ground_truth])
        item['OCR']['wer'] = round(ocr_wer, 3)

        # Calculate WER for the Prompt correcting section
        prompt_predicted = item['Prompt correcting']['predicted_label']
        prompt_wer = wer_only([prompt_predicted], [ground_truth])
        item['Prompt correcting']['wer'] = round(prompt_wer, 3)

    # Save the updated data back into the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# Example usage:
base_dir = os.path.join(results_llm, 'washington', 'Flor_model', 'mistral', 'method_1')  # The base directory containing train_25, train_50, etc.
process_json_files_in_subdirectories(base_dir)
