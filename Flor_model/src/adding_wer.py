import os
import json

from constants import results_outputs
from utils.flor.data.evaluation import wer_only, cer_only


# Assuming wer_only is already defined and available here
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

    # Prepare the lists of ground truth and predicted labels
    ground_truths = [item['ground_truth_label'] for item in data]
    predictions = [item['predicted_label'] for item in data]

    # Calculate WER for each pair using the wer_only function
    for item, gt, pred in zip(data, ground_truths, predictions):
        cer = cer_only([pred], [gt])
        wer = wer_only([pred], [gt])
        # Update the data structure with the WER result
        item['cer'] = round(cer, 3)
        item['wer'] = round(wer, 3)

    # Save the updated data back into the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# Example usage:
base_dir = os.path.join(results_outputs, 'washington')  # The base directory containing train_25, train_50, etc.
process_json_files_in_subdirectories(base_dir)
