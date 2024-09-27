import os
import json
import glob

from constants import results_llm
from evaluations.metrics_evaluation import cer_only, wer_only


def clean_predicted_label(predicted_label):
    # Remove unwanted phrases from the predicted_label
    unwanted_phrases = [
        "The corrected text line should be:",
        "Corrected text line:",
        "Here is the corrected text line:",
        # Add any other variations you might encounter
    ]
    was_cleaned = False  # Flag to indicate if cleaning occurred
    for phrase in unwanted_phrases:
        if phrase in predicted_label:
            predicted_label = predicted_label.replace(phrase, '').strip()
            was_cleaned = True  # Set the flag to True when cleaning occurs
    return predicted_label, was_cleaned


def process_json_file(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        # Access the necessary fields
        ground_truth_label = entry.get('ground_truth_label', '')
        prompt_correcting = entry.get('Prompt correcting', {})
        prompt_predicted_label = prompt_correcting.get('predicted_label', '')

        # Clean the predicted_label if it contains unwanted phrases
        cleaned_predicted_label, was_cleaned = clean_predicted_label(prompt_predicted_label)

        if was_cleaned:
            # Update the predicted_label
            prompt_correcting['predicted_label'] = cleaned_predicted_label

            # Recalculate CER and WER using the corrected predicted_label and ground_truth_label
            cer_corrected = cer_only([cleaned_predicted_label], [ground_truth_label])
            wer_corrected = wer_only([cleaned_predicted_label], [ground_truth_label])

            # Update the CER and WER in the Prompt correcting section
            prompt_correcting['cer'] = cer_corrected
            prompt_correcting['wer'] = wer_corrected

    # Save the updated data back to the JSON file (overwriting)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def process_json_files_in_directory(directory):
    # Get all JSON files in the directory and subdirectories
    json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)

    for file_path in json_files:
        print(f'Processing file: {file_path}')
        process_json_file(file_path)


# Usage example:
# Replace '/path/to/json/files' with the actual path to your directory
base_dir = os.path.join(results_llm, 'bentham', 'Flor_model', 'gpt-3.5-turbo',
                        'method_1')  # The base directory containing train_25, train_50_train_75, and train_100 directories
process_json_files_in_directory(base_dir)
