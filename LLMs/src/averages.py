import os
import json
from constants import results_llm

def calculate_average_wer_in_file(file_path):
    """
    Calculate the average WER for both OCR and Prompt correcting for a single JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary containing the average WER for OCR and Prompt correcting for the file.
    """
    ocr_wer_values = []
    prompt_wer_values = []

    # Open and load the JSON file
    with open(file_path, 'r') as json_file:
        try:
            evaluation_data = json.load(json_file)

            # Extract WER values from the JSON file
            for result in evaluation_data:
                ocr_wer = result['OCR'].get('wer')
                prompt_wer = result['Prompt correcting'].get('wer')

                if ocr_wer is not None:
                    ocr_wer_values.append(ocr_wer)
                if prompt_wer is not None:
                    prompt_wer_values.append(prompt_wer)

        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")
            return None

    # Calculate averages for OCR and Prompt correcting WER
    average_ocr_wer = sum(ocr_wer_values) / len(ocr_wer_values) if ocr_wer_values else 0
    average_prompt_wer = sum(prompt_wer_values) / len(prompt_wer_values) if prompt_wer_values else 0

    return {
        'file': os.path.basename(file_path),
        'average_ocr_wer': round(average_ocr_wer, 3),
        'average_prompt_wer': round(average_prompt_wer, 3)
    }


def calculate_average_wer_in_directory(directory_path):
    """
    Calculate and print the average WER for each JSON file in a directory.

    Args:
        directory_path (str): Path to the directory containing JSON files.
    """
    # Iterate over all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.json'):  # Process only JSON files
            file_path = os.path.join(directory_path, file_name)
            # Calculate WER for the individual file
            wer_stats = calculate_average_wer_in_file(file_path)
            if wer_stats:
                print(f"WER statistics for {file_name}: {wer_stats}")


# Usage example
# Replace '/path/to/json/files' with the actual path to your directory
base_dir = os.path.join(results_llm, 'iam', 'Flor_model', 'mistral', 'method_1')

# Iterate over subdirectories (train_25, train_50, etc.)
for sub_dir in ['train_25', 'train_50', 'train_75', 'train_100']:
    dir_path = os.path.join(base_dir, sub_dir)
    if os.path.exists(dir_path):
        print(f"\nProcessing {sub_dir} directory:")
        calculate_average_wer_in_directory(dir_path)
