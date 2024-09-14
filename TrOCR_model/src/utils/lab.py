import json
import os

from constans import results_outputs


def filter_ocr_results_by_filename(ocr_results_path, file_names_path, output_path):
    """
    Filters the OCR results JSON file, keeping only the entries with filenames present in the provided file names JSON.

    Args:
    - ocr_results_path (str): Path to the OCR results JSON file.
    - file_names_path (str): Path to the JSON file with the list of filenames to keep.
    - output_path (str): Path to save the filtered OCR results.
    """
    # Load the OCR results JSON
    with open(ocr_results_path, 'r') as ocr_file:
        ocr_results = json.load(ocr_file)

    # Load the filenames JSON
    with open(file_names_path, 'r') as file_names_file:
        file_names_data = json.load(file_names_file)

    # Extract the list of filenames from the file_names_data
    file_names_list = [entry["fileName"] for entry in file_names_data]

    # Filter the OCR results based on the filenames
    filtered_results = [result for result in ocr_results if result["file_name"] in file_names_list]

    # Save the filtered results to the output path
    with open(output_path, 'w') as output_file:
        json.dump(filtered_results, output_file, indent=4)

    print(f"Filtered results saved to {output_path}")


# Example usage
ocr_results_path = os.path.join(results_outputs, 'iam', 'train_25', 'results_2024-09-08_17-59-29.json')
file_names_path = os.path.join(results_outputs, 'iam', 'train_25',
                               'file_names.json')  # Path to the JSON file with filenames
output_path = os.path.join(results_outputs, 'iam', 'train_25',
                           'filtered_ocr_results.json')  # Path to save the filtered results

filter_ocr_results_by_filename(ocr_results_path, file_names_path, output_path)
