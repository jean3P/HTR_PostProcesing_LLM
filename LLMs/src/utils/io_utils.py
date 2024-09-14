# src/utils/io_utils.py

from datetime import datetime
import json
import os

from constants import results_from_TrOCR_path, results_from_Flor_path


def save_to_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def load_from_json(file_path):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)


def get_latest_result(test_dir):
    # List all files in the test directory
    files = [f for f in os.listdir(test_dir) if f.startswith('results_') and f.endswith('.json')]

    # If there are no result files, return None
    if not files:
        return None

    # Sort the files by name (which includes the date and time)
    files.sort()

    # The latest file will be the last one in the sorted list
    latest_file = files[-1]

    return os.path.join(test_dir, latest_file)


def create_testing_file(base_dir, dataset, partition, result, train_suggestion, llm_name, name_method):
    # Get current date and time in the format YYYY-MM-DD_HH-MM-SS
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the directory structure (without date)
    test_dir = os.path.join(base_dir, dataset, llm_name, name_method, partition)

    # Create the directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)

    # Define the result file name with the current date and time
    if train_suggestion == '':
        train_suggestion = 'empty'
    result_file = f"results_{train_suggestion}_{current_time}.json"

    # Full path to the JSON result file
    json_file_path = os.path.join(test_dir, result_file)

    save_to_json(result, json_file_path)

    return json_file_path


def get_latest_result_for_datasets(llms, datasets, train_sizes, type_model):
    results = []
    name_model_ocr = ""
    if type_model == "Flor_model":
        name_model_ocr = results_from_Flor_path
    elif type_model == "TrOCR_model":
        name_model_ocr = results_from_TrOCR_path
    for llm in llms:
        for dataset in datasets:
            for train_size in train_sizes:
                test_dir = os.path.join(name_model_ocr, dataset, train_size)
                latest_result_path = get_latest_result(test_dir)
                if latest_result_path:
                    results.append((llm, dataset, train_size, latest_result_path))
                else:
                    print(f"No result files found for {dataset} with {train_size}.")
    return results
