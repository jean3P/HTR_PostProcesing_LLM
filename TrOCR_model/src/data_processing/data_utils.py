import json
import os
from datetime import datetime
from utils.logger import setup_logger, log_status

# Set up the logger
logger = setup_logger()


def get_model_and_processor_paths_by_partition(base_dir, dataset_name, partition_name):
    """
    Generate the directories for model and processor for a specific dataset and partition.

    :param base_dir: The base directory where models and processors are saved.
    :param dataset_name: The name of the dataset (e.g., 'washington', 'bentham').
    :param partition_name: The name of the partition (e.g., 'train_25', 'train_50').
    :return: Tuple containing paths to model directory and processor directory.
    """
    dataset_dir = os.path.join(base_dir, dataset_name, partition_name)
    model_save_dir = os.path.join(dataset_dir, 'model')
    processor_save_dir = os.path.join(dataset_dir, 'processor')

    return model_save_dir, processor_save_dir


def create_directories_if_needed(directories):
    """
    Create necessary directories if they do not exist.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        log_status(f"Directory created: {directory}")


def save_to_json(dict_data, path_file):
    """Saves a dictionary to a JSON file.
    Args:
        dict_data (dict): The dictionary to save.
        path_file (str): The path to the file where the dictionary will be saved.
    """
    with open(path_file, 'w') as file:
        json.dump(dict_data, file, indent=4)


def create_testing_file(base_dir, dataset, partition, result):
    # Get current date and time in the format YYYY-MM-DD_HH-MM-SS
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the directory structure (without date)
    test_dir = os.path.join(base_dir, dataset, partition)

    # Create the directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)

    # Define the result file name with the current date and time
    result_file = f"results_{current_time}.json"

    # Full path to the JSON result file
    json_file_path = os.path.join(test_dir, result_file)

    save_to_json(result, json_file_path)

    return json_file_path


def dataset_has_content(model_save_dir):
    """
    Check if the model directory for a dataset already contains any files.
    If the directory doesn't exist, assume the model hasn't been trained yet.
    """
    if not os.path.exists(model_save_dir):
        return False  # If the directory doesn't exist, return False

    return len(os.listdir(model_save_dir)) > 0  # If the directory has content, we assume the model is already trained.



