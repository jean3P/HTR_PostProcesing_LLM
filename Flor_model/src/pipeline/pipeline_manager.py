import os
import time

from config.model_config import train_model, get_data_generator
from constants import model_outputs, results_outputs
from data_processing.data_utils import dataset_has_content, create_directories_if_needed
from evaluation.evaluate_model import evaluate_all_partitions


def train_for_partition(hdf5_file_path, dataset_name, partition):
    """
    Train the model for a specific dataset and partition if needed.
    """
    print(f"Starting training check for {dataset_name} - {partition}")
    start_time = time.time()

    # Create a data generator for the specific partition
    dtgen = get_data_generator(hdf5_file_path, partition)

    # Get save directories for the model
    save_dir = os.path.join(model_outputs, dataset_name, partition)
    model_save_dir = os.path.join(save_dir, 'model_flor')
    checkpoint_save_dir = os.path.join(model_save_dir, 'checkpoint_weights.hdf5')

    if dataset_has_content(model_save_dir):
        print(f"Model already exists for {dataset_name} - {partition}, skipping training.")
        return
    else:
        create_directories_if_needed([model_save_dir])

        print(f"Training started for {dataset_name} - {partition}")

        # Train the model and save it
        try:
            model = train_model(dtgen, model_save_dir, checkpoint_save_dir)
            print(f"Training completed for {dataset_name} - {partition}")
            # test_dir = os.path.join(results_outputs, dataset_name, partition)
            # if not dataset_has_content(test_dir):
            evaluate_all_partitions(dataset_name, dtgen, model, partition)
        except Exception as e:
            print(f"Training failed for {dataset_name} - {partition} due to: {e}")

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for {dataset_name} - {partition}: {elapsed_time:.2f} seconds")


def train_for_all_partitions(hdf5_file_path, dataset_name):
    """
    Train models for all partitions (25%, 50%, 75%, 100%) for a specific dataset.
    """
    partitions = ['train_25', 'train_50', 'train_75', 'train_100']
    # partitions = ['train_25']
    for partition in partitions:
        train_for_partition(hdf5_file_path, dataset_name, partition)


def main_training_pipeline(dataset_name, hdf5_file_path):
    """
    Main pipeline to handle training and evaluation for a dataset.
    """
    print(f"Starting training pipeline for {dataset_name}")

    # Train the model for all partitions
    train_for_all_partitions(hdf5_file_path, dataset_name)

    print(f"Pipeline completed for {dataset_name}")
