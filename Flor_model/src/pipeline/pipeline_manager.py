# ./pipeline/pipeline_manager.py

import os
import time

import h5py

from config.flor_parameters import HTRFlorConfig
from config.model_config import train_model, get_data_generator
from constants import model_outputs, results_outputs
from data_processing.data_utils import dataset_has_content, create_directories_if_needed
from evaluation.evaluate_model import evaluate_all_partitions
from utils.flor.network.model import HTRModel


def partition_exists(hdf5_file_path, partition):
    """
    Check if a partition exists in the HDF5 dataset.
    """
    with h5py.File(hdf5_file_path, 'r') as hf:
        return partition in hf


def train_for_partition(hdf5_file_path, dataset_name, partition):
    """
    Train the model for a specific dataset and partition if needed.
    """
    print(f"Starting training check for {dataset_name} - {partition}")
    start_time = time.time()

    # Check if the partition exists
    if not partition_exists(hdf5_file_path, partition):
        print(f"Partition {partition} does not exist in the dataset {dataset_name}, skipping.")
        return  # Skip training for this partition

    # Create a data generator for the specific partition
    dtgen = get_data_generator(hdf5_file_path, partition)

    # Get save directories for the model
    save_dir = os.path.join(model_outputs, dataset_name, partition)
    model_save_dir = os.path.join(save_dir, 'model_flor')
    checkpoint_save_dir = os.path.join(model_save_dir, 'checkpoint_weights.hdf5')

    if dataset_has_content(model_save_dir):
        print(f"Model already exists for {dataset_name} - {partition}, skipping training.")
        # Initialize the model with the same parameters used during training
        model = HTRModel(
            architecture="flor",
            input_size=HTRFlorConfig.INPUT_SIZE,
            vocab_size=dtgen.tokenizer.vocab_size,
            beam_width=HTRFlorConfig.BEAM_WIDTH,
            stop_tolerance=HTRFlorConfig.STOP_TOLERANCE,
            reduce_tolerance=HTRFlorConfig.REDUCE_TOLERANCE,
            reduce_factor=HTRFlorConfig.REDUCE_FACTOR
        )
        # Compile the model before loading weights
        model.compile(learning_rate=HTRFlorConfig.LEARNING_RATE)
        # Load the trained weights
        model.load_checkpoint(target=checkpoint_save_dir)
        # Proceed to evaluate the model
        evaluate_all_partitions(dataset_name, dtgen, model, partition)
    else:
        create_directories_if_needed([model_save_dir])

        print(f"Training started for {dataset_name} - {partition}")

        # Train the model and save it
        try:
            model = train_model(dtgen, model_save_dir, checkpoint_save_dir)
            print(f"Training completed for {dataset_name} - {partition}")
            # Evaluate the model after training
            evaluate_all_partitions(dataset_name, dtgen, model, partition)
        except Exception as e:
            print(f"Training failed for {dataset_name} - {partition} due to: {e}")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for {dataset_name} - {partition}: {elapsed_time:.2f} seconds")


def train_for_all_partitions(hdf5_file_path, dataset_name):
    """
    Train models for all partitions (25%, 50%, 75%, 100%) for a specific dataset.
    """
    partitions = ['train_25', 'train_50', 'train_75', 'train_100', 'train_25_empty_remaining_75',
                  'train_25_remaining_75', 'train_50_empty_remaining_50', 'train_50_remaining_50',
                  'train_75_empty_remaining_25', 'train_75_remaining_25']
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
