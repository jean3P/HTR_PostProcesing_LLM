import time
from data_processing.data_utils import get_model_and_processor_paths_by_partition, dataset_has_content, \
    create_directories_if_needed
from evaluation.evaluate_model import evaluate_all_partitions
from train.TrOCR_training import train_and_save_model
from utils.logger import setup_logger, log_status
from utils.utils import validate_hdf5_file

# Set up the logger
logger = setup_logger()


def train_for_partition(hdf5_file_path, base_dir, dataset_name, partition, training_config):
    """
    Train the model for a specific dataset and partition if needed.
    """
    log_status(f"Starting training check for {dataset_name} - {partition}", logger)
    start_time = time.time()

    # Get save directories for the model and processor for this partition
    model_save_dir, processor_save_dir = get_model_and_processor_paths_by_partition(base_dir, dataset_name, partition)

    # Check if model directory already contains files for this dataset
    if dataset_has_content(model_save_dir):
        log_status(f"Model and Processor already exist for {dataset_name}, skipping training for all partitions.", logger)
        return
    else:
        # Create directories if needed
        create_directories_if_needed([model_save_dir, processor_save_dir])

        # Log the status before starting training
        log_status(f"Training started for {dataset_name} - {partition}", logger)

        # Train and save the model and processor
        try:
            train_and_save_model(hdf5_file_path, model_save_dir, processor_save_dir, training_config, partition)
            log_status(f"Training completed for {dataset_name} - {partition}", logger)
        except Exception as e:
            logger.error(f"Training failed for {dataset_name} - {partition} due to: {e}")

        # Log the time taken for the partition
        elapsed_time = time.time() - start_time
        log_status(f"Elapsed time for {dataset_name} - {partition}: {elapsed_time:.2f} seconds", logger)

        # Log the final status after completion
        log_status(f"Completed training for {dataset_name} - {partition}", logger)


def train_for_all_partitions(hdf5_file_path, base_dir, dataset_name, training_config):
    """
    Train models for all partitions (e.g., train_25, train_50) for a specific dataset.
    """
    partitions = ['train_25', 'train_50', 'train_75', 'train_100']

    # Train for each partition unless the dataset already has content
    for partition in partitions:
        train_for_partition(hdf5_file_path, base_dir, dataset_name, partition, training_config)


def main_training_pipeline(dataset_name, hdf5_file_path, base_dir, training_config, output_path):
    """
    Main pipeline to handle the training process for a dataset.
    """
    log_status(f"Starting the training pipeline for dataset: {dataset_name}", logger)
    log_status("Pipeline initialization", logger)

    if validate_hdf5_file(hdf5_file_path, dataset_name):
        train_for_all_partitions(hdf5_file_path, base_dir, dataset_name, training_config)
    else:
        logger.warning(f"Training skipped for {dataset_name} due to missing HDF5 file.")

    log_status(f"Starting evaluation for dataset: {dataset_name}", logger)
    evaluate_all_partitions(hdf5_file_path, base_dir, dataset_name, output_path)

    log_status("Pipeline completed", logger)
