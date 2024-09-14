# main.py

import os
from config.TrOCR_config import TrainingConfig
from pipeline.pipeline_manager import main_training_pipeline
from constans import outputs_hdf5, model_outputs, results_outputs
from utils.logger import setup_logger, log_status

# Set up the logger
logger = setup_logger()


def run_training():
    datasets = ['iam']

    # Iterate over each dataset
    for dataset_name in datasets:
        # Construct the HDF5 file path for the current dataset
        hdf5_file_path = os.path.join(outputs_hdf5, dataset_name, f'{dataset_name}_dataset.hdf5')

        # Check if the HDF5 file exists
        if os.path.exists(hdf5_file_path):
            # If it exists, set up the training configuration and run the pipeline
            training_config = TrainingConfig()
            main_training_pipeline(dataset_name, hdf5_file_path, model_outputs, training_config, results_outputs)
        else:
            # Log a warning or message if the dataset file is not found
            log_status(f"Skipping dataset {dataset_name}: HDF5 file not found", logger)


if __name__ == "__main__":
    run_training()
