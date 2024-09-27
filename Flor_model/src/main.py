# main.py

import os
from constants import model_outputs, outputs_hdf5, results_outputs
from pipeline.pipeline_manager import main_training_pipeline
# from utils.logger import setup_logger, log_status
import tensorflow as tf
import psutil

# Set up the logger
# logger = setup_logger()


# Set logging level to show all messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING logs
#
# # Optionally set logging verbosity using TensorFlow's internal logging system
# tf.get_logger().setLevel('INFO')
#
# # Check for available GPUs
# Use the logger inside functions
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    raise SystemError("GPU device not found")

print("Found GPU at: {}".format(device_name))


def run_training():
    datasets = ['iam']
    print(f"DATASET: {datasets}")

    # Iterate over each dataset
    for dataset_name in datasets:
        # Construct the HDF5 file path for the current dataset
        hdf5_file_path = os.path.join(outputs_hdf5, dataset_name, f'{dataset_name}_dataset.hdf5')

        # Check if the HDF5 file exists
        if os.path.exists(hdf5_file_path):
            print(f"PATH: {hdf5_file_path}")
            # If it exists, set up the training configuration and run the pipeline
            main_training_pipeline(dataset_name, hdf5_file_path)
        else:
            # Log a warning or message if the dataset file is not found
            print(f"Skipping dataset {dataset_name}: HDF5 file not found")


if __name__ == "__main__":
    run_training()
