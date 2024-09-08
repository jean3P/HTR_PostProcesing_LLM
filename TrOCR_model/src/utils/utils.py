import gc
import os

import torch

from utils.logger import setup_logger, log_status

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set up the logger
logger = setup_logger()


def clear_cuda_cache():
    torch.cuda.empty_cache()  # Clear CUDA cache
    gc.collect()


def validate_hdf5_file(hdf5_file_path, dataset_name):
    """
    Validate that the HDF5 file exists.
    """
    if not os.path.exists(hdf5_file_path):
        log_status(f"HDF5 file not found for dataset {dataset_name}. Skipping training.", logger)
        return False
    log_status(f"HDF5 file found for {dataset_name}.", logger)
    return True
