# src/utils/logger.py

import logging
import os


def setup_logger(log_file_path=None):
    """Sets up the logger to append to the existing log file without deleting previous logs."""
    logger_name = log_file_path if log_file_path else 'default_logger'
    logger = logging.getLogger(logger_name)

    # Debug print to verify logger setup
    print(f"Setting up logger for {logger_name}")

    # Avoid adding multiple handlers to the logger
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # Always log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        # If a log file path is provided, append to that file
        if log_file_path:
            try:
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                print(f"Directory created or already exists: {os.path.dirname(log_file_path)}")
                file_handler = logging.FileHandler(log_file_path, mode='a')  # Append mode
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(file_handler)
                print(f"Log file handler added for {log_file_path}")
            except Exception as e:
                print(f"Failed to create log file handler: {e}")

    else:
        print("Logger already has handlers. Skipping handler addition.")

    return logger

