import logging
import os

import psutil


# Setup detailed logging
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training_pipeline.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def log_status(step_message, logger):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(
        f"{step_message} | Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB | CPU usage: {psutil.cpu_percent()}%")
