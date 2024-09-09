# src/utils/logger.py

import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("workflow.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
