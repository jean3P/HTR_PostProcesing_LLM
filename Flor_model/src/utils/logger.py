# import logging
# import os
# import psutil
#
# # Setup detailed logging
# def setup_logger():
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#
#     # Create a file handler for logging to a file
#     file_handler = logging.FileHandler('training_pipeline.log')
#     file_handler.setLevel(logging.INFO)
#
#     # Create a console handler for logging to the console
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#
#     # Define a logging format
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     console_handler.setFormatter(formatter)
#
#     # Add handlers to the logger
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
#
#     return logger
#
# def log_status(step_message, logger):
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
#     logger.info(f"{step_message} | Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB | CPU usage: {psutil.cpu_percent()}%")
