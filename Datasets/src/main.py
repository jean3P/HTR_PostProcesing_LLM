# main.py

import logging
import redis
from Dataset import Dataset
from constants import bentham_path, splits_bentham_path, washington_path, splits_washington_path, iam_path, \
    splits_iam_path

# Configure logging for the main module
logging.basicConfig(level=logging.INFO)

# Connect to Redis
r = redis.Redis(host='127.0.0.1', port=6379, db=0)

# Run the Flask app
if __name__ == '__main__':
    # Autoload Bentham dataset if necessary
    try:
        # Example usage
        input_size = (1024, 128, 1)
        max_text_length = 256

        # Initialize the Dataset class for Bentham
        bentham_dataset = Dataset(source=bentham_path, name="bentham")
        bentham_dataset.read_partitions()
        bentham_dataset.save_partitions(target_dir=splits_bentham_path, image_input_size=input_size,
                                        max_text_length=max_text_length)

        # Initialize the Dataset class for Washington (partition "cv0")
        washington_dataset = Dataset(source=washington_path, name="washington", partition_name="cv0")
        washington_dataset.read_partitions()
        washington_dataset.save_partitions(target_dir=splits_washington_path, image_input_size=input_size,
                                           max_text_length=max_text_length)

        # Initialize and process the IAM dataset
        iam_dataset = Dataset(source=iam_path, name="iam")
        iam_dataset.read_partitions()
        iam_dataset.save_partitions(target_dir=splits_iam_path, image_input_size=input_size,
                                    max_text_length=max_text_length)

        result = r.publish("project_channel", "datasets_done")
        logging.info(f"Message published to Redis with result: {result}")
    except Exception as e:
        logging.error(f"Error while loading the dataset: {e}")
