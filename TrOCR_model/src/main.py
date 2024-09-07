import os

from TrOCR_training import train_and_save_model
from constans import outputs_hdf5, datasets, model_outputs

class TrainingConfig:
    BATCH_SIZE = 4  # Adjust based on your GPU memory
    LEARNING_RATE = 5e-5
    EPOCHS = 3


def get_save_directories(base_dir, dataset_name, size_train):
    """
    Create and return the save directories for model, processor, logs, and checkpoints.
    """
    dataset_dir = os.path.join(base_dir, dataset_name, size_train)
    model_save_dir = os.path.join(dataset_dir, 'model')
    processor_save_dir = os.path.join(dataset_dir, 'processor')

    # Create directories if they don't exist
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(processor_save_dir, exist_ok=True)

    return model_save_dir, processor_save_dir


def check_and_train(hdf5_file_path, base_dir, dataset_name, training_config):
    """
    Check if the HDF5 file exists and train models sequentially for train_25, train_50, train_75, and train_100.
    """
    # Define the partition sizes to train
    partitions = ['train_25', 'train_50', 'train_75', 'train_100']

    # Loop through the partitions and train sequentially
    for size_train in partitions:
        model_save_dir, processor_save_dir = get_save_directories(base_dir, dataset_name, size_train)

        # Check if model and processor already exist
        model_exists = os.path.exists(os.path.join(model_save_dir, 'pytorch_model.bin'))
        processor_exists = os.path.exists(os.path.join(processor_save_dir, 'tokenizer_config.json'))

        if model_exists and processor_exists:
            print(f"Model and Processor already exist for {dataset_name} {size_train}, skipping training.")
        else:
            print(f"Training for {dataset_name} {size_train}")
            train_and_save_model(hdf5_file_path, model_save_dir, processor_save_dir, training_config, size_train)


def train_and_save_if_needed(hdf5_file_path, base_dir, dataset_name, training_config):
    """
    Main function to check if the HDF5 file exists and start the training process if needed.
    """
    # Ensure the HDF5 file exists
    if not os.path.exists(hdf5_file_path):
        print(f"HDF5 file not found for dataset {dataset_name}. Skipping training.")
        return

    # Check and train for each partition size
    check_and_train(hdf5_file_path, base_dir, dataset_name, training_config)


dataset_name = 'washington'
hdf5_file_path = os.path.join(outputs_hdf5, datasets[0], f'{datasets[0]}_dataset.hdf5')
training_config = TrainingConfig()  # Ensure this is defined correctly elsewhere

# Call the function to handle the full process
train_and_save_if_needed(hdf5_file_path, model_outputs, dataset_name, training_config)
