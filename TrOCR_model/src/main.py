# main.py

import os
from config.TrOCR_config import TrainingConfig
from pipeline.pipeline_manager import main_training_pipeline
from constans import outputs_hdf5, model_outputs, results_outputs


# Example usage
def run_training():
    dataset_name = 'washington'
    hdf5_file_path = os.path.join(outputs_hdf5, dataset_name, f'{dataset_name}_dataset.hdf5')
    training_config = TrainingConfig()

    # Call the main training pipeline to handle the full process
    main_training_pipeline(dataset_name, hdf5_file_path, model_outputs, training_config, results_outputs)


if __name__ == "__main__":
    run_training()
