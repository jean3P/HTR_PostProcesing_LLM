import os
from config.model_config import get_data_generator
from constants import outputs_hdf5
from utils.flor.network.model import HTRModel
from evaluation.evaluate_model import predict_and_evaluate


def setup_result_directories(base_dir, dataset_name, partition_name):
    """
    Creates result directories for the given dataset and partition.
    """
    main_dir = os.path.join(base_dir, dataset_name, partition_name)
    results_dir = os.path.join(main_dir, "results")
    remaining_dir = os.path.join(main_dir, "remaining_75")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(remaining_dir, exist_ok=True)

    return results_dir, remaining_dir


def initialize_data_generator(dataset_name, partition_name):
    hdf5_file_path = os.path.join(outputs_hdf5, dataset_name, f"{dataset_name}_dataset.hdf5")
    dtgen = get_data_generator(hdf5_file_path, partition_name)
    return dtgen


def load_trained_model(checkpoint_path, dtgen):
    """
    Load the trained HTR model with the given checkpoint.
    """
    model = HTRModel(
        architecture="flor",
        input_size=dtgen.input_size,
        vocab_size=dtgen.tokenizer.vocab_size,
        beam_width=10
    )
    model.compile(learning_rate=0.001)
    model.load_checkpoint(checkpoint_path)
    return model


def test_model(partition, dtgen, model, results_dir, remaining_dir):
    """
    Perform predictions and save results for both `test_xx` and its complementary test set.
    """
    # Predict and save results for `test_xx`
    predict_and_evaluate(
        model=model,
        dtgen=dtgen,
        output_path=results_dir,
        name_file=f"results_{partition}",
        partition=partition
    )

    # Predict and save results for the remaining complementary set
    remaining_partition = f"remaining_{partition.split('_')[-1]}"  # Example: `train_25` -> `remaining_75`
    predict_and_evaluate(
        model=model,
        dtgen=dtgen,
        output_path=remaining_dir,
        name_file=f"results_{remaining_partition}",
        partition=remaining_partition
    )


def run_testing_pipeline(base_dir, dataset_name, partitions, model_output_dir):
    """
    Run the entire pipeline to test the trained models on Washington dataset.
    """
    for partition in partitions:
        print(f"Processing partition: {partition}")

        # Step 1: Setup directories
        results_dir, remaining_dir = setup_result_directories(base_dir, dataset_name, partition)

        # Step 2: Initialize data generator
        dtgen = initialize_data_generator(dataset_name, partition)

        # Step 3: Load the trained model
        checkpoint_path = os.path.join(model_output_dir, dataset_name, partition, "checkpoint_weights.hdf5")
        model = load_trained_model(checkpoint_path, dtgen)

        # Step 4: Perform predictions and save results
        test_model(dataset_name, partition, dtgen, model, results_dir, remaining_dir)

        print(f"Completed partition: {partition}")

