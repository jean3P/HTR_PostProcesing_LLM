import datetime
import os
import time

import numpy as np

from config.model_config import get_data_generator
from constants import model_outputs, results_outputs
from data_processing.data_utils import create_testing_file
from utils.flor.data.evaluation import cer_only
from utils.flor.network.model import HTRModel


def predict_and_evaluate(model, dtgen, output_path, name_file, partition):
    """ Perform prediction and evaluate the model. """
    start_time = datetime.datetime.now()

    # Perform the model prediction
    predicts, _ = model.predict(
        x=dtgen.next_test_batch(),
        steps=dtgen.steps['test'],
        ctc_decode=True,
        verbose=1
    )

    # Ensure that predicts is flattened and decoded correctly
    predicts = [dtgen.tokenizer.decode(pred[0]) if isinstance(pred, list) else dtgen.tokenizer.decode(pred) for pred in
                predicts]
    # print(f"Decoded Predicts: {predicts}")

    # Ensure the structure of ground truth is correct
    ground_truth = [x.decode() for x in dtgen.dataset['test']['gt']]
    # print(f"Ground Truth: {ground_truth}")

    # Call save_evaluation_data only after confirming the structure
    save_evaluation_data(predicts, ground_truth, dtgen, output_path, name_file, partition)
    print("Evaluation completed in: ", datetime.datetime.now() - start_time)


def save_evaluation_data(predicts, ground_truth, dtgen, output_path, dataset, partition):
    evaluation_data = []

    for i, (pred, gt) in enumerate(zip(predicts, ground_truth)):
        try:
            print(f"Processing index: {i}, Pred: {pred}, Ground Truth: {gt}")
            cer = cer_only([pred], [gt])  # Calculate Character Error Rate (CER)

            # Ensure valid path extraction
            file_name = dtgen.dataset['test']['path'][i].decode('utf-8')
            print(f"File Name: {file_name}")

            evaluation_data.append({
                "file_name": file_name,
                "ground_truth_label": gt,
                "predicted_label": pred,
                "cer": cer
            })
        except Exception as e:
            print(f"Error processing index {i}: {e}")

    # Save the evaluation data using create_testing_file
    json_file_path = create_testing_file(output_path, dataset, partition, evaluation_data)
    print(f"Evaluation results saved to {json_file_path}")


def evaluate_test_data(dataset_name, partition, output_path, dtgen, model):
    """
    Evaluate the test set using the trained model for a specific partition.

    :param hdf5_file_path: Path to the HDF5 file containing the dataset.
    :param partition: Partition of the model used for training (e.g., 'train_25', 'train_50').
    :param output_path: Directory where the results will be saved.
    """

    # Log the start of the evaluation
    print(f"Evaluating the test set for partition: {partition} (dataset: {dataset_name})")
    start_time = time.time()

    # Perform prediction and evaluation using the predict_and_evaluate function
    predict_and_evaluate(model, dtgen, output_path, dataset_name, partition)

    # Log the total elapsed time for evaluation
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for evaluating partition {partition}: {elapsed_time:.2f} seconds")


def evaluate_all_partitions(dataset_name, dtgen, model, partition):
    """
    Evaluate all partitions (train_25, train_50, train_75, train_100) on the test set.

    :param hdf5_file_path: Path to the HDF5 file containing the dataset.
    :param base_dir: Base directory where models and processors are saved.
    :param dataset_name: Name of the dataset (e.g., 'washington').
    :param output_path: Directory where the evaluation results will be saved.
    """

    evaluate_test_data(dataset_name, partition, results_outputs, dtgen, model)

    print(f"Completed evaluation for dataset: {dataset_name}")
