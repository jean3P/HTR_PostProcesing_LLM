import os
import time

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from config.TrOCR_config import HandleDataTrOCR
from data_processing.data_utils import create_testing_file
from utils.logger import log_status, setup_logger
from utils.metrics_evaluation import cer_only, wer_only
from utils.utils import device

logger = setup_logger()


def ocr(pixel_values, processor, model):
    """Perform OCR on the input image."""
    pixel_values = pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def evaluate_test_data(hdf5_file_path, processor_save_dir, model_save_dir, partition, output_path, dataset):
    """
    Evaluate the test set using the trained model and processor for a specific partition.

    :param hdf5_file_path: Path to the HDF5 file containing the dataset.
    :param processor_save_dir: Directory where the processor is saved.
    :param model_save_dir: Directory where the trained model is saved.
    :param partition: Partition of the model used for training (e.g., 'train_25', 'train_50').
    :param output_path: Directory where the results will be saved.
    """

    # Load the processor and model
    processor = TrOCRProcessor.from_pretrained(processor_save_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_save_dir)
    model.to(device)

    # Load the test set
    test_dataset = HandleDataTrOCR(hdf5_file_path, 'test', processor)

    log_status(f"Evaluating the test set for partition: {partition}", logger)
    start_time = time.time()

    results = []

    # Iterate over the test set and perform evaluation
    for i in range(len(test_dataset)):
        data = test_dataset[i]
        pixel_values = data['pixel_values'].unsqueeze(0).to(device)  # Add batch dimension

        # Perform OCR on the image
        predicted_text = ocr(pixel_values, processor, model)
        file_name = test_dataset.image_file_names[i].decode('utf-8')

        # Decode the ground truth labels from token IDs to text
        labels_tensor = data['labels']
        labels_tensor[labels_tensor == -100] = processor.tokenizer.pad_token_id  # Replace padding tokens
        ground_truth_text = processor.tokenizer.decode(labels_tensor, skip_special_tokens=True)

        # Calculate CER (Character Error Rate)
        cer = cer_only([predicted_text], [ground_truth_text])
        wer = wer_only([predicted_text], [ground_truth_text])

        # Append the results
        results.append({
            'file_name': file_name,
            'ground_truth_label': ground_truth_text,
            'predicted_label': predicted_text,
            'cer': cer,
            "wer": wer,
        })

        # Log the result details
        log_status(f"File: {file_name} | Ground Truth: {ground_truth_text} | "
                   f"Prediction: {predicted_text} | CER: {cer:.4f}", logger)

    # Save the results to a JSON file
    json_file_path = create_testing_file(output_path, dataset, partition, results)
    log_status(f"Test evaluation completed for partition: {partition}. Results saved to {json_file_path}.", logger)

    # Close the dataset
    test_dataset.close()

    elapsed_time = time.time() - start_time
    log_status(f"Elapsed time for evaluating partition {partition}: {elapsed_time:.2f} seconds", logger)


def evaluate_all_partitions(hdf5_file_path, base_dir, dataset_name, output_path):
    """
    Evaluate all partitions (train_25, train_50, train_75, train_100) on the test set.

    :param hdf5_file_path: Path to the HDF5 file containing the dataset.
    :param base_dir: Base directory where models and processors are saved.
    :param dataset_name: Name of the dataset (e.g., 'washington').
    :param output_path: Directory where the evaluation results will be saved.
    """
    partitions = ['train_25', 'train_50', 'train_75', 'train_100']

    for partition in partitions:
        model_save_dir = os.path.join(base_dir, dataset_name, partition, 'model')
        processor_save_dir = os.path.join(base_dir, dataset_name, partition, 'processor')

        if os.path.exists(model_save_dir) and os.path.exists(processor_save_dir):
            evaluate_test_data(hdf5_file_path, processor_save_dir, model_save_dir, partition, output_path, dataset_name)
        else:
            log_status(f"Model or processor missing for partition {partition}. Skipping evaluation.", logger)

    log_status(f"Completed evaluation for dataset: {dataset_name}", logger)

