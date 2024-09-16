import os
import json
import logging
import re

from h5py import File
from constants import splits_bentham_path, splits_washington_path, splits_iam_path, llm_outputs_path
from my_graphql.types import FileInfo

# Define your paths here
DATASET_PATHS = {
    'bentham': splits_bentham_path,
    'washington': splits_washington_path,
    'iam': splits_iam_path
}


def load_partition_data(name_dataset, partition, number_of_rows):
    """Load partition data from HDF5 file."""
    hdf5_path = os.path.join(DATASET_PATHS.get(name_dataset), f'{name_dataset}_dataset.hdf5')

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"The dataset file {hdf5_path} does not exist.")

    with File(hdf5_path, "r") as f:
        if partition not in f:
            raise ValueError(f"Partition '{partition}' not found in the dataset.")

        # Global total across all partitions
        global_total = sum(len(f[f"{pt}/dt"]) for pt in ['train_100', 'train_75', 'train_50', 'train_25', 'valid', 'test'])

        total_count = len(f[f"{partition}/dt"])
        full_path = f.attrs['full_image_path']

        # Load partition data
        dt_data = f[f"{partition}/dt"][:number_of_rows]
        gt_data = f[f"{partition}/gt"][:number_of_rows]
        path_data = f[f"{partition}/path"][:number_of_rows]

        # Decode byte strings
        decoded_path_data = [path.decode('utf-8') for path in path_data]
        decoded_gt_data = [gt.decode('utf-8') for gt in gt_data]

        partition_data = [
            FileInfo(
                file_name=path,
                ground_truth=gt,
                image_data=list(dt.flatten())
            )
            for dt, gt, path in zip(dt_data, decoded_gt_data, decoded_path_data)
        ]

    return partition_data, global_total, full_path, total_count


def load_evaluation_results(name_dataset, name_method, partition, htr_model, llm_name, dict_name):
    """Load the most recent evaluation results from a JSON file."""
    eval_dir_path = os.path.join(llm_outputs_path, name_dataset, htr_model, llm_name, name_method, partition)

    if dict_name == 'noTraining':
        dict_name = 'empty'
    logging.info(f"Checking evaluation directory: {eval_dir_path}")

    if not os.path.exists(eval_dir_path):
        logging.warning(f"Evaluation directory not found for {name_dataset} - {name_method} - {partition}. Path: {eval_dir_path}")
        return []

    # Find all files that match the 'results_*.json' pattern
    result_files = [f for f in os.listdir(eval_dir_path) if f.startswith(f'results_{dict_name}_') and f.endswith('.json')]

    if not result_files:
        logging.warning(f"No results found for {name_dataset} - {partition} - dictionary: {dict_name}. Path: {eval_dir_path}")
        return []

    logging.info(f"Result files found: {len(result_files)}")

    # Sort the result files by the timestamp in the filename (most recent first)
    result_files.sort(reverse=True)

    # Load the most recent results file
    most_recent_file = result_files[0]
    eval_file_path = os.path.join(eval_dir_path, most_recent_file)

    with open(eval_file_path, 'r') as eval_file:
        eval_data = json.load(eval_file)

    logging.info(f"Loading evaluation file: {eval_file_path}")
    logging.info(f"File contents: {eval_data}")
    run_id = eval_data[0].get("run_id")
    # Convert evaluation data to FileInfo format
    evaluation_data = [
        FileInfo(
            file_name=item['file_name'],
            ground_truth=item['ground_truth_label'],
            predicted_text_ocr=item['OCR']['predicted_label'],
            cer_ocr=item['OCR']['cer'],
            predicted_text_llm=item['Prompt correcting']['predicted_label'],
            cer_llm=item['Prompt correcting']['cer'],
            confidence=item['Prompt correcting']['confidence'],
            justification=item['Prompt correcting']['justification'],
            wer_ocr=item['OCR']['wer'],
            wer_llm=item['Prompt correcting']['wer'],
            run_id=run_id,
            image_data=None
        )
        for item in eval_data
    ]

    return evaluation_data


def calculate_cer_statistics(evaluation_data):
    """Calculate average, minimum, and maximum CER for both OCR and LLM correction."""
    if not evaluation_data:
        return None  # No data to calculate

    # Collect valid CER values for both OCR and LLM
    cer_ocr_values = [result.cer_ocr for result in evaluation_data if result.cer_ocr is not None]
    cer_llm_values = [result.cer_llm for result in evaluation_data if result.cer_llm is not None]
    wer_ocr_values = [result.wer_ocr for result in evaluation_data if result.wer_ocr is not None]
    wer_llm_values = [result.wer_llm for result in evaluation_data if result.wer_llm is not None]

    # Ensure there are valid CER values for both OCR and LLM
    if not cer_ocr_values or not cer_llm_values:
        return None  # No valid CER values to calculate

    # Sum all CER and WER values
    total_cer_ocr = sum(cer_ocr_values)
    total_cer_llm = sum(cer_llm_values)
    total_wer_ocr = sum(wer_ocr_values)
    total_wer_llm = sum(wer_llm_values)

    # Calculate overall CER and WER reduction percentages
    cer_reduction_percentage = ((total_cer_ocr - total_cer_llm) / total_cer_ocr) * 100 if total_cer_ocr > 0 else 0
    wer_reduction_percentage = ((total_wer_ocr - total_wer_llm) / total_wer_ocr) * 100 if total_wer_ocr > 0 else 0

    # Return the calculated statistics
    return {
        'average_cer_ocr': round(sum(cer_ocr_values) / len(cer_ocr_values), 3) if cer_ocr_values else None,
        'min_cer_ocr': round(min(cer_ocr_values), 3) if cer_ocr_values else None,
        'max_cer_ocr': round(max(cer_ocr_values), 3) if cer_ocr_values else None,
        'average_cer_llm': round(sum(cer_llm_values) / len(cer_llm_values), 3) if cer_llm_values else None,
        'min_cer_llm': round(min(cer_llm_values), 3) if cer_llm_values else None,
        'max_cer_llm': round(max(cer_llm_values), 3) if cer_llm_values else None,
        'average_wer_llm': round(sum(wer_llm_values) / len(wer_llm_values), 3) if wer_llm_values else None,
        'average_wer_ocr': round(sum(wer_ocr_values) / len(wer_ocr_values), 3) if wer_ocr_values else None,
        'cer_reduction_percentage': round(cer_reduction_percentage, 3),
        'wer_reduction_percentage': round(wer_reduction_percentage, 3)
    }


def retrieve_log_info(log_file, run_id):
    log_entries = []
    capture = False  # Flag to start capturing logs

    # Regex patterns to detect the start and end of log blocks
    run_start_pattern = re.compile(
        rf"=== Running for '(?P<dataset>.+?)' with '(?P<train_size>.+?)' and suggestion dictionary '(?P<dict_suggestion>.+?)' "
        rf"\| (?P<method_name>.+?) \| Run ID: {run_id} ==="
    )
    run_complete_pattern = re.compile(
        rf"=== Evaluation for '(?P<dataset>.+?)' with '(?P<train_size>.+?)' and suggestion dictionary '(?P<dict_suggestion>.+?)' "
        rf"completed and results saved \| (?P<method_name>.+?) \| Run ID: {run_id} ==="
    )

    # Read the log file
    with open(log_file, 'r') as file:
        for line in file:
            # Check if we found the start of the run_id block
            if run_start_pattern.search(line):
                capture = True  # Start capturing logs for the specified run_id
                log_entries.append(line.strip())  # Include the start line

            # Capture all subsequent lines related to that run_id
            if capture:
                log_entries.append(line.strip())

            # If we find the completion log entry for the same run_id, stop capturing
            if run_complete_pattern.search(line) and capture:
                break  # Stop capturing after the complete line is found

    # Join all log entries into a single string to return
    return "\n".join(log_entries)

