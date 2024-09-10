import os
import json
import logging
from h5py import File
from constants import splits_bentham_path, splits_washington_path, splits_iam_path, evaluations_path, llm_outputs_path
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


def load_evaluation_results(name_dataset, name_method , partition, llm_name):
    """Load the most recent evaluation results from a JSON file."""
    eval_dir_path = os.path.join(llm_outputs_path, name_dataset, llm_name, name_method, partition)

    logging.info(f"Checking evaluation directory: {eval_dir_path}")

    if not os.path.exists(eval_dir_path):
        logging.warning(f"Evaluation directory not found for {name_dataset} - {name_method} - {partition}. Path: {eval_dir_path}")
        return []

    # Find all files that match the 'results_*.json' pattern
    result_files = [f for f in os.listdir(eval_dir_path) if f.startswith('results_') and f.endswith('.json')]

    if not result_files:
        logging.warning(f"No results found for {name_dataset} - {partition}. Path: {eval_dir_path}")
        return []

    logging.info(f"Result files found: {result_files}")

    # Sort the result files by the timestamp in the filename (most recent first)
    result_files.sort(reverse=True)

    # Load the most recent results file
    most_recent_file = result_files[0]
    eval_file_path = os.path.join(eval_dir_path, most_recent_file)

    with open(eval_file_path, 'r') as eval_file:
        eval_data = json.load(eval_file)

    logging.info(f"Loading evaluation file: {eval_file_path}")
    logging.info(f"File contents: {eval_data}")

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
            image_data=None
        )
        for item in eval_data
    ]

    return evaluation_data


def calculate_cer_statistics(evaluation_data):
    """Calculate average, minimum, and maximum CER for both OCR and LLM correction."""
    if not evaluation_data:
        return None  # No data to calculate

    cer_ocr_values = [result.cer_ocr for result in evaluation_data if result.cer_ocr is not None]
    cer_llm_values = [result.cer_llm for result in evaluation_data if result.cer_llm is not None]

    if not cer_ocr_values and not cer_llm_values:
        return None  # No valid CER values to calculate

    total_cer_reduction = 0
    for cer_ocr, cer_llm in zip(cer_ocr_values, cer_llm_values):
        if cer_ocr > 0:  # To avoid division by zero
            cer_reduction = ((cer_ocr - cer_llm) / cer_ocr) * 100
            total_cer_reduction += cer_reduction

    average_cer_reduction = total_cer_reduction / len(cer_ocr_values)

    return {
        'average_cer_ocr': round(sum(cer_ocr_values) / len(cer_ocr_values), 3) if cer_ocr_values else None,
        'min_cer_ocr': round(min(cer_ocr_values), 3) if cer_ocr_values else None,
        'max_cer_ocr': round(max(cer_ocr_values), 3) if cer_ocr_values else None,
        'average_cer_llm': round(sum(cer_llm_values) / len(cer_llm_values), 3) if cer_llm_values else None,
        'min_cer_llm': round(min(cer_llm_values), 3) if cer_llm_values else None,
        'max_cer_llm': round(max(cer_llm_values), 3) if cer_llm_values else None,
        'cer_reduction_percentage': round(average_cer_reduction, 3)
    }
