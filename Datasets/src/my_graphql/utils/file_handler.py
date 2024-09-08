import os
import json
import logging
from h5py import File
from constants import splits_bentham_path, splits_washington_path, splits_iam_path, evaluations_path
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


def load_evaluation_results(name_dataset, partition):
    """Load the most recent evaluation results from a JSON file."""
    eval_dir_path = os.path.join(evaluations_path, name_dataset, partition)

    if not os.path.exists(eval_dir_path):
        logging.warning(f"Evaluation directory not found for {name_dataset} - {partition}. Path: {eval_dir_path}")
        return []

    # Find all files that match the 'results_*.json' pattern
    result_files = [f for f in os.listdir(eval_dir_path) if f.startswith('results_') and f.endswith('.json')]

    if not result_files:
        logging.warning(f"No results found for {name_dataset} - {partition}. Path: {eval_dir_path}")
        return []

    # Sort the result files by the timestamp in the filename (most recent first)
    result_files.sort(reverse=True)

    # Load the most recent results file
    most_recent_file = result_files[0]
    eval_file_path = os.path.join(eval_dir_path, most_recent_file)

    with open(eval_file_path, 'r') as eval_file:
        eval_data = json.load(eval_file)

    # Convert evaluation data to FileInfo format
    return [
        FileInfo(
            file_name=item['file_name'],
            ground_truth=item['ground_truth_label'],
            predicted_text=item['predicted_label'],
            cer=item['cer'],
            image_data=None
        )
        for item in eval_data
    ]


def calculate_cer_statistics(evaluation_data):
    """Calculate average, minimum, and maximum CER."""
    if not evaluation_data:
        return None  # No data to calculate

    cer_values = [result.cer for result in evaluation_data if result.cer is not None]

    if not cer_values:
        return None  # No valid CER values to calculate

    return {
        'average_cer': round(sum(cer_values) / len(cer_values), 3),
        'min_cer': round(min(cer_values), 3),
        'max_cer': round(max(cer_values), 3)
    }