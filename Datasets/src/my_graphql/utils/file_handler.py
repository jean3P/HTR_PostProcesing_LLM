# repos/HTR_PostProcesing_LLM/Datasets/my_graphql/utils/file_handler.py

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
    # Paths to the standard and combined datasets
    standard_hdf5_path = os.path.join(DATASET_PATHS.get(name_dataset), f'{name_dataset}_dataset.hdf5')
    combined_hdf5_path = os.path.join(DATASET_PATHS.get(name_dataset), f'combined_{name_dataset}_dataset.hdf5')

    # Flags to indicate where the partition was found
    partition_found = False
    hdf5_path = ''

    # First, check in the standard dataset file
    if os.path.exists(standard_hdf5_path):
        with File(standard_hdf5_path, "r") as f:
            if partition in f:
                partition_found = True
                hdf5_path = standard_hdf5_path
                dataset_file = f
                # Global total across standard partitions
                global_total = sum(len(f[f"{pt}/dt"]) for pt in ['train_100', 'train_75', 'train_50', 'train_25', 'valid', 'test'])
    else:
        # Standard dataset file does not exist
        print(f"The standard dataset file {standard_hdf5_path} does not exist.")

    # If not found, check in the combined dataset file
    if not partition_found and os.path.exists(combined_hdf5_path):
        with File(combined_hdf5_path, "r") as f:
            if partition in f:
                partition_found = True
                hdf5_path = combined_hdf5_path
                dataset_file = f
                # Global total across all partitions in the combined dataset
                global_total = sum(len(f[f"{pt}/dt"]) for pt in f.keys())
    elif not partition_found:
        # Combined dataset file does not exist or partition not found
        raise FileNotFoundError(f"Partition '{partition}' not found in both standard and combined datasets.")

    if not partition_found:
        raise ValueError(f"Partition '{partition}' not found in any dataset.")

    # Now, load the partition data from the appropriate dataset file
    with File(hdf5_path, "r") as f:
        total_count = len(f[f"{partition}/dt"])
        full_path = f.attrs.get('full_image_path', '').decode('utf-8') if isinstance(f.attrs.get('full_image_path', ''), bytes) else f.attrs.get('full_image_path', '')

        # Load partition data
        dt_data = f[f"{partition}/dt"][:number_of_rows]
        gt_data = f[f"{partition}/gt"][:number_of_rows]
        path_data = f[f"{partition}/path"][:number_of_rows]

        # Decode byte strings
        decoded_path_data = [path.decode('utf-8') if isinstance(path, bytes) else path for path in path_data]
        decoded_gt_data = [gt.decode('utf-8') if isinstance(gt, bytes) else gt for gt in gt_data]

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

    # Handle different dictionary naming patterns
    if dict_name == 'noTraining':
        search_dict_name = 'empty'
    elif dict_name == name_dataset:
        # When using the dataset's own dictionary, filename uses dataset name
        search_dict_name = name_dataset
    else:
        # For cross-dataset dictionaries, use the dictionary name
        search_dict_name = dict_name

    logging.info(f"Checking evaluation directory: {eval_dir_path}")
    logging.info(f"Searching for dictionary: {dict_name} -> filename pattern: results_{search_dict_name}_*.json")

    if not os.path.exists(eval_dir_path):
        logging.warning(
            f"Evaluation directory not found for {name_dataset} - {name_method} - {partition}. Path: {eval_dir_path}")
        return []

    # Debug: Log all files in directory
    all_files = os.listdir(eval_dir_path)
    json_files = [f for f in all_files if f.endswith('.json')]
    logging.info(f"All JSON files in directory: {json_files}")

    # Find all files that match the 'results_*.json' pattern
    result_files = [f for f in json_files if f.startswith(f'results_{search_dict_name}_')]

    # Special handling for 'empty' dictionary - also check for 'no_suggestions' files
    if search_dict_name == 'empty' and not result_files:
        # Check for 'no_suggestions' files which are equivalent to 'empty'
        no_suggestions_files = [f for f in json_files if f.startswith('results_no_suggestions_')]
        if no_suggestions_files:
            logging.info(f"No 'results_empty_*.json' files found, but found {len(no_suggestions_files)} "
                        f"'results_no_suggestions_*.json' files (treating as equivalent)")
            result_files = no_suggestions_files

    if not result_files:
        # Try fallback: if looking for cross-dataset dictionary, check if dataset's own dictionary exists
        if dict_name != name_dataset and dict_name != 'noTraining':
            logging.info(f"No files found for {dict_name}, trying fallback to dataset dictionary: {name_dataset}")
            fallback_files = [f for f in json_files if f.startswith(f'results_{name_dataset}_')]
            if fallback_files:
                result_files = fallback_files
                search_dict_name = name_dataset
                logging.info(f"Using fallback files with pattern: results_{name_dataset}_*.json")

    if not result_files:
        logging.warning(
            f"No results found for {name_dataset} - {partition} - dictionary: {dict_name}. Path: {eval_dir_path}")
        logging.warning(f"Searched for pattern: results_{search_dict_name}_*.json")
        if search_dict_name == 'empty':
            logging.warning(f"Also searched for pattern: results_no_suggestions_*.json")
        return []

    logging.info(f"Result files found: {len(result_files)} - {result_files}")

    # Sort the result files by the timestamp in the filename (most recent first)
    result_files.sort(reverse=True)

    # Load the most recent results file
    most_recent_file = result_files[0]
    eval_file_path = os.path.join(eval_dir_path, most_recent_file)

    # Check if file exists and has content
    if not os.path.exists(eval_file_path):
        logging.error(f"Evaluation file does not exist: {eval_file_path}")
        return []

    file_size = os.path.getsize(eval_file_path)
    if file_size == 0:
        logging.error(f"Evaluation file is empty: {eval_file_path}")
        return []

    logging.info(f"Reading file: {eval_file_path} (size: {file_size} bytes)")

    try:
        with open(eval_file_path, 'r', encoding='utf-8') as eval_file:
            # Read raw content first for debugging
            content = eval_file.read()
            logging.info(f"File raw content (first 200 chars): {content[:200]}")

            if not content.strip():
                logging.error(f"File is empty or contains only whitespace: {eval_file_path}")
                return []

            # Parse JSON
            eval_data = json.loads(content)

            if not eval_data:
                logging.warning(f"JSON file contains no data: {eval_file_path}")
                return []

            if not isinstance(eval_data, list):
                logging.error(f"Expected list but got {type(eval_data)}: {eval_file_path}")
                return []

            logging.info(f"Successfully loaded {len(eval_data)} evaluation records")

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {eval_file_path}: {e}")
        logging.error(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
        # Try to show the problematic content around the error
        try:
            lines = content.split('\n')
            if e.lineno <= len(lines):
                logging.error(f"Problematic line: {lines[e.lineno - 1]}")
        except:
            pass
        return []
    except UnicodeDecodeError as e:
        logging.error(f"Unicode decode error in {eval_file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error reading {eval_file_path}: {e}")
        return []

    # Get run_id safely
    run_id = ""
    try:
        if eval_data and len(eval_data) > 0 and isinstance(eval_data[0], dict):
            run_id = eval_data[0].get("run_id", "")
        logging.info(f"Extracted run_id: {run_id}")
    except Exception as e:
        logging.warning(f"Could not extract run_id: {e}")

    def parse_confidence(confidence_str):
        """Helper function to parse the confidence score and handle non-integer values."""
        try:
            # Try to convert confidence value to an integer
            return int(confidence_str)
        except (ValueError, TypeError):
            # Return 0 if conversion fails
            return 0

    # Process evaluation data with error handling
    evaluation_data = []
    try:
        for i, item in enumerate(eval_data):
            try:
                if not isinstance(item, dict):
                    logging.warning(f"Skipping non-dict item at index {i}: {type(item)}")
                    continue

                # Check required fields exist
                required_fields = ['file_name', 'ground_truth_label', 'OCR', 'Prompt correcting']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    logging.warning(f"Skipping item {i} due to missing fields: {missing_fields}")
                    continue

                # Check OCR and Prompt correcting structures
                if not isinstance(item.get('OCR'), dict):
                    logging.warning(f"Skipping item {i}: OCR field is not a dict")
                    continue

                if not isinstance(item.get('Prompt correcting'), dict):
                    logging.warning(f"Skipping item {i}: 'Prompt correcting' field is not a dict")
                    continue

                # Create FileInfo object with safe field access
                file_info = FileInfo(
                    file_name=item.get('file_name', ''),
                    ground_truth=item.get('ground_truth_label', ''),
                    predicted_text_ocr=item.get('OCR', {}).get('predicted_label', ''),
                    cer_ocr=item.get('OCR', {}).get('cer', 0),
                    predicted_text_llm=item.get('Prompt correcting', {}).get('predicted_label', ''),
                    cer_llm=item.get('Prompt correcting', {}).get('cer', 0),
                    confidence=parse_confidence(item.get('Prompt correcting', {}).get('confidence', 0)),
                    justification=item.get('Prompt correcting', {}).get('justification', ''),
                    wer_ocr=item.get('OCR', {}).get('wer', 0),
                    wer_llm=item.get('Prompt correcting', {}).get('wer', 0),
                    run_id=run_id,
                    image_data=None
                )
                evaluation_data.append(file_info)

            except Exception as e:
                logging.error(f"Error processing evaluation item {i}: {e}")
                continue

        logging.info(f"Successfully processed {len(evaluation_data)} evaluation records out of {len(eval_data)} total")

    except Exception as e:
        logging.error(f"Error processing evaluation data: {e}")
        return []

    return evaluation_data

# Update the return statement in calculate_cer_statistics to match GraphQL field names:

def calculate_cer_statistics(evaluation_data):
    """Calculate average, minimum, and maximum CER for both OCR and LLM correction."""
    if not evaluation_data:
        return None  # No data to calculate

    # Collect valid CER and WER values for both OCR and LLM
    cer_ocr_values = [result.cer_ocr for result in evaluation_data if result.cer_ocr is not None]
    cer_llm_values = [result.cer_llm for result in evaluation_data if result.cer_llm is not None]
    wer_ocr_values = [result.wer_ocr for result in evaluation_data if result.wer_ocr is not None]
    wer_llm_values = [result.wer_llm for result in evaluation_data if result.wer_llm is not None]
    confidence_values = [result.confidence for result in evaluation_data if result.confidence is not None]

    # Calculate sum of all CER and WER values
    total_cer_ocr = sum(cer_ocr_values) if cer_ocr_values else 0
    total_cer_llm = sum(cer_llm_values) if cer_llm_values else 0
    total_wer_ocr = sum(wer_ocr_values) if wer_ocr_values else 0
    total_wer_llm = sum(wer_llm_values) if wer_llm_values else 0
    total_confidence = sum(confidence_values) if confidence_values else 0

    # Calculate CER and WER reduction percentages only if there are valid values
    cer_reduction_percentage = ((total_cer_ocr - total_cer_llm) / total_cer_ocr * 100) if total_cer_ocr > 0 else None
    wer_reduction_percentage = ((total_wer_ocr - total_wer_llm) / total_wer_ocr * 100) if total_wer_ocr > 0 else None

    # 🔧 FIXED: Return with snake_case field names to match GraphQL schema
    return {
        'min_cer_ocr': round(min(cer_ocr_values), 3) if cer_ocr_values else None,
        'max_cer_ocr': round(max(cer_ocr_values), 3) if cer_ocr_values else None,
        'min_cer_llm': round(min(cer_llm_values), 3) if cer_llm_values else None,
        'max_cer_llm': round(max(cer_llm_values), 3) if cer_llm_values else None,
        'average_cer_llm': round(total_cer_llm / len(cer_llm_values), 3) if cer_llm_values else None,
        'average_wer_llm': round(total_wer_llm / len(wer_llm_values), 3) if wer_llm_values else None,
        'average_cer_ocr': round(total_cer_ocr / len(cer_ocr_values), 3) if cer_ocr_values else None,
        'average_wer_ocr': round(total_wer_ocr / len(wer_ocr_values), 3) if wer_ocr_values else None,
        'average_confidence': round(total_confidence / len(confidence_values), 3) if confidence_values else None,
        'cer_reduction_percentage': round(cer_reduction_percentage, 3) if cer_reduction_percentage is not None else None,
        'wer_reduction_percentage': round(wer_reduction_percentage, 3) if wer_reduction_percentage is not None else None
    }


def retrieve_log_info(log_file, run_id):
    log_entries = []
    capture = False  # Flag to start capturing logs

    if not os.path.exists(log_file):
        return f"Log file '{log_file}' not found."

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
