# repos/HTR_PostProcesing_LLM/Datasets/Dataset.py

import os
import html
import multiprocessing
from functools import partial
import numpy as np
import h5py
from tqdm import tqdm

from constants import llm_outputs_path
from utils import preproc as pp
from utils.text_processing import correct_punctuation_spacing, find_latest_json_files, load_llm_corrected_labels


class Dataset:
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name, partition_name="cv1"):
        """
        Initialize the Dataset object.
        :param source: Path to the dataset source.
        :param name: Name of the dataset.
        :param partition_name: Partition name (default is 'cv1').
        """
        self.source = source
        self.name = name
        self.dataset = None
        self.partition_name = partition_name
        self.partitions = [
            'train_100', 'train_75', 'train_50', 'train_25',
            'valid', 'test', 'test_75', 'test_50', 'test_25'
        ]

    def read_partitions(self):
        """Read images and sentences from dataset"""
        if self.name == "bentham":
            dataset = self._bentham(self.partition_name)  # Call Bentham processor
        elif self.name == "washington":
            dataset = self._washington(self.partition_name)  # Call Washington processor
        elif self.name == "iam":
            print("Loading IAM dataset...")  # Debug print for IAM loading
            dataset = self._iam(self.partition_name)  # Call IAM processor
            print(f"IAM dataset loaded with {len(dataset['train']['dt'])} training samples.")

        if not self.dataset:
            self.dataset = self._init_dataset()

        total_train = len(dataset['train']['dt'])  # Get the total number of training examples
        print(f"Total training samples in {self.name}: {total_train}")

        # Split the training data into subsets of 100%, 75%, 50%, and 25%
        split_indices = {
            'train_100': total_train,
            'train_75': int(0.75 * total_train),
            'train_50': int(0.50 * total_train),
            'train_25': int(0.25 * total_train)
        }

        # Assign data to each train subset
        for subset in split_indices:
            self.dataset[subset]['path'] += dataset['train']['path'][:split_indices[subset]]
            self.dataset[subset]['dt'] += dataset['train']['dt'][:split_indices[subset]]
            self.dataset[subset]['gt'] += dataset['train']['gt'][:split_indices[subset]]

        # Add validation and test sets
        for y in ['valid', 'test']:
            self.dataset[y]['path'] += dataset[y]['path']
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

        # Calculate remaining data for new test sets
        for train_subset, test_subset in [('train_25', 'test_75'),
                                          ('train_50', 'test_50'),
                                          ('train_75', 'test_25')]:
            # Calculate the starting index of the remaining data
            train_size = len(self.dataset[train_subset]['dt'])
            remaining_start_index = train_size

            # Assign remaining data to the corresponding test set
            self.dataset[test_subset]['path'] += dataset['train']['path'][remaining_start_index:]
            self.dataset[test_subset]['dt'] += dataset['train']['dt'][remaining_start_index:]
            self.dataset[test_subset]['gt'] += dataset['train']['gt'][remaining_start_index:]

    def create_combined_dataset_with_llm_labels(self, target_dir, image_input_size, max_text_length,
                                                ocr_model, llm_model, method, train_partition, remaining_partition,
                                                mode):
        """
        Create new dataset partitions by combining the original training data with the remaining data,
        using LLM-corrected labels for the remaining data.
        """

        # Construct the directory path
        dataset_dir = os.path.join(llm_outputs_path, self.name, ocr_model, llm_model, method, train_partition,
                                   remaining_partition)
        if not os.path.exists(dataset_dir):
            print(f"Directory {dataset_dir} does not exist.")
            return

        # Find the latest JSON files in the directory
        latest_json_files = find_latest_json_files(dataset_dir)
        if not latest_json_files:
            print(f"No JSON files found in {dataset_dir}.")
            return

        # Extract the latest JSON files
        empty_json = latest_json_files.get('empty')
        dataset_json = latest_json_files.get('dataset')

        # Flags to indicate which partitions to create
        create_empty_partition = bool(empty_json)
        create_dataset_partition = bool(dataset_json)

        # Inform which partitions will be created
        print(f"Creating combined datasets for {self.name}:")

        if create_empty_partition:
            print(f"- Using LLM-corrected labels from {empty_json} for 'empty' partition.")
            # Create the combined dataset for the 'empty' JSON
            self._create_combined_partition(target_dir, image_input_size, max_text_length,
                                            ocr_model, llm_model, method,
                                            train_partition, remaining_partition, empty_json, mode,
                                            partition_suffix='_empty')
        if create_dataset_partition:
            print(f"- Using LLM-corrected labels from {dataset_json} for 'dataset' partition.")
            # Create the combined dataset for the 'dataset' JSON
            self._create_combined_partition(target_dir, image_input_size, max_text_length,
                                            ocr_model, llm_model, method,
                                            train_partition, remaining_partition, dataset_json, mode,
                                            partition_suffix='')
        if not create_empty_partition and not create_dataset_partition:
            print("No suitable JSON files found to create combined datasets.")

    def _create_combined_partition(self, target_dir, image_input_size, max_text_length,
                                   ocr_model, llm_model, method,
                                   train_partition, remaining_partition, json_file, mode, partition_suffix=''):
        """
        Helper method to create a combined dataset partition.
        """
        # Load LLM-corrected labels
        llm_labels = load_llm_corrected_labels(json_file, mode)

        # Load original training data
        train_data = {
            'dt': self.dataset[train_partition]['dt'],
            'gt': self.dataset[train_partition]['gt'],
            'path': self.dataset[train_partition]['path']
        }

        # Load remaining data and replace labels
        test_partition = f"test_{remaining_partition.split('_')[1]}"
        remaining_data = {
            'dt': [],
            'gt': [],
            'path': []
        }

        # Only include files for which we have LLM-corrected labels
        for idx, file_path in enumerate(self.dataset[test_partition]['dt']):
            file_name = os.path.basename(file_path)
            if file_name in llm_labels and len(llm_labels[file_name]) > 0:
                remaining_data['dt'].append(file_path)
                remaining_data['gt'].append(llm_labels[file_name])
                remaining_data['path'].append(file_path)
            else:
                print(f"No LLM label found for {file_name}, skipping.")

        # Ensure data and labels are aligned
        assert len(remaining_data['dt']) == len(remaining_data['gt']), "Mismatch in data and labels."

        print("Train data lengths:",
              f"dt={len(train_data['dt'])}",
              f"gt={len(train_data['gt'])}",
              f"path={len(train_data['path'])}")

        print("Remaining data lengths:",
              f"dt={len(remaining_data['dt'])}",
              f"gt={len(remaining_data['gt'])}",
              f"path={len(remaining_data['path'])}")

        # Combine data
        combined_data = {
            'dt': train_data['dt'] + remaining_data['dt'],
            'gt': train_data['gt'] + remaining_data['gt'],
            'path': train_data['path'] + remaining_data['path']
        }

        print("Combined data lengths:",
              f"dt={len(combined_data['dt'])}",
              f"gt={len(combined_data['gt'])}",
              f"path={len(combined_data['path'])}")

        # Save to HDF5
        print(f"Combined data: {len(combined_data['dt'])} samples")

        new_partition_name = f"{train_partition}{partition_suffix}_{remaining_partition}"
        self._save_combined_dataset(target_dir, combined_data, new_partition_name, image_input_size, max_text_length)

    def _bentham(self, partition_name):
        """Bentham dataset reader and processor"""
        source = os.path.join(self.source)
        pt_path = os.path.join(source, "Partitions")

        # Load paths for train, valid, and test splits
        paths = {
            "train": open(os.path.join(pt_path, "TrainLines.lst")).read().splitlines(),
            "valid": open(os.path.join(pt_path, "ValidationLines.lst")).read().splitlines(),
            "test": open(os.path.join(pt_path, "TestLines.lst")).read().splitlines()
        }

        # Load transcriptions (ground truth)
        transcriptions = os.path.join(source, "Transcriptions")
        gt_files = os.listdir(transcriptions)
        gt_dict = {}

        # Process the transcription files
        for gt_file in gt_files:
            text = " ".join(open(os.path.join(transcriptions, gt_file)).read().splitlines())
            text = html.unescape(text).replace("<gap/>", "")
            gt_dict[os.path.splitext(gt_file)[0]] = " ".join(text.split())

        # Path for the images (Lines)
        img_path = os.path.join(source, "Images", "Lines")
        dataset = {
            "train": {"path": [], "dt": [], "gt": []},
            "valid": {"path": [], "dt": [], "gt": []},
            "test": {"path": [], "dt": [], "gt": []}
        }

        # Process the dataset for each partition (train, valid, test)
        for partition in ["train", "valid", "test"]:
            for line in paths[partition]:
                if line not in gt_dict:
                    print(f"Warning: Missing ground truth for {line}, skipping.")
                    continue
                    # Check if the label is empty
                label = gt_dict[line].strip()
                if len(label) == 0:
                    print(f"Warning: Empty label for {line}, skipping.")
                    continue
                dataset[partition]['dt'].append(os.path.join(img_path, f"{line}.png"))
                dataset[partition]['gt'].append(gt_dict[line])
                dataset[partition]['path'].append(os.path.join(img_path, f"{line}.png"))

        return dataset

    def _washington(self, partition_name):
        """Washington dataset reader"""
        pt_path = os.path.join(self.source, "sets", partition_name)

        paths = {
            "train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
            "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
            "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()
        }

        # Load transcriptions (ground truth)
        lines = open(os.path.join(self.source, "ground_truth", "transcription.txt")).read().splitlines()
        gt_dict = {}

        for line in lines:
            split = line.split()
            split[1] = split[1].replace("-", "").replace("|", " ")
            split[1] = split[1].replace("s_pt", ".").replace("s_cm", ",")
            split[1] = split[1].replace("s_mi", "-").replace("s_qo", ":")
            split[1] = split[1].replace("s_sq", ";").replace("s_et", "V")
            split[1] = split[1].replace("s_bl", "(").replace("s_br", ")")
            split[1] = split[1].replace("s_qt", "'").replace("s_GW", "G.W.")
            split[1] = split[1].replace("s_", "")
            gt_dict[split[0]] = split[1]

        # Path for the images
        img_path = os.path.join(self.source, "data", "line_images_normalized")
        dataset = {
            "train": {"path": [], "dt": [], "gt": []},
            "valid": {"path": [], "dt": [], "gt": []},
            "test": {"path": [], "dt": [], "gt": []}
        }

        for i in ["train", "valid", "test"]:
            for line in paths[i]:
                dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                dataset[i]['gt'].append(gt_dict[line])
                dataset[i]['path'].append(f"{line}.png")

        return dataset

    def _init_dataset(self):
        """Initialize the dataset dictionary with empty lists for train subsets, valid, test, and new test subsets"""
        return {
            "train_100": {"path": [], "dt": [], "gt": []},  # 100% of training data
            "train_75": {"path": [], "dt": [], "gt": []},  # 75% of training data
            "train_50": {"path": [], "dt": [], "gt": []},  # 50% of training data
            "train_25": {"path": [], "dt": [], "gt": []},  # 25% of training data
            "valid": {"path": [], "dt": [], "gt": []},  # Validation data
            "test": {"path": [], "dt": [], "gt": []},  # Original test data
            "test_75": {"path": [], "dt": [], "gt": []},  # Remaining data for train_25
            "test_50": {"path": [], "dt": [], "gt": []},  # Remaining data for train_50
            "test_25": {"path": [], "dt": [], "gt": []}  # Remaining data for train_75
        }

    def _save_combined_dataset(self, target_dir, data, partition_name, image_input_size, max_text_length):
        """
        Saves the combined dataset to the HDF5 file under a new partition,
        including the 'valid' and 'test' partitions.
        """
        # Ensure the directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Generate the filename based on the dataset name
        filename = f"combined_{self.name}_dataset.hdf5"
        target = os.path.join(target_dir, filename)

        print(f"Saving partition '{partition_name}' with {len(data['dt'])} images")

        # Open the HDF5 file in append mode
        with h5py.File(target, "a") as hf:
            # Save the combined training partition
            if partition_name in hf:
                print(f"Partition {partition_name} already exists in {filename}, skipping.")
            else:
                print(f"Saving combined partition '{partition_name}' to {filename}.")
                self._save_partition(hf, data, partition_name, image_input_size, max_text_length)

            # Save 'valid' and 'test' partitions if they don't already exist
            for partition in ['valid', 'test']:
                if partition in hf:
                    print(f"Partition '{partition}' already exists in {filename}, skipping.")
                else:
                    print(f"Saving partition '{partition}' to {filename}.")
                    partition_data = {
                        'dt': self.dataset[partition]['dt'],
                        'gt': self.dataset[partition]['gt'],
                        'path': self.dataset[partition]['path']
                    }
                    self._save_partition(hf, partition_data, partition, image_input_size, max_text_length)

    def _save_partition(self, hf, data, partition_name, image_input_size, max_text_length):
        """
        Saves a single partition to the HDF5 file.
        """
        total_samples = len(data['dt'])
        size = (total_samples,) + image_input_size[:2]

        print(f"Initializing partition '{partition_name}' with size {size}")

        # Create datasets for images, labels, and paths
        hf.create_dataset(f"{partition_name}/dt", shape=size, maxshape=size,
                          dtype=np.uint8, compression="gzip", compression_opts=9)
        hf.create_dataset(f"{partition_name}/gt", (total_samples,), maxshape=(total_samples,),
                          dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip", compression_opts=9)
        hf.create_dataset(f"{partition_name}/path", (total_samples,), maxshape=(total_samples,),
                          dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip", compression_opts=9)

        pbar = tqdm(total=total_samples, desc=f"Processing {partition_name}")
        batch_size = 1024
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_size_actual = batch_end - batch_start

            # Preprocess images
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                images = pool.map(
                    partial(pp.preprocess, input_size=image_input_size),
                    data['dt'][batch_start:batch_end]
                )
                pool.close()
                pool.join()

            # Debug: Check the shape of each image in the batch
            for i, img in enumerate(images):
                if img.shape != (1024, 128):
                    print(
                        f"Error in image at index {batch_start + i}: shape {img.shape}, path: {data['dt'][batch_start + i]}")

            print(f"Writing batch {batch_start}:{batch_end}, images shape: {np.array(images).shape}")
            print(f"Batch {batch_start}:{batch_end} sizes: dt={len(data['dt'][batch_start:batch_end])}, "
                  f"gt={len(data['gt'][batch_start:batch_end])}, path={len(data['path'][batch_start:batch_end])}")

            # Save data to HDF5
            hf[f"{partition_name}/dt"][batch_start:batch_end] = images
            hf[f"{partition_name}/gt"][batch_start:batch_end] = [
                s[:max_text_length] for s in data['gt'][batch_start:batch_end]
            ]
            hf[f"{partition_name}/path"][batch_start:batch_end] = [
                os.path.basename(path) for path in data['path'][batch_start:batch_end]
            ]

            pbar.update(batch_size_actual)
        pbar.close()

    def save_partitions(self, target_dir, image_input_size, max_text_length):
        """
        Save images and sentences from dataset into a single HDF5 file,
        including different subsets of the training set (100%, 75%, 50%, 25%).
        """

        # Ensure the directory exists (this creates only the directory, not the file)
        os.makedirs(target_dir, exist_ok=True)

        # Generate the filename based on the dataset name (e.g., bentham_dataset.hdf5)
        filename = f"{self.name}_dataset.hdf5"
        target = os.path.join(target_dir, filename)

        if os.path.exists(target):
            # Check if file contents are the same before overwriting
            if self._check_existing_file(target):
                print(f"{filename} already exists with the same content, skipping.")
                return  # Skip saving if the data is identical

        if self.name == 'bentham':
            full_image_path = os.path.join(self.source, "Images", "Lines")
        elif self.name == 'washington':
            full_image_path = os.path.join(self.source, "data", "line_images_normalized")
        else:
            full_image_path = os.path.join(self.source, "lines")

            # Create the HDF5 file
        with h5py.File(target, "w") as hf:
            hf.attrs['full_image_path'] = full_image_path.encode('utf-8')

            # Save all subsets of the training data
            for subset in ['train_100', 'train_75', 'train_50', 'train_25']:
                self._save_subset(hf, subset, len(self.dataset[subset]['dt']), image_input_size, max_text_length)

            # Save validation, test sets, and new test subsets
            for subset in ['valid', 'test', 'test_75', 'test_50', 'test_25']:
                self._save_subset(hf, subset, len(self.dataset[subset]['dt']), image_input_size, max_text_length)

        pbar = tqdm(total=len(self.partitions))
        batch_size = 1024

        # Parallel image processing for all partitions
        for pt in self.dataset.keys():
            total_batches = len(self.dataset[pt]['gt'])
            for batch_start in range(0, total_batches, batch_size):
                batch_end = min(batch_start + batch_size, total_batches)
                batch_size_actual = batch_end - batch_start

                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                    images = pool.map(
                        partial(pp.preprocess, input_size=image_input_size),
                        self.dataset[pt]['dt'][batch_start:batch_end]
                    )
                    pool.close()
                    pool.join()

                with h5py.File(target, "a") as hf:
                    hf[f"{pt}/dt"][batch_start:batch_end] = images
                    hf[f"{pt}/gt"][batch_start:batch_end] = [s.encode() for s in
                                                             self.dataset[pt]['gt'][batch_start:batch_end]]
                    hf[f"{pt}/path"][batch_start:batch_end] = [os.path.basename(path).encode('utf-8') for path in
                                                               self.dataset[pt]['dt'][batch_start:batch_end]]

                pbar.update(batch_size_actual)

    def _save_subset(self, hf, partition, subset_size, image_input_size, max_text_length):
        """
        Save a subset of the partition (train_100, train_75, train_50, train_25, valid, test).

        :param hf: HDF5 file handler.
        :param partition: 'train_100', 'train_75', 'train_50', 'train_25', 'valid', 'test'.
        :param subset_size: The number of elements to include in the subset.
        :param image_input_size: The size of the input image (height, width, channels).
        :param max_text_length: Maximum text length for the ground truth labels.
        """
        size = (subset_size,) + image_input_size[:2]

        # Create dummy image data for testing purposes (replace with actual preprocessed images)
        dummy_image = np.zeros(size, dtype=np.uint8)

        # Get the actual ground truth data from the dataset
        ground_truth = [gt.encode('utf-8')[:max_text_length] for gt in self.dataset[partition]['gt'][:subset_size]]

        # Get the file names (paths)
        file_names = [os.path.basename(path).encode('utf-8') for path in self.dataset[partition]['dt'][:subset_size]]

        # Get the full paths
        full_paths = [path.encode('utf-8') for path in self.dataset[partition]['dt'][:subset_size]]

        # Save the data into the HDF5 file
        hf.create_dataset(f"{partition}/dt", data=dummy_image, compression="gzip", compression_opts=9)
        hf.create_dataset(f"{partition}/gt", data=ground_truth, compression="gzip", compression_opts=9)
        hf.create_dataset(f"{partition}/path", data=file_names, compression="gzip", compression_opts=9)

    def _check_existing_file(self, target):
        """
        Check if the content of the existing HDF5 file matches the current dataset.
        Return True if content is identical, otherwise False.
        """
        with h5py.File(target, 'r') as hf:
            for subset in self.partitions:
                if subset not in hf:
                    return False  # The partition doesn't exist, so the content is different

                # Compare lengths
                if len(hf[f"{subset}/dt"]) != len(self.dataset[subset]['dt']):
                    return False

                # Compare file paths (assuming paths are unique identifiers)
                existing_paths = [path.decode('utf-8') for path in hf[f"{subset}/path"]]
                current_paths = [os.path.basename(path) for path in self.dataset[subset]['dt']]
                if existing_paths != current_paths:
                    return False

        return True  # Data matches, no need to overwrite

    def _iam(self, partition_name):
        """IAM dataset reader"""
        pt_path = os.path.join(self.source, "largeWriterIndependentTextLineRecognitionTask")

        # Load paths for train, valid, and test sets
        paths = {
            "train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
            "valid": open(os.path.join(pt_path, "validationset1.txt")).read().splitlines() +
                     open(os.path.join(pt_path, "validationset2.txt")).read().splitlines(),
            "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()
        }

        lines_path = os.path.join(self.source, "ground_truth", "lines.txt")

        if not os.path.exists(lines_path):
            raise FileNotFoundError(f"Ground truth file does not exist: {lines_path}")

        lines = open(lines_path).read().splitlines()
        dataset = {
            "train": {"path": [], "dt": [], "gt": []},
            "valid": {"path": [], "dt": [], "gt": []},
            "test": {"path": [], "dt": [], "gt": []}
        }
        gt_dict = {}

        # Process the ground truth lines
        for line in lines:
            if not line or line.startswith("#"):
                continue
            split = line.split()
            corrected_gt = correct_punctuation_spacing(" ".join(split[8:]).replace("|", " "))
            gt_dict[split[0]] = corrected_gt

        # Process the IAM dataset for train, valid, and test partitions
        for i in ["train", "valid", "test"]:
            for line in paths[i]:
                try:
                    split = line.split("-")
                    img_file = f"{split[0]}-{split[1]}-{split[2]}.png"
                    img_path = os.path.join(self.source, "lines", img_file)
                    # print(img_path)

                    # Skip missing ground truth entries
                    if line not in gt_dict:
                        print(f"Warning: Missing ground truth for {line}")
                        continue

                    dataset[i]['gt'].append(gt_dict[line])
                    dataset[i]['dt'].append(img_path)
                    dataset[i]['path'].append(img_path)
                # except KeyError:
                # print(f"Warning: KeyError processing line {line}")
                except Exception:
                    pass

        # Ensure that training data exists
        if not dataset['train']['dt']:
            raise ValueError("No training data found for IAM dataset. Please check your files.")

        return dataset
