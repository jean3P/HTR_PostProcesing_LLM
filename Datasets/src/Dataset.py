import os
import html
import multiprocessing
from functools import partial
import numpy as np
import h5py
from tqdm import tqdm
from utils import preproc as pp
from utils.text_processing import correct_punctuation_spacing


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
        self.partitions = ['train_100', 'train_75', 'train_50', 'train_25', 'valid', 'test']

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
        print(f"total {self.name}: {total_train}")
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
        """Initialize the dataset dictionary with empty lists for train subsets, valid, and test"""
        return {
            "train_100": {"path": [], "dt": [], "gt": []},  # 100% of training data
            "train_75": {"path": [], "dt": [], "gt": []},  # 75% of training data
            "train_50": {"path": [], "dt": [], "gt": []},  # 50% of training data
            "train_25": {"path": [], "dt": [], "gt": []},  # 25% of training data
            "valid": {"path": [], "dt": [], "gt": []},  # Validation data
            "test": {"path": [], "dt": [], "gt": []}  # Test data
        }

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
            full_image_path = os.path.join(self.source, "Images", "Lines")  # Adjust based on dataset type
        elif self.name == 'washington':
            full_image_path = os.path.join(self.source, "data", "line_images_normalized")
        else:
            full_image_path = os.path.join(self.source, "lines")

            # Create the HDF5 file at the target location (this includes the filename)
        with h5py.File(target, "w") as hf:
            hf.attrs['full_image_path'] = full_image_path.encode('utf-8')

            # Save all subsets of the training data
            for subset in ['train_100', 'train_75', 'train_50', 'train_25']:
                self._save_subset(hf, subset, len(self.dataset[subset]['dt']), image_input_size, max_text_length)

            # Save validation and test sets
            self._save_subset(hf, 'valid', len(self.dataset['valid']['dt']), image_input_size, max_text_length)
            self._save_subset(hf, 'test', len(self.dataset['test']['dt']), image_input_size, max_text_length)

        pbar = tqdm(total=len(self.partitions))
        batch_size = 1024

        # Parallel image processing for train, valid, and test partitions
        for pt in self.partitions:
            for batch in range(0, len(self.dataset[pt]['gt']), batch_size):
                images = []

                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                    r = pool.map(partial(pp.preprocess, input_size=image_input_size),
                                 self.dataset[pt]['dt'][batch:batch + batch_size])
                    images.append(r)
                    pool.close()
                    pool.join()

                with h5py.File(target, "a") as hf:
                    hf[f"{pt}/dt"][batch:batch + batch_size] = images
                    hf[f"{pt}/gt"][batch:batch + batch_size] = [s.encode() for s in
                                                                self.dataset[pt]['gt'][batch:batch + batch_size]]
                    hf[f"{pt}/path"][batch:batch + batch_size] = [os.path.basename(path).encode('utf-8') for path in
                                                                  self.dataset[pt]['dt'][batch:batch + batch_size]]
                    pbar.update(batch_size)

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
                # except KeyError:
                    # print(f"Warning: KeyError processing line {line}")
                except Exception:
                    pass

        # Ensure that training data exists
        if not dataset['train']['dt']:
            raise ValueError("No training data found for IAM dataset. Please check your files.")

        return dataset
