import h5py
import os


class HDF5Handler:
    """
    Class to handle HDF5 files, open them, and access data.
    """

    def __init__(self, hdf5_path, partition, processor):
        self.hdf5_path = hdf5_path
        self.partition = partition
        self.processor = processor

        with h5py.File(self.hdf5_path, 'r') as f:
            self.image_data = f[f'{partition}/dt'][:]  # Load image data (PNG files)
            self.ground_truth = f[f'{partition}/gt'][:]  # Load ground truth text
            self.paths = f[f'{partition}/path'][:]  # Optional: Load paths (if you need them)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        gt_text = self.ground_truth[idx].decode('utf-8')

        # Preprocess the image and tokenize the text using the processor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(gt_text, padding="max_length", max_length=32, return_tensors="pt").input_ids

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": labels.squeeze()
        }

    def open_file(self):
        """
        Open the HDF5 file.
        """
        self.h5_file = h5py.File(self.file_path, 'r')
        print(f"File {self.file_path} opened successfully.")

    def list_groups(self):
        """
        List all groups and datasets in the HDF5 file.
        :return: List of groups in the file.
        """
        if self.h5_file is None:
            raise ValueError("The file is not open. Use 'open_file()' to open the file.")

        def visit(name):
            print(name)

        self.h5_file.visit(visit)

    def get_dataset(self, dataset_name):
        """
        Get a specific dataset from the HDF5 file.
        :param dataset_name: Name of the dataset (e.g., 'train_100/dt').
        :return: The dataset as a NumPy array.
        """
        if self.h5_file is None:
            raise ValueError("The file is not open. Use 'open_file()' to open the file.")

        if dataset_name not in self.h5_file:
            raise KeyError(f"Dataset {dataset_name} does not exist in the file.")

        dataset = self.h5_file[dataset_name][:]
        return dataset

    def get_dataset_shape(self, dataset_name):
        """
        Get the shape of a specific dataset.
        :param dataset_name: Name of the dataset (e.g., 'train_100/dt').
        :return: Shape of the dataset.
        """
        if self.h5_file is None:
            raise ValueError("The file is not open. Use 'open_file()' to open the file.")

        if dataset_name not in self.h5_file:
            raise KeyError(f"Dataset {dataset_name} does not exist in the file.")

        shape = self.h5_file[dataset_name].shape
        return shape

    def get_full_image_path(self):
        """
        Get the full image path from the file's attributes, if it exists.
        :return: Full image path as a string.
        """
        if self.h5_file is None:
            raise ValueError("The file is not open. Use 'open_file()' to open the file.")

        if 'full_image_path' not in self.h5_file.attrs:
            raise KeyError("Attribute 'full_image_path' does not exist in the file.")

        return self.h5_file.attrs['full_image_path']

    def get_ground_truth(self, partition):
        """
        Get ground truth data (gt) from a specific partition.
        :param partition: The partition name (e.g., 'train_100', 'valid', 'test').
        :return: Ground truth data as a NumPy array.
        """
        dataset_name = f"{partition}/gt"
        return self.get_dataset(dataset_name)

    def get_image_data(self, partition):
        """
        Get image data (dt) from a specific partition.
        :param partition: The partition name (e.g., 'train_100', 'valid', 'test').
        :return: Image data as a NumPy array.
        """
        dataset_name = f"{partition}/dt"
        return self.get_dataset(dataset_name)

    def close_file(self):
        """
        Close the HDF5 file.
        """
        if self.h5_file is not None:
            self.h5_file.close()
            print(f"File {self.file_path} closed.")