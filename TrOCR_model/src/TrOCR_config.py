from dataclasses import dataclass
import h5py
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import os

"""
Augmentations are applied to the training images to make the model robust to variations in input.
"""
train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
])


@dataclass(frozen=True)
class ModelConfig:
    # MODEL_NAME: str = 'microsoft/trocr-base-handwritten'
    MODEL_NAME: str = 'microsoft/trocr-small-printed'


class HandleDataTrOCR(Dataset):
    """
    CustomOCRDataset is a custom dataset class for OCR tasks.
    It handles loading, preprocessing of images, and conversion of text labels for model training.
    """

    def __init__(self, hdf5_file_path, partition, processor, max_target_length=128):
        """
        Initialize the dataset with HDF5 data.
        :param hdf5_file_path: Path to the HDF5 file containing the dataset.
        :param partition: The partition of the data to use ('train_100', 'train_25', 'valid', etc.).
        :param processor: The processor (TrOCRProcessor) for image preprocessing and tokenization.
        :param max_target_length: The maximum length of the text labels.
        """
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.partition = partition
        self.processor = processor
        self.max_target_length = max_target_length

        # Load the root directory from the HDF5 attributes (full image path)
        self.root_dir = self.hdf5_file.attrs['full_image_path']  # Leave as a string, no need to encode

        # Load the relative image file names and ground truth text labels from HDF5
        self.image_file_names = self.hdf5_file[f"{partition}/path"][:]  # Relative file names
        self.labels = self.hdf5_file[f"{partition}/gt"][:]  # Ground truth text labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # The relative image file name (stored in the HDF5)
        file_name = self.image_file_names[idx].decode('utf-8')

        # Combine root directory with the relative file name to get the full image path
        img_path = os.path.join(self.root_dir, file_name)  # Now both components are strings

        # Load the image using the full path
        image = Image.open(img_path).convert('RGB')

        # Apply augmentations
        image = train_transforms(image)

        # Preprocess the image to get pixel values
        pixel_values = self.processor(image, return_tensors='pt').pixel_values

        # The text (label) associated with the image
        text = self.labels[idx].decode('utf-8')

        # Tokenize the text and get the labels
        labels = self.processor.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_target_length
        ).input_ids

        # We are using -100 as the padding token
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        # Prepare the encoding dictionary for PyTorch training
        encoding = {
            "pixel_values": pixel_values.squeeze(),  # Remove the batch dimension
            "labels": torch.tensor(labels)
        }

        return encoding

    def close(self):
        """
        Close the HDF5 file when done.
        """
        if self.hdf5_file:
            self.hdf5_file.close()
