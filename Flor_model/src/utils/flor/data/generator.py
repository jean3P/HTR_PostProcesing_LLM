# ./flor/data/generator.py

"""
Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time.
"""
import json
import os

import h5py
import numpy as np

from utils.flor.data import preproc as pp


class DataGenerator:
    """Generator class with data streaming"""

    def __init__(self, source, batch_size, charset, max_text_length, train_partition, predict=False,
                 stream=False, external_labels=None):
        """
        Initialize the data generator.
        :param source: Path to the dataset source.
        :param batch_size: Number of samples per batch.
        :param charset: Character set for encoding.
        :param max_text_length: Maximum text length for encoding.
        :param train_partition: Training partition to use (e.g., 'train_100', 'train_75', etc.).
        :param predict: Whether to use the generator for prediction (default is False).
        :param stream: Whether to stream data from the HDF5 file.
        :param external_labels: Path to external labels for updating ground truth.
        """
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.batch_size = batch_size
        self.external_labels = external_labels
        self.train_partition = train_partition  # Store the training partition

        self.size = dict()
        self.steps = dict()
        self.index = dict()

        if stream:
            self.dataset = h5py.File(source, "r")

            # Load the specified training partition and valid/test sets
            for pt in [self.train_partition, 'valid', 'test']:
                self.size[pt] = self.dataset[pt]['gt'][:].shape[0]
                self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))
        else:
            self.dataset = dict()

            with h5py.File(source, "r") as f:
                for pt in [self.train_partition, 'valid', 'test']:
                    self.dataset[pt] = dict()
                    self.dataset[pt]['dt'] = np.array(f[pt]['dt'])
                    self.dataset[pt]['gt'] = np.array(f[pt]['gt'])
                    self.dataset[pt]['path'] = np.array(f[pt]['path'])  # assuming names are stored here

                    self.size[pt] = len(self.dataset[pt]['gt'])
                    self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))
        self.stream = stream
        self.arange = np.arange(len(self.dataset[self.train_partition]['gt']))
        np.random.seed(42)

    # def update_labels_with_external(self):
    #     """Updates the labels in the training dataset with those provided in an external JSON file."""
    #     with open(self.external_labels, 'r') as file:
    #         new_labels = json.load(file)
    #
    #     # Update training set labels
    #     path_to_index = {path.decode('utf-8'): i for i, path in enumerate(self.dataset['train']['path'])}
    #     for item in new_labels:
    #         filename = item["file_name"]
    #         new_label = item["MISTRAL"]["predicted_label"]
    #         if filename in path_to_index:
    #             index = path_to_index[filename]
    #             if new_label.strip():  # This will be False for empty strings and strings with only whitespaces
    #                 self.dataset['train']['gt'][index] = new_label.strip().encode('utf-8')
    #                 print(f"Updated {filename}: {new_label.strip()}")
    #             else:
    #                 print(f"Skipped updating {filename} due to empty label.")

    def next_train_batch(self):
        """Get the next batch from the specified train partition (yield)"""

        self.index[self.train_partition] = 0

        while True:
            if self.index[self.train_partition] >= self.size[self.train_partition]:
                self.index[self.train_partition] = 0

                if not self.stream:
                    np.random.shuffle(self.arange)
                    self.dataset[self.train_partition]['dt'] = self.dataset[self.train_partition]['dt'][self.arange]
                    self.dataset[self.train_partition]['gt'] = self.dataset[self.train_partition]['gt'][self.arange]

            index = self.index[self.train_partition]
            until = index + self.batch_size
            self.index[self.train_partition] = until

            x_train = self.dataset[self.train_partition]['dt'][index:until]
            x_train = pp.augmentation(x_train,
                                      rotation_range=1.5,
                                      scale_range=0.05,
                                      height_shift_range=0.025,
                                      width_shift_range=0.05)
            x_train = pp.normalization(x_train)

            y_train = [self.tokenizer.encode(y) for y in self.dataset[self.train_partition]['gt'][index:until]]
            y_train = [np.pad(y, (0, self.tokenizer.maxlen - len(y))) for y in y_train]
            y_train = np.asarray(y_train, dtype=np.int16)

            yield (x_train, y_train)

    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        self.index['valid'] = 0

        while True:
            if self.index['valid'] >= self.size['valid']:
                self.index['valid'] = 0

            index = self.index['valid']
            until = index + self.batch_size
            self.index['valid'] = until

            x_valid = self.dataset['valid']['dt'][index:until]
            x_valid = pp.normalization(x_valid)

            y_valid = [self.tokenizer.encode(y) for y in self.dataset['valid']['gt'][index:until]]
            y_valid = [np.pad(y, (0, self.tokenizer.maxlen - len(y))) for y in y_valid]
            y_valid = np.asarray(y_valid, dtype=np.int16)

            # names = self.dataset['valid']['path'][index:until]

            yield (x_valid, y_valid)

    def next_test_batch(self):
        """Return model predict parameters"""

        self.index['test'] = 0

        while True:
            if self.index['test'] >= self.size['test']:
                self.index['test'] = 0
                break

            index = self.index['test']
            until = index + self.batch_size
            self.index['test'] = until

            x_test = self.dataset['test']['dt'][index:until]
            x_test = pp.normalization(x_test)

            # names = self.dataset['test']['path'][index:until]  # Extract filenames

            yield x_test


class Tokenizer:
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        if isinstance(text, bytes):
            text = text.decode()

        encoded = []
        for item in " ".join(text.split()):
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
