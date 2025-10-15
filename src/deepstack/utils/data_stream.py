"""
DataStream for DeepStack Poker Training (Python)
Handles loading, batching, and shuffling of training/validation data.
"""
import numpy as np
import os

class DataStream:
    def __init__(self, data_path, batch_size=32, use_gpu=False):
        self.data = {}
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        # Load training data
        train_prefix = os.path.join(data_path, 'train_samples')
        self.data['train_inputs'] = np.fromfile(os.path.join(train_prefix, 'train.inputs'), dtype=np.float32)
        self.data['train_targets'] = np.fromfile(os.path.join(train_prefix, 'train.targets'), dtype=np.float32)
        self.data['train_mask'] = np.fromfile(os.path.join(train_prefix, 'train.mask'), dtype=np.float32)
        # Load validation data
        self.data['valid_inputs'] = np.fromfile(os.path.join(train_prefix, 'valid.inputs'), dtype=np.float32)
        self.data['valid_targets'] = np.fromfile(os.path.join(train_prefix, 'valid.targets'), dtype=np.float32)
        self.data['valid_mask'] = np.fromfile(os.path.join(train_prefix, 'valid.mask'), dtype=np.float32)
        self.train_data_count = len(self.data['train_inputs'])
        self.valid_data_count = len(self.data['valid_inputs'])
        # Reshape as needed (user should specify shapes)
    def get_train_batch_count(self):
        return self.train_data_count // self.batch_size
    def get_valid_batch_count(self):
        return self.valid_data_count // self.batch_size
    def start_epoch(self):
        # Shuffle training data
        idx = np.random.permutation(self.train_data_count)
        for key in ['train_inputs', 'train_targets', 'train_mask']:
            self.data[key] = self.data[key][idx]
    def get_batch(self, set_type, batch_index):
        start = batch_index * self.batch_size
        end = start + self.batch_size
        inputs = self.data[f'{set_type}_inputs'][start:end]
        targets = self.data[f'{set_type}_targets'][start:end]
        mask = self.data[f'{set_type}_mask'][start:end]
        return inputs, targets, mask
