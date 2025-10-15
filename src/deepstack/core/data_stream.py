"""
DataStream: Handles data loading, batching, and shuffling for DeepStack neural net training.
Ported from DeepStack Lua data_stream.lua.
"""
import torch
import os
import numpy as np

class DataStream:
    def __init__(self, data_path, train_batch_size, use_gpu=False):
        self.data = {}
        # Load validation data
        valid_prefix = os.path.join(data_path, 'valid')
        self.data['valid_mask'] = torch.load(valid_prefix + '.mask')
        self.data['valid_mask'] = self.data['valid_mask'].repeat(1, 2)
        self.data['valid_targets'] = torch.load(valid_prefix + '.targets')
        self.data['valid_inputs'] = torch.load(valid_prefix + '.inputs')
        self.valid_data_count = self.data['valid_inputs'].shape[0]
        assert self.valid_data_count >= train_batch_size, 'Validation data count must be >= train batch size!'
        self.valid_batch_count = self.valid_data_count // train_batch_size
        # Load training data
        train_prefix = os.path.join(data_path, 'train')
        self.data['train_mask'] = torch.load(train_prefix + '.mask')
        self.data['train_mask'] = self.data['train_mask'].repeat(1, 2)
        self.data['train_inputs'] = torch.load(train_prefix + '.inputs')
        self.data['train_targets'] = torch.load(train_prefix + '.targets')
        self.train_data_count = self.data['train_inputs'].shape[0]
        assert self.train_data_count >= train_batch_size, 'Training data count must be >= train batch size!'
        self.train_batch_count = self.train_data_count // train_batch_size
        # Move to GPU if needed
        if use_gpu:
            for key in self.data:
                self.data[key] = self.data[key].cuda()
        self.train_batch_size = train_batch_size

    def get_valid_batch_count(self):
        return self.valid_batch_count

    def get_train_batch_count(self):
        return self.train_batch_count

    def start_epoch(self):
        # Shuffle training data each epoch
        shuffle = torch.randperm(self.train_data_count)
        self.data['train_inputs'] = self.data['train_inputs'][shuffle]
        self.data['train_targets'] = self.data['train_targets'][shuffle]
        self.data['train_mask'] = self.data['train_mask'][shuffle]

    def get_batch(self, inputs, targets, mask, batch_index):
        assert inputs.shape[0] == targets.shape[0] == mask.shape[0]
        start = batch_index * self.train_batch_size
        end = start + self.train_batch_size
        batch_inputs = inputs[start:end]
        batch_targets = targets[start:end]
        batch_mask = mask[start:end]
        return batch_inputs, batch_targets, batch_mask

    def get_train_batch(self, batch_index):
        return self.get_batch(self.data['train_inputs'], self.data['train_targets'], self.data['train_mask'], batch_index)

    def get_valid_batch(self, batch_index):
        return self.get_batch(self.data['valid_inputs'], self.data['valid_targets'], self.data['valid_mask'], batch_index)
