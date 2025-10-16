"""
DataStream: Handles data loading, batching, and shuffling for DeepStack neural net training.
Ported from DeepStack Lua data_stream.lua.
"""
import torch
import os
import numpy as np
import math

class DataStream:
    def __init__(self, data_path, train_batch_size, use_gpu=False):
        self.data = {}
        # Load validation data
        self.data['valid_mask'] = torch.load(os.path.join(data_path, 'valid_mask.pt'))
        self.data['valid_targets'] = torch.load(os.path.join(data_path, 'valid_targets.pt'))
        self.data['valid_inputs'] = torch.load(os.path.join(data_path, 'valid_inputs.pt'))
        # Optional: per-sample street info for analysis
        valid_street_path = os.path.join(data_path, 'valid_street.pt')
        if os.path.exists(valid_street_path):
            self.data['valid_street'] = torch.load(valid_street_path)
        # Ensure valid_mask matches target dimension: if mask is per-player, repeat; if already full, keep
        if self.data['valid_mask'].shape[1] * 2 == self.data['valid_targets'].shape[1]:
            self.data['valid_mask'] = self.data['valid_mask'].repeat(1, 2)
        self.valid_data_count = self.data['valid_inputs'].shape[0]
        # Load training data
        self.data['train_mask'] = torch.load(os.path.join(data_path, 'train_mask.pt'))
        self.data['train_inputs'] = torch.load(os.path.join(data_path, 'train_inputs.pt'))
        self.data['train_targets'] = torch.load(os.path.join(data_path, 'train_targets.pt'))
        # Optional: per-sample street info for analysis
        train_street_path = os.path.join(data_path, 'train_street.pt')
        if os.path.exists(train_street_path):
            self.data['train_street'] = torch.load(train_street_path)
        # Ensure train_mask matches target dimension
        if self.data['train_mask'].shape[1] * 2 == self.data['train_targets'].shape[1]:
            self.data['train_mask'] = self.data['train_mask'].repeat(1, 2)
        self.train_data_count = self.data['train_inputs'].shape[0]

        # Auto-adjust batch size to fit available data across both splits
        adjusted_batch = max(1, min(train_batch_size, self.valid_data_count, self.train_data_count))
        self.train_batch_size = adjusted_batch
        # Use ceil to include a final partial batch when needed
        self.valid_batch_count = max(1, math.ceil(self.valid_data_count / self.train_batch_size))
        self.train_batch_count = max(1, math.ceil(self.train_data_count / self.train_batch_size))

        # Move to GPU if needed
        if use_gpu:
            for key in self.data:
                self.data[key] = self.data[key].cuda()
        # train_batch_size already set to adjusted value above

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

    def get_batch(self, split, batch_index):
        """Get a batch from the specified split (train or valid)."""
        inputs = self.data[f'{split}_inputs']
        targets = self.data[f'{split}_targets']
        mask = self.data[f'{split}_mask']
        
        assert inputs.shape[0] == targets.shape[0] == mask.shape[0]
        start = batch_index * self.train_batch_size
        end = start + self.train_batch_size
        batch_inputs = inputs[start:end]
        batch_targets = targets[start:end]
        batch_mask = mask[start:end]
        return batch_inputs, batch_targets, batch_mask

    def get_batch_with_street(self, split, batch_index):
        """Get a batch including street tensor if available.
        Returns (inputs, targets, mask, street_or_none)
        """
        batch_inputs, batch_targets, batch_mask = self.get_batch(split, batch_index)
        street_key = f'{split}_street'
        street = None
        if street_key in self.data:
            street_full = self.data[street_key]
            start = batch_index * self.train_batch_size
            end = start + self.train_batch_size
            street = street_full[start:end]
        return batch_inputs, batch_targets, batch_mask, street
