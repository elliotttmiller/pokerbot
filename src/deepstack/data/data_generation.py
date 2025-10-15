"""
Data generation utilities for DeepStack training pipeline.
Ported from DeepStack Lua data_generation.lua.
"""
import numpy as np
from deepstack.data.random_card_generator import generate_random_cards
from deepstack.data.range_generator import generate_range

def generate_training_data(train_data_count, valid_data_count, data_path, batch_size=32):
    """
    Generates training and validation data by sampling random poker situations and solving them.
    Args:
        train_data_count: number of training examples
        valid_data_count: number of validation examples
        data_path: path prefix for saving files
        batch_size: batch size for generation
    """
    def generate_data_file(data_count, file_name):
        # Placeholder: generate random features and targets
        inputs = np.random.rand(data_count, 10)
        targets = np.random.rand(data_count, 10)
        mask = np.ones((data_count, 10))
        np.save(file_name + '.inputs.npy', inputs)
        np.save(file_name + '.targets.npy', targets)
        np.save(file_name + '.mask.npy', mask)

    print('Generating validation data ...')
    generate_data_file(valid_data_count, data_path + 'valid')
    print('Generating training data ...')
    generate_data_file(train_data_count, data_path + 'train')
    print('Done')
