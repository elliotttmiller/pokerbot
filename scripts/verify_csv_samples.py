import numpy as np
import os

input_dir = 'data/deepstacked_training/samples/train_samples'
files = [
    'train_inputs.csv',
    'train_mask.csv',
    'train_targets.csv',
    'valid_inputs.csv',
    'valid_mask.csv',
    'valid_targets.csv'
]

for fname in files:
    path = os.path.join(input_dir, fname)
    try:
        arr = np.loadtxt(path, delimiter=',')
        print(f"{fname}: shape={arr.shape}, dtype={arr.dtype}, sum={arr.sum()}")
    except Exception as e:
        print(f"{fname}: Error - {e}")
