import os
import torch
import json

# Load config
config_path = 'scripts/config/training.json'
with open(config_path, 'r') as f:
    config = json.load(f)


# Get sample data directory from config and resolve to absolute path
sample_dir = config.get('data_path')
if not sample_dir:
    print('No data_path found in config.')
    exit(1)


# If path is relative, resolve from current working directory (project root)
sample_dir_abs = os.path.abspath(sample_dir)

# List expected sample files
expected_files = [
    'train_inputs.pt',
    'train_mask.pt',
    'train_targets.pt',
    'valid_inputs.pt',
    'valid_mask.pt',
    'valid_targets.pt'
]

all_ok = True
for fname in expected_files:
    fpath = os.path.join(sample_dir_abs, fname)
    if not os.path.isfile(fpath):
        print(f"Missing file: {fpath}")
        all_ok = False
        continue
    try:
        tensor = torch.load(fpath)
        print(f"{fname}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, sum={tensor.sum().item()}")
    except Exception as e:
        print(f"Error loading {fpath}: {e}")
        all_ok = False

if all_ok:
    print("All sample files are present and loadable. Pipeline is correctly wired.")
else:
    print("Some sample files are missing or invalid. Check above messages.")
