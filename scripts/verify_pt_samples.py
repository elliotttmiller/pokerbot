import os
import torch

def verify_pt_file(pt_file):
    try:
        tensor = torch.load(pt_file)
        print(f"{pt_file}:")
        print(f"  shape: {tuple(tensor.shape)}")
        print(f"  dtype: {tensor.dtype}")
        print(f"  sum: {tensor.sum().item()}")
    except Exception as e:
        print(f"Error loading {pt_file}: {e}")

# Directory containing .pt files
dir_path = os.path.join('data', 'deepstacked_training', 'samples', 'train_samples')

for fname in os.listdir(dir_path):
    if fname.endswith('.pt'):
        verify_pt_file(os.path.join(dir_path, fname))
