import os
import torchfile
import torch

# Directory containing sample files
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/deepstacked_training/samples/train_samples'))
pt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/deepstacked_training/samples/pt_samples'))
os.makedirs(pt_dir, exist_ok=True)

# List of file types to convert
file_types = [
    'train.inputs', 'train.mask', 'train.targets',
    'valid.inputs', 'valid.mask', 'valid.targets'
]

def convert_to_pt(filename):
    path = os.path.join(data_dir, filename)
    try:
        arr = torchfile.load(path)
    except Exception as e:
        print(f"Failed to load {filename} with torchfile: {e}")
        return
    tensor = torch.tensor(arr)
    pt_path = os.path.join(pt_dir, filename + '.pt')
    torch.save(tensor, pt_path)
    print(f"Converted {filename} to {pt_path}")

if __name__ == "__main__":
    for f in file_types:
        convert_to_pt(f)
    print("All sample files converted to PyTorch format.")
