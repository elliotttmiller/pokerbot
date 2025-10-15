import sys
import os
import torch
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python convert_csv_to_pt.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.isfile(csv_file):
    print(f"File not found: {csv_file}")
    sys.exit(1)

# Load CSV as pandas DataFrame
try:
    df = pd.read_csv(csv_file, header=None)
except Exception as e:
    print(f"Error reading {csv_file}: {e}")
    sys.exit(1)

# Convert to PyTorch tensor
try:
    tensor = torch.tensor(df.values, dtype=torch.float32)
except Exception as e:
    print(f"Error converting to tensor: {e}")
    sys.exit(1)

# Save as .pt file
pt_file = os.path.splitext(csv_file)[0] + '.pt'
try:
    torch.save(tensor, pt_file)
    print(f"Saved: {pt_file}")
except Exception as e:
    print(f"Error saving {pt_file}: {e}")
    sys.exit(1)
