#!/usr/bin/env python3
"""
Quick dataset shape checker for DeepStack training samples.
- Prints shapes of train/valid inputs, targets, masks
- Infers bucket count from inputs
- Verifies targets/masks dimensions

Usage:
    python scripts/check_dataset_shape.py [data_path]
    
Examples:
    python scripts/check_dataset_shape.py
    python scripts/check_dataset_shape.py src/train_samples
    python scripts/check_dataset_shape.py src/train_samples_production
"""
import argparse
import os
import sys
import torch

# Default path - can be overridden via command line
DEF_PATH = "src/train_samples"

def main(data_path: str = DEF_PATH):
    """Check and display shapes of training data tensors."""
    if not os.path.exists(data_path):
        print(f"Error: data path does not exist: {data_path}")
        sys.exit(1)
    
    print(f"Checking dataset in: {data_path}\n")
    
    paths = {
        'train_inputs': os.path.join(data_path, 'train_inputs.pt'),
        'train_targets': os.path.join(data_path, 'train_targets.pt'),
        'train_mask': os.path.join(data_path, 'train_mask.pt'),
        'valid_inputs': os.path.join(data_path, 'valid_inputs.pt'),
        'valid_targets': os.path.join(data_path, 'valid_targets.pt'),
        'valid_mask': os.path.join(data_path, 'valid_mask.pt'),
    }
    
    tensors = {}
    missing = []
    for k, v in paths.items():
        if os.path.exists(v):
            try:
                tensors[k] = torch.load(v)
            except Exception as e:
                print(f"Error loading {k}: {e}")
        else:
            missing.append(k)
    
    if not tensors:
        print("No tensor files found!")
        sys.exit(1)
    
    for k, t in tensors.items():
        print(f"{k:14s}: shape={tuple(t.shape)}")
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
    
    if 'train_inputs' in tensors:
        inp_dim = tensors['train_inputs'].shape[1]
        buckets = (inp_dim - 1) // 2
        print(f"\nInferred buckets from inputs: {buckets}")
    if 'train_targets' in tensors:
        tgt_dim = tensors['train_targets'].shape[1]
        print(f"Targets width: {tgt_dim} (should be 2*buckets)")
    if 'train_mask' in tensors:
        m_dim = tensors['train_mask'].shape[1]
        print(f"Mask width   : {m_dim} (should match targets width)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Check shapes of DeepStack training data tensors',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('data_path', nargs='?', default=DEF_PATH,
                       help=f'Path to training samples directory (default: {DEF_PATH})')
    args = parser.parse_args()
    main(args.data_path)
