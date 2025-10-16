#!/usr/bin/env python3
"""
Quick dataset shape checker for DeepStack training samples.
- Prints shapes of train/valid inputs, targets, masks
- Infers bucket count from inputs
- Verifies targets/masks dimensions
"""
import os
import sys
import torch

DEF_PATH = r"C:/Users/AMD/pokerbot/src/train_samples"

def main(data_path: str = DEF_PATH):
    if not os.path.exists(data_path):
        print(f"Error: data path does not exist: {data_path}")
        sys.exit(1)
    paths = {
        'train_inputs': os.path.join(data_path, 'train_inputs.pt'),
        'train_targets': os.path.join(data_path, 'train_targets.pt'),
        'train_mask': os.path.join(data_path, 'train_mask.pt'),
        'valid_inputs': os.path.join(data_path, 'valid_inputs.pt'),
        'valid_targets': os.path.join(data_path, 'valid_targets.pt'),
        'valid_mask': os.path.join(data_path, 'valid_mask.pt'),
    }
    tensors = {k: torch.load(v) for k, v in paths.items() if os.path.exists(v)}
    for k, t in tensors.items():
        print(f"{k:14s}: shape={tuple(t.shape)}")
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
    main(sys.argv[1] if len(sys.argv) > 1 else DEF_PATH)
