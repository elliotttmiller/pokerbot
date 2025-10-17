#!/usr/bin/env python3
"""
Quick data generation script with optimized settings.

Usage:
  # Generate small dataset for testing (fast)
  python scripts/generate_quick_data.py --samples 100 --cfr-iters 500

  # Generate medium dataset for development (moderate)
  python scripts/generate_quick_data.py --samples 5000 --cfr-iters 1500

  # Generate large dataset for production (slow but championship-level)
  python scripts/generate_quick_data.py --samples 50000 --cfr-iters 2500
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from deepstack.data.data_generation import generate_training_data
import json

def main():
    parser = argparse.ArgumentParser(description='Generate training data with optimized defaults')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of training samples (default: 1000)')
    parser.add_argument('--validation-samples', type=int, default=None,
                       help='Number of validation samples (default: samples // 5)')
    parser.add_argument('--cfr-iters', type=int, default=2000,
                       help='CFR iterations per sample (default: 2000, paper recommends 2000+)')
    parser.add_argument('--output', type=str, default='src/train_samples',
                       help='Output directory for training data')
    parser.add_argument('--bucket-weights', type=str, default='',
                       help='Path to bucket_weights.json for adaptive sampling')
    
    args = parser.parse_args()
    
    valid_samples = args.validation_samples or max(100, args.samples // 5)
    
    print("="*70)
    print("DeepStack Training Data Generation")
    print("="*70)
    print(f"Training samples: {args.samples}")
    print(f"Validation samples: {valid_samples}")
    print(f"CFR iterations: {args.cfr_iters}")
    print(f"Output: {args.output}")
    print("="*70)
    print()
    
    # Load bucket weights if provided
    bucket_weights = None
    if args.bucket_weights and os.path.exists(args.bucket_weights):
        try:
            with open(args.bucket_weights, 'r') as f:
                weights = json.load(f)
                if len(weights) == 169:
                    import numpy as np
                    bucket_weights = np.array(weights, dtype=np.float32)
                    print(f"✓ Loaded bucket weights from {args.bucket_weights}")
                    print(f"  Boosted buckets: {(bucket_weights > 1.5).sum()}")
                else:
                    print(f"⚠ Bucket weights file has wrong length ({len(weights)} != 169), ignoring")
        except Exception as e:
            print(f"⚠ Could not load bucket weights: {e}")
        print()
    
    # Generate data
    generate_training_data(
        train_count=args.samples,
        valid_count=valid_samples,
        output_path=args.output,
        cfr_iterations=args.cfr_iters,
        bucket_sampling_weights=bucket_weights
    )
    
    print()
    print("="*70)
    print("✓ Data generation complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Validate data: python scripts/validate_data.py")
    print("  2. Train model: python scripts/train_deepstack.py")
    print("  3. Validate model: python scripts/validate_deepstack_model.py")
    print("="*70)

if __name__ == '__main__':
    main()
