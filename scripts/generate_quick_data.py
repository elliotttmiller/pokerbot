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
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath:
    for p in pythonpath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)
# Fallback: always add src path directly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

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
                       help='CFR iterations per sample (default: 2000, championship: 2500+)')
    parser.add_argument('--output', type=str, default='src/train_samples',
                       help='Output directory for training data')
    parser.add_argument('--bucket-weights', type=str, default='',
                       help='Path to bucket_weights.json for adaptive sampling')
    parser.add_argument('--championship-bet-sizing', action='store_true', default=True,
                       help='Use per-street bet sizing abstractions (recommended, default: True)')
    parser.add_argument('--simple-bet-sizing', dest='championship_bet_sizing', action='store_false',
                       help='Use simple pot-sized bets only')
    parser.add_argument('--adaptive-cfr', action='store_true', default=False,
                       help='Use adaptive CFR iterations based on situation complexity (experimental)')
    
    args = parser.parse_args()
    
    valid_samples = args.validation_samples or max(100, args.samples // 5)
    
    print("="*70)
    print("DeepStack Training Data Generation")
    print("="*70)
    print(f"Training samples: {args.samples}")
    print(f"Validation samples: {valid_samples}")
    print(f"CFR iterations: {args.cfr_iters}")
    print(f"Championship bet sizing: {args.championship_bet_sizing}")
    print(f"Adaptive CFR: {args.adaptive_cfr}")
    print(f"Output: {args.output}")
    
    # Warn if settings are suboptimal
    if args.samples < 10000:
        print()
        print("⚠ WARNING: Low sample count detected")
        print(f"  Current: {args.samples} samples")
        print("  Recommended minimum: 10,000 samples")
        print("  Championship-level: 100,000+ samples")
        print("  This may result in poor model performance")
    
    if args.cfr_iters < 2000:
        print()
        print("⚠ WARNING: Low CFR iterations detected")
        print(f"  Current: {args.cfr_iters} iterations")
        print("  Recommended: 2000+ iterations")
        print("  Championship-level: 2500+ iterations")
    
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
        bucket_sampling_weights=bucket_weights,
        use_championship_bet_sizing=args.championship_bet_sizing,
        use_adaptive_cfr=args.adaptive_cfr
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
