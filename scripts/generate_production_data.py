#!/usr/bin/env python3
"""
Production-level training data generation for championship performance.

This script generates large-scale, high-quality training data using:
- Championship-level CFR iterations (2500+)
- Per-street bet sizing abstractions
- Large sample counts (100K-1M+)
- Optional adaptive CFR for efficiency

Usage:
  # Standard production (100K samples, ~10-20 hours)
  python scripts/generate_production_data.py --samples 100000

  # Championship-level (1M samples, ~4-7 days)
  python scripts/generate_production_data.py --samples 1000000 --cfr-iters 2500

  # With adaptive CFR for efficiency
  python scripts/generate_production_data.py --samples 500000 --adaptive-cfr
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
import time
from deepstack.data.data_generation import generate_training_data
import json


def estimate_time(samples: int, cfr_iters: int) -> str:
    """Estimate generation time based on samples and CFR iterations."""
    # Rough estimate: 2-3 samples/sec with CFR=2000
    # Scales roughly linearly with CFR iterations
    samples_per_sec = 2.5 * (2000 / cfr_iters)
    total_seconds = samples / samples_per_sec
    
    hours = int(total_seconds / 3600)
    minutes = int((total_seconds % 3600) / 60)
    
    if hours > 24:
        days = hours / 24
        return f"~{days:.1f} days"
    elif hours > 0:
        return f"~{hours} hours {minutes} minutes"
    else:
        return f"~{minutes} minutes"


def main():
    parser = argparse.ArgumentParser(
        description='Production-level training data generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard production run (100K samples)
  python scripts/generate_production_data.py --samples 100000
  
  # Championship-level (1M samples)
  python scripts/generate_production_data.py --samples 1000000 --cfr-iters 2500
  
  # With adaptive CFR for efficiency
  python scripts/generate_production_data.py --samples 500000 --adaptive-cfr

Sample Count Guidelines:
  - Minimum viable: 10,000 samples
  - Production: 100,000 samples
  - Championship: 500,000 - 1,000,000 samples
  - World-class: 10,000,000+ samples (as per DeepStack paper)

CFR Iteration Guidelines:
  - Minimum: 1500 iterations
  - Recommended: 2000 iterations
  - Championship: 2500+ iterations
        """
    )
    
    parser.add_argument('--samples', type=int, default=100000,
                       help='Number of training samples (default: 100000)')
    parser.add_argument('--validation-samples', type=int, default=None,
                       help='Number of validation samples (default: samples // 5)')
    parser.add_argument('--cfr-iters', type=int, default=2500,
                       help='CFR iterations per sample (default: 2500)')
    parser.add_argument('--output', type=str, default='src/train_samples_production',
                       help='Output directory for training data')
    parser.add_argument('--bucket-weights', type=str, default='',
                       help='Path to bucket_weights.json for adaptive sampling')
    parser.add_argument('--adaptive-cfr', action='store_true', default=False,
                       help='Use adaptive CFR iterations (adjust based on complexity)')
    parser.add_argument('--yes', '-y', action='store_true', default=False,
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    valid_samples = args.validation_samples or max(1000, args.samples // 5)
    
    # Calculate estimated time
    est_time = estimate_time(args.samples + valid_samples, args.cfr_iters)
    
    print("="*70)
    print("PRODUCTION-LEVEL TRAINING DATA GENERATION")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Training samples: {args.samples:,}")
    print(f"  Validation samples: {valid_samples:,}")
    print(f"  Total samples: {args.samples + valid_samples:,}")
    print(f"  CFR iterations: {args.cfr_iters}")
    print(f"  Adaptive CFR: {args.adaptive_cfr}")
    print(f"  Championship bet sizing: True")
    print(f"  Output directory: {args.output}")
    print()
    print(f"Estimated generation time: {est_time}")
    print()
    
    # Quality check
    quality_level = "Unknown"
    if args.samples >= 1000000:
        quality_level = "Championship (excellent!)"
    elif args.samples >= 500000:
        quality_level = "High (very good)"
    elif args.samples >= 100000:
        quality_level = "Production (good)"
    elif args.samples >= 10000:
        quality_level = "Development (acceptable)"
    else:
        quality_level = "Testing only (insufficient for production)"
    
    print(f"Expected quality level: {quality_level}")
    
    if args.cfr_iters < 2000:
        print()
        print("⚠ WARNING: CFR iterations below recommended threshold")
        print(f"  Current: {args.cfr_iters}")
        print("  Recommended: 2000+ for production")
        print("  Championship: 2500+")
    
    if args.samples < 10000:
        print()
        print("⚠ WARNING: Sample count too low for production use")
        print(f"  Current: {args.samples:,}")
        print("  Minimum recommended: 10,000")
        print("  Production recommended: 100,000+")
    
    print("="*70)
    
    # Confirmation prompt
    if not args.yes:
        print()
        response = input("Proceed with data generation? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return
    
    print()
    print("Starting production data generation...")
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
    
    # Track time
    start_time = time.time()
    
    # Generate data
    try:
        generate_training_data(
            train_count=args.samples,
            valid_count=valid_samples,
            output_path=args.output,
            cfr_iterations=args.cfr_iters,
            bucket_sampling_weights=bucket_weights,
            use_championship_bet_sizing=True,
            use_adaptive_cfr=args.adaptive_cfr
        )
    except KeyboardInterrupt:
        print()
        print("="*70)
        print("⚠ Generation interrupted by user")
        print("="*70)
        return
    except Exception as e:
        print()
        print("="*70)
        print(f"❌ Error during generation: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return
    
    # Calculate actual time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time / 3600)
    minutes = int((elapsed_time % 3600) / 60)
    seconds = int(elapsed_time % 60)
    
    print()
    print("="*70)
    print("✓ PRODUCTION DATA GENERATION COMPLETE!")
    print("="*70)
    print()
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Data saved to: {args.output}")
    print()
    print("Next steps:")
    print("  1. Validate data: python scripts/validate_data.py")
    print("  2. Train model: python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu")
    print("  3. Validate model: python scripts/validate_deepstack_model.py")
    print("="*70)


if __name__ == '__main__':
    main()
