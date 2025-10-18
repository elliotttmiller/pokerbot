#!/usr/bin/env python3
"""
Diagnostic script to profile and optimize data generation performance.

Usage:
  python scripts/diagnose_generation.py --samples 10
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from deepstack.data.data_generation import ImprovedDataGenerator


def profile_single_sample(cfr_iters=1000, street=1):
    """Profile generation of a single sample."""
    print(f"\nProfiling single sample (CFR={cfr_iters}, street={street})...")
    
    gen = ImprovedDataGenerator(
        num_hands=169,
        cfr_iterations=cfr_iters,
        verbose=False
    )
    
    # Sample situation
    start = time.time()
    situation = gen.sample_random_situation()
    sample_time = time.time() - start
    
    # Solve
    start = time.time()
    inputs, targets = gen.solve_situation(situation)
    solve_time = time.time() - start
    
    total_time = sample_time + solve_time
    
    print(f"  Sampling:  {sample_time*1000:.1f} ms")
    print(f"  Solving:   {solve_time*1000:.1f} ms ({solve_time/cfr_iters*1000:.2f} ms/iter)")
    print(f"  Total:     {total_time*1000:.1f} ms")
    print(f"  Throughput: {1/total_time:.2f} samples/sec")
    
    return total_time


def profile_multiprocessing(num_samples=50, cfr_iters=1000):
    """Profile multiprocessing generation."""
    print(f"\nProfiling MP generation ({num_samples} samples, CFR={cfr_iters})...")
    
    gen = ImprovedDataGenerator(
        num_hands=169,
        cfr_iterations=cfr_iters,
        verbose=True
    )
    
    start = time.time()
    inputs, targets, masks, streets = gen.generate_dataset(
        num_samples=num_samples,
        output_path='temp_diagnostic',
        dataset_type='diagnostic',
        use_multiprocessing=True
    )
    elapsed = time.time() - start
    
    throughput = num_samples / elapsed
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Per sample: {elapsed/num_samples:.2f}s")
    
    return throughput


def test_config_loading():
    """Test configuration loading with analytics."""
    print("\nTesting config loading...")
    
    # Load latest analytics config
    from glob import glob
    config_dir = Path(__file__).parent.parent / 'config' / 'data_generation' / 'parameters'
    files = list(config_dir.glob('analytics_*.json'))
    
    if not files:
        print("  ⚠ No analytics configs found")
        return
    
    latest = max(files, key=lambda p: p.stat().st_mtime)
    print(f"  Latest: {latest.name}")
    
    import json
    with open(latest) as f:
        cfg = json.load(f)
    
    print(f"  Street weights: {cfg.get('street_distribution', {})}")
    print(f"  Bet override keys: {list(cfg.get('bet_sizing_override', {}).keys())}")
    print(f"  Smoothing alpha: {cfg.get('smoothing_alpha')}")
    
    # Test generator creation
    gen = ImprovedDataGenerator(
        num_hands=169,
        cfr_iterations=1000,
        verbose=False,
        street_sampling_weights=list(cfg.get('street_distribution', {}).values()),
        bet_sizing_override=cfg.get('bet_sizing_override'),
        smoothing_alpha=cfg.get('smoothing_alpha', 0.15)
    )
    
    print(f"  ✓ Generator created successfully")
    print(f"  Street probs: {gen.street_probs}")
    print(f"  Bet override: {gen.bet_sizing_override}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose generation performance')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples for MP test')
    parser.add_argument('--cfr-iters', type=int, default=1000,
                       help='CFR iterations')
    parser.add_argument('--skip-single', action='store_true',
                       help='Skip single sample profiling')
    parser.add_argument('--skip-mp', action='store_true',
                       help='Skip multiprocessing test')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATION PERFORMANCE DIAGNOSTIC")
    print("="*70)
    
    # Test config loading
    test_config_loading()
    
    # Single sample profiling
    if not args.skip_single:
        for iters in [500, 1000, 2000]:
            profile_single_sample(cfr_iters=iters, street=1)
    
    # Multiprocessing profiling
    if not args.skip_mp:
        profile_multiprocessing(num_samples=args.samples, cfr_iters=args.cfr_iters)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    
    print("\nRecommendations:")
    print("  1. Use 1000-1500 CFR iterations for validation sets")
    print("  2. Use 2000-2500 CFR iterations for training sets")
    print("  3. Target throughput: 15-30 samples/sec with MP")
    print("  4. Chunksize should be num_samples // (cpu_count * 4)")
    print()


if __name__ == '__main__':
    main()
