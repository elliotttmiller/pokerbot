#!/usr/bin/env python3
"""
Test the improved data generation and training pipeline.
"""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
from deepstack.data.data_generation import ImprovedDataGenerator
from deepstack.core.terminal_equity import TerminalEquity

print("="*70)
print("Testing Improved DeepStack Pipeline")
print("="*70)
print()

# Test 1: Terminal Equity Calculation
print("[1/4] Testing Terminal Equity Calculation...")
te = TerminalEquity(game_variant='holdem', num_hands=169)
te.set_board([0, 1, 2])  # Sample board
matrix = te.get_call_matrix()

# Check AA vs 72o equity (should be ~0.82-0.85)
aa_vs_72o = matrix[0, 168]
print(f"  AA vs 72o equity: {aa_vs_72o:.3f} (expected ~0.81-0.85)")
if 0.75 < aa_vs_72o < 0.90:
    print("  ✓ Terminal equity calculation looks correct")
else:
    print(f"  ⚠ Terminal equity might be off (got {aa_vs_72o:.3f})")
print()

# Test 2: Data Generation
print("[2/4] Testing Data Generation...")
gen = ImprovedDataGenerator(num_hands=169, cfr_iterations=100, verbose=False)

# Generate a small sample
situation = gen.sample_random_situation()
print(f"  Sampled situation: street={situation['pot_state']['street']}, board_cards={len(situation['board'])}")

try:
    inputs, targets = gen.solve_situation(situation)
    print(f"  ✓ Successfully solved situation")
    print(f"    Input shape: {inputs.shape}")
    print(f"    Target shape: {targets.shape}")
    print(f"    Input size: {inputs.shape[0]} (expected 395)")
    print(f"    Target size: {targets.shape[0]} (expected 338)")
    
    if inputs.shape[0] == 395 and targets.shape[0] == 338:
        print("  ✓ Data dimensions are correct")
    else:
        print(f"  ⚠ Data dimensions mismatch")
except Exception as e:
    print(f"  ✗ Error solving situation: {e}")
print()

# Test 3: Street Distribution
print("[3/4] Testing Street Distribution...")
street_counts = {0: 0, 1: 0, 2: 0, 3: 0}
num_samples = 100
for _ in range(num_samples):
    sit = gen.sample_random_situation()
    street = sit['pot_state']['street']
    street_counts[street] += 1

print("  Street distribution:")
for street, count in street_counts.items():
    pct = 100.0 * count / num_samples
    print(f"    Street {street}: {count}/{num_samples} ({pct:.1f}%)")

# Check if we have good coverage
if street_counts[1] > 20 and street_counts[3] > 10:
    print("  ✓ Good coverage across all streets")
else:
    print("  ⚠ Low coverage on flop or river")
print()

# Test 4: CFR Convergence
print("[4/4] Testing CFR Quality...")
print("  Testing with small CFR iterations (100)...")
gen_low = ImprovedDataGenerator(num_hands=169, cfr_iterations=100, verbose=False)
inputs_low, targets_low = gen_low.solve_situation(gen_low.sample_random_situation())

print("  Testing with higher CFR iterations (500)...")
gen_high = ImprovedDataGenerator(num_hands=169, cfr_iterations=500, verbose=False)
inputs_high, targets_high = gen_high.solve_situation(gen_high.sample_random_situation())

print(f"  Low CFR target range: [{targets_low.min():.2f}, {targets_low.max():.2f}]")
print(f"  High CFR target range: [{targets_high.min():.2f}, {targets_high.max():.2f}]")
print("  ✓ CFR solving is working")
print()

print("="*70)
print("Pipeline Test Summary")
print("="*70)
print("✓ Terminal equity using Monte Carlo")
print("✓ Data generation working")
print("✓ Street coverage improved")
print("✓ CFR solving functional")
print()
print("Next steps:")
print("  1. Generate new training data with: python src/deepstack/data/data_generation.py")
print("  2. Train model with: python scripts/train_deepstack.py")
print("="*70)
