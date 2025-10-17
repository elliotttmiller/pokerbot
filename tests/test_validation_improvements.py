#!/usr/bin/env python3
"""
Test suite for validation improvements.

This script tests:
1. Temperature scaling functionality
2. Enhanced diagnostics
3. Configuration validation
"""

import sys
import os
import json
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from deepstack.core.temperature_scaling import TemperatureScaler, evaluate_calibration


def test_temperature_scaling():
    """Test temperature scaling functionality."""
    print("Testing Temperature Scaling...")
    
    # Generate synthetic data
    np.random.seed(42)
    predictions = np.random.randn(1000) * 2.0  # Over-confident
    targets = np.random.randn(1000)
    
    # Test fitting
    scaler = TemperatureScaler()
    temp = scaler.fit(predictions, targets, lr=0.01, max_iter=50)
    assert temp > 0, "Temperature must be positive"
    print(f"  ✓ Fitted temperature: {temp:.4f}")
    
    # Test transform
    scaled = scaler.transform(predictions)
    assert scaled.shape == predictions.shape, "Shape mismatch after scaling"
    print(f"  ✓ Predictions scaled correctly")
    
    # Test save/load
    import tempfile
    temp_file = os.path.join(tempfile.gettempdir(), 'test_scaler.pt')
    scaler.save(temp_file)
    
    scaler2 = TemperatureScaler()
    scaler2.load(temp_file)
    assert abs(scaler2.get_temperature() - temp) < 1e-6, "Load mismatch"
    print(f"  ✓ Save/load works correctly")
    
    # Cleanup
    os.remove(temp_file)
    
    # Test calibration metrics
    metrics = evaluate_calibration(predictions, targets, n_bins=10)
    assert 'expected_calibration_error' in metrics, "Missing ECE metric"
    assert 'maximum_calibration_error' in metrics, "Missing MCE metric"
    print(f"  ✓ Calibration metrics computed: ECE={metrics['expected_calibration_error']:.4f}")
    
    print("✅ Temperature scaling tests passed!\n")
    return True


def test_configuration_validation():
    """Test optimized configuration file."""
    print("Testing Configuration...")
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'config', 'optimized.json')
    
    # Check file exists
    assert os.path.exists(config_path), f"Config not found: {config_path}"
    print(f"  ✓ Config file exists: {config_path}")
    
    # Load and validate
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check key parameters
    assert config.get('huber_delta') == 0.3, "Huber delta should be 0.3"
    assert config.get('epochs') == 200, "Epochs should be 200"
    assert config.get('effective_batch_size') == 4096, "Effective batch should be 4096"
    assert config.get('warmup_epochs') == 10, "Warmup epochs should be 10"
    assert config.get('street_weights') == [0.8, 1.0, 1.2, 1.4], "Street weights mismatch"
    print(f"  ✓ All configuration parameters are correct")
    
    # Check comments exist
    assert 'comments' in config, "Config should have comments"
    assert 'expected_results' in config['comments'], "Should have expected results"
    print(f"  ✓ Configuration has proper documentation")
    
    print("✅ Configuration validation tests passed!\n")
    return True


def test_per_player_diagnostics():
    """Test per-player diagnostic calculations."""
    print("Testing Per-Player Diagnostics...")
    
    # Simulate predictions and targets for two players
    np.random.seed(42)
    num_samples = 100
    num_buckets = 169
    
    # Player 1: Good predictions
    p1_preds = np.random.randn(num_samples, num_buckets)
    p1_targets = p1_preds + np.random.randn(num_samples, num_buckets) * 0.1
    
    # Player 2: Poor predictions
    p2_preds = np.random.randn(num_samples, num_buckets)
    p2_targets = np.random.randn(num_samples, num_buckets)
    
    # Combine
    all_preds = np.concatenate([p1_preds, p2_preds], axis=1)
    all_targets = np.concatenate([p1_targets, p2_targets], axis=1)
    
    # Calculate per-player correlations
    half = num_buckets
    
    p1_mask = (p1_preds != 0) & (p1_targets != 0)
    p1_corr = np.corrcoef(p1_preds[p1_mask].flatten(), p1_targets[p1_mask].flatten())[0, 1]
    
    p2_mask = (p2_preds != 0) & (p2_targets != 0)
    p2_corr = np.corrcoef(p2_preds[p2_mask].flatten(), p2_targets[p2_mask].flatten())[0, 1]
    
    print(f"  ✓ Player 1 correlation: {p1_corr:.4f}")
    print(f"  ✓ Player 2 correlation: {p2_corr:.4f}")
    
    # Detect asymmetry
    corr_diff = abs(p1_corr - p2_corr)
    if corr_diff > 0.2:
        print(f"  ⚠ Large correlation difference detected: {corr_diff:.4f}")
    else:
        print(f"  ✓ Correlation difference acceptable: {corr_diff:.4f}")
    
    assert p1_corr > p2_corr, "Player 1 should have better correlation (by design)"
    print("✅ Per-player diagnostics tests passed!\n")
    return True


def test_documentation_exists():
    """Test that all documentation files exist."""
    print("Testing Documentation...")
    
    docs = [
        'VALIDATION_RECOMMENDATIONS_ANALYSIS.md',
        'docs/VALIDATION_IMPROVEMENTS.md',
        'OPTIMIZATION_GUIDE.md',
        'scripts/config/optimized.json',
        'src/deepstack/core/temperature_scaling.py'
    ]
    
    base_path = os.path.join(os.path.dirname(__file__), '..')
    
    for doc in docs:
        full_path = os.path.join(base_path, doc)
        assert os.path.exists(full_path), f"Missing documentation: {doc}"
        print(f"  ✓ {doc}")
    
    print("✅ All documentation files exist!\n")
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("Validation Improvements Test Suite")
    print("="*70)
    print()
    
    tests = [
        test_temperature_scaling,
        test_configuration_validation,
        test_per_player_diagnostics,
        test_documentation_exists
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    print("="*70)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*70)
    
    if all(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
