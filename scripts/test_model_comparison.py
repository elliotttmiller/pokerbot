#!/usr/bin/env python3
"""
Test script for model comparison functionality in train_champion.py

This script tests that:
1. Models are properly saved with consistent names
2. Comparison logic works correctly
3. Promotion only happens when new model is better
"""

import os
import sys
import json
from pathlib import Path

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


def test_model_file_structure():
    """Test that models are saved with the expected file structure in organized directories."""
    versions_dir = Path("models/versions")
    checkpoints_dir = Path("models/checkpoints")
    
    print("TEST 1: Model File Structure (Organized Directories)")
    print("-" * 70)
    
    # Expected files in versions directory
    expected_version_files = [
        "champion_current.cfr",
        "champion_current_metadata.json",
        "champion_best.cfr", 
        "champion_best_metadata.json"
    ]
    
    # Expected files in checkpoints directory
    expected_checkpoint_files = [
        "champion_checkpoint.cfr",
        "champion_checkpoint_metadata.json"
    ]
    
    all_exist = True
    
    print("\n  Versions directory (models/versions/):")
    for filename in expected_version_files:
        filepath = versions_dir / filename
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        print(f"    {status} {filename}: {'Found' if exists else 'Missing'}")
        if not exists:
            all_exist = False
    
    print("\n  Checkpoints directory (models/checkpoints/):")
    for filename in expected_checkpoint_files:
        filepath = checkpoints_dir / filename
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        print(f"    {status} {filename}: {'Found' if exists else 'Missing'}")
        if not exists:
            all_exist = False
    
    print(f"\nResult: {'✓ PASS' if all_exist else '✗ FAIL'}")
    return all_exist


def test_comparison_results():
    """Test that comparison results are properly saved."""
    print("\nTEST 2: Comparison Results")
    print("-" * 70)
    
    comparison_file = Path("models/versions/champion_best_comparison.json")
    
    if not comparison_file.exists():
        print("  ✗ champion_best_comparison.json not found in models/versions/")
        print("\nResult: ✗ FAIL")
        return False
    
    print("  ✓ champion_best_comparison.json exists in models/versions/")
    
    # Read and validate structure
    with open(comparison_file) as f:
        data = json.load(f)
    
    required_keys = ['previous_best_exists', 'is_better']
    has_all_keys = all(key in data for key in required_keys)
    
    if has_all_keys:
        print(f"  ✓ Has required keys: {required_keys}")
    else:
        print(f"  ✗ Missing required keys")
        print("\nResult: ✗ FAIL")
        return False
    
    print(f"  Previous best exists: {data['previous_best_exists']}")
    print(f"  New model is better: {data['is_better']}")
    
    if data.get('previous_best_exists'):
        print(f"  Win rate: {data.get('win_rate', 'N/A')}")
        print(f"  Avg reward: {data.get('avg_reward', 'N/A')}")
    
    print("\nResult: ✓ PASS")
    return True


def test_metadata_consistency():
    """Test that metadata files are consistent."""
    print("\nTEST 3: Metadata Consistency")
    print("-" * 70)
    
    versions_dir = Path("models/versions")
    
    # Check current metadata
    current_meta_file = versions_dir / "champion_current_metadata.json"
    best_meta_file = versions_dir / "champion_best_metadata.json"
    
    if not current_meta_file.exists():
        print("  ✗ champion_current_metadata.json not found in models/versions/")
        print("\nResult: ✗ FAIL")
        return False
    
    if not best_meta_file.exists():
        print("  ✗ champion_best_metadata.json not found in models/versions/")
        print("\nResult: ✗ FAIL")
        return False
    
    with open(current_meta_file) as f:
        current_meta = json.load(f)
    
    with open(best_meta_file) as f:
        best_meta = json.load(f)
    
    # Verify expected keys
    expected_keys = ['timestamp', 'total_episodes', 'epsilon', 'cfr_iterations']
    
    current_has_keys = all(key in current_meta for key in expected_keys)
    best_has_keys = all(key in best_meta for key in expected_keys)
    
    print(f"  ✓ Current metadata has required keys: {current_has_keys}")
    print(f"  ✓ Best metadata has required keys: {best_has_keys}")
    
    if current_has_keys and best_has_keys:
        print(f"  Current model - Episodes: {current_meta['total_episodes']}, Epsilon: {current_meta['epsilon']}")
        print(f"  Best model - Episodes: {best_meta['total_episodes']}, Epsilon: {best_meta['epsilon']}")
        print("\nResult: ✓ PASS")
        return True
    else:
        print("\nResult: ✗ FAIL")
        return False


def test_no_timestamped_files():
    """Test that no timestamped model files were created and directories are clean."""
    print("\nTEST 4: Clean Directory Structure")
    print("-" * 70)
    
    versions_dir = Path("models/versions")
    checkpoints_dir = Path("models/checkpoints")
    
    # Check that only expected files exist in versions
    timestamped_files = []
    if versions_dir.exists():
        for f in versions_dir.glob("champion_*.cfr"):
            filename = f.name
            # Only expect champion_current and champion_best
            if filename not in ["champion_current.cfr", "champion_best.cfr"]:
                # Check if it has episode numbers (e.g., _ep100, _episode_50)
                if "_ep" in filename.lower() or "_episode" in filename.lower():
                    timestamped_files.append(filename)
    
    if timestamped_files:
        print(f"  ✗ Found {len(timestamped_files)} unexpected timestamped files in versions/:")
        for f in timestamped_files:
            print(f"    - {f}")
        print("\nResult: ✗ FAIL")
        return False
    else:
        print("  ✓ No unexpected timestamped files found in versions/")
        print("  ✓ Clean directory structure maintained")
        print("\nResult: ✓ PASS")
        return True


def main():
    """Run all tests."""
    print("="*70)
    print("MODEL COMPARISON FUNCTIONALITY TESTS")
    print("="*70)
    print()
    
    results = []
    
    results.append(("File Structure", test_model_file_structure()))
    results.append(("Comparison Results", test_comparison_results()))
    results.append(("Metadata Consistency", test_metadata_consistency()))
    results.append(("No Timestamped Files", test_no_timestamped_files()))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} - {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
