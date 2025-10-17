#!/usr/bin/env python3
"""
Unified Model Validation Script

This script consolidates all validation functionality.
Replaces: validate_deepstack_model.py, validate_data.py

Usage:
  # Validate model
  python scripts/validate_model.py --type model
  python scripts/validate_model.py --type model --model models/versions/best_model.pt
  
  # Validate data
  python scripts/validate_model.py --type data
  python scripts/validate_model.py --type data --data-path src/train_samples
  
  # Comprehensive validation (both)
  python scripts/validate_model.py --type all
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
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import json
from pathlib import Path


def validate_data(data_path: str):
    """Validate training data quality."""
    print("="*70)
    print("DATA VALIDATION")
    print("="*70)
    print()
    
    # Import here to avoid dependency issues
    import torch
    import numpy as np
    
    data_path = Path(data_path)
    
    # Check if data exists
    required_files = [
        'train_inputs.pt', 'train_targets.pt', 'train_mask.pt', 'train_street.pt',
        'valid_inputs.pt', 'valid_targets.pt', 'valid_mask.pt', 'valid_street.pt',
        'targets_scaling.pt'
    ]
    
    missing = []
    for f in required_files:
        if not (data_path / f).exists():
            missing.append(f)
    
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        print(f"   Path: {data_path}")
        return False
    
    print("✓ All required files present")
    print()
    
    # Load and analyze data
    train_inputs = torch.load(data_path / 'train_inputs.pt')
    train_targets = torch.load(data_path / 'train_targets.pt')
    train_mask = torch.load(data_path / 'train_mask.pt')
    train_street = torch.load(data_path / 'train_street.pt')
    
    valid_inputs = torch.load(data_path / 'valid_inputs.pt')
    valid_targets = torch.load(data_path / 'valid_targets.pt')
    
    scaling = torch.load(data_path / 'targets_scaling.pt')
    
    print("Dataset statistics:")
    print(f"  Training samples:   {len(train_inputs):,}")
    print(f"  Validation samples: {len(valid_inputs):,}")
    print(f"  Input dimensions:   {train_inputs.shape[1]}")
    print(f"  Output dimensions:  {train_targets.shape[1]}")
    print()
    
    # Street distribution
    print("Street distribution (training):")
    street_counts = {}
    for s in [0, 1, 2, 3]:
        count = (train_street == s).sum().item()
        pct = 100.0 * count / len(train_street)
        street_name = ['Preflop', 'Flop', 'Turn', 'River'][s]
        print(f"  {street_name:8s}: {count:,} ({pct:.1f}%)")
        street_counts[s] = count
    print()
    
    # Check for issues
    issues = []
    if len(train_inputs) < 10000:
        issues.append(f"Training samples too low ({len(train_inputs):,} < 10K)")
    if any(street_counts[s] == 0 for s in [1, 2, 3]):
        issues.append("Missing street coverage (some streets have 0 samples)")
    
    if issues:
        print("⚠ Data quality issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    else:
        print("✓ Data quality: Good")
        print()
    
    print("="*70)
    return len(issues) == 0


def validate_model(model_path: str, data_path: str):
    """Validate trained model performance."""
    print("="*70)
    print("MODEL VALIDATION")
    print("="*70)
    print()
    
    # Import DeepStack validation
    import torch
    import numpy as np
    from pathlib import Path
    
    model_path = Path(model_path)
    data_path = Path(data_path)
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print()
    
    # Load validation data
    print("[1/4] Loading validation data...")
    try:
        valid_inputs = torch.load(data_path / 'valid_inputs.pt')
        valid_targets = torch.load(data_path / 'valid_targets.pt')
        valid_mask = torch.load(data_path / 'valid_mask.pt')
        valid_street = torch.load(data_path / 'valid_street.pt')
        scaling = torch.load(data_path / 'targets_scaling.pt')
        print(f"  ✓ Loaded {len(valid_inputs):,} validation samples")
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return False
    
    # Load model
    print("[2/4] Loading trained model...")
    try:
        from deepstack.core.net_builder import NetBuilder
        
        # Infer num_buckets from targets
        num_buckets = valid_targets.shape[1] // 2
        print(f"  Inferred num_buckets from targets: {num_buckets}")
        
        # Build network
        net_builder = NetBuilder(
            num_buckets=num_buckets,
            hidden_sizes=[500, 500, 500, 500, 500, 500, 500],
            activation='prelu'
        )
        model = net_builder.build_network()
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compute metrics
    print("[3/4] Computing validation metrics...")
    try:
        with torch.no_grad():
            predictions = model(valid_inputs)
            
            # Compute loss
            from deepstack.core.masked_huber_loss import MaskedHuberLoss
            criterion = MaskedHuberLoss(delta=0.3, reduction='mean')
            loss = criterion(predictions, valid_targets, valid_mask)
            
            # Compute MAE and RMSE
            diff = (predictions - valid_targets) * valid_mask
            mae = diff.abs().sum() / valid_mask.sum()
            rmse = torch.sqrt((diff ** 2).sum() / valid_mask.sum())
            
            print("  ✓ Validation complete")
            print()
            print("  Metrics:")
            print(f"    Huber Loss: {loss.item():.6f}")
            print(f"    MAE:        {mae.item():.6f}")
            print(f"    RMSE:       {rmse.item():.6f}")
            print(f"    Samples:    {valid_mask.sum().item():.0f}")
    except Exception as e:
        print(f"  ❌ Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze predictions
    print("[4/4] Analyzing predictions...")
    try:
        # De-standardize for analysis
        mean = scaling['mean']
        std = scaling['std']
        
        pred_destd = predictions * std + mean
        target_destd = valid_targets * std + mean
        
        # Compute correlation
        mask_bool = valid_mask.bool()
        pred_masked = pred_destd[mask_bool].numpy()
        target_masked = target_destd[mask_bool].numpy()
        
        correlation = np.corrcoef(pred_masked, target_masked)[0, 1]
        
        # Other stats
        mae_destd = np.abs(pred_masked - target_masked).mean()
        rmse_destd = np.sqrt(((pred_masked - target_masked) ** 2).mean())
        
        # Sign mismatch
        sign_mismatch = ((pred_masked > 0) != (target_masked > 0)).mean()
        
        print("  ✓ Analysis complete")
        print()
        print("  Statistics (de-standardized):")
        print(f"    Correlation:       {correlation:.6f}")
        print(f"    MAE:              {mae_destd:.6f}")
        print(f"    RMSE:             {rmse_destd:.6f}")
        print(f"    Sign mismatch:    {sign_mismatch:.6f}")
        print()
        
        # Quality assessment
        if correlation > 0.85:
            quality = "Championship-level ✅"
        elif correlation > 0.75:
            quality = "Production quality ✅"
        elif correlation > 0.65:
            quality = "Development quality ⚠️"
        else:
            quality = "Needs improvement ❌"
        
        print(f"  Model Quality: {quality}")
        print(f"  Correlation: {correlation:.6f}")
        
    except Exception as e:
        print(f"  ❌ Error analyzing predictions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("="*70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Unified validation for data and models'
    )
    
    parser.add_argument('--type', type=str, choices=['data', 'model', 'all'],
                       default='model', help='Validation type')
    parser.add_argument('--model', type=str, default='models/versions/best_model.pt',
                       help='Path to model file')
    parser.add_argument('--data-path', type=str, default='src/train_samples',
                       help='Path to training data')
    
    args = parser.parse_args()
    
    success = True
    
    if args.type in ['data', 'all']:
        success = validate_data(args.data_path) and success
        if args.type == 'all':
            print()
    
    if args.type in ['model', 'all']:
        success = validate_model(args.model, args.data_path) and success
    
    if success:
        print()
        print("✓ Validation passed")
        return 0
    else:
        print()
        print("❌ Validation failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
