#!/usr/bin/env python3
"""
Validate trained DeepStack model

This script validates a trained DeepStack value network by:
1. Loading the model
2. Testing on validation data
3. Computing performance metrics
4. Visualizing predictions vs targets

Usage:
    python scripts/validate_deepstack_model.py
    python scripts/validate_deepstack_model.py --model models/pretrained/best_model.pt
"""

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import numpy as np
from deepstack.core.net_builder import NetBuilder
from deepstack.core.data_stream import DataStream
from deepstack.core.masked_huber_loss import MaskedHuberLoss


def validate_model(model_path, data_path, num_buckets=36, batch_size=32):
    """Validate a trained model."""
    print("="*70)
    print("DeepStack Model Validation")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print()
    
    # Load data
    print("[1/4] Loading validation data...")
    data_stream = DataStream(data_path, batch_size, use_gpu=False)
    print(f"  ✓ Loaded {data_stream.valid_data_count} validation samples")
    
    # Load model
    print("[2/4] Loading trained model...")
    # Infer architecture from saved model
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Determine hidden sizes from layer dimensions
    hidden_sizes = []
    for key in sorted(state_dict.keys()):
        if 'model' in key and 'weight' in key and 'Linear' not in key:
            weight = state_dict[key]
            if len(weight.shape) == 2:  # Linear layer
                hidden_sizes.append(weight.shape[0])
    
    # Remove output layer size
    if hidden_sizes:
        hidden_sizes = hidden_sizes[:-1]
    else:
        hidden_sizes = [256, 256, 128]  # Default
    
    print(f"  Architecture: {num_buckets} buckets, hidden={hidden_sizes}")
    
    net = NetBuilder.build_net(num_buckets=num_buckets, 
                               hidden_sizes=hidden_sizes,
                               activation='prelu',
                               use_gpu=False)
    net.load_state_dict(state_dict)
    net.eval()
    print(f"  ✓ Model loaded successfully")
    
    # Validate
    print("[3/4] Computing validation metrics...")
    criterion = MaskedHuberLoss(delta=1.0)
    
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    num_samples = 0
    num_batches = data_stream.get_valid_batch_count()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs, targets, mask = data_stream.get_batch('valid', batch_idx)
            
            # Forward pass
            outputs = net(inputs)
            
            # Compute metrics
            loss = criterion(outputs, targets, mask)
            total_loss += loss.item()
            
            # Masked metrics
            masked_outputs = outputs * mask
            masked_targets = targets * mask
            
            mae = torch.mean(torch.abs(masked_outputs - masked_targets))
            mse = torch.mean((masked_outputs - masked_targets) ** 2)
            
            total_mae += mae.item()
            total_mse += mse.item()
            num_samples += mask.sum().item()
            
            # Store for visualization
            all_predictions.append(masked_outputs.cpu().numpy())
            all_targets.append(masked_targets.cpu().numpy())
    
    # Compute averages
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_mse = total_mse / num_batches
    rmse = np.sqrt(avg_mse)
    
    print(f"  ✓ Validation complete")
    print()
    print("  Metrics:")
    print(f"    Huber Loss: {avg_loss:.6f}")
    print(f"    MAE:        {avg_mae:.6f}")
    print(f"    RMSE:       {rmse:.6f}")
    print(f"    Samples:    {num_samples:.0f}")
    
    # Analyze predictions
    print("[4/4] Analyzing predictions...")
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Remove zeros (masked values)
    nonzero_mask = (all_targets != 0) & (all_predictions != 0)
    pred_nonzero = all_predictions[nonzero_mask]
    target_nonzero = all_targets[nonzero_mask]
    
    if len(pred_nonzero) > 0:
        # Compute correlation
        correlation = np.corrcoef(pred_nonzero.flatten(), target_nonzero.flatten())[0, 1]
        
        # Compute relative error
        rel_error = np.abs(pred_nonzero - target_nonzero) / (np.abs(target_nonzero) + 1e-8)
        avg_rel_error = np.mean(rel_error)
        
        print(f"  ✓ Analysis complete")
        print()
        print("  Statistics (non-masked values):")
        print(f"    Correlation:       {correlation:.6f}")
        print(f"    Avg Relative Err:  {avg_rel_error:.6f} ({avg_rel_error*100:.2f}%)")
        print(f"    Prediction Range:  [{pred_nonzero.min():.6f}, {pred_nonzero.max():.6f}]")
        print(f"    Target Range:      [{target_nonzero.min():.6f}, {target_nonzero.max():.6f}]")
    
    print()
    print("="*70)
    print("Validation Summary")
    print("="*70)
    
    # Overall assessment
    if avg_loss < 0.15:
        quality = "EXCELLENT ✓✓✓"
    elif avg_loss < 0.25:
        quality = "GOOD ✓✓"
    elif avg_loss < 0.35:
        quality = "FAIR ✓"
    else:
        quality = "NEEDS MORE TRAINING"
    
    print(f"Model Quality: {quality}")
    print(f"Loss: {avg_loss:.6f}")
    
    if len(pred_nonzero) > 0:
        print(f"Correlation: {correlation:.6f}")
        
        if correlation > 0.95:
            print("Prediction accuracy: EXCELLENT")
        elif correlation > 0.85:
            print("Prediction accuracy: GOOD")
        elif correlation > 0.70:
            print("Prediction accuracy: FAIR")
        else:
            print("Prediction accuracy: POOR - Consider retraining")
    
    print()
    print("Model ready for deployment!" if avg_loss < 0.25 else "Consider training for more epochs.")
    print("="*70)
    
    return {
        'loss': avg_loss,
        'mae': avg_mae,
        'rmse': rmse,
        'correlation': correlation if len(pred_nonzero) > 0 else 0.0,
        'quality': quality
    }


def main():
    parser = argparse.ArgumentParser(description='Validate trained DeepStack model')
    parser.add_argument('--model', type=str, 
                       default='models/pretrained/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data-path', type=str,
                       default='data/deepstacked_training/samples/train_samples',
                       help='Path to validation data')
    parser.add_argument('--num-buckets', type=int, default=36,
                       help='Number of hand buckets')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for validation')
    
    args = parser.parse_args()
    
    # Validate
    results = validate_model(args.model, args.data_path, 
                           args.num_buckets, args.batch_size)


if __name__ == '__main__':
    main()
