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
    python scripts/validate_deepstack_model.py --model models/versions/best_model.pt
"""

import argparse
import sys
import os
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

import torch
import numpy as np
from deepstack.core.net_builder import NetBuilder
from deepstack.core.data_stream import DataStream
from deepstack.core.masked_huber_loss import MaskedHuberLoss
from deepstack.core.temperature_scaling import TemperatureScaler, evaluate_calibration


def validate_model(model_path, data_path, num_buckets: int | None = None, batch_size=32):
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
    # Try to load scaling metadata for de-standardization
    scaling_mean = None
    scaling_std = None
    scaling_path = os.path.join(data_path, 'targets_scaling.pt')
    if os.path.exists(scaling_path):
        try:
            scaling = torch.load(scaling_path, map_location='cpu')
            scaling_mean = scaling.get('mean')
            scaling_std = scaling.get('std')
            if isinstance(scaling_mean, torch.Tensor):
                scaling_mean = scaling_mean.float()
            if isinstance(scaling_std, torch.Tensor):
                scaling_std = scaling_std.float()
            print("  ✓ Loaded target scaling for de-standardization")
        except Exception:
            print("  [WARN] Failed to load targets_scaling.pt; proceeding without de-standardization")
    
    # Load model
    print("[2/4] Loading trained model...")
    raw_obj = torch.load(model_path, map_location='cpu')
    # Support both checkpoint dicts and raw state_dicts
    if isinstance(raw_obj, dict) and 'model_state_dict' in raw_obj:
        state_dict = raw_obj['model_state_dict']
        ckpt_config = raw_obj.get('config', {})
    else:
        state_dict = raw_obj
        ckpt_config = {}

    def reconstruct_hidden_from_state(sd: dict) -> list:
        # Find linear weight layers in order: model.<idx>.weight
        weight_items = [(k, v) for k, v in sd.items() if k.startswith('model.') and k.endswith('.weight') and v.dim() == 2]
        # Sort by numeric index within 'model.<n>.weight'
        def idx_of(k: str) -> int:
            try:
                return int(k.split('.')[1])
            except Exception:
                return 1_000_000
        weight_items.sort(key=lambda kv: idx_of(kv[0]))
        # Extract out_features for all but last (last is output layer)
        outs = [w.shape[0] for (_, w) in weight_items]
        if len(outs) >= 2:
            return outs[:-1]
        return []

    # Prefer config hidden_sizes if available via latest checkpoint in checkpoints dir
    hidden_sizes = []
    if 'hidden_sizes' in ckpt_config and ckpt_config['hidden_sizes']:
        hidden_sizes = ckpt_config['hidden_sizes']
    else:
        # Try reading most recent checkpoint for config
        checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'models', 'checkpoints')
        try:
            if os.path.exists(checkpoints_dir):
                ckpts = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
                if ckpts:
                    latest = max(ckpts, key=os.path.getmtime)
                    latest_obj = torch.load(latest, map_location='cpu')
                    if isinstance(latest_obj, dict):
                        cfg = latest_obj.get('config', {})
                        if cfg and cfg.get('hidden_sizes'):
                            hidden_sizes = cfg['hidden_sizes']
        except Exception:
            pass
        # Fallback: reconstruct from state_dict
        if not hidden_sizes:
            hidden_sizes = reconstruct_hidden_from_state(state_dict) or [256, 256, 128]
    
    # Infer bucket count from targets (robust to extra input features)
    if not num_buckets or num_buckets <= 0:
        target_dim = int(data_stream.data['valid_targets'].shape[1])
        num_buckets = target_dim // 2
        print(f"  Inferred num_buckets from targets: {num_buckets}")
    print(f"  Architecture: {num_buckets} buckets, hidden={hidden_sizes}")
    
    # Provide actual input size to be robust to extra features
    input_dim = int(data_stream.data['valid_inputs'].shape[1])
    net = NetBuilder.build_net(num_buckets=num_buckets, 
                               hidden_sizes=hidden_sizes,
                               activation='prelu',
                               use_gpu=False,
                               input_size=input_dim)
    try:
        net.load_state_dict(state_dict)
    except Exception as e:
        print(f"[WARN] Exact load failed ({e}); attempting non-strict load...")
        missing = net.load_state_dict(state_dict, strict=False)
        if missing.missing_keys or missing.unexpected_keys:
            print(f"[WARN] Non-strict load report: missing={missing.missing_keys}, unexpected={missing.unexpected_keys}")
    net.eval()
    print(f"  ✓ Model loaded successfully")
    
    # Validate
    print("[3/4] Computing validation metrics...")
    # Use lower Huber delta for more sensitive error penalization
    criterion = MaskedHuberLoss(delta=0.5)
    
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    num_samples = 0
    num_batches = data_stream.get_valid_batch_count()
    
    all_predictions = []
    all_targets = []
    all_streets = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs, targets, mask = data_stream.get_batch('valid', batch_idx)
            # Forward pass
            outputs = net(inputs)
            # Compute metrics
            loss = criterion(outputs, targets, mask)
            total_loss += loss.item()
            # Masked metrics (always apply mask before any metric)
            masked_outputs = outputs * mask
            masked_targets = targets * mask
            # De-standardize for reporting, then re-apply mask so masked entries stay zero
            if scaling_mean is not None and scaling_std is not None:
                mean = scaling_mean.view(1, -1)
                std = scaling_std.view(1, -1)
                outputs_ds = outputs * std + mean
                targets_ds = targets * std + mean
                masked_outputs = outputs_ds * mask
                masked_targets = targets_ds * mask
            mae = torch.mean(torch.abs(masked_outputs - masked_targets))
            mse = torch.mean((masked_outputs - masked_targets) ** 2)
            total_mae += mae.item()
            total_mse += mse.item()
            num_samples += mask.sum().item()
            all_predictions.append(masked_outputs.cpu().numpy())
            all_targets.append(masked_targets.cpu().numpy())
            if 'valid_street' in data_stream.data:
                # Repeat street for each sample in this batch
                batch_streets = data_stream.data['valid_street'][batch_idx*data_stream.train_batch_size : (batch_idx+1)*data_stream.train_batch_size]
                all_streets.append(batch_streets.cpu().numpy())
    # Player-half alignment check (diagnostic)
    if all_predictions and all_targets:
        P = np.concatenate(all_predictions)
        T = np.concatenate(all_targets)
        num_outputs = P.shape[1]
        half = num_outputs // 2
        # Correlation for each half
        corr_p1 = np.corrcoef(P[:, :half].flatten(), T[:, :half].flatten())[0, 1]
        corr_p2 = np.corrcoef(P[:, half:].flatten(), T[:, half:].flatten())[0, 1]
        print(f"  [DIAG] Player 1 half correlation: {corr_p1:.4f}")
        print(f"  [DIAG] Player 2 half correlation: {corr_p2:.4f}")
        if corr_p1 < -0.1 or corr_p2 < -0.1:
            print("  [WARN] Negative correlation detected in player halves: check alignment and mask application!")
    
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
    streets_concat = None
    if len(all_streets) > 0:
        streets_concat = np.concatenate(all_streets)
    
    # Remove zeros (masked values)
    nonzero_mask = (all_targets != 0) & (all_predictions != 0)
    pred_nonzero = all_predictions[nonzero_mask]
    target_nonzero = all_targets[nonzero_mask]
    
    # Compute basic coverage diagnostics before filtering
    total_vals = all_targets.size
    masked_vals = np.count_nonzero(all_targets == 0)
    covered_vals = total_vals - masked_vals
    coverage_ratio = covered_vals / max(1, total_vals)

    if len(pred_nonzero) > 0:
        # Compute correlation
        correlation = np.corrcoef(pred_nonzero.flatten(), target_nonzero.flatten())[0, 1]
        
        # Compute relative error
        rel_error = np.abs(pred_nonzero - target_nonzero) / (np.abs(target_nonzero) + 1e-8)
        avg_rel_error = np.mean(rel_error)
        # Compute de-standardized MAE/RMSE (these arrays are de-standardized already)
        mae_ds = np.mean(np.abs(pred_nonzero - target_nonzero))
        rmse_ds = np.sqrt(np.mean((pred_nonzero - target_nonzero) ** 2))
        # Sign mismatch rate (directional error)
        sign_mismatch = np.mean(np.sign(pred_nonzero) != np.sign(target_nonzero))
        # Simple calibration: slope and intercept of least-squares fit target ~ a*pred + b
        try:
            x = pred_nonzero.reshape(-1, 1)
            X = np.concatenate([x, np.ones_like(x)], axis=1)
            theta, *_ = np.linalg.lstsq(X, target_nonzero.reshape(-1, 1), rcond=None)
            calib_slope = float(theta[0, 0])
            calib_bias = float(theta[1, 0])
        except Exception:
            calib_slope, calib_bias = float('nan'), float('nan')
        
        print(f"  ✓ Analysis complete")
        print()
        print("  Statistics (non-masked values):")
        print(f"    Correlation:       {correlation:.6f}")
        print(f"    MAE (de-std):      {mae_ds:.6f}")
        print(f"    RMSE (de-std):     {rmse_ds:.6f}")
        print(f"    Avg Relative Err:  {avg_rel_error:.6f} ({avg_rel_error*100:.2f}%)")
        print(f"    Sign mismatch:     {sign_mismatch:.6f}")
        print(f"    Calibration:       slope={calib_slope:.4f}, bias={calib_bias:.4f}")
        print(f"    Prediction Range:  [{pred_nonzero.min():.6f}, {pred_nonzero.max():.6f}]")
        print(f"    Target Range:      [{target_nonzero.min():.6f}, {target_nonzero.max():.6f}]")
        print()
        print("  Coverage:")
        print(f"    Non-masked ratio:  {coverage_ratio:.4f} ({covered_vals}/{total_vals})")
        # Per-street correlations if street data is available
        if streets_concat is not None:
            print()
            print("  Per-street correlation (0=pre,1=flop,2=turn,3=river):")
            for street in [0, 1, 2, 3]:
                idx = streets_concat == street
                # Need to map sample-wise street to per-output vectors: aggregate by samples
                if idx.any():
                    # Collect batch of outputs/targets for those samples
                    # all_predictions/all_targets have shape [N, output_dim]; compute corr per flattened
                    p = all_predictions[idx]
                    t = all_targets[idx]
                    # Mask zero entries again just in case
                    mask_nonzero = (t != 0) & (p != 0)
                    if mask_nonzero.any():
                        corr = np.corrcoef(p[mask_nonzero].flatten(), t[mask_nonzero].flatten())[0, 1]
                        print(f"    Street {street}: corr={corr:.6f} on {mask_nonzero.sum()} values")
            # Also print street coverage
            print()
            print("  Per-street coverage (non-masked counts):")
            for street in [0, 1, 2, 3]:
                idx = streets_concat == street
                count = int((all_targets[idx] != 0).sum())
                print(f"    Street {street}: {count} values")
        # Concise per-bucket correlation summary (across all samples)
        print()
        print("  Per-bucket correlation summary:")
        # Compute per-output correlation, then aggregate by player half
        P = all_predictions
        T = all_targets
        num_outputs = P.shape[1]
        corrs = np.zeros(num_outputs)
        for j in range(num_outputs):
            pj = P[:, j]
            tj = T[:, j]
            nz = (pj != 0) & (tj != 0)
            if nz.any():
                corrs[j] = np.corrcoef(pj[nz], tj[nz])[0, 1]
            else:
                corrs[j] = 0.0
        # Player halves
        half = num_outputs // 2
        avg_p1 = float(np.nanmean(corrs[:half]))
        avg_p2 = float(np.nanmean(corrs[half:]))
        avg_all = float(np.nanmean(corrs))
        print(f"    Avg corr (P1 half): {avg_p1:.4f} | (P2 half): {avg_p2:.4f} | (All): {avg_all:.4f}")
        # Show worst and best 5 buckets indices for quick inspection
        worst_idx = np.argsort(corrs)[:5]
        best_idx = np.argsort(-corrs)[:5]
        print(f"    Worst 5 dims: {[(int(i), float(corrs[i])) for i in worst_idx]}")
        print(f"    Best  5 dims: {[(int(i), float(corrs[i])) for i in best_idx]}")

        # Export per-bucket correlations (averaged across players) for downstream weighting
        try:
            bucket_corrs = []
            nb = num_outputs // 2
            for b in range(nb):
                # average both players' dims for this bucket
                vals = []
                if not np.isnan(corrs[b]):
                    vals.append(float(corrs[b]))
                if not np.isnan(corrs[nb + b]):
                    vals.append(float(corrs[nb + b]))
                bucket_corrs.append(float(np.mean(vals)) if vals else 0.0)
            # Unified reports directory at repo root: models/reports
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            reports_dir = os.path.join(repo_root, 'models', 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            out_json = os.path.join(reports_dir, 'per_bucket_corrs.json')
            with open(out_json, 'w') as f:
                import json as _json
                _json.dump({
                    'num_buckets': int(nb),
                    'bucket_corrs': bucket_corrs
                }, f, indent=2)
            print(f"  ✓ Exported per-bucket correlations to {out_json}")
            # Save diagnostics JSON with actionable notes
            diag = {
                'correlation': float(correlation),
                'mae_ds': float(mae_ds),
                'rmse_ds': float(rmse_ds),
                'avg_relative_error': float(avg_rel_error),
                'sign_mismatch_rate': float(sign_mismatch),
                'calibration': {'slope': calib_slope, 'bias': calib_bias},
                'coverage_ratio': float(coverage_ratio),
                'avg_corr_player1_half': avg_p1,
                'avg_corr_player2_half': avg_p2,
                'avg_corr_all': avg_all,
                'worst_dims': [(int(i), float(corrs[int(i)])) for i in worst_idx],
                'best_dims': [(int(i), float(corrs[int(i)])) for i in best_idx]
            }
            try:
                if streets_concat is not None:
                    per_street = {}
                    for s in [0,1,2,3]:
                        idx = streets_concat == s
                        if idx.any():
                            p = all_predictions[idx]
                            t = all_targets[idx]
                            nz = (p!=0) & (t!=0)
                            if nz.any():
                                per_street[str(s)] = float(np.corrcoef(p[nz].flatten(), t[nz].flatten())[0,1])
                    diag['per_street_correlation'] = per_street
            except Exception:
                pass
            with open(os.path.join(reports_dir, 'diagnostics.json'), 'w') as f:
                _json.dump(diag, f, indent=2)
        except Exception as _e:
            print(f"  [WARN] Failed to export per-bucket correlations: {_e}")

        # Temperature Scaling for Calibration
        print()
        print("  Temperature Scaling Analysis:")
        try:
            temp_scaler = TemperatureScaler()
            # Use validation data for temperature scaling
            temp_value = temp_scaler.fit(pred_nonzero, target_nonzero, lr=0.01, max_iter=50)
            print(f"    Optimal Temperature: {temp_value:.4f}")
            
            # Apply temperature scaling
            scaled_preds = temp_scaler.transform(pred_nonzero)
            
            # Re-compute calibration with scaled predictions
            try:
                x_scaled = scaled_preds.reshape(-1, 1) if isinstance(scaled_preds, np.ndarray) else scaled_preds.cpu().numpy().reshape(-1, 1)
                X_scaled = np.concatenate([x_scaled, np.ones_like(x_scaled)], axis=1)
                theta_scaled, *_ = np.linalg.lstsq(X_scaled, target_nonzero.reshape(-1, 1), rcond=None)
                calib_slope_scaled = float(theta_scaled[0, 0])
                calib_bias_scaled = float(theta_scaled[1, 0])
                print(f"    Calibration after scaling: slope={calib_slope_scaled:.4f}, bias={calib_bias_scaled:.4f}")
                
                # Improvement metrics
                slope_improvement = abs(1.0 - calib_slope_scaled) - abs(1.0 - calib_slope)
                bias_improvement = abs(calib_bias_scaled) - abs(calib_bias)
                if slope_improvement < 0:
                    print(f"    ✓ Slope improved by {abs(slope_improvement):.4f}")
                if bias_improvement < 0:
                    print(f"    ✓ Bias improved by {abs(bias_improvement):.4f}")
            except Exception as e:
                print(f"    [WARN] Could not compute post-scaling calibration: {e}")
            
            # Save temperature scaler for deployment
            try:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                reports_dir = os.path.join(repo_root, 'models', 'reports')
                temp_path = os.path.join(reports_dir, 'temperature_scaler.pt')
                temp_scaler.save(temp_path)
                print(f"    ✓ Saved temperature scaler to {temp_path}")
            except Exception:
                pass
                
        except Exception as e:
            print(f"    [WARN] Temperature scaling failed: {e}")
        
        # Per-player diagnostics
        print()
        print("  Per-Player Diagnostics:")
        half = num_outputs // 2
        # Player 1 metrics
        p1_preds = all_predictions[:, :half]
        p1_targets = all_targets[:, :half]
        p1_mask = (p1_preds != 0) & (p1_targets != 0)
        if p1_mask.any():
            p1_corr = float(np.corrcoef(p1_preds[p1_mask].flatten(), p1_targets[p1_mask].flatten())[0, 1])
            p1_mae = float(np.mean(np.abs(p1_preds[p1_mask] - p1_targets[p1_mask])))
            p1_sign_mismatch = float(np.mean(np.sign(p1_preds[p1_mask]) != np.sign(p1_targets[p1_mask])))
            print(f"    Player 1: corr={p1_corr:.4f}, MAE={p1_mae:.4f}, sign_mismatch={p1_sign_mismatch:.4f}")
        
        # Player 2 metrics
        p2_preds = all_predictions[:, half:]
        p2_targets = all_targets[:, half:]
        p2_mask = (p2_preds != 0) & (p2_targets != 0)
        if p2_mask.any():
            p2_corr = float(np.corrcoef(p2_preds[p2_mask].flatten(), p2_targets[p2_mask].flatten())[0, 1])
            p2_mae = float(np.mean(np.abs(p2_preds[p2_mask] - p2_targets[p2_mask])))
            p2_sign_mismatch = float(np.mean(np.sign(p2_preds[p2_mask]) != np.sign(p2_targets[p2_mask])))
            print(f"    Player 2: corr={p2_corr:.4f}, MAE={p2_mae:.4f}, sign_mismatch={p2_sign_mismatch:.4f}")
        
        # Flag if asymmetric (suggests alignment issue)
        if p1_mask.any() and p2_mask.any():
            corr_diff = abs(p1_corr - p2_corr)
            if corr_diff > 0.2:
                print(f"    ⚠ Large correlation difference ({corr_diff:.4f}) - check player alignment!")
        
        # Print actionable recommendations based on diagnostics
        print()
        print("Recommendations:")
        recs = []
        if calib_slope < 0.8 or calib_slope > 1.2:
            recs.append("Model is miscalibrated: adjust learning rate schedule or add EMA/bigger batch; consider temperature scaling in evaluation.")
        if sign_mismatch > 0.35:
            recs.append("High sign mismatch: network struggles with directionality; increase CFR iterations and emphasize postflop buckets using derive_bucket_weights.")
        if avg_all < 0.2 and avg_loss > 0.35:
            recs.append("Low correlation and high loss: generate more data with higher CFR iterations (>=2000) and train longer on GPU with AMP.")
        if streets_concat is not None:
            # Check for under-covered streets
            street_counts = {}
            for s in [0,1,2,3]:
                idx = streets_concat == s
                street_counts[s] = int((all_targets[idx]!=0).sum())
            if street_counts.get(1,0) < 1e4 or street_counts.get(3,0) < 1e4:
                recs.append("Low coverage on flop/river: bias sampling towards later streets or increase dataset size.")
        if avg_rel_error > 3.0:
            recs.append("Very high relative error: verify target standardization/denormalization and loss masking; consider lowering Huber delta.")
        if avg_p1 < 0 or avg_p2 < 0:
            recs.append("Negative per-player correlations: double-check player-half alignment and mask application.")
        if not recs:
            recs.append("Continue training for more epochs with cosine decay and monitor analyzer trends; iterate bucket weighting from latest diagnostics.")
        
        # Enhanced recommendations with priority levels
        print()
        print("  CRITICAL (Fix Immediately):")
        critical_recs = [r for r in recs if any(word in r.lower() for word in ['negative', 'very high', 'miscalibrated'])]
        if critical_recs:
            for r in critical_recs:
                print(f"    🔴 {r}")
        else:
            print("    ✓ No critical issues detected")
        
        print()
        print("  HIGH PRIORITY (Address Soon):")
        high_recs = [r for r in recs if any(word in r.lower() for word in ['low correlation', 'low coverage', 'high sign'])]
        if high_recs:
            for r in high_recs:
                print(f"    🟡 {r}")
        else:
            print("    ✓ No high priority issues")
        
        print()
        print("  OPTIMIZATION (Iterative Improvement):")
        other_recs = [r for r in recs if r not in critical_recs and r not in high_recs]
        if other_recs:
            for r in other_recs:
                print(f"    🟢 {r}")
        else:
            print("    ✓ Model is performing well")
    
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
    
    result = {
        'loss': avg_loss,
        'mae': avg_mae,
        'rmse': rmse,
        'correlation': correlation if len(pred_nonzero) > 0 else 0.0,
        'quality': quality
    }
    # Attach per-bucket correlations if computed
    try:
        result['per_bucket_corrs'] = bucket_corrs  # type: ignore[name-defined]
        result['num_buckets'] = int(num_outputs // 2)
    except Exception:
        pass
    return result


def main():
    parser = argparse.ArgumentParser(description='Validate trained DeepStack model')
    parser.add_argument('--model', type=str, 
                       default='models/versions/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data-path', type=str,
                       default=r'C:/Users/AMD/pokerbot/src/train_samples',
                       help='Path to validation data')
    parser.add_argument('--num-buckets', type=int, default=0,
                       help='Number of hand buckets (0=infer from data)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for validation')
    
    args = parser.parse_args()
    
    # Validate
    results = validate_model(args.model, args.data_path, 
                           args.num_buckets, args.batch_size)


if __name__ == '__main__':
    main()
