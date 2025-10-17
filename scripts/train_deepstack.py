#!/usr/bin/env python3
"""
DeepStack Neural Network Training Script

This script trains the DeepStack value network using pre-generated training samples.
The network learns to estimate counterfactual values at lookahead leaves, which is
essential for DeepStack's depth-limited continual re-solving.

Usage:
    python scripts/train_deepstack.py --config scripts/config/training.json
    python scripts/train_deepstack.py --epochs 100 --batch-size 64 --lr 0.0005
    python scripts/train_deepstack.py --use-gpu --verbose
"""


import argparse
import json
import os
import sys
from pathlib import Path
import math
import random
import numpy as np
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
import torch.optim as optim
import torch.nn as nn
from deepstack.core.net_builder import NetBuilder
from deepstack.core.data_stream import DataStream
from deepstack.core.masked_huber_loss import MaskedHuberLoss
# Robust import for analyzer whether run as module or script
try:
    from train_analyzer import TrainAnalyzer  # same folder
except Exception:
    try:
        from scripts.train_analyzer import TrainAnalyzer
    except Exception:
        scripts_dir = os.path.dirname(__file__)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from train_analyzer import TrainAnalyzer


class DeepStackTrainer:
    """
    Trainer for DeepStack value network with best practices:
    - Early stopping with patience
    - Learning rate scheduling
    - Model checkpointing
    - Validation monitoring
    """
    
    def __init__(self, config):
        self.config = config
        self.use_gpu = config.get('use_gpu', False) and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.fresh = config.get('fresh', False)
        self.use_street_weighting = bool(config.get('use_street_weighting', True))
        self.street_weights = config.get('street_weights', [0.8, 1.0, 1.2, 1.4])

        # Reproducibility
        seed = int(config.get('seed', 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.use_gpu:
            torch.cuda.manual_seed_all(seed)
            # Performance-focused cuDNN settings
            deterministic = bool(config.get('deterministic', False))
            try:
                import torch.backends.cudnn as cudnn
                if deterministic:
                    torch.use_deterministic_algorithms(True)
                    cudnn.deterministic = True
                    cudnn.benchmark = False
                else:
                    cudnn.benchmark = True
            except Exception:
                pass
        
        # Initialize data stream first (to infer bucket count from data)
        data_path = config['data_path']
        batch_size = config.get('batch_size', 32)
        self.data_stream = DataStream(data_path, batch_size, use_gpu=self.use_gpu)
        # If DataStream adjusted the batch size, reflect it for logging/training
        if batch_size != self.data_stream.train_batch_size:
            print(f"[INFO] Adjusted batch size from {batch_size} to {self.data_stream.train_batch_size} based on data availability")
            batch_size = self.data_stream.train_batch_size

        # Gradient accumulation to achieve larger effective batch sizes
        effective_batch = int(self.config.get('effective_batch_size', batch_size))
        if effective_batch < 1:
            effective_batch = batch_size
        self.per_device_batch_size = batch_size
        self.effective_batch_size = max(effective_batch, batch_size)
        self.accumulation_steps = max(1, math.ceil(self.effective_batch_size / self.per_device_batch_size))

        # Infer number of buckets from TARGETS (robust to extra input features)
        input_dim = int(self.data_stream.data['train_inputs'].shape[1])
        target_dim = int(self.data_stream.data['train_targets'].shape[1])
        inferred_buckets = target_dim // 2

        # Compare with config and warn if mismatch
        cfg_buckets = config.get('num_buckets', inferred_buckets)
        if cfg_buckets != inferred_buckets:
            print(f"[WARN] Config num_buckets={cfg_buckets} mismatches dataset shape -> using inferred {inferred_buckets}")
        num_buckets = inferred_buckets

        # Initialize network with inferred buckets
        hidden_sizes = config.get('hidden_sizes', [256, 256, 128])
        activation = config.get('activation', 'prelu')
        
        self.net = NetBuilder.build_net(
            num_buckets=num_buckets,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_gpu=self.use_gpu,
            input_size=input_dim
        )
        # Optional compile for speed (PyTorch 2.x). Enable only on GPU to avoid CPU compiler requirements.
        self.use_torch_compile = bool(config.get('use_torch_compile', True)) and self.use_gpu
        if self.use_torch_compile:
            try:
                self.net = torch.compile(self.net)  # type: ignore[attr-defined]
                print("  torch.compile: enabled")
            except Exception:
                print("  torch.compile: unavailable; skipping")
        
        # Initialize training components
        # Slightly smaller delta in standardized units and normalize by valid fraction for stability
        self.criterion = MaskedHuberLoss(delta=float(config.get('huber_delta', 0.5)), normalize_by_valid_fraction=True)
        lr = float(config.get('lr', 0.001))
        weight_decay = float(config.get('weight_decay', 0.01))
        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        # Mixed precision (AMP) setup
        self.use_amp = self.use_gpu
        try:
            # New API (PyTorch 2.1+)
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        except Exception:
            # Fallback to older API
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Scheduler: warmup + cosine decay (per-epoch)
        self.warmup_epochs = int(config.get('warmup_epochs', 5))
        self.base_lr = lr
        self.min_lr = float(config.get('min_lr', 1e-5))
        
        # Training parameters and directories
        self.epochs = config.get('epochs', 50)
        # Standardized directories per repo convention (no legacy pretrained dir)
        self.checkpoint_dir = config.get('checkpoint_dir', 'models/checkpoints')
        self.versions_dir = config.get('versions_dir', 'models/versions')
        # Ensure dirs exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.versions_dir, exist_ok=True)

        # Exponential Moving Average (EMA) of weights
        self.ema_decay = float(config.get('ema_decay', 0.999))
        self.ema_state = None
        if self.ema_decay > 0:
            self._init_ema()

        # Early stopping
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience = int(config.get('early_stop_patience', 10))
        self.min_delta = float(config.get('early_stop_min_delta', 0.0))
        self.patience_counter = 0
        
        print(f"[INFO] Initialized DeepStack Trainer")
        print(f"  Device: {self.device}")
        print(f"  Network: {num_buckets} buckets, hidden={hidden_sizes}, activation={activation}")
        print(f"  Training samples: {self.data_stream.train_data_count}")
        print(f"  Validation samples: {self.data_stream.valid_data_count}")
        print(f"  Batch size: {batch_size}")
        if self.accumulation_steps > 1:
            print(f"  Effective batch size: {self.effective_batch_size} via accumulation x{self.accumulation_steps}")
        print(f"  Learning rate: {lr}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Early stop: patience={self.patience}, min_delta={self.min_delta}")
        print(f"  Optimizer: AdamW (weight_decay={weight_decay})")
        print(f"  AMP: {'enabled' if self.use_amp else 'disabled'} | EMA: {'enabled' if self.ema_decay > 0 else 'disabled'} | Warmup epochs: {self.warmup_epochs} | min_lr: {self.min_lr}")
        print(f"  Street weighting: {'enabled' if self.use_street_weighting else 'disabled'} weights={self.street_weights}")
        # Analyzer for per-epoch diagnostics
        scaling_path = os.path.join(self.config['data_path'], 'targets_scaling.pt')
        scaling = None
        if os.path.exists(scaling_path):
            try:
                scaling = torch.load(scaling_path, map_location='cpu')
            except Exception:
                scaling = None
        # Unified reports directory at repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        unified_reports = os.path.join(repo_root, 'models', 'reports')
        self.analyzer = TrainAnalyzer(report_dir=unified_reports, scaling=scaling)

    def _init_ema(self):
        """Initialize EMA state from current model parameters."""
        self.ema_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def _update_ema(self):
        if self.ema_state is None:
            return
        with torch.no_grad():
            model_state = self.net.state_dict()
            for k, v in model_state.items():
                self.ema_state[k].mul_(self.ema_decay).add_(v.detach(), alpha=1.0 - self.ema_decay)

    def _apply_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def _get_current_state(self):
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def _get_ema_state(self):
        return self.ema_state

    def _lr_multiplier_for_epoch(self, epoch: int) -> float:
        # 1-based epoch
        if epoch <= max(1, self.warmup_epochs):
            return epoch / max(1, self.warmup_epochs)
        # Cosine decay over remaining epochs
        progress = (epoch - self.warmup_epochs) / max(1, (self.epochs - self.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.net.train()
        self.data_stream.start_epoch()
        
        total_loss = 0.0
        num_batches = self.data_stream.get_train_batch_count()
        
        # Gradient accumulation: step optimizer every accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx in range(num_batches):
            # Get batch
            inputs, targets, mask, streets = self.data_stream.get_batch_with_street('train', batch_idx)
            
            # Already tensors from DataStream, just move to device if needed
            if self.use_gpu and not inputs.is_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                mask = mask.cuda()
            
            # Forward pass with AMP
            # Optional street-balanced sample weights
            sample_weights = None
            if streets is not None and self.use_street_weighting:
                # Heavier weight on later streets where signal is harder
                # Mapping via config street_weights: [pre, flop, turn, river]
                sw = torch.ones((inputs.shape[0],), device=inputs.device)
                st = streets if streets.device == inputs.device else streets.to(inputs.device)
                w = self.street_weights
                pre_w, flop_w, turn_w, river_w = float(w[0]), float(w[1]), float(w[2]), float(w[3])
                sw = sw * pre_w
                sw = torch.where(st == 1, torch.ones_like(sw) * flop_w, sw)
                sw = torch.where(st == 2, torch.ones_like(sw) * turn_w, sw)
                sw = torch.where(st == 3, torch.ones_like(sw) * river_w, sw)
                sample_weights = sw

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.net(inputs)
                    batch_loss = self.criterion(outputs, targets, mask, sample_weights=sample_weights)
            else:
                outputs = self.net(inputs)
                batch_loss = self.criterion(outputs, targets, mask, sample_weights=sample_weights)
            # Accumulate gradients
            loss = batch_loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            
            # Optimizer step on schedule
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                # Update EMA after optimizer step
                self._update_ema()
                self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += batch_loss.item()
        
        avg_loss = total_loss / num_batches
        # Log one mini-batch snapshot to analyzer for train phase (last batch)
        try:
            self.analyzer.log_epoch(epoch, 'train', outputs.detach().cpu(), targets.detach().cpu(), mask.detach().cpu(), streets.detach().cpu() if streets is not None else None, loss_value=avg_loss)
        except Exception:
            pass
        return avg_loss
    
    def validate(self, epoch: int | None = None):
        """Validate on validation set."""
        self.net.eval()
        
        total_loss = 0.0
        num_batches = self.data_stream.get_valid_batch_count()
        
        # Accumulate for full-epoch analyzer logging
        outs_all = []
        tgs_all = []
        msk_all = []
        str_all = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Get batch
                inputs, targets, mask, streets = self.data_stream.get_batch_with_street('valid', batch_idx)
                
                # Move to device if needed
                if self.use_gpu and not inputs.is_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    mask = mask.cuda()
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.net(inputs)
                else:
                    outputs = self.net(inputs)
                loss = self.criterion(outputs, targets, mask)
                
                total_loss += loss.item()
                outs_all.append(outputs.detach().cpu())
                tgs_all.append(targets.detach().cpu())
                msk_all.append(mask.detach().cpu())
                if streets is not None:
                    str_all.append(streets.detach().cpu())
        
        avg_loss = total_loss / num_batches
        # Log full validation epoch snapshot
        try:
            if outs_all:
                outputs_cat = torch.cat(outs_all, dim=0)
                targets_cat = torch.cat(tgs_all, dim=0)
                mask_cat = torch.cat(msk_all, dim=0)
                streets_cat = torch.cat(str_all, dim=0) if str_all else None
                self.analyzer.log_epoch(epoch if epoch is not None else 0, 'valid', outputs_cat, targets_cat, mask_cat, streets_cat, loss_value=avg_loss)
        except Exception:
            pass
        return avg_loss

    def _is_state_dict_compatible(self, state: dict) -> bool:
        """Check if loaded state_dict is compatible with current network."""
        try:
            current = self.net.state_dict()
            # Quick check: compare shapes for common keys
            for k, v in current.items():
                if k in state:
                    if hasattr(v, 'shape') and hasattr(state[k], 'shape'):
                        if tuple(v.shape) != tuple(state[k].shape):
                            return False
                else:
                    # Missing key in loaded state
                    return False
            return True
        except Exception:
            return False
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint to standardized checkpoints directory
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model to standardized versions directory
        if is_best:
            state_to_save = self._get_ema_state() if self.ema_state is not None else self.net.state_dict()
            best_versions_path = os.path.join(self.versions_dir, 'best_model.pt')
            torch.save(state_to_save, best_versions_path)
            print(f"  ✓ Saved best model to {best_versions_path}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("Starting DeepStack Neural Network Training")
        print("="*70)
        
        # Load existing best model if available and not forcing fresh start
        best_model_path = os.path.join(self.versions_dir, 'best_model.pt')
        if os.path.exists(best_model_path) and not self.fresh:
            print(f"[INFO] Loading existing best model for continual training...")
            state = None
            try:
                state = torch.load(best_model_path, map_location=self.device)
                if self._is_state_dict_compatible(state):
                    self.net.load_state_dict(state)
                else:
                    print("[WARN] Existing best model is incompatible with current architecture; skipping load.")
            except Exception as e:
                print(f"[WARN] Could not load existing best model due to mismatch or error: {e}")
                print("       Proceeding with randomly initialized weights.")
        elif self.fresh:
            print("[INFO] Fresh start requested: ignoring any existing best model.")
        
        for epoch in range(1, self.epochs + 1):
            # Set learning rate via warmup+cosine schedule
            lr_mult = self._lr_multiplier_for_epoch(epoch)
            for pg in self.optimizer.param_groups:
                pg['lr'] = max(self.min_lr, self.base_lr * lr_mult)
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate with EMA weights if available
            current_state = None
            if self.ema_state is not None:
                current_state = self._get_current_state()
                self._apply_state_dict(self._get_ema_state())
            val_loss = self.validate(epoch)
            if current_state is not None:
                self._apply_state_dict(current_state)
            
            # Print progress
            print(f"Epoch {epoch:3d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # (No ReduceLROnPlateau; LR managed by warmup+cosine)
            
            # Check for improvement
            is_best = val_loss < (self.best_val_loss - self.min_delta)
            if is_best:
                self.best_val_loss = val_loss
                self.best_model_state = self.net.state_dict()
                self.patience_counter = 0
                print(f"  ★ New best validation loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint every 10 epochs or if best
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs")
                print(f"       Best validation loss: {self.best_val_loss:.6f}")
                break
        
        # Save final model
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Checkpoints dir: {self.checkpoint_dir}")
        print(f"Versions dir: {self.versions_dir}")

        # Load best model for final save and export to standardized versions directory
        if self.best_model_state is not None:
            self.net.load_state_dict(self.best_model_state)
            final_versions_path = os.path.join(self.versions_dir, 'final_model.pt')
            torch.save(self.net.state_dict(), final_versions_path)
            print(f"Final best model saved to: {final_versions_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train DeepStack value network',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='scripts/config/training.json',
                      help='Path to training configuration JSON file')
    
    # Override options
    parser.add_argument('--data-path', type=str, default=r'C:/Users/AMD/pokerbot/src/train_samples',
                      help='Override data path')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Override batch size')
    parser.add_argument('--effective-batch-size', type=int, default=None,
                      help='Target effective batch size via gradient accumulation')
    parser.add_argument('--lr', type=float, default=None,
                      help='Override learning rate')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Force GPU usage (if available)')
    parser.add_argument('--no-gpu', action='store_true',
                      help='Force CPU usage')
    # Deprecated: model-dir (pretrained) is removed; keep arg for compatibility but ignore
    parser.add_argument('--model-dir', type=str, default=None,
                      help='[DEPRECATED] No effect; models are saved in models/versions and checkpoints in models/checkpoints')
    parser.add_argument('--verbose', action='store_true',
                      help='Verbose output')
    parser.add_argument('--fresh', action='store_true',
                      help='Ignore existing best model and start fresh')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Apply overrides
    if args.data_path is not None:
        config['data_path'] = args.data_path
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    if args.effective_batch_size is not None:
        config['effective_batch_size'] = args.effective_batch_size
    if args.use_gpu:
        config['use_gpu'] = True
    if args.no_gpu:
        config['use_gpu'] = False
    # model_dir override is deprecated and ignored to enforce standardized paths
    if args.fresh:
        config['fresh'] = True
    
    # Validate data path
    data_path = config['data_path']
    if not os.path.exists(data_path):
        print(f"Error: Data path not found: {data_path}")
        print("Tip: Generate training samples with src/deepstack/data/data_generation.py")
        sys.exit(1)
    
    # Print configuration
    if args.verbose:
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()
    
    # Create trainer and train
    trainer = DeepStackTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
