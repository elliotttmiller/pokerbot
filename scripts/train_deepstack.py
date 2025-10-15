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

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.optim as optim
import torch.nn as nn
from deepstack.core.net_builder import NetBuilder
from deepstack.core.data_stream import DataStream
from deepstack.core.masked_huber_loss import MaskedHuberLoss


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
        
        # Initialize network
        num_buckets = config['num_buckets']
        hidden_sizes = config.get('hidden_sizes', [256, 256, 128])
        activation = config.get('activation', 'prelu')
        
        self.net = NetBuilder.build_net(
            num_buckets=num_buckets,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_gpu=self.use_gpu
        )
        
        # Initialize data stream
        data_path = config['data_path']
        batch_size = config.get('batch_size', 32)
        self.data_stream = DataStream(data_path, batch_size, use_gpu=self.use_gpu)
        
        # Initialize training components
        self.criterion = MaskedHuberLoss(delta=1.0)
        lr = config.get('lr', 0.001)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training parameters
        self.epochs = config.get('epochs', 50)
        self.model_save_dir = config.get('model_save_dir', 'models/pretrained')
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience = 10
        self.patience_counter = 0
        
        print(f"[INFO] Initialized DeepStack Trainer")
        print(f"  Device: {self.device}")
        print(f"  Network: {num_buckets} buckets, hidden={hidden_sizes}, activation={activation}")
        print(f"  Training samples: {self.data_stream.train_data_count}")
        print(f"  Validation samples: {self.data_stream.valid_data_count}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Epochs: {self.epochs}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.net.train()
        self.data_stream.start_epoch()
        
        total_loss = 0.0
        num_batches = self.data_stream.get_train_batch_count()
        
        for batch_idx in range(num_batches):
            # Get batch
            inputs, targets, mask = self.data_stream.get_batch('train', batch_idx)
            
            # Already tensors from DataStream, just move to device if needed
            if self.use_gpu and not inputs.is_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                mask = mask.cuda()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets, mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate on validation set."""
        self.net.eval()
        
        total_loss = 0.0
        num_batches = self.data_stream.get_valid_batch_count()
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Get batch
                inputs, targets, mask = self.data_stream.get_batch('valid', batch_idx)
                
                # Move to device if needed
                if self.use_gpu and not inputs.is_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    mask = mask.cuda()
                
                # Forward pass
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets, mask)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.model_save_dir, 'best_model.pt')
            torch.save(self.net.state_dict(), best_path)
            print(f"  ✓ Saved best model to {best_path}")
        
        # Save final model
        final_path = os.path.join(self.model_save_dir, 'final_model.pt')
        torch.save(self.net.state_dict(), final_path)
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("Starting DeepStack Neural Network Training")
        print("="*70)
        
        # Load existing best model if available
        best_model_path = os.path.join(self.model_save_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            print(f"[INFO] Loading existing best model for continual training...")
            self.net.load_state_dict(torch.load(best_model_path))
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Print progress
            print(f"Epoch {epoch:3d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
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
        print(f"Model saved to: {self.model_save_dir}")
        
        # Load best model for final save
        if self.best_model_state is not None:
            self.net.load_state_dict(self.best_model_state)
            final_path = os.path.join(self.model_save_dir, 'final_model.pt')
            torch.save(self.net.state_dict(), final_path)
            print(f"Final best model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train DeepStack value network',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='scripts/config/training.json',
                      help='Path to training configuration JSON file')
    
    # Override options
    parser.add_argument('--data-path', type=str, default=None,
                      help='Override data path')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                      help='Override learning rate')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Force GPU usage (if available)')
    parser.add_argument('--no-gpu', action='store_true',
                      help='Force CPU usage')
    parser.add_argument('--model-dir', type=str, default=None,
                      help='Override model save directory')
    parser.add_argument('--verbose', action='store_true',
                      help='Verbose output')
    
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
    if args.use_gpu:
        config['use_gpu'] = True
    if args.no_gpu:
        config['use_gpu'] = False
    if args.model_dir is not None:
        config['model_save_dir'] = args.model_dir
    
    # Validate data path
    data_path = config['data_path']
    if not os.path.exists(data_path):
        print(f"Error: Data path not found: {data_path}")
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
