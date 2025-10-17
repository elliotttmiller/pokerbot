#!/usr/bin/env python3
"""
Unified Model Training Script

This script consolidates all training functionality with profile-based configurations.
Replaces: train_deepstack.py, train_champion.py, and provides unified interface

Usage:
  # Using predefined profiles
  python scripts/train_model.py --profile testing
  python scripts/train_model.py --profile development  
  python scripts/train_model.py --profile production
  python scripts/train_model.py --profile championship
  
  # Custom configuration
  python scripts/train_model.py --config config/training/custom.json
  
  # Override specific parameters
  python scripts/train_model.py --profile production --epochs 300 --batch-size 2048
  
  # With GPU
  python scripts/train_model.py --profile championship --use-gpu
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
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
import torch.nn as nn
from deepstack.core.net_builder import NetBuilder
from deepstack.core.data_stream import DataStream
from deepstack.core.masked_huber_loss import MaskedHuberLoss

try:
    from train_analyzer import TrainAnalyzer
except Exception:
    try:
        from scripts.train_analyzer import TrainAnalyzer
    except Exception:
        scripts_dir = os.path.dirname(__file__)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from train_analyzer import TrainAnalyzer


# Predefined training profiles
PROFILES = {
    'testing': {
        'data_path': 'src/train_samples_test',
        'epochs': 20,
        'batch_size': 256,
        'lr': 0.001,
        'huber_delta': 0.5,
        'early_stop_patience': 5,
        'description': 'Quick testing (minutes)',
    },
    'development': {
        'data_path': 'src/train_samples_dev',
        'epochs': 100,
        'batch_size': 1024,
        'lr': 0.0005,
        'huber_delta': 0.3,
        'early_stop_patience': 15,
        'use_street_weighting': True,
        'street_weights': [0.8, 1.0, 1.2, 1.4],
        'description': 'Development iteration',
    },
    'production': {
        'data_path': 'src/train_samples_production',
        'epochs': 200,
        'batch_size': 1024,
        'effective_batch_size': 4096,
        'lr': 0.0005,
        'min_lr': 1e-6,
        'huber_delta': 0.3,
        'warmup_epochs': 10,
        'weight_decay': 0.01,
        'ema_decay': 0.999,
        'early_stop_patience': 20,
        'use_street_weighting': True,
        'street_weights': [0.8, 1.0, 1.2, 1.4],
        'use_torch_compile': True,
        'description': 'Production quality',
    },
    'championship': {
        'data_path': 'src/train_samples_championship',
        'epochs': 200,
        'batch_size': 1024,
        'effective_batch_size': 4096,
        'lr': 0.0005,
        'min_lr': 1e-6,
        'huber_delta': 0.3,
        'warmup_epochs': 10,
        'weight_decay': 0.01,
        'ema_decay': 0.999,
        'early_stop_patience': 25,
        'use_street_weighting': True,
        'street_weights': [0.6, 1.0, 1.4, 2.0],  # More aggressive
        'use_torch_compile': True,
        'description': 'Championship-level',
    },
}


def load_config_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Unified model training with profile support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  testing      : 20 epochs, quick validation (~minutes)
  development  : 100 epochs, moderate training (~hours)
  production   : 200 epochs, production quality (~2-4 hours GPU)
  championship : 200 epochs, championship-level (~2-4 hours GPU)

Examples:
  # Quick test
  python scripts/train_model.py --profile testing
  
  # Production with GPU
  python scripts/train_model.py --profile production --use-gpu
  
  # Championship with custom epochs
  python scripts/train_model.py --profile championship --epochs 300 --use-gpu
        """
    )
    
    # Profile or config selection
    parser.add_argument('--profile', type=str, choices=list(PROFILES.keys()),
                       help='Use predefined profile')
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')
    
    # Parameter overrides
    parser.add_argument('--data-path', type=str,
                       help='Path to training data (overrides profile)')
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs (overrides profile)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (overrides profile)')
    parser.add_argument('--lr', type=float,
                       help='Learning rate (overrides profile)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU if available')
    parser.add_argument('--fresh', action='store_true',
                       help='Train from scratch (ignore existing checkpoints)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    # Output options
    parser.add_argument('--versions-dir', type=str, default='models/versions',
                       help='Directory for model versions')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                       help='Directory for checkpoints')
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.config:
        config = load_config_file(args.config)
        source = f"config file: {args.config}"
    elif args.profile:
        config = PROFILES[args.profile].copy()
        source = f"profile: {args.profile}"
    else:
        # Default to development
        config = PROFILES['development'].copy()
        source = "profile: development (default)"
        print("⚠ No profile or config specified, using 'development' profile")
        print()
    
    # Apply parameter overrides
    if args.data_path:
        config['data_path'] = args.data_path
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['lr'] = args.lr
    if args.use_gpu:
        config['use_gpu'] = True
    if args.fresh:
        config['fresh'] = True
    if args.verbose:
        config['verbose'] = True
    
    # Set output directories
    config['versions_dir'] = args.versions_dir
    config['checkpoint_dir'] = args.checkpoint_dir
    
    # Set defaults
    config.setdefault('num_buckets', 169)
    config.setdefault('bucket_count', 169)
    config.setdefault('hidden_sizes', [500, 500, 500, 500, 500, 500, 500])
    config.setdefault('activation', 'prelu')
    config.setdefault('seed', 42)
    config.setdefault('deterministic', False)
    
    # Display configuration
    print("="*70)
    print("UNIFIED MODEL TRAINING")
    print("="*70)
    print()
    print(f"Configuration source: {source}")
    if 'description' in config:
        print(f"Description: {config['description']}")
    print()
    print("Training parameters:")
    print(f"  Data path:        {config['data_path']}")
    print(f"  Epochs:           {config['epochs']}")
    print(f"  Batch size:       {config['batch_size']}")
    print(f"  Learning rate:    {config['lr']}")
    print(f"  Huber delta:      {config.get('huber_delta', 0.5)}")
    print(f"  Use GPU:          {config.get('use_gpu', False)}")
    print(f"  Fresh training:   {config.get('fresh', False)}")
    print()
    print("Model architecture:")
    print(f"  Buckets:          {config['num_buckets']}")
    print(f"  Hidden layers:    {config['hidden_sizes']}")
    print(f"  Activation:       {config['activation']}")
    print()
    print("="*70)
    print()
    
    # Import and run DeepStack trainer
    from deepstack.core.train_deepstack import DeepStackTrainer
    
    # Create trainer
    trainer = DeepStackTrainer(config)
    
    # Train
    trainer.train()
    
    print()
    print("="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Validate: python scripts/validate_model.py")
    print("  2. Analyze: python scripts/run_analysis_report.py")
    print("="*70)


if __name__ == '__main__':
    main()
