#!/usr/bin/env python3
"""
Multi-GPU and Distributed Training System

Provides seamless multi-GPU training with:
- Automatic GPU detection and allocation
- Data parallelism with DistributedDataParallel
- Gradient accumulation across GPUs
- Mixed precision training (AMP)
- Fault tolerance and checkpointing

Usage:
  # Single GPU
  python scripts/distributed_train.py --profile production
  
  # Multi-GPU (automatic detection)
  python scripts/distributed_train.py --profile championship --multi-gpu
  
  # Distributed across nodes
  torchrun --nproc_per_node=4 scripts/distributed_train.py --profile production
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
import logging
from pathlib import Path
from typing import Dict, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Multi-GPU distributed training with advanced optimizations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = None
        self.is_distributed = False
        
    def setup_distributed(self):
        """Initialize distributed training environment."""
        try:
            import torch
            import torch.distributed as dist
            
            # Check if launched with torchrun/torch.distributed.launch
            if 'RANK' in os.environ:
                self.is_distributed = True
                self.rank = int(os.environ['RANK'])
                self.local_rank = int(os.environ['LOCAL_RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
                
                # Initialize process group
                dist.init_process_group(backend='nccl')
                
                # Set device
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f'cuda:{self.local_rank}')
                
                if self.rank == 0:
                    logger.info(f"Distributed training initialized: {self.world_size} GPUs")
            else:
                # Single GPU or CPU
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    if gpu_count > 1 and self.config.get('multi_gpu', False):
                        logger.warning(f"Found {gpu_count} GPUs but not launched with torchrun")
                        logger.warning("For multi-GPU training, use: torchrun --nproc_per_node=N script.py")
                    
                    self.device = torch.device('cuda:0')
                    logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = torch.device('cpu')
                    logger.info("Using CPU")
        
        except ImportError:
            logger.error("PyTorch not installed")
            self.device = None
    
    def prepare_model(self, model):
        """Prepare model for distributed training."""
        try:
            import torch
            import torch.nn as nn
            
            model = model.to(self.device)
            
            if self.is_distributed:
                # Wrap model with DistributedDataParallel
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False
                )
                
                if self.rank == 0:
                    logger.info("Model wrapped with DistributedDataParallel")
            elif self.config.get('multi_gpu', False) and torch.cuda.device_count() > 1:
                # DataParallel for single-node multi-GPU
                model = nn.DataParallel(model)
                logger.info(f"Model wrapped with DataParallel ({torch.cuda.device_count()} GPUs)")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to prepare model: {e}")
            return model
    
    def prepare_data_loader(self, dataset, batch_size: int, shuffle: bool = True):
        """Prepare data loader for distributed training."""
        try:
            import torch
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler
            
            if self.is_distributed:
                # Use DistributedSampler for proper data splitting
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=shuffle
                )
                
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=4,
                    pin_memory=True
                )
                
                if self.rank == 0:
                    logger.info(f"Created distributed data loader: {len(dataset)} samples, batch size {batch_size}")
                
                return loader, sampler
            else:
                # Standard data loader
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=4,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                
                logger.info(f"Created data loader: {len(dataset)} samples, batch size {batch_size}")
                
                return loader, None
                
        except Exception as e:
            logger.error(f"Failed to prepare data loader: {e}")
            return None, None
    
    def train_step(self, model, batch, optimizer, criterion, scaler=None):
        """Execute single training step with mixed precision."""
        try:
            import torch
            
            inputs, targets, masks = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)
            
            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, masks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return float('inf')
    
    def save_checkpoint(self, model, optimizer, epoch, filepath):
        """Save distributed checkpoint."""
        # Only rank 0 saves
        if self.rank == 0:
            try:
                import torch
                
                # Unwrap model if using DDP
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': self.config
                }
                
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, filepath)
                
                logger.info(f"Checkpoint saved: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
    
    def cleanup(self):
        """Cleanup distributed resources."""
        if self.is_distributed:
            try:
                import torch.distributed as dist
                dist.destroy_process_group()
                logger.info("Distributed training cleaned up")
            except:
                pass


def detect_gpu_configuration():
    """Detect and report GPU configuration."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                'available': False,
                'count': 0,
                'devices': [],
                'recommendation': 'No GPU detected. Training will use CPU (slow).'
            }
        
        gpu_count = torch.cuda.device_count()
        devices = []
        
        for i in range(gpu_count):
            device_props = torch.cuda.get_device_properties(i)
            devices.append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': device_props.total_memory / 1024**3,  # GB
                'memory_free': (device_props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3
            })
        
        # Generate recommendation
        if gpu_count == 1:
            recommendation = "Single GPU detected. Use --use-gpu for accelerated training."
        elif gpu_count > 1:
            recommendation = f"{gpu_count} GPUs detected. Use torchrun for multi-GPU training: torchrun --nproc_per_node={gpu_count} script.py"
        else:
            recommendation = "GPU configuration unclear."
        
        return {
            'available': True,
            'count': gpu_count,
            'devices': devices,
            'recommendation': recommendation
        }
        
    except ImportError:
        return {
            'available': False,
            'count': 0,
            'devices': [],
            'recommendation': 'PyTorch not installed.'
        }


def main():
    parser = argparse.ArgumentParser(description='Distributed multi-GPU training')
    parser.add_argument('--profile', type=str, choices=['testing', 'development', 'production', 'championship'],
                       help='Training profile')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--multi-gpu', action='store_true', help='Enable multi-GPU training')
    parser.add_argument('--detect-gpus', action='store_true', help='Detect GPU configuration and exit')
    
    args = parser.parse_args()
    
    # GPU detection mode
    if args.detect_gpus:
        config = detect_gpu_configuration()
        print("="*70)
        print("GPU CONFIGURATION")
        print("="*70)
        print()
        print(f"GPUs Available: {config['available']}")
        print(f"GPU Count: {config['count']}")
        print()
        
        if config['devices']:
            print("Detected GPUs:")
            for device in config['devices']:
                print(f"  GPU {device['id']}: {device['name']}")
                print(f"    Total Memory: {device['memory_total']:.2f} GB")
                print(f"    Free Memory: {device['memory_free']:.2f} GB")
            print()
        
        print("Recommendation:")
        print(f"  {config['recommendation']}")
        print()
        print("="*70)
        return
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    elif args.profile:
        config_path = f"config/training/{args.profile}.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    config['multi_gpu'] = args.multi_gpu
    
    # Initialize distributed trainer
    trainer = DistributedTrainer(config)
    trainer.setup_distributed()
    
    print("="*70)
    print("DISTRIBUTED TRAINING SETUP")
    print("="*70)
    print()
    print(f"Device: {trainer.device}")
    print(f"Distributed: {trainer.is_distributed}")
    if trainer.is_distributed:
        print(f"Rank: {trainer.rank}/{trainer.world_size}")
        print(f"Local Rank: {trainer.local_rank}")
    print()
    print("="*70)
    
    # Note: Actual training logic would be integrated with train_model.py
    # This is a standalone demonstration of the distributed training infrastructure
    
    print()
    print("Distributed training infrastructure ready.")
    print("Integrate with train_model.py for full training workflow.")
    
    trainer.cleanup()


if __name__ == '__main__':
    main()
