"""Evaluation package initialization."""

from .trainer import UnifiedTrainer, Trainer
from .distributed_trainer import DistributedTrainer, AsyncDistributedTrainer

__all__ = [
    'UnifiedTrainer',
    'Trainer',  # Alias for backward compatibility
    'DistributedTrainer',
    'AsyncDistributedTrainer',
]
