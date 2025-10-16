"""Evaluation package initialization."""

from .trainer import UnifiedTrainer, Trainer

__all__ = [
    'UnifiedTrainer',
    'Trainer',  # Alias for backward compatibility
    'DistributedTrainer',
    'AsyncDistributedTrainer',
]
