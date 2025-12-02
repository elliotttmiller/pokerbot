"""
Perception Module - QWEN 2.5-7B VL Inference Pipeline

This module provides Colab-optimized perception for poker game state detection
using a fine-tuned QWEN 2.5-7B Vision-Language model with Unsloth 4-bit quantization.

Compatible with Google Colab Free Tier (T4 GPU, 16GB VRAM).
"""

from .detector import QWENPokerDetector
from .data_models import GameState, Card, BettingRound
from .coco_loader import COCOPokerDataset

__all__ = [
    'QWENPokerDetector',
    'GameState',
    'Card',
    'BettingRound',
    'COCOPokerDataset'
]
