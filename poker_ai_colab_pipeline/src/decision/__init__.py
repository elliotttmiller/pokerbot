"""
Decision Module - DeepStack Continual Re-Solving Engine

This module provides the official DeepStack implementation for poker decision-making:
- Depth-limited lookahead tree construction
- CFR+ solver with linear discounting
- Neural network terminal value estimation
- Continual re-solving API

Based on Moravčík et al., 2017 and DeepStack-Leduc reference implementation.
"""

from .solver import DeepStackSolver
from .value_network import ValueNetwork, load_value_network
from .cfr import CFRSolver
from .action import Action, ActionType

__all__ = [
    'DeepStackSolver',
    'ValueNetwork',
    'load_value_network',
    'CFRSolver',
    'Action',
    'ActionType'
]
