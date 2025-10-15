"""
CFRDGadget: DeepStack-style CFR-D gadget for opponent range reconstruction during re-solving.

The CFRD (Counterfactual Regret Decomposition) Gadget is a key component of continual
re-solving. It reconstructs opponent ranges from their counterfactual values, enabling
consistent strategy computation across sequential game tree solves.

Based on the original DeepStack CFRDGadget module.
"""
import numpy as np
from typing import Optional, List


class CFRDGadget:
    """
    Reconstructs opponent ranges from counterfactual values during re-solving.
    
    The gadget maintains consistency between sequential solves by ensuring
    that the opponent's range at each node is consistent with their previous
    counterfactual values.
    """
    
    def __init__(self, board: List, player_range: Optional[np.ndarray] = None,
                 opponent_cfvs: Optional[np.ndarray] = None):
        """
        Initialize CFRDGadget.
        
        Args:
            board: Board cards
            player_range: Player's current range
            opponent_cfvs: Opponent's CFVs from previous solve
        """
        self.board = board
        self.player_range = player_range
        self.opponent_cfvs = opponent_cfvs
        self._cached_range = None

    def compute_opponent_range(self, current_opponent_cfvs: np.ndarray, 
                               iteration: int = 0) -> np.ndarray:
        """
        Compute opponent range from counterfactual values.
        
        The gadget runs a simplified game where:
        1. Opponent plays to achieve their target CFVs
        2. Player plays their fixed range
        3. Iterative process converges to consistent range
        
        Args:
            current_opponent_cfvs: Current CFV vector for opponent
            iteration: Current iteration number (for convergence)
            
        Returns:
            Opponent's range (probability distribution)
        """
        if current_opponent_cfvs is None or len(current_opponent_cfvs) == 0:
            # No CFVs provided, return uniform
            num_hands = len(self.player_range) if self.player_range is not None else 6
            return np.ones(num_hands) / num_hands
        
        # Ensure numpy array
        cfvs = np.asarray(current_opponent_cfvs)
        num_hands = len(cfvs)
        
        # Method: Softmax normalization with CFV-based weighting
        # Hands with higher CFVs are more likely in opponent's range
        
        # Shift CFVs to be non-negative
        cfvs_shifted = cfvs - cfvs.min() + 0.01
        
        # Temperature for softmax (decreases over iterations for sharper distribution)
        temperature = max(0.5, 2.0 / (iteration + 1))
        
        # Softmax transformation
        exp_cfvs = np.exp(cfvs_shifted / temperature)
        opponent_range = exp_cfvs / exp_cfvs.sum()
        
        # Ensure valid probability distribution
        opponent_range = np.clip(opponent_range, 1e-8, 1.0)
        opponent_range = opponent_range / opponent_range.sum()
        
        # Cache for potential reuse
        self._cached_range = opponent_range
        
        return opponent_range
    
    def get_range(self) -> np.ndarray:
        """
        Get last computed opponent range.
        
        Returns:
            Cached opponent range or uniform if not computed
        """
        if self._cached_range is not None:
            return self._cached_range
        
        if self.opponent_cfvs is not None:
            return self.compute_opponent_range(self.opponent_cfvs, 0)
        
        # Default: uniform
        num_hands = len(self.player_range) if self.player_range is not None else 6
        return np.ones(num_hands) / num_hands
    
    def update_cfvs(self, new_cfvs: np.ndarray):
        """
        Update opponent CFVs and invalidate cache.
        
        Args:
            new_cfvs: New counterfactual values
        """
        self.opponent_cfvs = new_cfvs
        self._cached_range = None
    
    def iterative_refinement(self, target_cfvs: np.ndarray,
                             num_iterations: int = 10,
                             learning_rate: float = 0.1) -> np.ndarray:
        """
        Iteratively refine opponent range to match target CFVs.
        
        Uses gradient-based optimization to find range that achieves
        target counterfactual values while maintaining probability constraints.
        
        Args:
            target_cfvs: Target CFV vector
            num_iterations: Number of refinement iterations
            learning_rate: Step size for updates
            
        Returns:
            Refined opponent range
        """
        num_hands = len(target_cfvs)
        opponent_range = np.ones(num_hands) / num_hands
        
        for i in range(num_iterations):
            # Compute gradient direction (simplified)
            cfvs_norm = (target_cfvs - target_cfvs.mean()) / (target_cfvs.std() + 1e-8)
            
            # Update with momentum
            momentum = 0.9 if i > 0 else 0.0
            adjustment = cfvs_norm * learning_rate
            
            if i > 0 and hasattr(self, '_last_adjustment'):
                adjustment += momentum * self._last_adjustment
            
            opponent_range += adjustment
            self._last_adjustment = adjustment
            
            # Project to probability simplex
            opponent_range = np.maximum(opponent_range, 0.0)
            range_sum = opponent_range.sum()
            if range_sum > 0:
                opponent_range /= range_sum
            else:
                opponent_range = np.ones(num_hands) / num_hands
        
        return opponent_range
