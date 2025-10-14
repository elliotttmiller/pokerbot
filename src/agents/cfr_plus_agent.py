"""
Enhanced CFR+ Algorithm Implementation

Implements the CFR+ algorithm used by Libratus with additional enhancements:
1. Regret matching+ with negative regret reset
2. Action pruning based on regret thresholds
3. Linear CFR (LCFR) with discounting for faster convergence
4. Weighted strategy averaging

Based on Libratus paper and poker-ai repository:
https://github.com/elliotttmiller/poker-ai

Key improvements over vanilla CFR:
- Faster convergence to Nash equilibrium
- More efficient use of compute (pruning)
- Better handling of recent vs old iterations (LCFR)
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .cfr_agent import CFRAgent, InfoSet


class CFRPlusAgent(CFRAgent):
    """
    Enhanced CFR+ agent with pruning and linear discounting.
    
    Improvements over base CFR:
    1. **Regret Matching+**: Negative regrets are reset to 0
    2. **Action Pruning**: Skip actions with very negative regret
    3. **Linear CFR**: Discount old regrets and strategies
    4. **Weighted Averaging**: Recent iterations weighted more heavily
    """
    
    def __init__(
        self,
        name: str = "CFRPlus",
        strategy_interval: int = 100,
        prune_threshold: float = -200.0,
        lcfr_threshold: int = 400,
        discount_interval: int = 10
    ):
        """
        Initialize CFR+ agent.
        
        Args:
            name: Agent name
            strategy_interval: Interval for updating average strategy
            prune_threshold: Regret threshold for action pruning (negative)
            lcfr_threshold: Iteration to start linear CFR discounting
            discount_interval: Interval for discounting old regrets/strategies
        """
        super().__init__(name)
        
        # CFR+ specific parameters
        self.strategy_interval = strategy_interval
        self.prune_threshold = prune_threshold
        self.lcfr_threshold = lcfr_threshold
        self.discount_interval = discount_interval
        
        print(f"CFR+ Agent '{name}' initialized with enhancements:")
        print(f"  Prune threshold: {prune_threshold}")
        print(f"  LCFR threshold: {lcfr_threshold} iterations")
        print(f"  Discount interval: {discount_interval}")
    
    def train(self, num_iterations: int = 1000):
        """
        Train using CFR+ algorithm with enhancements.
        
        Args:
            num_iterations: Number of CFR iterations to run
        """
        print(f"Training CFR+ for {num_iterations} iterations...")
        
        for i in range(num_iterations):
            # Run standard CFR iteration
            super().train(num_iterations=1)
            
            # CFR+ enhancement: Reset negative regrets to 0
            if (i + 1) % 10 == 0:
                self._reset_negative_regrets()
            
            # Linear CFR: Apply discounting after threshold
            if i > self.lcfr_threshold and i % self.discount_interval == 0:
                self._apply_discounting(i)
            
            # Progress logging
            if (i + 1) % 100 == 0 and i > 0:
                avg_regret = self._compute_average_regret()
                print(f"  Iteration {i + 1}/{num_iterations} - Avg Regret: {avg_regret:.6f}")
        
        print(f"✓ CFR+ training complete ({self.iterations} total iterations)")
    
    def _reset_negative_regrets(self):
        """
        CFR+ Enhancement: Reset negative regrets to 0.
        
        This speeds up convergence by forgetting bad past decisions faster.
        """
        for infoset in self.infosets.values():
            # Reset negative values in regret_sum to 0
            infoset.regret_sum = np.maximum(infoset.regret_sum, 0.0)
    
    def _apply_discounting(self, iteration: int):
        """
        Linear CFR: Discount old regrets and strategies.
        
        This weights recent iterations more heavily than old ones,
        leading to faster convergence.
        
        Args:
            iteration: Current iteration number
        """
        # Calculate discount factor
        # d = t / (t + 1) where t = iteration / discount_interval
        t = iteration / self.discount_interval
        discount = t / (t + 1.0)
        
        # Apply discount to all infosets
        for infoset in self.infosets.values():
            infoset.regret_sum *= discount
            infoset.strategy_sum *= discount
    
    def _should_prune_action(self, infoset: InfoSet, action_idx: int) -> bool:
        """
        Determine if an action should be pruned based on regret.
        
        Actions with very negative regret are unlikely to be optimal
        and can be skipped during traversal to save computation.
        
        Args:
            infoset: Information set
            action_idx: Action index to check
        
        Returns:
            True if action should be pruned, False otherwise
        """
        if action_idx >= len(infoset.regret_sum):
            return False
        
        regret = infoset.regret_sum[action_idx]
        return regret < self.prune_threshold
    
    def _compute_average_regret(self) -> float:
        """
        Compute average regret across all infosets.
        
        Returns:
            Average regret value
        """
        if not self.infosets:
            return 0.0
        
        total_regret = 0.0
        count = 0
        
        for infoset in self.infosets.values():
            total_regret += np.sum(np.maximum(infoset.regret_sum, 0.0))
            count += len(infoset.regret_sum)
        
        return total_regret / max(count, 1)
    
    def get_strategy_with_pruning(self, infoset_key: str) -> np.ndarray:
        """
        Get current strategy for an infoset with action pruning.
        
        Args:
            infoset_key: Information set identifier
        
        Returns:
            Strategy array (probability distribution over actions)
        """
        # Get base strategy
        strategy = self.get_strategy(infoset_key)
        
        if infoset_key not in self.infosets:
            return strategy
        
        infoset = self.infosets[infoset_key]
        
        # Zero out probabilities for pruned actions
        pruned_mask = np.array([
            0.0 if self._should_prune_action(infoset, i) else 1.0
            for i in range(len(strategy))
        ])
        
        strategy = strategy * pruned_mask
        
        # Renormalize
        total = np.sum(strategy)
        if total > 0:
            strategy = strategy / total
        else:
            # If all actions pruned, use uniform over non-pruned (shouldn't happen)
            strategy = pruned_mask / max(np.sum(pruned_mask), 1.0)
        
        return strategy
    
    def get_training_stats(self) -> Dict:
        """
        Get detailed training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        stats = {
            'iterations': self.iterations,
            'infosets': len(self.infosets),
            'average_regret': self._compute_average_regret(),
            'algorithm': 'CFR+',
            'prune_threshold': self.prune_threshold,
            'lcfr_enabled': self.iterations > self.lcfr_threshold
        }
        
        # Count pruned actions
        pruned_count = 0
        total_actions = 0
        
        for infoset in self.infosets.values():
            for i in range(len(infoset.regret_sum)):
                total_actions += 1
                if self._should_prune_action(infoset, i):
                    pruned_count += 1
        
        stats['pruned_actions'] = pruned_count
        stats['total_actions'] = total_actions
        stats['prune_percentage'] = (pruned_count / max(total_actions, 1)) * 100
        
        return stats


def create_cfr_plus_agent(name: str = "CFRPlus", **kwargs) -> CFRPlusAgent:
    """
    Factory function to create a CFR+ agent with default settings.
    
    Args:
        name: Agent name
        **kwargs: Additional arguments for CFRPlusAgent
    
    Returns:
        Initialized CFR+ agent
    """
    return CFRPlusAgent(name=name, **kwargs)
