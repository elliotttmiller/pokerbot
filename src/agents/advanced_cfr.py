"""Advanced CFR implementation with pruning and linear discounting.

This module implements sophisticated CFR variants:
- CFR with Pruning (CFRp): Skip actions with low regret
- Linear CFR (LCFR): Apply discounting for faster convergence
- Progressive training: Multi-phase curriculum learning
"""

import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..game import Action, Card, GameState, HandEvaluator
from .cfr_agent import CFRAgent, InfoSet


class AdvancedCFRAgent(CFRAgent):
    """
    Advanced CFR agent with pruning, linear discounting, and progressive training.
    
    Implements techniques from Pluribus and DeepStack:
    - CFR with Pruning (CFRp)
    - Linear CFR discounting
    - Progressive multi-phase training
    """
    
    def __init__(self, 
                 name: str = "AdvancedCFR",
                 regret_floor: int = -310000000,
                 use_linear_cfr: bool = True):
        """
        Initialize Advanced CFR agent.
        
        Args:
            name: Agent name
            regret_floor: Minimum regret value (prevents overflow)
            use_linear_cfr: Whether to use linear CFR discounting
        """
        super().__init__(name)
        self.regret_floor = regret_floor
        self.use_linear_cfr = use_linear_cfr
        
    def train_progressive(self,
                         num_iterations: int = 50000,
                         warmup_threshold: int = 1000,
                         prune_threshold: int = 10000,
                         lcfr_threshold: int = 50000,
                         discount_interval: int = 1000,
                         strategy_interval: int = 100,
                         game_state: Optional[GameState] = None,
                         verbose: bool = True):
        """
        Progressive training with multiple phases.
        
        Phase 1 (0-warmup): Pure CFR exploration
        Phase 2 (warmup-prune): Start pruning low-regret actions
        Phase 3 (prune-lcfr): Mixed CFR/CFRp with exploration
        Phase 4 (lcfr+): Linear CFR with discounting
        
        Args:
            num_iterations: Total training iterations
            warmup_threshold: Iterations before pruning starts
            prune_threshold: Iterations before aggressive pruning
            lcfr_threshold: Iterations before linear discounting
            discount_interval: How often to apply discounting
            strategy_interval: How often to update strategy
            game_state: Game state to use
            verbose: Print progress
        """
        if game_state is None:
            game_state = GameState(num_players=2)
        
        for t in range(1, num_iterations + 1):
            game_state.reset()
            
            # Determine training phase
            if t < warmup_threshold:
                # Phase 1: Pure CFR
                method = "CFR (warmup)"
                for player_idx in range(2):
                    game_state.reset()
                    self._cfr_iteration(game_state, player_idx, 1.0, 1.0)
                    
            elif t < prune_threshold:
                # Phase 2: Start pruning
                method = "CFR with light pruning"
                for player_idx in range(2):
                    game_state.reset()
                    c = self.regret_floor // 10  # Light pruning threshold
                    self._cfrp_iteration(game_state, player_idx, 1.0, 1.0, c)
                    
            elif t < lcfr_threshold:
                # Phase 3: Mixed CFR/CFRp (95% pruned, 5% full exploration)
                method = "Mixed CFR/CFRp"
                for player_idx in range(2):
                    game_state.reset()
                    if random.random() < 0.05:
                        # 5% full exploration
                        self._cfr_iteration(game_state, player_idx, 1.0, 1.0)
                    else:
                        # 95% pruned search
                        c = self.regret_floor
                        self._cfrp_iteration(game_state, player_idx, 1.0, 1.0, c)
                        
            else:
                # Phase 4: Full CFRp with linear discounting
                method = "Linear CFR"
                for player_idx in range(2):
                    game_state.reset()
                    c = self.regret_floor
                    self._cfrp_iteration(game_state, player_idx, 1.0, 1.0, c)
                
                # Apply linear discounting
                if self.use_linear_cfr and t % discount_interval == 0:
                    self._apply_linear_discount(t, discount_interval)
            
            # Update strategy periodically
            if t % strategy_interval == 0:
                self._update_average_strategy()
            
            self.iterations = t
            
            # Progress reporting
            if verbose and (t % 1000 == 0 or t == num_iterations):
                print(f"Iteration {t}/{num_iterations} - Method: {method}")
    
    def _cfrp_iteration(self,
                       game_state: GameState,
                       player_idx: int,
                       p0: float,
                       p1: float,
                       c: int) -> float:
        """
        CFR iteration with pruning.
        
        Prunes actions with regret below threshold c.
        
        Args:
            game_state: Current game state
            player_idx: Player whose strategy we're computing
            p0: Reach probability for player 0
            p1: Reach probability for player 1
            c: Pruning threshold
        
        Returns:
            Expected utility for player_idx
        """
        # Terminal node
        if game_state.is_hand_complete():
            winners = game_state.get_winners()
            if player_idx in winners:
                return game_state.pot / len(winners)
            else:
                return 0.0
        
        player = game_state.players[player_idx]
        current_player = player_idx  # Simplified
        
        # Get available actions
        available_actions = self._get_available_actions(game_state, player_idx)
        
        # Create infoset key
        infoset_key = self.create_infoset_key(
            player.hand,
            game_state.community_cards,
            ""
        )
        
        # Get infoset
        infoset = self.get_infoset(infoset_key, available_actions)
        
        # Get strategy
        if current_player == player_idx:
            strategy = infoset.get_strategy(p0 if player_idx == 0 else p1)
        else:
            strategy = infoset.get_average_strategy()
        
        # PRUNING: Skip actions with low regret
        action_utilities = np.zeros(len(available_actions))
        explored_actions = []
        
        for action_idx, action in enumerate(available_actions):
            # Check if action should be pruned
            if current_player == player_idx and infoset.regret_sum[action_idx] < c:
                # Prune this action
                action_utilities[action_idx] = 0.0
                continue
            
            explored_actions.append(action_idx)
            
            # Make a copy of game state and apply action
            next_state = self._apply_action_copy(game_state, player_idx, action)
            
            # Recurse
            if current_player == player_idx:
                if player_idx == 0:
                    action_utilities[action_idx] = self._cfrp_iteration(
                        next_state, player_idx, p0 * strategy[action_idx], p1, c
                    )
                else:
                    action_utilities[action_idx] = self._cfrp_iteration(
                        next_state, player_idx, p0, p1 * strategy[action_idx], c
                    )
            else:
                action_utilities[action_idx] = self._cfrp_iteration(
                    next_state, player_idx, p0, p1, c
                )
        
        # Compute node utility
        node_utility = np.sum(strategy * action_utilities)
        
        # Update regrets if it's our turn (only for explored actions)
        if current_player == player_idx:
            for action_idx in explored_actions:
                regret = action_utilities[action_idx] - node_utility
                opponent_reach = p1 if player_idx == 0 else p0
                infoset.update_regret(action_idx, opponent_reach * regret)
                
                # Apply regret floor
                if infoset.regret_sum[action_idx] < self.regret_floor:
                    infoset.regret_sum[action_idx] = self.regret_floor
        
        return node_utility
    
    def _apply_linear_discount(self, t: int, discount_interval: int):
        """
        Apply linear CFR discounting to all infosets.
        
        Recent iterations are weighted more heavily than older ones.
        
        Args:
            t: Current iteration
            discount_interval: Interval for discounting
        """
        d = (t / discount_interval) / ((t / discount_interval) + 1)
        
        for infoset in self.infosets.values():
            for i in range(infoset.num_actions):
                infoset.regret_sum[i] *= d
                infoset.strategy_sum[i] *= d
    
    def _update_average_strategy(self):
        """
        Update average strategy for all infosets.
        
        This helps stabilize learning by averaging strategies over time.
        """
        for infoset in self.infosets.values():
            # Get current strategy
            strategy = infoset.get_strategy(1.0)
            
            # Accumulate into strategy_sum (already done in get_strategy)
            # This method is called to ensure strategy updates happen regularly
            pass
