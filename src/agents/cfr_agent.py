"""Counterfactual Regret Minimization (CFR) agent implementation.

Based on concepts from:
- DeepStack-Leduc (tree-based CFR)
- coms4995-finalproj (ESMCCFR implementation)
- Research papers on CFR algorithms
"""

import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..game import Action, Card, GameState, HandEvaluator


class InfoSet:
    """Represents an information set in poker."""
    
    def __init__(self, actions: List[Action]):
        """
        Initialize information set.
        
        Args:
            actions: Available actions at this infoset
        """
        self.actions = actions
        self.num_actions = len(actions)
        
        # Regret and strategy tracking
        self.regret_sum = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
        self.strategy = np.ones(self.num_actions) / self.num_actions
    
    def get_strategy(self, realization_weight: float = 1.0) -> np.ndarray:
        """
        Get current strategy using regret matching.
        
        Args:
            realization_weight: Weight for strategy accumulation
        
        Returns:
            Probability distribution over actions
        """
        # Regret matching
        positive_regrets = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(positive_regrets)
        
        if normalizing_sum > 0:
            self.strategy = positive_regrets / normalizing_sum
        else:
            # Uniform strategy if no positive regrets
            self.strategy = np.ones(self.num_actions) / self.num_actions
        
        # Accumulate strategy
        self.strategy_sum += realization_weight * self.strategy
        
        return self.strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """
        Get average strategy over all iterations.
        
        Returns:
            Average probability distribution
        """
        normalizing_sum = np.sum(self.strategy_sum)
        
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        else:
            return np.ones(self.num_actions) / self.num_actions
    
    def update_regret(self, action_idx: int, regret: float):
        """Update regret for an action."""
        self.regret_sum[action_idx] += regret


class CFRAgent:
    """
    Vanilla Counterfactual Regret Minimization agent.
    
    This is the foundational CFR algorithm that can be extended to:
    - Monte Carlo CFR (MCCFR)
    - External Sampling MCCFR (ESMCCFR)
    - CFR+
    """
    
    def __init__(self, name: str = "CFR"):
        """Initialize CFR agent."""
        self.name = name
        self.infosets: Dict[str, InfoSet] = {}
        self.iterations = 0
    
    def get_infoset(self, infoset_key: str, actions: List[Action]) -> InfoSet:
        """
        Get or create information set.
        
        Args:
            infoset_key: String key identifying the infoset
            actions: Available actions
        
        Returns:
            InfoSet object
        """
        if infoset_key not in self.infosets:
            self.infosets[infoset_key] = InfoSet(actions)
        return self.infosets[infoset_key]
    
    def create_infoset_key(self,
                          hole_cards: List[Card],
                          community_cards: List[Card],
                          history: str) -> str:
        """
        Create unique key for information set.
        
        Args:
            hole_cards: Player's private cards
            community_cards: Public community cards
            history: Betting history string
        
        Returns:
            Unique infoset identifier
        """
        # Sort hole cards for canonical representation
        hole_str = ''.join(sorted([str(c) for c in hole_cards]))
        community_str = ''.join([str(c) for c in community_cards])
        
        return f"{hole_str}|{community_str}|{history}"
    
    def train(self,
             num_iterations: int = 10000,
             game_state: Optional[GameState] = None):
        """
        Train the CFR agent through self-play.
        
        Args:
            num_iterations: Number of training iterations
            game_state: Game state to use (creates new if None)
        """
        if game_state is None:
            game_state = GameState(num_players=2)
        
        for i in range(num_iterations):
            game_state.reset()
            
            # Alternate who goes first
            for player_idx in range(2):
                game_state.reset()
                self._cfr_iteration(game_state, player_idx, 1.0, 1.0)
            
            self.iterations += 1
            
            if (i + 1) % 1000 == 0:
                print(f"CFR Iteration {i + 1}/{num_iterations}")
    
    def _cfr_iteration(self,
                      game_state: GameState,
                      player_idx: int,
                      p0: float,
                      p1: float) -> float:
        """
        Single CFR iteration (recursive).
        
        Args:
            game_state: Current game state
            player_idx: Player whose strategy we're computing
            p0: Reach probability for player 0
            p1: Reach probability for player 1
        
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
            ""  # Simplified history
        )
        
        # Get infoset
        infoset = self.get_infoset(infoset_key, available_actions)
        
        # Get strategy
        if current_player == player_idx:
            strategy = infoset.get_strategy(p0 if player_idx == 0 else p1)
        else:
            strategy = infoset.get_average_strategy()
        
        # Compute action utilities
        action_utilities = np.zeros(len(available_actions))
        
        for action_idx, action in enumerate(available_actions):
            # Make a copy of game state and apply action
            next_state = self._apply_action_copy(game_state, player_idx, action)
            
            # Recurse
            if current_player == player_idx:
                if player_idx == 0:
                    action_utilities[action_idx] = self._cfr_iteration(
                        next_state, player_idx, p0 * strategy[action_idx], p1
                    )
                else:
                    action_utilities[action_idx] = self._cfr_iteration(
                        next_state, player_idx, p0, p1 * strategy[action_idx]
                    )
            else:
                action_utilities[action_idx] = self._cfr_iteration(
                    next_state, player_idx, p0, p1
                )
        
        # Compute node utility
        node_utility = np.sum(strategy * action_utilities)
        
        # Update regrets if it's our turn
        if current_player == player_idx:
            for action_idx in range(len(available_actions)):
                regret = action_utilities[action_idx] - node_utility
                opponent_reach = p1 if player_idx == 0 else p0
                infoset.update_regret(action_idx, opponent_reach * regret)
        
        return node_utility
    
    def _get_available_actions(self,
                               game_state: GameState,
                               player_idx: int) -> List[Action]:
        """Get available actions for player."""
        player = game_state.players[player_idx]
        actions = [Action.FOLD]
        
        if game_state.current_bet == 0:
            actions.append(Action.CHECK)
        else:
            actions.append(Action.CALL)
        
        if player.stack > 0:
            actions.append(Action.RAISE)
        
        return actions
    
    def _apply_action_copy(self,
                          game_state: GameState,
                          player_idx: int,
                          action: Action) -> GameState:
        """Apply action to a copy of game state."""
        # In production, this should deep copy the game state
        # For now, simplified
        raise_amount = 20 if action == Action.RAISE else 0
        game_state.apply_action(player_idx, action, raise_amount)
        return game_state
    
    def choose_action(self,
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> Tuple[Action, int]:
        """
        Choose action using trained strategy.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot: Current pot
            current_bet: Current bet to call
            player_stack: Player's stack
            opponent_bet: Opponent's bet
        
        Returns:
            Tuple of (Action, raise_amount)
        """
        # Get infoset
        available_actions = [Action.FOLD, Action.CALL if current_bet > 0 else Action.CHECK]
        if player_stack > 0:
            available_actions.append(Action.RAISE)
        
        infoset_key = self.create_infoset_key(hole_cards, community_cards, "")
        
        if infoset_key in self.infosets:
            infoset = self.infosets[infoset_key]
            strategy = infoset.get_average_strategy()
            
            # Sample action according to strategy
            action_idx = np.random.choice(len(available_actions), p=strategy)
            action = available_actions[action_idx]
        else:
            # Fallback to random if infoset not seen
            action = random.choice(available_actions)
        
        raise_amount = 0
        if action == Action.RAISE:
            raise_amount = min(pot // 2, player_stack)
        
        return action, raise_amount
    
    def save_strategy(self, filepath: str):
        """Save trained strategy to file."""
        strategy = {
            'infosets': self.infosets,
            'iterations': self.iterations
        }
        with open(filepath, 'wb') as f:
            pickle.dump(strategy, f)
    
    def load_strategy(self, filepath: str):
        """Load trained strategy from file."""
        with open(filepath, 'rb') as f:
            strategy = pickle.load(f)
            self.infosets = strategy['infosets']
            self.iterations = strategy['iterations']
