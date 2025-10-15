"""Counterfactual Regret Minimization (CFR) agent implementation with CFR+ enhancements.

This implementation includes:
- Vanilla CFR algorithm
- CFR+ enhancements (regret matching+, action pruning, linear CFR)
- Integration from poker-ai repository concepts

Based on concepts from:
- DeepStack-Leduc (tree-based CFR)
- Libratus (CFR+ algorithm)
- coms4995-finalproj (ESMCCFR implementation)
- poker-ai repository (enhanced features)
"""

import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from deepstack.game.game_state import Action, GameState
from deepstack.game.card import Card
from deepstack.utils.hand_evaluator import HandEvaluator


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
        Get or create information set. Ensures action consistency.
        
        Args:
            infoset_key: String key identifying the infoset
            actions: Available actions
        
        Returns:
            InfoSet object
        """
        if infoset_key in self.infosets:
            infoset = self.infosets[infoset_key]
            # If actions mismatch, reinitialize infoset
            if len(infoset.actions) != len(actions) or any(a != b for a, b in zip(infoset.actions, actions)):
                self.infosets[infoset_key] = InfoSet(actions)
        else:
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
    
    # ========================================================================
    # CFR+ ENHANCEMENTS (from poker-ai integration)
    # ========================================================================
    
    def enable_cfr_plus(
        self,
        use_regret_matching_plus: bool = True,
        use_pruning: bool = True,
        use_linear_cfr: bool = True,
        prune_threshold: float = -200.0,
        lcfr_threshold: int = 400,
        discount_interval: int = 10
    ):
        """
        Enable CFR+ enhancements for faster convergence.
        
        CFR+ improvements:
        1. Regret matching+: Reset negative regrets to 0
        2. Action pruning: Skip actions with very negative regret
        3. Linear CFR: Discount old regrets and strategies
        
        Args:
            use_regret_matching_plus: Enable regret matching+
            use_pruning: Enable action pruning
            use_linear_cfr: Enable linear CFR discounting
            prune_threshold: Regret threshold for pruning (negative)
            lcfr_threshold: Iteration to start LCFR
            discount_interval: Interval for discounting
        """
        self.use_regret_matching_plus = use_regret_matching_plus
        self.use_pruning = use_pruning
        self.use_linear_cfr = use_linear_cfr
        self.prune_threshold = prune_threshold
        self.lcfr_threshold = lcfr_threshold
        self.discount_interval = discount_interval
        
        print(f"CFR+ enhancements enabled for {self.name}:")
        if use_regret_matching_plus:
            print(f"  [OK] Regret matching+ (reset negative regrets)")
        if use_pruning:
            print(f"  [OK] Action pruning (threshold: {prune_threshold})")
        if use_linear_cfr:
            print(f"  [OK] Linear CFR (starts at iteration {lcfr_threshold})")
    
    def train_with_cfr_plus(self, num_iterations: int = 1000):
        """
        Train using CFR+ enhancements.
        
        Args:
            num_iterations: Number of CFR+ iterations
        """
        # Enable CFR+ if not already enabled
        if not hasattr(self, 'use_regret_matching_plus'):
            self.enable_cfr_plus()
        
        print(f"Training {self.name} with CFR+ for {num_iterations} iterations...")
        
        for i in range(num_iterations):
            # Run standard CFR iteration
            self.train(num_iterations=1)
            
            # CFR+ enhancement: Reset negative regrets
            if self.use_regret_matching_plus and (i + 1) % 10 == 0:
                self._reset_negative_regrets()
            
            # Linear CFR: Apply discounting
            if self.use_linear_cfr and i > self.lcfr_threshold and i % self.discount_interval == 0:
                self._apply_discounting(i)
            
            # Progress logging
            if (i + 1) % 100 == 0 and i > 0:
                avg_regret = self._compute_average_regret()
                print(f"  Iteration {i + 1}/{num_iterations} - Avg Regret: {avg_regret:.6f}")
        
        print(f"[OK] CFR+ training complete ({self.iterations} total iterations)")


class AdvancedCFRAgent(CFRAgent):
    """Advanced CFR with pruning and linear discounting (merged from advanced_cfr)."""

    def __init__(self, name: str = "AdvancedCFR", regret_floor: int = -310000000, use_linear_cfr: bool = True):
        super().__init__(name)
        self.regret_floor = regret_floor
        self.use_linear_cfr = use_linear_cfr

    def train_progressive(
        self,
        num_iterations: int = 50000,
        warmup_threshold: int = 1000,
        prune_threshold: int = 10000,
        lcfr_threshold: int = 50000,
        discount_interval: int = 1000,
        strategy_interval: int = 100,
        game_state: Optional[GameState] = None,
        verbose: bool = True,
    ) -> None:
        if game_state is None:
            game_state = GameState(num_players=2)

        for t in range(1, num_iterations + 1):
            game_state.reset()

            if t < warmup_threshold:
                method = "CFR (warmup)"
                for player_idx in range(2):
                    game_state.reset()
                    self._cfr_iteration(game_state, player_idx, 1.0, 1.0)
            elif t < prune_threshold:
                method = "CFR with light pruning"
                for player_idx in range(2):
                    game_state.reset()
                    c = self.regret_floor // 10
                    self._cfrp_iteration(game_state, player_idx, 1.0, 1.0, c)
            elif t < lcfr_threshold:
                method = "Mixed CFR/CFRp"
                for player_idx in range(2):
                    game_state.reset()
                    if random.random() < 0.05:
                        self._cfr_iteration(game_state, player_idx, 1.0, 1.0)
                    else:
                        c = self.regret_floor
                        self._cfrp_iteration(game_state, player_idx, 1.0, 1.0, c)
            else:
                method = "Linear CFR"
                for player_idx in range(2):
                    game_state.reset()
                    c = self.regret_floor
                    self._cfrp_iteration(game_state, player_idx, 1.0, 1.0, c)

                if self.use_linear_cfr and t % discount_interval == 0:
                    self._apply_linear_discount(t, discount_interval)

            if t % strategy_interval == 0:
                self._update_average_strategy()

            self.iterations = t

            if verbose and (t % 1000 == 0 or t == num_iterations):
                print(f"Iteration {t}/{num_iterations} - Method: {method}")

    def _cfrp_iteration(
        self,
        game_state: GameState,
        player_idx: int,
        p0: float,
        p1: float,
        c: int,
    ) -> float:
        if game_state.is_hand_complete():
            winners = game_state.get_winners()
            if player_idx in winners:
                return game_state.pot / len(winners)
            return 0.0

        current_player = player_idx
        available_actions = self._get_available_actions(game_state, player_idx)
        infoset_key = self.create_infoset_key(
            game_state.players[player_idx].hand,
            game_state.community_cards,
            "",
        )
        infoset = self.get_infoset(infoset_key, available_actions)

        if current_player == player_idx:
            strategy = infoset.get_strategy(p0 if player_idx == 0 else p1)
        else:
            strategy = infoset.get_average_strategy()

        action_utilities = np.zeros(len(available_actions))
        explored_actions: List[int] = []

        for action_idx, action in enumerate(available_actions):
            if current_player == player_idx and infoset.regret_sum[action_idx] < c:
                action_utilities[action_idx] = 0.0
                continue

            explored_actions.append(action_idx)
            next_state = self._apply_action_copy(game_state, player_idx, action)

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

        node_utility = np.sum(strategy * action_utilities)

        if current_player == player_idx:
            for action_idx in explored_actions:
                regret = action_utilities[action_idx] - node_utility
                opponent_reach = p1 if player_idx == 0 else p0
                infoset.update_regret(action_idx, opponent_reach * regret)
                if infoset.regret_sum[action_idx] < self.regret_floor:
                    infoset.regret_sum[action_idx] = self.regret_floor

        return node_utility

    def _apply_linear_discount(self, t: int, discount_interval: int) -> None:
        d = (t / discount_interval) / ((t / discount_interval) + 1)
        for infoset in self.infosets.values():
            for i in range(infoset.num_actions):
                infoset.regret_sum[i] *= d
                infoset.strategy_sum[i] *= d

    def _update_average_strategy(self) -> None:
        for infoset in self.infosets.values():
            infoset.get_strategy(1.0)
    
    def _reset_negative_regrets(self):
        """CFR+: Reset negative regrets to 0 for faster convergence."""
        for infoset in self.infosets.values():
            infoset.regret_sum = np.maximum(infoset.regret_sum, 0.0)
    
    def _apply_discounting(self, iteration: int):
        """
        Linear CFR: Discount old regrets and strategies.
        
        Args:
            iteration: Current iteration number
        """
        t = iteration / self.discount_interval
        discount = t / (t + 1.0)
        
        for infoset in self.infosets.values():
            infoset.regret_sum *= discount
            infoset.strategy_sum *= discount
    
    def _should_prune_action(self, infoset: InfoSet, action_idx: int) -> bool:
        """
        Determine if action should be pruned based on regret.
        
        Args:
            infoset: Information set
            action_idx: Action index
        
        Returns:
            True if action should be pruned
        """
        if not hasattr(self, 'use_pruning') or not self.use_pruning:
            return False
        
        if action_idx >= len(infoset.regret_sum):
            return False
        
        return infoset.regret_sum[action_idx] < self.prune_threshold
    
    def _compute_average_regret(self) -> float:
        """Compute average regret across all infosets."""
        if not self.infosets:
            return 0.0
        
        total_regret = 0.0
        count = 0
        
        for infoset in self.infosets.values():
            total_regret += np.sum(np.maximum(infoset.regret_sum, 0.0))
            count += len(infoset.regret_sum)
        
        return total_regret / max(count, 1)
    
    def get_training_stats(self) -> Dict:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        stats = {
            'iterations': self.iterations,
            'infosets': len(self.infosets),
            'average_regret': self._compute_average_regret(),
            'cfr_plus_enabled': hasattr(self, 'use_regret_matching_plus')
        }
        
        if hasattr(self, 'use_pruning'):
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
    
    # ========================================================================
    # END CFR+ ENHANCEMENTS
    # ========================================================================
    
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
    
    def visualize_strategy(self, infoset_key: str = None):
        """
        Visualize CFR strategy for a given infoset or all infosets.
        Converts and optimizes data for compatibility with pokerbot workflow.
        """
        import matplotlib.pyplot as plt
        if infoset_key:
            infosets = {infoset_key: self.infosets[infoset_key]} if infoset_key in self.infosets else {}
        else:
            infosets = self.infosets
        for key, infoset in infosets.items():
            avg_strategy = infoset.get_average_strategy()
            actions = [str(a) for a in infoset.actions]
            plt.figure(figsize=(6, 3))
            plt.bar(actions, avg_strategy)
            plt.title(f"CFR Strategy for Infoset: {key}")
            plt.xlabel("Actions")
            plt.ylabel("Probability")
            plt.tight_layout()
            plt.show()
            print(f"Visualized CFR strategy for infoset: {key}")
