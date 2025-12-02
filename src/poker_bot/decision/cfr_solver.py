"""
CFR Solver - The "Brain" of the Strategist.

Performs continual re-solving using Counterfactual Regret Minimization.
When it's our turn, this module runs a limited-depth search of the 
immediate future to find the best action.

Key features:
- Limited-depth lookahead search
- Uses ValueNetwork for leaf node estimation
- CFR+ with linear discounting for fast convergence
- Real-time computation (sub-second decisions)

References:
- DeepStack paper: Continual re-solving algorithm
- CFR+: Tammelin et al. 2015
- DeepStack-Leduc: Reference implementation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from ..core.data_models import GameState, Action, ActionType, Card


class NodeType(Enum):
    """Types of nodes in the game tree."""
    PLAYER = "player"
    OPPONENT = "opponent"
    CHANCE = "chance"
    TERMINAL = "terminal"


@dataclass
class SearchNode:
    """Node in the CFR search tree."""
    node_type: NodeType
    player: int = 0  # 0 = player, 1 = opponent
    pot: float = 0.0
    street: int = 0  # 0-3 for preflop-river
    actions: List[str] = field(default_factory=list)
    children: Dict[str, 'SearchNode'] = field(default_factory=dict)
    
    # CFR data
    regret_sum: np.ndarray = None
    strategy_sum: np.ndarray = None
    
    def get_average_strategy(self, num_hands: int) -> np.ndarray:
        """Get average strategy from strategy sum."""
        if self.strategy_sum is None:
            num_actions = len(self.actions)
            return np.ones((num_hands, num_actions)) / num_actions
        
        total = self.strategy_sum.sum(axis=1, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        return self.strategy_sum / total


class CFRSolver:
    """
    CFR Solver for real-time poker decisions.
    
    The "Brain" of the Strategist. It receives a perfect game state
    and computes the mathematically optimal action. It knows nothing
    of pixels or GUIs.
    
    Usage:
        solver = CFRSolver(num_buckets=169)
        action = solver.solve(game_state, iterations=1000)
    """
    
    def __init__(self,
                 num_buckets: int = 169,
                 value_network: Optional[Any] = None,
                 lookahead_depth: int = 3,
                 use_cfr_plus: bool = True,
                 discount_factor: float = 0.95):
        """
        Initialize CFR solver.
        
        Args:
            num_buckets: Number of hand buckets (169 for Hold'em preflop)
            value_network: Optional ValueNetwork for leaf estimation
            lookahead_depth: Depth limit for search tree
            use_cfr_plus: Use CFR+ (regret matching+)
            discount_factor: Discount for linear CFR
        """
        self.num_buckets = num_buckets
        self.value_network = value_network
        self.lookahead_depth = lookahead_depth
        self.use_cfr_plus = use_cfr_plus
        self.discount_factor = discount_factor
        
        # Search tree
        self.root: Optional[SearchNode] = None
        
        # Ranges
        self.player_range: Optional[np.ndarray] = None
        self.opponent_range: Optional[np.ndarray] = None
        
        # Action abstraction (standard bet sizes as pot fractions)
        self.bet_sizes = [0.5, 0.75, 1.0, 1.5, 2.0]
    
    def solve(self, game_state: GameState, 
              iterations: int = 1000,
              time_limit_ms: Optional[int] = None) -> Action:
        """
        Solve for optimal action at given game state.
        
        Main entry point for real-time decision making.
        
        Args:
            game_state: Current game state from perception
            iterations: Number of CFR iterations
            time_limit_ms: Optional time limit in milliseconds
            
        Returns:
            Optimal Action to take
        """
        import time
        start_time = time.time()
        
        # Initialize ranges (uniform if not set)
        if self.player_range is None:
            self.player_range = np.ones(self.num_buckets) / self.num_buckets
        if self.opponent_range is None:
            self.opponent_range = np.ones(self.num_buckets) / self.num_buckets
        
        # Build search tree
        self.root = self._build_tree(game_state)
        
        # Run CFR iterations
        for i in range(iterations):
            # Check time limit
            if time_limit_ms:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms >= time_limit_ms:
                    break
            
            # CFR traversal
            self._cfr_traverse(
                self.root, 
                self.player_range.copy(),
                self.opponent_range.copy(),
                i
            )
        
        # Extract optimal action
        return self._get_best_action(game_state)
    
    def _build_tree(self, game_state: GameState) -> SearchNode:
        """Build limited-depth search tree."""
        # Determine actions
        actions = self._get_actions(game_state)
        
        # Create root node
        root = SearchNode(
            node_type=NodeType.PLAYER,
            player=0,
            pot=game_state.pot_size,
            street=list(['preflop', 'flop', 'turn', 'river']).index(game_state.street.value),
            actions=actions
        )
        
        # Initialize CFR arrays
        num_actions = len(actions)
        root.regret_sum = np.zeros((self.num_buckets, num_actions))
        root.strategy_sum = np.zeros((self.num_buckets, num_actions))
        
        # Build children recursively (limited depth)
        self._build_children(root, game_state, depth=0)
        
        return root
    
    def _build_children(self, node: SearchNode, game_state: GameState, depth: int):
        """Recursively build child nodes."""
        if depth >= self.lookahead_depth:
            return
        
        for action in node.actions:
            # Create child state (simplified)
            child_type = NodeType.OPPONENT if node.node_type == NodeType.PLAYER else NodeType.PLAYER
            child_pot = node.pot
            
            # Adjust pot based on action
            if action == 'call':
                child_pot += game_state.current_bet
            elif action.startswith('raise'):
                try:
                    multiplier = float(action.split('_')[1])
                    raise_amount = int(node.pot * multiplier)
                    child_pot += raise_amount
                except (IndexError, ValueError):
                    child_pot += game_state.current_bet * 2
            
            # Create child node
            child_actions = self._get_actions(game_state)  # Simplified
            child = SearchNode(
                node_type=child_type,
                player=1 - node.player,
                pot=child_pot,
                street=node.street,
                actions=child_actions
            )
            
            # Initialize CFR arrays
            num_actions = len(child_actions)
            child.regret_sum = np.zeros((self.num_buckets, num_actions))
            child.strategy_sum = np.zeros((self.num_buckets, num_actions))
            
            node.children[action] = child
            
            # Recurse
            if action != 'fold':
                self._build_children(child, game_state, depth + 1)
    
    def _get_actions(self, game_state: GameState) -> List[str]:
        """Get available actions as strings."""
        actions = ['fold']
        
        if game_state.current_bet == 0:
            actions.append('check')
        else:
            actions.append('call')
        
        # Add raise sizes
        for size in self.bet_sizes:
            actions.append(f'raise_{size}')
        
        return actions
    
    def _cfr_traverse(self, node: SearchNode,
                      player_range: np.ndarray,
                      opponent_range: np.ndarray,
                      iteration: int) -> np.ndarray:
        """
        CFR traversal with alternating updates.
        
        Returns counterfactual values for the traversing player.
        """
        # Terminal or leaf node
        if not node.children or node.node_type == NodeType.TERMINAL:
            return self._evaluate_terminal(node, player_range, opponent_range)
        
        num_actions = len(node.actions)
        num_hands = self.num_buckets
        
        # Get current strategy via regret matching
        strategy = self._get_strategy(node)
        
        # Initialize CFV arrays
        action_cfvs = np.zeros((num_hands, num_actions))
        
        # Traverse each action
        for a, action in enumerate(node.actions):
            if action not in node.children:
                # Leaf action - use value network or simple evaluation
                action_cfvs[:, a] = self._evaluate_leaf(node, action, player_range, opponent_range)
            else:
                child = node.children[action]
                
                if node.player == 0:  # Player node
                    # Traverse with action probability weighted ranges
                    action_cfvs[:, a] = self._cfr_traverse(
                        child,
                        player_range * strategy[:, a],
                        opponent_range,
                        iteration
                    )
                else:  # Opponent node
                    action_cfvs[:, a] = self._cfr_traverse(
                        child,
                        player_range,
                        opponent_range * strategy[:, a],
                        iteration
                    )
        
        # Compute counterfactual values
        cfv = (strategy * action_cfvs).sum(axis=1)
        
        # Update regrets
        if node.player == 0:  # Only update player's nodes
            for a in range(num_actions):
                regret = action_cfvs[:, a] - cfv
                
                if self.use_cfr_plus:
                    # CFR+: use regret matching+
                    node.regret_sum[:, a] = np.maximum(0, node.regret_sum[:, a] + regret)
                else:
                    node.regret_sum[:, a] += regret
            
            # Update strategy sum with discounting
            discount = (iteration / (iteration + 1)) ** self.discount_factor
            node.strategy_sum += player_range.reshape(-1, 1) * strategy * discount
        
        return cfv
    
    def _get_strategy(self, node: SearchNode) -> np.ndarray:
        """Get current strategy via regret matching."""
        if node.regret_sum is None:
            num_actions = len(node.actions)
            return np.ones((self.num_buckets, num_actions)) / num_actions
        
        # Regret matching
        positive_regrets = np.maximum(node.regret_sum, 0)
        regret_sum = positive_regrets.sum(axis=1, keepdims=True)
        
        # Uniform strategy for hands with no regret
        uniform = np.ones_like(positive_regrets) / positive_regrets.shape[1]
        
        # Compute strategy
        strategy = np.where(
            regret_sum > 0,
            positive_regrets / regret_sum,
            uniform
        )
        
        return strategy
    
    def _evaluate_terminal(self, node: SearchNode,
                          player_range: np.ndarray,
                          opponent_range: np.ndarray) -> np.ndarray:
        """Evaluate terminal node."""
        # Simplified terminal evaluation
        # In full implementation, this would compute showdown values
        return np.zeros(self.num_buckets)
    
    def _evaluate_leaf(self, node: SearchNode, action: str,
                      player_range: np.ndarray,
                      opponent_range: np.ndarray) -> np.ndarray:
        """Evaluate leaf node (end of lookahead)."""
        # Use value network if available
        if self.value_network is not None:
            pot_normalized = node.pot / 2000.0  # Normalize pot
            player_cfv, _ = self.value_network.get_value(
                player_range, opponent_range, pot_normalized
            )
            return player_cfv
        
        # Simple heuristic evaluation
        if action == 'fold':
            return np.full(self.num_buckets, -node.pot * 0.5)
        elif action == 'call':
            return np.zeros(self.num_buckets)
        else:
            # Raising has positive expectation for strong hands
            return np.linspace(-0.2, 0.3, self.num_buckets) * node.pot
    
    def _get_best_action(self, game_state: GameState) -> Action:
        """Extract best action from solved tree."""
        if self.root is None or self.root.strategy_sum is None:
            return Action(action_type=ActionType.CALL, amount=0)
        
        # Get average strategy
        avg_strategy = self.root.get_average_strategy(self.num_buckets)
        
        # Weight by player range to get overall action probabilities
        action_probs = (self.player_range.reshape(-1, 1) * avg_strategy).sum(axis=0)
        action_probs = action_probs / (action_probs.sum() + 1e-8)
        
        # Select best action
        best_idx = np.argmax(action_probs)
        best_action_str = self.root.actions[best_idx]
        
        # Convert to Action object
        return self._convert_action(best_action_str, game_state, action_probs[best_idx])
    
    def _convert_action(self, action_str: str, game_state: GameState,
                       confidence: float) -> Action:
        """Convert action string to Action object."""
        if action_str == 'fold':
            return Action(
                action_type=ActionType.FOLD,
                amount=0,
                confidence=confidence
            )
        elif action_str == 'check':
            return Action(
                action_type=ActionType.CHECK,
                amount=0,
                confidence=confidence
            )
        elif action_str == 'call':
            return Action(
                action_type=ActionType.CALL,
                amount=0,
                confidence=confidence
            )
        elif action_str.startswith('raise'):
            try:
                multiplier = float(action_str.split('_')[1])
                raise_amount = int(game_state.pot_size * multiplier)
                raise_amount = min(raise_amount, game_state.player.stack)
            except (IndexError, ValueError):
                raise_amount = game_state.current_bet * 2
            
            return Action(
                action_type=ActionType.RAISE,
                amount=raise_amount,
                confidence=confidence
            )
        
        # Default to call
        return Action(action_type=ActionType.CALL, amount=0, confidence=confidence)
    
    def update_range(self, action: str, is_opponent: bool = False):
        """Update range based on observed action."""
        # Simplified range update
        # In full implementation, would use Bayes' rule
        pass
    
    def reset(self):
        """Reset solver for new hand."""
        self.root = None
        self.player_range = np.ones(self.num_buckets) / self.num_buckets
        self.opponent_range = np.ones(self.num_buckets) / self.num_buckets


__all__ = ['CFRSolver', 'SearchNode', 'NodeType']
