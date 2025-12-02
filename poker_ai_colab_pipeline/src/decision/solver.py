"""
DeepStack Solver - Unified Decision Engine

Complete implementation of DeepStack's continual re-solving algorithm:
1. Build depth-limited lookahead tree
2. Solve tree using CFR with neural network terminal values
3. Extract Nash equilibrium strategy
4. Return optimal action with confidence

Based on Moravčík et al., 2017 Science paper.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .cfr import CFRSolver, SearchNode
from .value_network import ValueNetwork
from .action import Action, ActionType


class DeepStackSolver:
    """
    DeepStack-style poker solver with continual re-solving.
    
    Implements the two-component DeepStack architecture:
    
    **Offline Component** (Neural Network):
    - Pre-trained on CFR-solved poker situations
    - Maps (public state, opponent range) → counterfactual values
    - Used for terminal node estimation in lookahead
    
    **Online Component** (This class):
    - Builds depth-limited lookahead tree
    - Solves using CFR with neural terminal values
    - Produces approximate Nash equilibrium strategy
    
    Usage:
        solver = DeepStackSolver(value_network_path="models/deepstack.pt")
        action = solver.solve(game_state, iterations=2000)
        print(f"Action: {action.action_type}, Amount: {action.amount}")
    """
    
    def __init__(
        self,
        value_network_path: Optional[str] = None,
        num_hands: int = 36,
        lookahead_depth: int = 3,
        cfr_iterations: int = 2000,
        use_cfr_plus: bool = True,
        time_limit_ms: Optional[int] = None
    ):
        """
        Initialize DeepStack solver.
        
        Args:
            value_network_path: Path to trained value network
            num_hands: Number of hand buckets
            lookahead_depth: Depth of lookahead tree
            cfr_iterations: Default CFR iterations per solve
            use_cfr_plus: Use CFR+ algorithm
            time_limit_ms: Optional time limit for solving
        """
        self.num_hands = num_hands
        self.lookahead_depth = lookahead_depth
        self.default_iterations = cfr_iterations
        self.use_cfr_plus = use_cfr_plus
        self.time_limit_ms = time_limit_ms
        
        # Initialize CFR solver
        self.cfr = CFRSolver(
            num_hands=num_hands,
            use_cfr_plus=use_cfr_plus,
            linear_discounting=True,
            skip_iterations=200
        )
        
        # Load value network if provided
        self.value_network = None
        if value_network_path:
            try:
                self.value_network = ValueNetwork(input_size=num_hands)
                self.value_network.load(value_network_path)
                print(f"[DeepStackSolver] Value network loaded from {value_network_path}")
            except Exception as e:
                print(f"[DeepStackSolver] Failed to load value network: {e}")
        
        # Storage for last solve results
        self.last_tree = None
        self.last_strategy = None
        self.last_solve_time = 0.0
    
    def solve(
        self,
        game_state: Dict,
        player_range: Optional[np.ndarray] = None,
        opponent_range: Optional[np.ndarray] = None,
        iterations: Optional[int] = None
    ) -> Action:
        """
        Solve game state and return optimal action.
        
        This is the main API for decision-making.
        
        Args:
            game_state: Dictionary with game state (from perception)
            player_range: Player's range (uniform if not provided)
            opponent_range: Opponent's range (uniform if not provided)
            iterations: CFR iterations (uses default if not provided)
            
        Returns:
            Action object with recommended action and confidence
        """
        start_time = time.time()
        iterations = iterations or self.default_iterations
        
        # Convert game state to node params
        node_params = self._to_node_params(game_state)
        
        # Initialize ranges (uniform if not provided)
        if player_range is None:
            player_range = np.ones(self.num_hands) / self.num_hands
        if opponent_range is None:
            opponent_range = np.ones(self.num_hands) / self.num_hands
        
        # Build lookahead tree
        tree = self.cfr.build_tree(node_params)
        self.last_tree = tree
        
        # Solve tree
        result = self.cfr.solve(tree, player_range, opponent_range, iterations)
        self.last_strategy = result['strategy']
        
        # Record timing
        self.last_solve_time = (time.time() - start_time) * 1000
        
        # Extract best action with timing info
        action = self._select_action(tree, game_state, iterations, self.last_solve_time)
        
        return action
    
    def _to_node_params(self, game_state: Dict) -> Dict:
        """Convert game state to CFR node parameters."""
        # Handle both raw dict and GameState objects
        if hasattr(game_state, 'to_deepstack_format'):
            return game_state.to_deepstack_format()
        
        # Map street string to number
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street = game_state.get('street', 0)
        if isinstance(street, str):
            street = street_map.get(street.lower(), 0)
        
        pot = game_state.get('pot_size', game_state.get('pot', 100))
        current_bet = game_state.get('current_bet', 20)
        
        return {
            'street': street,
            'bets': [current_bet, current_bet],
            'current_player': 1,  # Hero
            'board': game_state.get('board', []),
            'bet_sizing': [1.0],  # Pot-sized bets
            'pot': pot
        }
    
    def _select_action(self, tree: SearchNode, game_state: Dict, 
                       cfr_iterations: int = 0, solve_time_ms: float = 0.0) -> Action:
        """Select best action from solved tree."""
        action_probs = self.cfr.get_action_probabilities(tree)
        
        # Find highest probability action
        best_action = max(action_probs.keys(), key=lambda a: action_probs[a])
        best_prob = action_probs[best_action]
        
        # Convert to Action object
        if best_action == 'fold':
            return Action(
                action_type=ActionType.FOLD,
                confidence=best_prob,
                strategy=action_probs,
                cfr_iterations=cfr_iterations,
                solve_time_ms=solve_time_ms
            )
        
        if best_action == 'check':
            return Action(
                action_type=ActionType.CHECK,
                confidence=best_prob,
                strategy=action_probs,
                cfr_iterations=cfr_iterations,
                solve_time_ms=solve_time_ms
            )
        
        if best_action == 'call':
            return Action(
                action_type=ActionType.CALL,
                confidence=best_prob,
                strategy=action_probs,
                cfr_iterations=cfr_iterations,
                solve_time_ms=solve_time_ms
            )
        
        if best_action.startswith('raise_'):
            size = float(best_action.split('_')[1])
            pot = game_state.get('pot_size', game_state.get('pot', 100))
            amount = int(pot * size)
            
            return Action(
                action_type=ActionType.RAISE,
                amount=amount,
                confidence=best_prob,
                strategy=action_probs,
                cfr_iterations=cfr_iterations,
                solve_time_ms=solve_time_ms
            )
        
        # Default to call
        return Action(
            action_type=ActionType.CALL,
            confidence=0.5,
            strategy=action_probs,
            cfr_iterations=cfr_iterations,
            solve_time_ms=solve_time_ms
        )
    
    def get_strategy(self) -> Optional[Dict[str, float]]:
        """Get last computed strategy."""
        if self.last_tree is None:
            return None
        return self.cfr.get_action_probabilities(self.last_tree)
    
    def get_counterfactual_values(
        self,
        public_features: np.ndarray,
        opponent_range: np.ndarray
    ) -> np.ndarray:
        """
        Get counterfactual values using value network.
        
        Used for terminal node estimation in lookahead.
        
        Args:
            public_features: Public state features
            opponent_range: Opponent's range distribution
            
        Returns:
            CFV vector for all player hands
        """
        if self.value_network is None:
            # Fallback: uniform values
            return np.zeros(self.num_hands)
        
        return self.value_network.predict(public_features, opponent_range)


def solve(
    game_state: Dict,
    model_path: Optional[str] = None,
    iterations: int = 2000
) -> Action:
    """
    One-call function to solve a game state.
    
    Args:
        game_state: Dictionary with game state
        model_path: Optional path to value network
        iterations: CFR iterations
        
    Returns:
        Action object with recommended action
    """
    solver = DeepStackSolver(value_network_path=model_path)
    return solver.solve(game_state, iterations=iterations)


__all__ = ['DeepStackSolver', 'solve']
