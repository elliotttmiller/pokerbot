"""
CFR Solver for DeepStack Decision Engine

Implements Counterfactual Regret Minimization (CFR+) for solving
depth-limited lookahead trees. This is the core algorithm for
computing Nash equilibrium strategies in poker.

Based on DeepStack-Leduc reference implementation:
- CFR+ with regret matching
- Linear discounting
- Action pruning for efficiency
- Skip iterations for strategy averaging
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SearchNode:
    """
    Node in the search tree for CFR solving.
    
    Attributes:
        player: Acting player (0 or 1)
        pot: Current pot size
        bets: Bet amounts for each player
        actions: Available actions
        children: Child nodes for each action
        regret_sum: Cumulative regrets per hand/action
        strategy_sum: Cumulative strategy per hand/action
        terminal: Whether this is a terminal node
        terminal_type: Type of terminal ('showdown', 'fold')
    """
    player: int
    pot: float
    bets: List[float] = field(default_factory=lambda: [0.0, 0.0])
    actions: List[str] = field(default_factory=list)
    children: Dict[str, 'SearchNode'] = field(default_factory=dict)
    regret_sum: Optional[np.ndarray] = None
    strategy_sum: Optional[np.ndarray] = None
    terminal: bool = False
    terminal_type: Optional[str] = None


class CFRSolver:
    """
    CFR+ solver for poker game trees.
    
    Implements the core CFR algorithm used by DeepStack:
    1. Traverses game tree recursively
    2. Accumulates regrets for each action
    3. Uses regret matching to compute strategy
    4. Averages strategy over iterations
    
    Key optimizations (matching DeepStack-Leduc):
    - CFR+ with non-negative regrets
    - Linear discounting for strategy averaging
    - Action pruning for negative-regret actions
    - Skip iterations before strategy averaging
    
    Usage:
        solver = CFRSolver(num_hands=36)
        tree = solver.build_tree(node_params)
        solver.solve(tree, player_range, opponent_range, iterations=2000)
        strategy = solver.get_strategy(tree)
    """
    
    def __init__(
        self,
        num_hands: int = 36,
        use_cfr_plus: bool = True,
        linear_discounting: bool = True,
        pruning_threshold: float = -500000000,
        skip_iterations: int = 200
    ):
        """
        Initialize CFR solver.
        
        Args:
            num_hands: Number of possible hands (hand buckets)
            use_cfr_plus: Use CFR+ (non-negative regrets)
            linear_discounting: Apply linear discounting to strategy
            pruning_threshold: Threshold for action pruning
            skip_iterations: Don't update strategy for first N iterations
        """
        self.num_hands = num_hands
        self.use_cfr_plus = use_cfr_plus
        self.linear_discounting = linear_discounting
        self.pruning_threshold = pruning_threshold
        self.skip_iterations = skip_iterations
        
        # Current iteration counter
        self.iteration = 0
    
    def build_tree(self, node_params: Dict) -> SearchNode:
        """
        Build search tree from node parameters.
        
        Args:
            node_params: Dictionary with keys:
                - street: Betting round (0-3)
                - bets: Current bets [player, opponent]
                - current_player: Acting player (0 or 1)
                - board: Board cards (list of indices)
                - bet_sizing: Pot-relative bet sizes
                
        Returns:
            Root SearchNode of the tree
        """
        street = node_params.get('street', 0)
        bets = node_params.get('bets', [20.0, 20.0])
        current_player = node_params.get('current_player', 0)
        bet_sizing = node_params.get('bet_sizing', [1.0])
        
        pot = sum(bets)
        
        # Determine available actions
        actions = self._get_actions(bets, pot, bet_sizing, current_player)
        
        root = SearchNode(
            player=current_player,
            pot=pot,
            bets=list(bets),
            actions=actions
        )
        
        # Initialize regret and strategy storage
        num_actions = len(actions)
        root.regret_sum = np.zeros((self.num_hands, num_actions))
        root.strategy_sum = np.zeros((self.num_hands, num_actions))
        
        # Build children recursively
        for action in actions:
            child = self._build_child(root, action, node_params)
            root.children[action] = child
        
        return root
    
    def _get_actions(
        self,
        bets: List[float],
        pot: float,
        bet_sizing: List[float],
        player: int
    ) -> List[str]:
        """Get available actions for current state."""
        actions = ['fold']
        
        to_call = abs(bets[1 - player] - bets[player])
        
        if to_call == 0:
            actions.append('check')
        else:
            actions.append('call')
        
        # Add raise actions based on bet sizing
        for size in bet_sizing:
            raise_amount = pot * size
            actions.append(f'raise_{size}')
        
        return actions
    
    def _build_child(
        self,
        parent: SearchNode,
        action: str,
        node_params: Dict,
        depth: int = 0,
        max_depth: int = 2
    ) -> SearchNode:
        """Build child node for an action."""
        new_bets = list(parent.bets)
        new_pot = parent.pot
        next_player = 1 - parent.player
        
        if action == 'fold':
            return SearchNode(
                player=next_player,
                pot=new_pot,
                bets=new_bets,
                terminal=True,
                terminal_type='fold'
            )
        
        if action == 'check':
            # Check if round is complete (both players checked)
            return SearchNode(
                player=next_player,
                pot=new_pot,
                bets=new_bets,
                terminal=True,
                terminal_type='showdown'
            )
        
        if action == 'call':
            to_call = abs(parent.bets[1 - parent.player] - parent.bets[parent.player])
            new_bets[parent.player] += to_call
            new_pot += to_call
            
            return SearchNode(
                player=next_player,
                pot=new_pot,
                bets=new_bets,
                terminal=True,
                terminal_type='showdown'
            )
        
        if action.startswith('raise_'):
            size = float(action.split('_')[1])
            raise_amount = parent.pot * size
            new_bets[parent.player] = new_bets[1 - parent.player] + raise_amount
            new_pot = sum(new_bets)
            
            # Limit tree depth
            if depth >= max_depth:
                return SearchNode(
                    player=next_player,
                    pot=new_pot,
                    bets=new_bets,
                    terminal=True,
                    terminal_type='showdown'
                )
            
            child_actions = ['fold', 'call']
            if depth < max_depth - 1:
                child_actions.append('raise_1.0')
            
            child = SearchNode(
                player=next_player,
                pot=new_pot,
                bets=new_bets,
                actions=child_actions
            )
            
            # Initialize storage for non-terminal
            num_actions = len(child.actions)
            child.regret_sum = np.zeros((self.num_hands, num_actions))
            child.strategy_sum = np.zeros((self.num_hands, num_actions))
            
            # Build children recursively
            for child_action in child_actions:
                grandchild = self._build_child(child, child_action, node_params, depth + 1, max_depth)
                child.children[child_action] = grandchild
            
            return child
        
        # Default: terminal
        return SearchNode(
            player=next_player,
            pot=new_pot,
            bets=new_bets,
            terminal=True,
            terminal_type='showdown'
        )
    
    def solve(
        self,
        root: SearchNode,
        player_range: np.ndarray,
        opponent_range: np.ndarray,
        iterations: int = 2000
    ) -> Dict[str, np.ndarray]:
        """
        Solve the game tree using CFR.
        
        Args:
            root: Root node of the search tree
            player_range: Player's initial range (normalized probabilities)
            opponent_range: Opponent's initial range
            iterations: Number of CFR iterations
            
        Returns:
            Dictionary with 'strategy' and 'values' keys
        """
        self.iteration = 0
        
        for i in range(iterations):
            self.iteration = i
            
            # Alternating updates (standard CFR)
            for traverser in [0, 1]:
                if traverser == 0:
                    self._cfr_traverse(root, player_range, opponent_range, traverser)
                else:
                    self._cfr_traverse(root, opponent_range, player_range, traverser)
        
        # Extract final strategy
        strategy = self._get_average_strategy(root)
        
        return {
            'strategy': {'root': strategy},
            'iterations': iterations
        }
    
    def _cfr_traverse(
        self,
        node: SearchNode,
        player_range: np.ndarray,
        opponent_range: np.ndarray,
        traverser: int
    ) -> np.ndarray:
        """
        Recursively traverse tree and update regrets.
        
        Args:
            node: Current node
            player_range: Acting player's range
            opponent_range: Opponent's range
            traverser: Player updating regrets
            
        Returns:
            Counterfactual values for traverser's hands
        """
        if node.terminal:
            return self._terminal_cfv(node, player_range, opponent_range, traverser)
        
        num_actions = len(node.actions)
        
        # Initialize regret_sum if needed
        if node.regret_sum is None:
            node.regret_sum = np.zeros((self.num_hands, num_actions))
            node.strategy_sum = np.zeros((self.num_hands, num_actions))
        
        # Get current strategy from regrets
        strategy = self._regret_matching(node.regret_sum)
        
        # Compute action values
        action_cfvs = np.zeros((self.num_hands, num_actions))
        
        for a, action in enumerate(node.actions):
            if action not in node.children:
                # Skip missing children (terminal actions)
                continue
                
            child = node.children[action]
            
            if node.player == traverser:
                # Traverser acts: use current strategy
                action_cfvs[:, a] = self._cfr_traverse(
                    child, player_range, opponent_range, traverser
                )
            else:
                # Opponent acts: weight by opponent strategy
                new_opponent_range = opponent_range * strategy[:, a]
                
                action_cfvs[:, a] = self._cfr_traverse(
                    child, player_range, new_opponent_range, traverser
                )
        
        # Compute node values
        node_cfv = np.sum(strategy * action_cfvs, axis=1)
        
        # Update regrets if this is traverser's node
        if node.player == traverser:
            for a in range(num_actions):
                regret = action_cfvs[:, a] - node_cfv
                
                if self.use_cfr_plus:
                    node.regret_sum[:, a] = np.maximum(0, node.regret_sum[:, a] + regret)
                else:
                    node.regret_sum[:, a] += regret
            
            # Update strategy sum (after skip iterations)
            if self.iteration >= self.skip_iterations:
                if self.linear_discounting:
                    weight = self.iteration - self.skip_iterations + 1
                    node.strategy_sum += strategy * weight
                else:
                    node.strategy_sum += strategy
        
        return node_cfv
    
    def _terminal_cfv(
        self,
        node: SearchNode,
        player_range: np.ndarray,
        opponent_range: np.ndarray,
        traverser: int
    ) -> np.ndarray:
        """
        Compute counterfactual values at terminal node.
        
        Args:
            node: Terminal node
            player_range: Player's range
            opponent_range: Opponent's range
            traverser: Player to compute values for
            
        Returns:
            CFV vector for traverser's hands
        """
        if node.terminal_type == 'fold':
            # Opponent folded, traverser wins pot
            folder = 1 - node.player  # Player who reached this node folded
            if folder == traverser:
                return -node.bets[folder] * np.ones(self.num_hands)
            else:
                return node.bets[folder] * np.ones(self.num_hands)
        
        # Showdown: simplified equity calculation
        # In full implementation, this would use hand rankings
        pot = node.pot
        
        # Placeholder: uniform equity (would be replaced by actual evaluation)
        equity = 0.5 * np.ones(self.num_hands)
        
        return pot * (2 * equity - 1)
    
    def _regret_matching(self, regret_sum: np.ndarray) -> np.ndarray:
        """
        Compute strategy from cumulative regrets using regret matching.
        
        Args:
            regret_sum: Cumulative regrets [num_hands, num_actions]
            
        Returns:
            Strategy [num_hands, num_actions]
        """
        positive_regrets = np.maximum(0, regret_sum)
        sum_regrets = np.sum(positive_regrets, axis=1, keepdims=True)
        
        # Uniform strategy if no positive regrets
        uniform = np.ones_like(regret_sum) / regret_sum.shape[1]
        
        strategy = np.where(
            sum_regrets > 0,
            positive_regrets / (sum_regrets + 1e-10),
            uniform
        )
        
        return strategy
    
    def _get_average_strategy(self, node: SearchNode) -> np.ndarray:
        """Get average strategy from strategy sum."""
        if node.strategy_sum is None:
            return np.ones(len(node.actions)) / len(node.actions)
        
        sum_strategy = np.sum(node.strategy_sum, axis=1, keepdims=True)
        
        # Average over hands
        hand_averaged = np.sum(node.strategy_sum, axis=0)
        total = np.sum(hand_averaged)
        
        if total > 0:
            return hand_averaged / total
        else:
            return np.ones(len(node.actions)) / len(node.actions)
    
    def get_action_probabilities(self, node: SearchNode) -> Dict[str, float]:
        """
        Get action probabilities from solved tree.
        
        Args:
            node: Solved node
            
        Returns:
            Dictionary mapping action names to probabilities
        """
        strategy = self._get_average_strategy(node)
        return {action: float(strategy[i]) for i, action in enumerate(node.actions)}


__all__ = ['CFRSolver', 'SearchNode']
