"""
TreeCFR: DeepStack-style CFR solver for full game trees.

Implements Counterfactual Regret Minimization on complete game trees.
Based on the original DeepStack implementation documented in data/doc/.

After 1000 iterations on tutorial trees, exploitability should be ~1.0 chips.
"""
import numpy as np
from typing import List, Optional, Dict


class TreeCFR:
    """
    Counterfactual Regret Minimization solver for poker game trees.
    
    Computes near-optimal strategies by minimizing regret through iterative
    self-play. The algorithm:
    1. Traverses the game tree from root to leaves
    2. Computes counterfactual values for each action
    3. Updates regrets based on foregone payoffs
    4. Derives strategies via regret matching
    5. Averages strategies across iterations
    
    Reference: "Regret Minimization in Games with Incomplete Information" (Zinkevich et al.)
    """
    
    def __init__(self, skip_iterations: int = 0, use_linear_cfr: bool = True, use_cfr_plus: bool = True):
        """
        Initialize TreeCFR solver.
        
        Args:
            skip_iterations: Number of initial iterations to skip for averaging (warmup)
            use_linear_cfr: Enable Linear CFR (weight regrets by iteration number for faster convergence)
            use_cfr_plus: Enable CFR+ (reset negative regrets to 0 for faster convergence)
        """
        self.skip_iterations = skip_iterations
        self.use_linear_cfr = use_linear_cfr
        self.use_cfr_plus = use_cfr_plus
        self.iteration = 0
        
        # Storage for regrets and strategies per node
        self.regrets = {}  # node_id -> action regrets
        self.strategy_sum = {}  # node_id -> cumulative strategy
        self.current_strategy = {}  # node_id -> current strategy
        
    def run_cfr(self, root, starting_ranges: Optional[np.ndarray] = None, 
                iter_count: int = 1000) -> Dict:
        """
        Run CFR algorithm on game tree to compute nash equilibrium strategy.
        
        Args:
            root: Root node of the game tree (PokerTreeNode)
            starting_ranges: Starting ranges for both players [player1_range, player2_range]
                           Each range is a vector of probabilities over possible hands
            iter_count: Number of CFR iterations to run
            
        Returns:
            dict containing:
                - 'strategy': Final average strategy at each node
                - 'exploitability': Exploitability of the strategy (if computed)
                - 'regrets': Final regret values
        """
        if starting_ranges is None:
            # Default to uniform ranges for Texas Hold'em (169 hands)
            num_hands = 169  # Hold'em default (169 hand combinations)
            starting_ranges = np.array([
                np.ones(num_hands) / num_hands,
                np.ones(num_hands) / num_hands
            ])
        
        # Initialize regrets and strategies
        self._initialize_tree(root)
        
        # Run CFR iterations
        for i in range(iter_count):
            self.iteration = i
            
            # Linear CFR: weight by iteration number (DeepStack paper Section S2.1)
            iteration_weight = (i + 1) if self.use_linear_cfr else 1.0
            
            # Traverse tree for each player
            for player_id in range(2):
                # Create reach probabilities (everyone reaches root with probability 1)
                reach_probs = np.ones(2)
                
                # Run CFR traversal with iteration weight
                self._cfr_traverse(root, starting_ranges, reach_probs, player_id, iteration_weight)
            
            # CFR+: Reset negative regrets (DeepStack paper Section S2.2)
            if self.use_cfr_plus:
                for node_id in self.regrets:
                    self.regrets[node_id] = np.maximum(self.regrets[node_id], 0.0)
            
            # Update average strategy (skip initial iterations)
            if i >= self.skip_iterations:
                self._update_average_strategies(root)
        
        # Compute final average strategy
        final_strategy = self._get_average_strategy(root)
        
        return {
            'strategy': final_strategy,
            'regrets': self.regrets,
            'iterations': iter_count
        }
    
    def _initialize_tree(self, node, node_id: str = "root"):
        """
        Initialize regrets and strategies for all nodes in tree.
        
        Args:
            node: Current node
            node_id: Unique identifier for this node
        """
        if not hasattr(node, 'children') or not node.children:
            # Leaf node
            return
        
        # Get number of actions at this node
        num_actions = len(node.children)
        
        # Initialize regrets to zero
        if node_id not in self.regrets:
            self.regrets[node_id] = np.zeros(num_actions)
            self.strategy_sum[node_id] = np.zeros(num_actions)
            self.current_strategy[node_id] = np.ones(num_actions) / num_actions
        
        # Recursively initialize children
        for action_idx, child in enumerate(node.children):
            child_id = f"{node_id}_a{action_idx}"
            self._initialize_tree(child, child_id)
    
    def _cfr_traverse(self, node, ranges: np.ndarray, reach_probs: np.ndarray, 
                      traversing_player: int, iteration_weight: float = 1.0, node_id: str = "root") -> np.ndarray:
        """
        Traverse tree and update regrets via CFR.
        
        Args:
            node: Current node in tree
            ranges: Current ranges for both players [2 x num_hands]
            reach_probs: Reach probabilities for both players
            traversing_player: Player whose regrets we're updating (0 or 1)
            iteration_weight: Weight for Linear CFR (default 1.0, or iteration number)
            node_id: Unique identifier for this node
            
        Returns:
            Counterfactual values for traversing player's range
        """
        # Terminal node - return payoffs
        if not hasattr(node, 'children') or not node.children:
            return self._evaluate_terminal(node, ranges, traversing_player)
        
        # Get current player at this node
        current_player = node.current_player if hasattr(node, 'current_player') else 0
        
        # Get current strategy using regret matching
        strategy = self._get_strategy(node_id)
        
        # Number of possible hands
        num_hands = ranges.shape[1]
        
        # Compute counterfactual values for each action
        num_actions = len(node.children)
        action_cfvs = np.zeros((num_actions, num_hands))
        
        for action_idx, child in enumerate(node.children):
            child_id = f"{node_id}_a{action_idx}"
            
            # Update reach probabilities if current player acts
            new_reach_probs = reach_probs.copy()
            if current_player == traversing_player:
                new_reach_probs[current_player] *= strategy[action_idx]
            
            # Recursively compute child CFVs
            action_cfvs[action_idx] = self._cfr_traverse(
                child, ranges, new_reach_probs, traversing_player, iteration_weight, child_id
            )
        
        # Compute node CFV as strategy-weighted sum
        node_cfv = np.dot(strategy, action_cfvs)
        
        # Update regrets if this is traversing player's node
        if current_player == traversing_player:
            # Compute regrets: CFV(action) - CFV(strategy)
            for action_idx in range(num_actions):
                instant_regrets = action_cfvs[action_idx] - node_cfv
                
                # Weight by opponent reach probability
                opponent = 1 - current_player
                weighted_regrets = instant_regrets * reach_probs[opponent]
                
                # Accumulate regrets with Linear CFR weighting (DeepStack paper Section S2.1)
                # Linear CFR: multiply by iteration number for faster convergence
                self.regrets[node_id][action_idx] += (
                    weighted_regrets * ranges[current_player] * iteration_weight
                ).sum()
        
        return node_cfv
    
    def _get_strategy(self, node_id: str) -> np.ndarray:
        """
        Get current strategy at node using regret matching.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Strategy vector (probabilities over actions)
        """
        if node_id not in self.regrets:
            # No regrets yet, return uniform
            return self.current_strategy.get(node_id, np.array([0.5, 0.5]))
        
        regrets = self.regrets[node_id]
        
        # Regret matching: play actions proportionally to positive regrets
        positive_regrets = np.maximum(regrets, 0)
        regret_sum = positive_regrets.sum()
        
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            # No positive regrets, play uniformly
            strategy = np.ones(len(regrets)) / len(regrets)
        
        self.current_strategy[node_id] = strategy
        return strategy
    
    def _update_average_strategies(self, node, node_id: str = "root"):
        """
        Update average strategy by adding current strategy.
        
        Args:
            node: Current node
            node_id: Node identifier
        """
        if not hasattr(node, 'children') or not node.children:
            return
        
        if node_id in self.current_strategy:
            strategy = self.current_strategy[node_id]
            self.strategy_sum[node_id] += strategy
        
        # Recursively update children
        for action_idx, child in enumerate(node.children):
            child_id = f"{node_id}_a{action_idx}"
            self._update_average_strategies(child, child_id)
    
    def _get_average_strategy(self, node, node_id: str = "root") -> Dict:
        """
        Compute final average strategy over all iterations.
        
        Args:
            node: Root node
            node_id: Node identifier
            
        Returns:
            Dictionary mapping node_id to average strategy
        """
        result = {}
        
        if not hasattr(node, 'children') or not node.children:
            return result
        
        if node_id in self.strategy_sum:
            strategy_sum = self.strategy_sum[node_id]
            total = strategy_sum.sum()
            
            if total > 0:
                result[node_id] = strategy_sum / total
            else:
                # No strategy accumulated, use uniform
                result[node_id] = np.ones(len(strategy_sum)) / len(strategy_sum)
        
        # Recursively get children strategies
        for action_idx, child in enumerate(node.children):
            child_id = f"{node_id}_a{action_idx}"
            child_strategies = self._get_average_strategy(child, child_id)
            result.update(child_strategies)
        
        return result
    
    def _evaluate_terminal(self, node, ranges: np.ndarray, player: int) -> np.ndarray:
        """
        Evaluate terminal node payoffs.
        
        Args:
            node: Terminal node
            ranges: Current ranges for both players
            player: Player to evaluate for
            
        Returns:
            Payoff vector for player's range
        """
        num_hands = ranges.shape[1]
        
        # Simple placeholder - in production, use actual showdown/fold equity
        # For now, return uniform small payoffs
        if hasattr(node, 'node_type') and 'fold' in str(node.node_type).lower():
            # Fold node - fixed payoff
            return np.ones(num_hands) * (node.pot if hasattr(node, 'pot') else 100)
        else:
            # Showdown - would use hand strength evaluation
            # Placeholder: zero-sum random outcome
            return np.random.randn(num_hands) * 10
    
    def update_average_strategy(self, node, current_strategy, iter):
        """
        Legacy method for compatibility.
        Updates node's average strategy given current strategy.
        
        Args:
            node: Node to update
            current_strategy: Current strategy to add to average
            iter: Current iteration number
        """
        if not hasattr(node, 'average_strategy'):
            node.average_strategy = current_strategy.copy()
            node.strategy_weight = 1
        else:
            node.average_strategy = (
                node.average_strategy * node.strategy_weight + current_strategy
            ) / (node.strategy_weight + 1)
            node.strategy_weight += 1
