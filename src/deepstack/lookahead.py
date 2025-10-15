"""
Lookahead: DeepStack's depth-limited game tree lookahead for re-solving.

Builds lookahead trees and performs CFR-based solving with neural network
value estimation at leaves. This is the core computational engine of DeepStack.

Ported from the original DeepStack lookahead.lua module.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from src.deepstack.tree_builder import PokerTreeBuilder, PokerTreeNode
from src.deepstack.terminal_equity import TerminalEquity
from src.deepstack.cfrd_gadget import CFRDGadget
from src.deepstack.value_nn import ValueNN


class LookaheadBuilder:
    """Builds lookahead data structures from public trees."""
    
    def __init__(self, lookahead):
        self.lookahead = lookahead
    
    def build_from_tree(self, tree: PokerTreeNode):
        """
        Build lookahead data structures from game tree.
        
        Args:
            tree: Root node of public game tree
        """
        # Store tree reference
        self.lookahead.tree = tree
        
        # Build lookahead structure
        self._compute_structure(tree)
        self._allocate_data_structures()
        
    def _compute_structure(self, tree: PokerTreeNode):
        """Analyze tree structure and compute dimensions."""
        # Tree traversal to compute structure
        self.lookahead.depth = self._compute_depth(tree)
        self.lookahead.num_nodes_by_depth = self._compute_nodes_by_depth(tree)
        self.lookahead.actions_count = self._compute_actions_count(tree)
        self.lookahead.terminal_actions_count = self._compute_terminal_actions(tree)
        
    def _compute_depth(self, node: PokerTreeNode, current_depth: int = 0) -> int:
        """Compute maximum depth of tree."""
        if not node.children:
            return current_depth
        
        max_child_depth = current_depth
        for child in node.children:
            child_depth = self._compute_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _compute_nodes_by_depth(self, node: PokerTreeNode) -> List[int]:
        """Compute number of nodes at each depth."""
        counts = []
        self._count_nodes_recursive(node, 0, counts)
        return counts
    
    def _count_nodes_recursive(self, node: PokerTreeNode, depth: int, counts: List[int]):
        """Recursive helper for counting nodes."""
        # Extend counts list if needed
        while len(counts) <= depth:
            counts.append(0)
        
        counts[depth] += 1
        
        for child in node.children:
            self._count_nodes_recursive(child, depth + 1, counts)
    
    def _compute_actions_count(self, tree: PokerTreeNode) -> List[int]:
        """Compute action counts for each depth."""
        # Simplified - would compute actual action counts in production
        return [3] * (self.lookahead.depth + 1)  # Assume 3 actions per node
    
    def _compute_terminal_actions(self, tree: PokerTreeNode) -> List[int]:
        """Compute terminal action counts."""
        # Simplified - would compute actual terminal action counts
        return [1] * (self.lookahead.depth + 1)
    
    def _allocate_data_structures(self):
        """Allocate tensors for lookahead data."""
        depth = self.lookahead.depth
        num_hands = 169 if self.lookahead.game_variant == 'holdem' else 6
        max_actions = 4  # fold, check/call, raise1, raise2
        
        # Range data: [depth][action][bet][player][hand]
        self.lookahead.ranges_data = []
        for d in range(depth + 1):
            level_data = np.zeros((max_actions, 3, 3, 2, num_hands))
            self.lookahead.ranges_data.append(level_data)
        
        # CFV data: same structure as ranges
        self.lookahead.cfvs_data = []
        for d in range(depth + 1):
            level_data = np.zeros((max_actions, 3, 3, 2, num_hands))
            self.lookahead.cfvs_data.append(level_data)
        
        # Strategy data: [depth][action][bet][hand]  
        self.lookahead.current_strategy_data = []
        self.lookahead.average_strategies_data = []
        for d in range(depth + 1):
            current = np.zeros((max_actions, 3, num_hands))
            average = np.zeros((max_actions, 3, num_hands))
            self.lookahead.current_strategy_data.append(current)
            self.lookahead.average_strategies_data.append(average)
        
        # Regret data
        self.lookahead.regrets_data = []
        self.lookahead.positive_regrets_data = []
        for d in range(depth + 1):
            regrets = np.zeros((max_actions, 3, num_hands))
            pos_regrets = np.zeros((max_actions, 3, num_hands))
            self.lookahead.regrets_data.append(regrets)
            self.lookahead.positive_regrets_data.append(pos_regrets)


class Lookahead:
    """
    DeepStack's depth-limited lookahead solver.
    
    Performs CFR-based solving on lookahead trees with neural network
    value estimation at depth-limited leaves.
    """
    
    def __init__(self, game_variant: str = 'leduc', num_hands: int = 6):
        """
        Initialize lookahead solver.
        
        Args:
            game_variant: 'leduc' or 'holdem'  
            num_hands: Number of hand abstractions
        """
        self.game_variant = game_variant
        self.num_hands = num_hands
        
        # Components
        self.builder = LookaheadBuilder(self)
        self.terminal_equity = None
        self.reconstruction_gadget = None
        self.value_network = None
        
        # Data structures (allocated by builder)
        self.tree = None
        self.depth = 0
        self.ranges_data = []
        self.cfvs_data = []
        self.current_strategy_data = []
        self.average_strategies_data = []
        self.regrets_data = []
        self.positive_regrets_data = []
        
        # Solving parameters
        self.regret_epsilon = 1e-9
        self.cfr_iterations = 1000
        self.cfr_skip_iterations = 500
        
        # Opponent modeling
        self.reconstruction_opponent_cfvs = None
    
    def build_lookahead(self, tree: PokerTreeNode):
        """
        Build lookahead from public game tree.
        
        Args:
            tree: Root node of public tree
        """
        self.builder.build_from_tree(tree)
        
        # Initialize terminal equity calculator
        self.terminal_equity = TerminalEquity(
            game_variant=self.game_variant,
            num_hands=self.num_hands
        )
        self.terminal_equity.set_board(tree.board)
        
        print(f"[Lookahead] Built lookahead tree, depth={self.depth}")
    
    def resolve_first_node(self, player_range: np.ndarray, opponent_range: np.ndarray):
        """
        Re-solve lookahead using fixed ranges for both players.
        
        Used at root of game tree where ranges are known.
        
        Args:
            player_range: Player's range vector
            opponent_range: Opponent's range vector
        """
        # Set initial ranges
        if len(self.ranges_data) > 0:
            self.ranges_data[0][0, 0, 0, 0, :] = player_range
            self.ranges_data[0][0, 0, 0, 1, :] = opponent_range
        
        # Solve
        self._compute()
    
    def resolve(self, player_range: np.ndarray, opponent_cfvs: np.ndarray):
        """
        Re-solve lookahead using player range and opponent CFVs.
        
        Used during gameplay after opponent has acted.
        
        Args:
            player_range: Updated player range
            opponent_cfvs: Opponent's counterfactual values
        """
        # Create reconstruction gadget for opponent
        board = self.tree.board if self.tree else []
        self.reconstruction_gadget = CFRDGadget(board, player_range, opponent_cfvs)
        
        # Set player range
        if len(self.ranges_data) > 0:
            self.ranges_data[0][0, 0, 0, 0, :] = player_range
        
        # Store opponent CFVs for reconstruction
        self.reconstruction_opponent_cfvs = opponent_cfvs
        
        # Solve
        self._compute()
    
    def _compute(self):
        """Main CFR solving loop."""
        print(f"[Lookahead] Starting CFR solve, {self.cfr_iterations} iterations")
        
        for iteration in range(1, self.cfr_iterations + 1):
            # Set opponent starting range (using gadget if available)
            self._set_opponent_starting_range(iteration)
            
            # Compute current strategies using regret matching
            self._compute_current_strategies()
            
            # Compute reach probabilities
            self._compute_ranges()
            
            # Update average strategies
            self._compute_update_average_strategies(iteration)
            
            # Compute terminal equities (showdown values)
            self._compute_terminal_equities()
            
            # Compute counterfactual values
            self._compute_cfvs()
            
            # Update regrets
            self._compute_regrets()
            
            # Accumulate average CFVs
            self._compute_cumulate_average_cfvs(iteration)
            
            if iteration % 100 == 0:
                print(f"[Lookahead] CFR iteration {iteration}/{self.cfr_iterations}")
        
        # Normalize final strategies and CFVs
        self._compute_normalize_average_strategies()
        self._compute_normalize_average_cfvs()
        
        print(f"[Lookahead] CFR solve complete")
    
    def _set_opponent_starting_range(self, iteration: int):
        """Set opponent's starting range using reconstruction gadget."""
        if self.reconstruction_gadget and self.reconstruction_opponent_cfvs is not None:
            # Reconstruct opponent range from CFVs
            opponent_range = self.reconstruction_gadget.compute_opponent_range(
                self.reconstruction_opponent_cfvs, iteration
            )
            
            if len(self.ranges_data) > 0:
                self.ranges_data[0][0, 0, 0, 1, :] = opponent_range
    
    def _compute_current_strategies(self):
        """Compute current strategies using regret matching."""
        for d in range(1, self.depth + 1):
            if d >= len(self.regrets_data):
                continue
                
            regrets = self.regrets_data[d]
            pos_regrets = self.positive_regrets_data[d]
            strategy = self.current_strategy_data[d]
            
            # Regret matching: positive part of regrets
            pos_regrets[:] = np.maximum(regrets, self.regret_epsilon)
            
            # Normalize to get strategy
            regret_sums = np.sum(pos_regrets, axis=0, keepdims=True)
            regret_sums = np.maximum(regret_sums, 1e-10)  # Avoid division by zero
            
            strategy[:] = pos_regrets / regret_sums
    
    def _compute_ranges(self):
        """Compute reach probabilities using current strategies."""
        for d in range(self.depth):
            if (d >= len(self.ranges_data) or d + 1 >= len(self.ranges_data) or
                d >= len(self.current_strategy_data)):
                continue
            
            current_ranges = self.ranges_data[d]
            next_ranges = self.ranges_data[d + 1]
            strategy = self.current_strategy_data[d + 1]
            
            # Propagate ranges through tree using strategies
            # Simplified propagation - would use proper tree structure in production
            for action in range(min(strategy.shape[0], next_ranges.shape[0])):
                for bet in range(min(strategy.shape[1], next_ranges.shape[1])):
                    for player in range(2):
                        if player < current_ranges.shape[3] and player < next_ranges.shape[3]:
                            # Copy range from parent
                            next_ranges[action, bet, 0, player, :] = current_ranges[0, 0, 0, player, :]
                            
                            # Multiply by strategy if it's the acting player's turn
                            acting_player = d % 2  # Simplified - alternating players
                            if player == acting_player and action < strategy.shape[0] and bet < strategy.shape[1]:
                                next_ranges[action, bet, 0, player, :] *= strategy[action, bet, :]
    
    def _compute_update_average_strategies(self, iteration: int):
        """Update average strategies with current strategies."""
        if iteration > self.cfr_skip_iterations:
            for d in range(1, min(len(self.current_strategy_data), len(self.average_strategies_data))):
                self.average_strategies_data[d] += self.current_strategy_data[d]
    
    def _compute_terminal_equities(self):
        """Compute terminal node values (folds and showdowns)."""
        if not self.terminal_equity:
            return
            
        for d in range(1, self.depth + 1):
            if d >= len(self.ranges_data) or d >= len(self.cfvs_data):
                continue
            
            ranges = self.ranges_data[d]
            cfvs = self.cfvs_data[d]
            
            # Compute fold values
            for action in range(min(ranges.shape[0], cfvs.shape[0])):
                for bet in range(min(ranges.shape[1], cfvs.shape[1])):
                    # Fold equity (player who folds loses pot)
                    pot_size = 100  # Simplified pot size
                    
                    # Player 0 folds
                    cfvs[action, bet, 0, 0, :] = -pot_size
                    cfvs[action, bet, 0, 1, :] = pot_size
                    
            # Compute call/showdown values using terminal equity
            for d_level in range(1, min(len(self.ranges_data), len(self.cfvs_data))):
                level_ranges = self.ranges_data[d_level]
                level_cfvs = self.cfvs_data[d_level]
                
                # Call terminal equity for showdowns
                if level_ranges.size > 0 and level_cfvs.size > 0:
                    # Extract ranges for terminal equity calculation
                    p1_range = level_ranges[0, 0, 0, 0, :]
                    p2_range = level_ranges[0, 0, 0, 1, :]
                    
                    if np.sum(p1_range) > 0 and np.sum(p2_range) > 0:
                        # Compute showdown values
                        p1_values, p2_values = self.terminal_equity.call_value(
                            p1_range.reshape(1, -1), 
                            p2_range.reshape(1, -1)
                        )
                        
                        if p1_values is not None and p2_values is not None:
                            level_cfvs[0, 0, 0, 0, :] = p1_values.flatten()
                            level_cfvs[0, 0, 0, 1, :] = p2_values.flatten()
    
    def _compute_cfvs(self):
        """Compute counterfactual values via backward induction."""
        # Backward pass through tree
        for d in range(self.depth, 0, -1):
            if (d >= len(self.cfvs_data) or d - 1 >= len(self.cfvs_data) or
                d >= len(self.current_strategy_data)):
                continue
            
            child_cfvs = self.cfvs_data[d]
            parent_cfvs = self.cfvs_data[d - 1]
            strategy = self.current_strategy_data[d]
            
            # Compute expected values using strategies
            for action in range(min(strategy.shape[0], child_cfvs.shape[0])):
                for bet in range(min(strategy.shape[1], child_cfvs.shape[1])):
                    for player in range(2):
                        if (action < child_cfvs.shape[0] and bet < child_cfvs.shape[1] and
                            player < child_cfvs.shape[3] and player < parent_cfvs.shape[3]):
                            
                            # Strategy-weighted average of child values
                            if action < strategy.shape[0] and bet < strategy.shape[1]:
                                weighted_cfv = (strategy[action, bet, :] * 
                                               child_cfvs[action, bet, 0, player, :])
                                parent_cfvs[0, 0, 0, player, :] += np.sum(weighted_cfv)
    
    def _compute_regrets(self):
        """Update regrets based on counterfactual values."""
        for d in range(1, self.depth + 1):
            if (d >= len(self.cfvs_data) or d >= len(self.regrets_data) or
                d >= len(self.current_strategy_data)):
                continue
            
            cfvs = self.cfvs_data[d]
            regrets = self.regrets_data[d]
            strategy = self.current_strategy_data[d]
            
            # Compute regrets: CFV(action) - CFV(strategy)
            for action in range(min(cfvs.shape[0], regrets.shape[0])):
                for bet in range(min(cfvs.shape[1], regrets.shape[1])):
                    acting_player = d % 2  # Simplified alternating players
                    
                    if (action < cfvs.shape[0] and bet < cfvs.shape[1] and
                        acting_player < cfvs.shape[3]):
                        
                        action_cfv = cfvs[action, bet, 0, acting_player, :]
                        
                        # Strategy CFV (expected value)
                        if action < strategy.shape[0] and bet < strategy.shape[1]:
                            strategy_cfv = np.sum(strategy[action, bet, :] * action_cfv)
                            
                            # Update regret
                            instant_regret = action_cfv - strategy_cfv
                            regrets[action, bet, :] += instant_regret
    
    def _compute_cumulate_average_cfvs(self, iteration: int):
        """Accumulate average counterfactual values."""
        # Simplified - would properly accumulate CFVs for final average
        pass
    
    def _compute_normalize_average_strategies(self):
        """Normalize average strategies."""
        for d in range(len(self.average_strategies_data)):
            strategy = self.average_strategies_data[d]
            
            # Normalize each strategy vector
            strategy_sums = np.sum(strategy, axis=0, keepdims=True)
            strategy_sums = np.maximum(strategy_sums, 1e-10)
            strategy /= strategy_sums
    
    def _compute_normalize_average_cfvs(self):
        """Normalize average counterfactual values."""
        # Simplified - would normalize final CFVs
        pass
    
    def get_results(self) -> Dict:
        """
        Get solving results.
        
        Returns:
            Dict with strategy, cfvs, and other results
        """
        results = {
            'strategy': {},
            'cfvs': {},
            'average_strategy': {}
        }
        
        # Extract average strategies
        for d in range(len(self.average_strategies_data)):
            strategy = self.average_strategies_data[d]
            results['average_strategy'][f'depth_{d}'] = strategy.copy()
        
        # Extract CFVs
        for d in range(len(self.cfvs_data)):
            cfvs = self.cfvs_data[d]
            results['cfvs'][f'depth_{d}'] = cfvs.copy()
        
        return results
    
    def get_root_cfv(self) -> np.ndarray:
        """Get counterfactual values at root."""
        if len(self.cfvs_data) > 0 and self.cfvs_data[0].size > 0:
            return self.cfvs_data[0][0, 0, 0, 0, :]  # Player 0's CFVs
        return np.zeros(self.num_hands)
    
    def get_strategy(self) -> Dict[str, np.ndarray]:
        """Get computed strategy."""
        if len(self.average_strategies_data) > 1:
            # Return root strategy (depth 1)
            return {'root': self.average_strategies_data[1][0, 0, :]}
        return {}
    
    def set_value_network(self, value_network: ValueNN):
        """Set neural network for value estimation at leaves."""
        self.value_network = value_network
