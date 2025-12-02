"""
Tests for CFR Solver.

Tests the CFR algorithm on known, simple scenarios.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
from poker_bot.core.data_models import (
    GameState, Card, PlayerState, BettingRound, ActionType
)
from poker_bot.decision.cfr_solver import CFRSolver, SearchNode, NodeType


class TestCFRSolver:
    """Test CFRSolver class."""
    
    def test_initialization(self):
        """Test solver initialization."""
        solver = CFRSolver(num_buckets=10, lookahead_depth=2)
        
        assert solver.num_buckets == 10
        assert solver.lookahead_depth == 2
        assert solver.use_cfr_plus == True
    
    def test_solve_basic(self):
        """Test basic solve functionality."""
        solver = CFRSolver(num_buckets=10, lookahead_depth=2)
        
        # Create simple game state
        game_state = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[],
            pot_size=100,
            current_bet=20,
            player=PlayerState(stack=1000),
            opponent=PlayerState(stack=1000),
            street=BettingRound.PREFLOP,
            available_actions=[ActionType.FOLD, ActionType.CALL, ActionType.RAISE]
        )
        
        # Solve with small number of iterations
        action = solver.solve(game_state, iterations=100)
        
        # Should return a valid action
        assert action.action_type in [ActionType.FOLD, ActionType.CHECK, 
                                       ActionType.CALL, ActionType.RAISE]
    
    def test_get_actions(self):
        """Test action generation."""
        solver = CFRSolver(num_buckets=10)
        
        game_state = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[],
            pot_size=100,
            current_bet=20,
            street=BettingRound.PREFLOP
        )
        
        actions = solver._get_actions(game_state)
        
        # Should have fold, call, and raise options
        assert 'fold' in actions
        assert 'call' in actions
        assert any(a.startswith('raise') for a in actions)
    
    def test_build_tree(self):
        """Test tree building."""
        solver = CFRSolver(num_buckets=10, lookahead_depth=2)
        
        game_state = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[],
            pot_size=100,
            current_bet=20,
            street=BettingRound.PREFLOP
        )
        
        tree = solver._build_tree(game_state)
        
        # Root should be player node
        assert tree.node_type == NodeType.PLAYER
        assert len(tree.actions) > 0
        
        # Should have regret arrays
        assert tree.regret_sum is not None
        assert tree.strategy_sum is not None
    
    def test_regret_matching(self):
        """Test regret matching strategy computation."""
        solver = CFRSolver(num_buckets=5)
        
        # Create node with known regrets
        node = SearchNode(
            node_type=NodeType.PLAYER,
            actions=['fold', 'call', 'raise']
        )
        node.regret_sum = np.array([
            [0.0, 1.0, 2.0],  # Hand 0: prefers raise
            [2.0, 1.0, 0.0],  # Hand 1: prefers fold
            [1.0, 1.0, 1.0],  # Hand 2: indifferent
            [0.0, 0.0, 0.0],  # Hand 3: no regret yet
            [-1.0, 1.0, 0.5], # Hand 4: negative regret for fold
        ])
        
        strategy = solver._get_strategy(node)
        
        # Strategy should be probability distribution per hand
        assert strategy.shape == (5, 3)
        
        # Each row should sum to ~1
        row_sums = strategy.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5), decimal=5)
    
    def test_reset(self):
        """Test solver reset."""
        solver = CFRSolver(num_buckets=10)
        
        # Set some state
        solver.player_range = np.ones(10) / 10
        solver.root = SearchNode(node_type=NodeType.PLAYER)
        
        # Reset
        solver.reset()
        
        # Should be cleared
        assert solver.root is None
        np.testing.assert_array_almost_equal(
            solver.player_range, 
            np.ones(10) / 10
        )
    
    def test_time_limited_solve(self):
        """Test time-limited solving."""
        solver = CFRSolver(num_buckets=10, lookahead_depth=2)
        
        game_state = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[],
            pot_size=100,
            current_bet=20,
            street=BettingRound.PREFLOP
        )
        
        # Solve with time limit
        import time
        start = time.time()
        action = solver.solve(game_state, iterations=10000, time_limit_ms=100)
        elapsed_ms = (time.time() - start) * 1000
        
        # Should complete within time limit (with some margin)
        assert elapsed_ms < 500  # Allow for overhead
        
        # Should still return valid action
        assert action is not None


class TestSearchNode:
    """Test SearchNode class."""
    
    def test_initialization(self):
        """Test node initialization."""
        node = SearchNode(
            node_type=NodeType.PLAYER,
            player=0,
            pot=100.0,
            actions=['fold', 'call']
        )
        
        assert node.node_type == NodeType.PLAYER
        assert node.player == 0
        assert node.pot == 100.0
        assert len(node.actions) == 2
    
    def test_get_average_strategy(self):
        """Test average strategy computation."""
        node = SearchNode(
            node_type=NodeType.PLAYER,
            actions=['fold', 'call', 'raise']
        )
        
        # Set strategy sum
        node.strategy_sum = np.array([
            [10.0, 20.0, 70.0],  # 10%, 20%, 70%
            [50.0, 50.0, 0.0],   # 50%, 50%, 0%
        ])
        
        avg_strategy = node.get_average_strategy(num_hands=2)
        
        # Check normalization
        np.testing.assert_array_almost_equal(
            avg_strategy[0], 
            [0.1, 0.2, 0.7], 
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            avg_strategy[1], 
            [0.5, 0.5, 0.0], 
            decimal=5
        )
    
    def test_get_average_strategy_uniform(self):
        """Test uniform strategy when no data."""
        node = SearchNode(
            node_type=NodeType.PLAYER,
            actions=['fold', 'call', 'raise']
        )
        
        # No strategy sum set
        node.strategy_sum = None
        
        avg_strategy = node.get_average_strategy(num_hands=2)
        
        # Should be uniform
        expected = 1.0 / 3.0
        np.testing.assert_array_almost_equal(
            avg_strategy,
            np.full((2, 3), expected),
            decimal=5
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
