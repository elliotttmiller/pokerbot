"""
Test suite for DeepStack core modules.

Tests the ported DeepStack components to ensure they work correctly
and produce reasonable results.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deepstack.core.tree_builder import PokerTreeBuilder, PokerTreeNode, PLAYERS
from deepstack.core.tree_cfr import TreeCFR
from deepstack.core.terminal_equity import TerminalEquity
from deepstack.core.cfrd_gadget import CFRDGadget
from deepstack.core.resolving import Resolving


def test_tree_builder():
    """Test that tree builder creates valid game trees."""
    print("\n=== Testing TreeBuilder ===")
    
    builder = PokerTreeBuilder(game_variant='leduc')
    
    # Build a simple tree
    params = {
        'street': 0,
        'bets': [20, 20],
        'current_player': PLAYERS['P1'],
        'board': [],
        'limit_to_street': True,
        'bet_sizing': [1.0]  # Pot-sized bets
    }
    
    root = builder.build_tree(params)
    
    # Verify root node
    assert root is not None, "Root node should exist"
    assert root.street == 0, "Street should be 0"
    assert root.bets == [20, 20], "Bets should match"
    assert len(root.children) > 0, "Root should have children"
    
    # Check that actions are present
    assert len(root.actions) > 0, "Root should have actions"
    assert any('fold' in str(a).lower() or 'call' in str(a).lower() or 
               'raise' in str(a).lower() for a in root.actions), \
        "Actions should include fold/call/raise"
    
    print(f"✓ Tree built successfully")
    print(f"  - Root node type: {root.node_type}")
    print(f"  - Number of children: {len(root.children)}")
    print(f"  - Actions: {root.actions}")
    
    # Count total nodes
    def count_nodes(node):
        if not node.children:
            return 1
        return 1 + sum(count_nodes(child) for child in node.children)
    
    total_nodes = count_nodes(root)
    print(f"  - Total nodes in tree: {total_nodes}")
    
    return root


def test_terminal_equity():
    """Test terminal equity calculation."""
    print("\n=== Testing TerminalEquity ===")
    
    equity = TerminalEquity(game_variant='leduc', num_hands=6)
    
    # Set board (e.g., King of Spades)
    board = ['K']
    equity.set_board(board)
    
    # Get call matrix
    call_matrix = equity.get_call_matrix()
    
    assert call_matrix is not None, "Call matrix should exist"
    assert call_matrix.shape == (6, 6), f"Call matrix shape should be (6,6), got {call_matrix.shape}"
    
    # Check that equity is between 0 and 1
    assert np.all(call_matrix >= 0) and np.all(call_matrix <= 1), \
        "Equity values should be between 0 and 1"
    
    print(f"✓ Terminal equity computed successfully")
    print(f"  - Call matrix shape: {call_matrix.shape}")
    print(f"  - Sample equities:")
    for i in range(min(3, 6)):
        for j in range(min(3, 6)):
            print(f"    Hand {i} vs Hand {j}: {call_matrix[i][j]:.3f}")
    
    # Test expected value computation
    player_range = np.ones(6) / 6  # Uniform range
    opponent_range = np.ones(6) / 6
    
    ev = equity.compute_expected_value(player_range, opponent_range, pot_size=100)
    
    print(f"  - Expected value (uniform ranges): {ev.mean():.2f}")
    
    return equity


def test_cfrd_gadget():
    """Test CFRDGadget for range reconstruction."""
    print("\n=== Testing CFRDGadget ===")
    
    num_hands = 6
    board = ['K']
    
    player_range = np.ones(num_hands) / num_hands
    opponent_cfvs = np.array([10, 20, 30, 15, 25, 35])  # Sample CFVs
    
    gadget = CFRDGadget(board, player_range, opponent_cfvs)
    
    # Compute opponent range
    opponent_range = gadget.compute_opponent_range(opponent_cfvs, 0)
    
    assert opponent_range is not None, "Opponent range should be computed"
    assert len(opponent_range) == num_hands, f"Range length should be {num_hands}"
    assert np.abs(opponent_range.sum() - 1.0) < 0.01, \
        f"Range should sum to 1.0, got {opponent_range.sum()}"
    assert np.all(opponent_range >= 0) and np.all(opponent_range <= 1), \
        "Range probabilities should be between 0 and 1"
    
    print(f"✓ CFRDGadget computed opponent range successfully")
    print(f"  - Input CFVs: {opponent_cfvs}")
    print(f"  - Output range: {opponent_range}")
    print(f"  - Range sum: {opponent_range.sum():.4f}")
    
    return gadget


def test_tree_cfr():
    """Test TreeCFR solver."""
    print("\n=== Testing TreeCFR ===")
    
    # Build a simple tree
    builder = PokerTreeBuilder(game_variant='leduc')
    params = {
        'street': 0,
        'bets': [20, 20],
        'current_player': PLAYERS['P1'],
        'board': [],
        'limit_to_street': True,
        'bet_sizing': [1.0]
    }
    root = builder.build_tree(params)
    
    # Run CFR
    cfr = TreeCFR(skip_iterations=100)
    num_hands = 6
    starting_ranges = np.array([
        np.ones(num_hands) / num_hands,
        np.ones(num_hands) / num_hands
    ])
    
    print(f"  - Running CFR with 500 iterations...")
    result = cfr.run_cfr(root, starting_ranges, iter_count=500)
    
    assert result is not None, "CFR should return result"
    assert 'strategy' in result, "Result should contain strategy"
    assert 'regrets' in result, "Result should contain regrets"
    
    print(f"✓ CFR completed successfully")
    print(f"  - Number of iterations: {result.get('iterations', 0)}")
    print(f"  - Number of nodes with strategies: {len(result['strategy'])}")
    
    # Check strategy at root
    if 'root' in result['strategy']:
        root_strategy = result['strategy']['root']
        print(f"  - Root strategy: {root_strategy}")
        print(f"  - Strategy sum: {root_strategy.sum():.4f}")
    
    return result


def test_resolving():
    """Test Resolving API."""
    print("\n=== Testing Resolving ===")
    
    num_hands = 6
    resolving = Resolving(num_hands=num_hands, game_variant='leduc')
    
    # Test resolve_first_node
    node_params = {
        'street': 0,
        'bets': [20, 20],
        'current_player': PLAYERS['P1'],
        'board': [],
        'bet_sizing': [1.0]
    }
    
    player_range = np.ones(num_hands) / num_hands
    opponent_range = np.ones(num_hands) / num_hands
    
    print(f"  - Running resolve_first_node with 200 iterations...")
    resolving.resolve_first_node(node_params, player_range, opponent_range, 
                                  iterations=200)
    
    # Check results
    actions = resolving.get_possible_actions()
    assert len(actions) > 0, "Should have possible actions"
    
    print(f"✓ Resolving completed successfully")
    print(f"  - Possible actions: {actions}")
    
    # Get action strategies
    for action in actions[:3]:  # Show first 3 actions
        prob = resolving.get_action_strategy(action)
        print(f"  - P({action}) = {prob:.4f}")
    
    # Get root CFVs
    root_cfv = resolving.get_root_cfv()
    print(f"  - Root CFV shape: {root_cfv.shape}")
    print(f"  - Root CFV mean: {root_cfv.mean():.2f}")
    
    return resolving


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("DeepStack Core Module Tests")
    print("="*60)
    
    try:
        # Run tests
        test_tree_builder()
        test_terminal_equity()
        test_cfrd_gadget()
        test_tree_cfr()
        test_resolving()
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
