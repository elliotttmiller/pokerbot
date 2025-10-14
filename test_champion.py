#!/usr/bin/env python3
"""Test suite for Champion Agent."""

import sys

from src.agents import ChampionAgent, CFRAgent, DQNAgent
from src.game import Action, Card, Rank, Suit


def test_champion_agent_creation():
    """Test champion agent initialization."""
    print("Testing champion agent creation...")
    try:
        # Test with pre-trained models
        agent1 = ChampionAgent(name="Test1", use_pretrained=True)
        assert agent1.name == "Test1"
        assert agent1.cfr is not None
        assert agent1.model_loader is not None
        assert agent1.data_manager is not None
        
        # Test without pre-trained models
        agent2 = ChampionAgent(name="Test2", use_pretrained=False)
        assert agent2.name == "Test2"
        assert agent2.cfr is not None
        assert agent2.model_loader is None
        assert agent2.data_manager is None
        
        print("✓ Champion agent creation tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Champion agent creation tests failed: {e}\n")
        return False


def test_champion_agent_weights():
    """Test ensemble weight normalization."""
    print("Testing ensemble weight normalization...")
    try:
        # Test custom weights
        agent = ChampionAgent(
            name="TestWeights",
            use_pretrained=False,
            cfr_weight=0.5,
            dqn_weight=0.3,
            equity_weight=0.2
        )
        
        # Weights should sum to 1.0
        total = agent.cfr_weight + agent.dqn_weight + agent.equity_weight
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"
        
        # Test with unbalanced weights
        agent2 = ChampionAgent(
            name="TestWeights2",
            use_pretrained=False,
            cfr_weight=2.0,
            dqn_weight=1.0,
            equity_weight=1.0
        )
        
        total2 = agent2.cfr_weight + agent2.dqn_weight + agent2.equity_weight
        assert abs(total2 - 1.0) < 0.001, f"Weights sum to {total2}, expected 1.0"
        
        print("✓ Ensemble weight normalization tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Ensemble weight normalization tests failed: {e}\n")
        return False


def test_champion_agent_action_selection():
    """Test champion agent action selection."""
    print("Testing champion agent action selection...")
    try:
        agent = ChampionAgent(name="TestActions", use_pretrained=True)
        
        # Test premium hand preflop - should be aggressive
        hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)]
        community_cards = []
        action, raise_amt = agent.choose_action(
            hole_cards, community_cards, 100, 20, 1000, 20
        )
        assert action in [Action.CALL, Action.RAISE], f"Expected CALL or RAISE, got {action}"
        
        # Test weak hand with very high bet - ensemble should tend to fold
        hole_cards = [Card(Rank.TWO, Suit.CLUBS), Card(Rank.SEVEN, Suit.DIAMONDS)]
        community_cards = []
        action, raise_amt = agent.choose_action(
            hole_cards, community_cards, 100, 200, 1000, 200
        )
        # With very high bet (200 vs 100 pot), weak hand should fold or possibly call
        # Note: Ensemble might occasionally call due to CFR exploration
        assert action in [Action.FOLD, Action.CALL], \
            f"Expected FOLD or CALL for weak hand with very high bet, got {action}"
        
        # Test post-flop decision - any reasonable action is acceptable
        hole_cards = [Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS)]
        community_cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.FOUR, Suit.CLUBS)
        ]
        action, raise_amt = agent.choose_action(
            hole_cards, community_cards, 300, 50, 900, 50
        )
        assert action in [Action.FOLD, Action.CALL, Action.RAISE]
        
        print("✓ Champion agent action selection tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Champion agent action selection tests failed: {e}\n")
        return False


def test_champion_agent_hand_notation():
    """Test hand notation conversion."""
    print("Testing hand notation conversion...")
    try:
        agent = ChampionAgent(name="TestNotation", use_pretrained=False)
        
        # Test pocket pair
        hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)]
        notation = agent._get_hand_notation(hole_cards)
        assert notation == "AA", f"Expected 'AA', got '{notation}'"
        
        # Test suited hand (lower rank first)
        hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)]
        notation = agent._get_hand_notation(hole_cards)
        assert notation == "KAS", f"Expected 'KAS', got '{notation}'"
        
        # Test offsuit hand (lower rank first)
        hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        notation = agent._get_hand_notation(hole_cards)
        assert notation == "KAO", f"Expected 'KAO', got '{notation}'"
        
        # Test order (lower rank first)
        hole_cards = [Card(Rank.FIVE, Suit.HEARTS), Card(Rank.ACE, Suit.SPADES)]
        notation = agent._get_hand_notation(hole_cards)
        assert notation == "5AO", f"Expected '5AO', got '{notation}'"
        
        # Test 7-2 offsuit (worst hand)
        hole_cards = [Card(Rank.SEVEN, Suit.CLUBS), Card(Rank.TWO, Suit.DIAMONDS)]
        notation = agent._get_hand_notation(hole_cards)
        assert notation == "27O", f"Expected '27O', got '{notation}'"
        
        print("✓ Hand notation conversion tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Hand notation conversion tests failed: {e}\n")
        return False


def test_champion_agent_equity_decisions():
    """Test equity-based decision making."""
    print("Testing equity-based decision making...")
    try:
        agent = ChampionAgent(name="TestEquity", use_pretrained=True)
        
        # Test premium hand (should raise)
        hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)]
        action, raise_amt = agent._get_equity_based_action(
            hole_cards, [], 100, 20, 1000
        )
        assert action == Action.RAISE, f"Expected RAISE for AA, got {action}"
        assert raise_amt > 0, "Expected positive raise amount"
        
        # Test weak hand (should fold to bet)
        hole_cards = [Card(Rank.TWO, Suit.CLUBS), Card(Rank.SEVEN, Suit.DIAMONDS)]
        action, raise_amt = agent._get_equity_based_action(
            hole_cards, [], 100, 50, 1000
        )
        assert action == Action.FOLD, f"Expected FOLD for 72o, got {action}"
        
        # Test post-flop (should return None)
        hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        community_cards = [Card(Rank.ACE, Suit.HEARTS)]
        action, raise_amt = agent._get_equity_based_action(
            hole_cards, community_cards, 100, 20, 1000
        )
        assert action is None, "Expected None for post-flop equity decision"
        
        print("✓ Equity-based decision making tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Equity-based decision making tests failed: {e}\n")
        return False


def test_champion_agent_save_load():
    """Test saving and loading strategies."""
    print("Testing save/load functionality...")
    try:
        # Create and train agent
        agent1 = ChampionAgent(name="SaveTest", use_pretrained=False)
        agent1.cfr.iterations = 1000  # Simulate training
        
        # Save strategy
        agent1.save_strategy("/tmp/test_champion")
        
        # Create new agent and load
        agent2 = ChampionAgent(name="LoadTest", use_pretrained=False)
        agent2.load_strategy("/tmp/test_champion")
        
        # Verify loaded correctly
        assert agent2.cfr.iterations == 1000, "CFR iterations not loaded correctly"
        
        print("✓ Save/load functionality tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Save/load functionality tests failed: {e}\n")
        return False


def test_champion_agent_training_mode():
    """Test training mode setting."""
    print("Testing training mode...")
    try:
        agent = ChampionAgent(name="TrainTest", use_pretrained=False)
        
        # Initially in training mode
        assert agent.training_mode == True
        assert agent.epsilon > 0
        
        # Set to evaluation mode
        agent.set_training_mode(False)
        assert agent.training_mode == False
        assert agent.epsilon == 0.0
        
        # Set back to training mode
        agent.set_training_mode(True)
        assert agent.training_mode == True
        
        print("✓ Training mode tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Training mode tests failed: {e}\n")
        return False


def test_champion_integration():
    """Test that champion agent integrates CFR and DQN properly."""
    print("Testing CFR and DQN integration...")
    try:
        agent = ChampionAgent(name="IntegrationTest", use_pretrained=False)
        
        # Verify CFR component exists
        assert isinstance(agent.cfr, CFRAgent)
        assert agent.cfr.name == "IntegrationTest_CFR"
        
        # Verify DQN components exist
        assert agent.state_size == 60
        assert agent.action_size == 3
        assert agent.memory is not None
        
        # Verify ensemble weights
        assert agent.cfr_weight > 0
        assert agent.dqn_weight > 0
        assert agent.equity_weight > 0
        
        print("✓ CFR and DQN integration tests passed\n")
        return True
    except Exception as e:
        print(f"✗ CFR and DQN integration tests failed: {e}\n")
        return False


def run_all_tests():
    """Run all champion agent tests."""
    print("\n" + "="*70)
    print("CHAMPION AGENT TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        test_champion_agent_creation,
        test_champion_agent_weights,
        test_champion_agent_action_selection,
        test_champion_agent_hand_notation,
        test_champion_agent_equity_decisions,
        test_champion_agent_save_load,
        test_champion_agent_training_mode,
        test_champion_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    if failed == 0:
        print("✓ All champion agent tests passed!")
        return 0
    else:
        print(f"✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
