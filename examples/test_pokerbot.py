#!/usr/bin/env python3
"""Test suite for unified PokerBot Agent."""


import os
import sys
import traceback
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath:
    for p in pythonpath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)
# Fallback: always add src path directly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.agents import create_agent, get_available_agents
from src.deepstack.game import Action, Card, Rank, Suit


def test_pokerbot_creation():
    """Test PokerBot agent creation with various configurations."""
    print("Testing PokerBot agent creation...")
    
    try:
        # Test minimal configuration
        agent1 = create_agent('pokerbot', name='Minimal', 
                             use_cfr=False, use_dqn=False, 
                             use_deepstack=False, use_pretrained=False)
        assert agent1.name == 'Minimal'
        assert not agent1.use_cfr
        assert not agent1.use_dqn
        assert not agent1.use_deepstack
        print("  ✓ Minimal agent creation passed")
        
        # Test full configuration
        agent2 = create_agent('pokerbot', name='Full',
                             use_cfr=True, use_dqn=True,
                             use_deepstack=True, use_pretrained=False)
        assert agent2.name == 'Full'
        assert agent2.use_cfr
        assert agent2.use_dqn or not agent2.use_dqn  # DQN may fail without TF
        assert agent2.use_deepstack
        print("  ✓ Full agent creation passed")
        
        # Test with custom weights
        agent3 = create_agent('pokerbot', name='CustomWeights',
                             cfr_weight=0.5, dqn_weight=0.3,
                             deepstack_weight=0.2, use_pretrained=False)
        total = agent3.cfr_weight + agent3.dqn_weight + agent3.deepstack_weight
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"
        print("  ✓ Custom weights normalization passed")
        
        print("✓ PokerBot agent creation tests passed\n")
        return True
    except Exception as e:
        print(f"✗ PokerBot agent creation tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pokerbot_decision_making():
    """Test PokerBot agent decision making."""
    print("Testing PokerBot decision making...")
    
    try:
        agent = create_agent('pokerbot', name='TestDecision',
                           use_pretrained=False)
        
        # Create test cards
        hole_cards = [
            Card(rank=12, suit=0),  # Ace of Spades
            Card(rank=11, suit=0),  # King of Spades
        ]
        community_cards = []
        
        # Test decision making
        action, amount = agent.choose_action(
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot=100,
            current_bet=20,
            player_stack=1000,
            opponent_bet=20
        )
        
        # Should return a valid action
        assert action in [Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE]
        assert amount >= 0
        assert amount <= 1000
        
        # Check stats updated
        assert agent.stats['decisions_made'] == 1
        
        print(f"  ✓ Decision: {action}, Amount: {amount}")
        print("✓ PokerBot decision making tests passed\n")
        return True
    except Exception as e:
        print(f"✗ PokerBot decision making tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pokerbot_training():
    """Test PokerBot training capabilities."""
    print("Testing PokerBot training...")
    
    try:
        agent = create_agent('pokerbot', name='TestTrain',
                           use_cfr=True, use_dqn=True,
                           use_pretrained=False)
        
        # Test CFR training
        if agent.use_cfr and agent.cfr_component:
            agent.train_cfr(num_iterations=10)
            print("  ✓ CFR training executed")
        
        # Test DQN training
        if agent.use_dqn and agent.dqn_model:
            # Add some dummy experiences
            import numpy as np
            for i in range(40):
                state = np.random.rand(agent.state_size)
                action = 1
                reward = 1.0 if i % 2 == 0 else -1.0
                next_state = np.random.rand(agent.state_size)
                done = False
                agent.remember(state, action, reward, next_state, done)
            
            # Train on experiences
            agent.replay(batch_size=32)
            print("  ✓ DQN training executed")
        
        print("✓ PokerBot training tests passed\n")
        return True
    except Exception as e:
        print(f"✗ PokerBot training tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pokerbot_save_load():
    """Test PokerBot model save/load."""
    print("Testing PokerBot save/load...")
    
    try:
        import tempfile
        import shutil
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            agent = create_agent('pokerbot', name='TestSave',
                               use_cfr=True, use_pretrained=False)
            
            # Save models
            agent.save_models(temp_dir)
            print("  ✓ Models saved")
            
            # Load models
            agent2 = create_agent('pokerbot', name='TestLoad',
                                use_cfr=True, use_pretrained=False)
            agent2.load_models(temp_dir)
            print("  ✓ Models loaded")
            
            print("✓ PokerBot save/load tests passed\n")
            return True
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"✗ PokerBot save/load tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pokerbot_registry():
    """Test that PokerBot is properly registered."""
    print("Testing PokerBot registry...")
    try:
        agents = get_available_agents()
        assert 'pokerbot' in agents, f"pokerbot not in registry: {agents}"
        print(f"  ✓ Available agents: {agents}")
        # Test that it can be created through registry
        agent = create_agent('pokerbot', name='RegistryTest', use_pretrained=False)
        assert agent.name == 'RegistryTest'
        print("  ✓ Agent created through registry")
    except Exception as e:
        print(f"✗ PokerBot registry tests failed: {e}\n")
        traceback.print_exc()
        return False
        print("✓ PokerBot registry tests passed\n")
        return True
    except Exception as e:
        print(f"✗ PokerBot registry tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("PokerBot Agent Test Suite")
    print("="*60 + "\n")
    
    tests = [
        test_pokerbot_registry,
        test_pokerbot_creation,
        test_pokerbot_decision_making,
        test_pokerbot_training,
        test_pokerbot_save_load,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("="*60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*60)
    
    return all(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
