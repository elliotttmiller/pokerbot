#!/usr/bin/env python3
"""
Quick validation script for PokerBot agent.
Tests basic functionality without requiring full training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import create_agent, get_available_agents
from deepstack.game import Card, Action


def test_agent_registry():
    """Test that PokerBot is available."""
    print("="*60)
    print("Test 1: Agent Registry")
    print("="*60)
    agents = get_available_agents()
    print(f"Available agents: {agents}")
    assert 'pokerbot' in agents, "PokerBot not in registry!"
    print("✓ PokerBot registered\n")


def test_agent_creation_variations():
    """Test different agent configurations."""
    print("="*60)
    print("Test 2: Agent Configuration Variations")
    print("="*60)
    
    configs = [
        {
            'name': 'Minimal',
            'use_cfr': False,
            'use_dqn': False,
            'use_deepstack': False,
            'use_pretrained': False
        },
        {
            'name': 'CFR-Only',
            'use_cfr': True,
            'use_dqn': False,
            'use_deepstack': False,
            'use_pretrained': False
        },
        {
            'name': 'Full',
            'use_cfr': True,
            'use_dqn': True,
            'use_deepstack': True,
            'use_pretrained': False
        }
    ]
    
    for i, config in enumerate(configs, 1):
        name = config['name']
        agent = create_agent('pokerbot', **config)
        print(f"{i}. {name}: CFR={agent.use_cfr}, DQN={agent.use_dqn}, "
              f"DeepStack={agent.use_deepstack}")
        assert agent.name == name
    
    print("✓ All configurations work\n")


def test_decision_making():
    """Test agent decision making with different hands."""
    print("="*60)
    print("Test 3: Decision Making")
    print("="*60)
    
    agent = create_agent('pokerbot', name='DecisionTest', use_pretrained=False)
    
    test_cases = [
        {
            'name': 'Strong Hand (AA)',
            'hole_cards': [Card(rank=12, suit=0), Card(rank=12, suit=1)],
            'community_cards': [],
            'pot': 100,
            'current_bet': 20,
            'player_stack': 1000,
            'opponent_bet': 20
        },
        {
            'name': 'Weak Hand (72o)',
            'hole_cards': [Card(rank=5, suit=0), Card(rank=0, suit=1)],
            'community_cards': [],
            'pot': 100,
            'current_bet': 50,
            'player_stack': 1000,
            'opponent_bet': 50
        },
        {
            'name': 'Post-Flop',
            'hole_cards': [Card(rank=11, suit=0), Card(rank=10, suit=0)],
            'community_cards': [
                Card(rank=12, suit=0),
                Card(rank=9, suit=1),
                Card(rank=8, suit=2)
            ],
            'pot': 200,
            'current_bet': 0,
            'player_stack': 800,
            'opponent_bet': 0
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        action, amount = agent.choose_action(**{k: v for k, v in test.items() if k != 'name'})
        print(f"{i}. {test['name']}: {action.name}, Amount: {amount}")
        assert action in [Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE]
        assert 0 <= amount <= test['player_stack']
    
    print("✓ Decision making works\n")


def test_ensemble_voting():
    """Test that ensemble decision making combines components."""
    print("="*60)
    print("Test 4: Ensemble Voting")
    print("="*60)
    
    agent = create_agent('pokerbot', 
                        name='EnsembleTest',
                        use_cfr=True,
                        use_dqn=True,
                        use_deepstack=True,
                        use_pretrained=False,
                        cfr_weight=0.5,
                        dqn_weight=0.3,
                        deepstack_weight=0.2)
    
    # Check weights are normalized
    total = agent.cfr_weight + agent.dqn_weight + agent.deepstack_weight
    assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, not 1.0"
    print(f"Weights normalized: CFR={agent.cfr_weight:.2f}, "
          f"DQN={agent.dqn_weight:.2f}, DeepStack={agent.deepstack_weight:.2f}")
    
    # Make a decision (should use multiple components)
    hole_cards = [Card(rank=11, suit=0), Card(rank=11, suit=1)]
    action, amount = agent.choose_action(
        hole_cards=hole_cards,
        community_cards=[],
        pot=100,
        current_bet=20,
        player_stack=1000,
        opponent_bet=20
    )
    
    stats = agent.get_stats()
    print(f"Decision made using {stats['decisions_made']} components")
    print(f"Stats: {stats}")
    
    print("✓ Ensemble voting works\n")


def test_training_hooks():
    """Test training functionality."""
    print("="*60)
    print("Test 5: Training Hooks")
    print("="*60)
    
    agent = create_agent('pokerbot',
                        name='TrainTest',
                        use_cfr=True,
                        use_dqn=True,
                        use_pretrained=False)
    
    # Test CFR training
    if agent.use_cfr:
        print("Training CFR component (10 iterations)...")
        agent.train_cfr(num_iterations=10)
        print("✓ CFR training works")
    
    # Test DQN training
    if agent.use_dqn and agent.dqn_model:
        print("Training DQN component...")
        import numpy as np
        for i in range(35):
            state = np.random.rand(agent.state_size)
            action = i % agent.action_size
            reward = 1.0 if i % 2 == 0 else -1.0
            next_state = np.random.rand(agent.state_size)
            done = False
            agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size=32)
        print("✓ DQN training works")
    
    print()


def test_save_load():
    """Test model persistence."""
    print("="*60)
    print("Test 6: Save/Load")
    print("="*60)
    
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create and save
        agent1 = create_agent('pokerbot', name='SaveTest', 
                             use_cfr=True, use_pretrained=False)
        agent1.save_models(temp_dir)
        print(f"✓ Models saved to {temp_dir}")
        
        # Load
        agent2 = create_agent('pokerbot', name='LoadTest',
                             use_cfr=True, use_pretrained=False)
        agent2.load_models(temp_dir)
        print("✓ Models loaded successfully")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print()


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("PokerBot Agent Validation")
    print("="*60 + "\n")
    
    try:
        test_agent_registry()
        test_agent_creation_variations()
        test_decision_making()
        test_ensemble_voting()
        test_training_hooks()
        test_save_load()
        
        print("="*60)
        print("✓ ALL VALIDATION TESTS PASSED")
        print("="*60)
        print("\nPokerBot agent is ready for use!")
        print("Try: python scripts/train.py --agent-type pokerbot --mode smoketest\n")
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ VALIDATION FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
