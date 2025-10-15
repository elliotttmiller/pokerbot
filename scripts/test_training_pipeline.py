#!/usr/bin/env python3
"""
End-to-End Test for Champion Agent Training Pipeline

This script runs a complete test of the training pipeline to verify
all components work correctly.

Usage:
    python scripts/test_training_pipeline.py
"""

import os
import sys
# Ensure parent directory is in sys.path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import subprocess
import numpy as np
from src.utils import Logger
from src.agents import ChampionAgent, CFRAgent, DQNAgent, RandomAgent, FixedStrategyAgent
from src.deepstack.evaluation import Evaluator, Trainer
from src.deepstack.game import GameState, Card, Rank, Suit
from src.deepstack.utils.bucketer import Bucketer
from src.deepstack.core.masked_huber_loss import masked_huber_loss
from src.deepstack.core.strategy_filling import StrategyFilling


def test_imports():
    """Test that all required modules can be imported."""
    logger = Logger(verbose=True)
    logger.info("TEST 1: Testing imports...")
    
    try:
        from src.agents import ChampionAgent, CFRAgent, DQNAgent, RandomAgent, FixedStrategyAgent
        from src.deepstack.evaluation import Evaluator, Trainer
        from src.deepstack.game import GameState, Card, Rank, Suit
        logger.info("  [OK] All imports successful")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"  [FAIL] Import failed: {e}")
        return False


def test_agent_creation():
    """Test creating a Champion Agent."""
    logger = Logger(verbose=True)
    logger.info("\nTEST 2: Testing agent creation...")
    
    try:
        from src.agents import ChampionAgent
        
        agent = ChampionAgent(name="TestAgent", use_pretrained=False)
        
        assert agent.state_size == 60
        assert agent.action_size == 3
        assert agent.cfr is not None
        assert agent.model is not None
        
        logger.info("  [OK] Agent created successfully")
        logger.info(f"    State size: {agent.state_size}")
        logger.info(f"    Action size: {agent.action_size}")
        logger.info(f"    Epsilon: {agent.epsilon}")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline():
    """Test the training pipeline with minimal settings."""
    logger = Logger(verbose=True)
    logger.info("\nTEST 3: Testing training pipeline...")
    logger.info("  Running minimal training (this may take 1-2 minutes)...")
    
    try:
        import subprocess
        
        # Run training with very minimal settings
        result = subprocess.run(
            [
                sys.executable,
                "scripts/train_champion.py",
                "--mode", "smoketest",
                "--episodes", "5",
                "--cfr-iterations", "10"
            ],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            logger.info("  [OK] Training completed successfully")
            
            # Check if model files were created
            model_files = [
                "models/versions/champion_current.cfr",
                "models/versions/champion_current.keras"
            ]
            
            all_exist = True
            for filepath in model_files:
                if os.path.exists(filepath):
                    logger.info(f"    [OK] {filepath} created")
                else:
                    logger.info(f"    [WARN] {filepath} not found")
                    all_exist = False
            
            return all_exist
        else:
            logger.error(f"  [FAIL] Training failed with exit code {result.returncode}")
            error_msg = result.stderr[-500:] if result.stderr else "No error output"
            logger.error(f"  Error output: {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("  [FAIL] Training timed out")
        return False
    except Exception as e:
        logger.error(f"  [FAIL] Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test loading a trained model."""
    logger = Logger(verbose=True)
    logger.info("\nTEST 4: Testing model loading...")
    
    try:
        from src.agents import ChampionAgent
        
        # Load the trained model
        agent = ChampionAgent(name="LoadedAgent", use_pretrained=False)
        agent.load_strategy("models/versions/champion_current")
        
        # Verify loaded state
        assert agent.cfr.iterations > 0
        assert len(agent.cfr.infosets) > 0
        
        logger.info("  [OK] Model loaded successfully")
        logger.info(f"    CFR iterations: {agent.cfr.iterations}")
        logger.info(f"    Information sets: {len(agent.cfr.infosets)}")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_decision_making():
    """Test that the agent can make decisions."""
    logger = Logger(verbose=True)
    logger.info("\nTEST 5: Testing agent decision making...")
    
    try:
        from src.agents import ChampionAgent
        from src.deepstack.game import Card, Rank, Suit
        
        # Load trained agent
        agent = ChampionAgent(name="TestAgent", use_pretrained=False)
        agent.load_strategy("models/versions/champion_current")
        
        # Test decision making
        hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        community_cards = []
        pot = 100
        current_bet = 20
        player_stack = 1000
        
        action, raise_amt = agent.choose_action(
            hole_cards, community_cards, pot,
            current_bet, player_stack, current_bet  # opponent_bet same as current_bet in simple case
        )
        
        logger.info("  [OK] Agent made decision successfully")
        logger.info(f"    Action: {action}")
        logger.info(f"    Raise amount: {raise_amt}")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] Decision making failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_script():
    """Test the validation script."""
    logger = Logger(verbose=True)
    logger.info("\nTEST 6: Testing validation script...")
    logger.info("  Running validation (this may take 30-60 seconds)...")
    
    try:
        import subprocess
        
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_training.py",
                "--model", "models/versions/champion_current",
                "--hands", "25"
            ],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            capture_output=True,
            text=True,
            timeout=90
        )
        
        if result.returncode == 0:
            logger.info("  [OK] Validation completed successfully")
            
            # Check if validation results were created
            if os.path.exists("models/versions/champion_current_validation.json"):
                logger.info("    [OK] Validation results saved")
                return True
            else:
                logger.info("    [WARN] Validation results not found")
                return False
        else:
            logger.error(f"  [FAIL] Validation failed with exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("  [FAIL] Validation timed out")
        return False
    except Exception as e:
        logger.error(f"  [FAIL] Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deepstack_module_integrations():
    """Test direct integration and correctness of DeepStack modules."""
    logger = Logger(verbose=True)
    logger.info("\nTEST 7: Testing DeepStack module integrations...")
    try:
        # Bucketer
        from src.deepstack.utils.bucketer import Bucketer
        bucketer = Bucketer()
        bucket_idx = bucketer.get_bucket([], [])
        assert isinstance(bucket_idx, int)
        logger.info(f"  [OK] Bucketer abstraction returns bucket index: {bucket_idx}")

        # Masked Huber Loss
        from src.deepstack.core.masked_huber_loss import masked_huber_loss
        import numpy as np
        y_true = np.ones((2, 13))
        y_pred = np.zeros((2, 13))
        mask = np.ones((2, 13))
        loss = masked_huber_loss(y_true, y_pred, mask)
        assert loss >= 0
        logger.info(f"  [OK] Masked Huber loss returns valid value: {loss}")

        # Strategy Filling
        strategy_filler = StrategyFilling()
        strategy = np.array([[0.2, 0.3, 0.5], [0.0, 0.0, 0.0]])
        filled = strategy_filler.fill_missing(strategy)
        assert np.allclose(filled.sum(axis=1), 1.0)
        logger.info(f"  [OK] Strategy filling normalizes strategies: {filled}")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] DeepStack module integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        # Yes, all imports should ideally be declared at the top of the file for clarity and maintainability.
        # However, in test functions, sometimes imports are placed inside the function to:
        #   - Test import errors in isolation
        #   - Avoid unnecessary imports if the test is not run
        #   - Reduce startup time for scripts with many dependencies
        # For production code, move all imports to the top unless you have a specific reason to keep them inside functions.

def test_agent_evaluation():
    """Test agent benchmarking using Evaluator."""
    logger = Logger(verbose=True)
    logger.info("\nTEST 8: Testing agent evaluation and benchmarking...")
    try:
        from src.agents import ChampionAgent, RandomAgent
        from src.deepstack.evaluation import Evaluator
        agent = ChampionAgent(name="EvalAgent", use_pretrained=False)
        opponent = RandomAgent()
        evaluator = Evaluator([agent, opponent])
        results = evaluator.evaluate_agents(num_hands=10, verbose=False)
        assert 'wins' in results and 'losses' in results
        logger.info(f"  [OK] Evaluation results: {results}")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] Agent evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger = Logger(verbose=True)
    
    logger.info("="*70)
    logger.info("CHAMPION AGENT TRAINING PIPELINE - END-TO-END TEST")
    logger.info("="*70)
    logger.info("")
    
    start_time = time.time()
    
    tests = [
        ("Imports", test_imports),
        ("Agent Creation", test_agent_creation),
        ("Training Pipeline", test_training_pipeline),
        ("Model Loading", test_model_loading),
        ("Decision Making", test_agent_decision_making),
        ("Validation", test_validation_script),
        ("DeepStack Module Integrations", test_deepstack_module_integrations),
        ("Agent Evaluation", test_agent_evaluation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"\nTest '{test_name}' crashed: {e}")
            results[test_name] = False
    
    elapsed = time.time() - start_time
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "[OK] PASSED" if passed_test else "[FAIL] FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info(f"Time: {elapsed:.1f} seconds")
    logger.info("="*70)
    
    if passed == total:
        logger.info("\n[OK] ALL TESTS PASSED! Training pipeline is working correctly.")
        return 0
    else:
        logger.error(f"\n[FAIL] {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
