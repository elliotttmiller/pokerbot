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
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import Logger


def test_imports():
    """Test that all required modules can be imported."""
    logger = Logger(verbose=True)
    logger.info("TEST 1: Testing imports...")
    
    try:
        from src.agents import ChampionAgent, CFRAgent, DQNAgent, RandomAgent, FixedStrategyAgent
        from src.evaluation import Evaluator, Trainer
        from src.game import GameState, Card, Rank, Suit
        logger.info("  ✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"  ✗ Import failed: {e}")
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
        
        logger.info("  ✓ Agent created successfully")
        logger.info(f"    State size: {agent.state_size}")
        logger.info(f"    Action size: {agent.action_size}")
        logger.info(f"    Epsilon: {agent.epsilon}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Agent creation failed: {e}")
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
            logger.info("  ✓ Training completed successfully")
            
            # Check if model files were created
            model_files = [
                "models/champion_final.cfr",
                "models/champion_final.keras"
            ]
            
            all_exist = True
            for filepath in model_files:
                if os.path.exists(filepath):
                    logger.info(f"    ✓ {filepath} created")
                else:
                    logger.info(f"    ✗ {filepath} not found")
                    all_exist = False
            
            return all_exist
        else:
            logger.error(f"  ✗ Training failed with exit code {result.returncode}")
            logger.error(f"  Error output: {result.stderr[-500:]}")  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("  ✗ Training timed out")
        return False
    except Exception as e:
        logger.error(f"  ✗ Training pipeline test failed: {e}")
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
        agent.load_strategy("models/champion_final")
        
        # Verify loaded state
        assert agent.cfr.iterations > 0
        assert len(agent.cfr.infosets) > 0
        
        logger.info("  ✓ Model loaded successfully")
        logger.info(f"    CFR iterations: {agent.cfr.iterations}")
        logger.info(f"    Information sets: {len(agent.cfr.infosets)}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_decision_making():
    """Test that the agent can make decisions."""
    logger = Logger(verbose=True)
    logger.info("\nTEST 5: Testing agent decision making...")
    
    try:
        from src.agents import ChampionAgent
        from src.game import Card, Rank, Suit
        
        # Load trained agent
        agent = ChampionAgent(name="TestAgent", use_pretrained=False)
        agent.load_strategy("models/champion_final")
        
        # Test decision making
        hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        community_cards = []
        pot = 100
        current_bet = 20
        player_stack = 1000
        
        action, raise_amt = agent.choose_action(
            hole_cards, community_cards, pot,
            current_bet, player_stack, current_bet
        )
        
        logger.info("  ✓ Agent made decision successfully")
        logger.info(f"    Action: {action}")
        logger.info(f"    Raise amount: {raise_amt}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Decision making failed: {e}")
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
                "--model", "models/champion_final",
                "--hands", "25"
            ],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            capture_output=True,
            text=True,
            timeout=90
        )
        
        if result.returncode == 0:
            logger.info("  ✓ Validation completed successfully")
            
            # Check if validation results were created
            if os.path.exists("models/champion_final_validation.json"):
                logger.info("    ✓ Validation results saved")
                return True
            else:
                logger.info("    ✗ Validation results not found")
                return False
        else:
            logger.error(f"  ✗ Validation failed with exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("  ✗ Validation timed out")
        return False
    except Exception as e:
        logger.error(f"  ✗ Validation test failed: {e}")
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
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info(f"Time: {elapsed:.1f} seconds")
    logger.info("="*70)
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Training pipeline is working correctly.")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
