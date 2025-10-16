#!/usr/bin/env python3
"""
Validation Script for PokerBot Agent Training

This script validates that the trained PokerBot Agent is working correctly
and has actually improved through the training process.

Usage:
    python scripts/validate_training.py --model models/pokerbot_final
    python scripts/validate_training.py --model models/pokerbot_final --hands 500
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import create_agent
from src.agents.cfr_agent import CFRAgent
from src.agents.fixed_strategy_agent import FixedStrategyAgent
from src.agents.random_agent import RandomAgent
# from src.evaluation import Evaluator  # Commented out, module not found
from src.deepstack.game import GameState
from src.deepstack.utils import Logger


class TrainingValidator:
    """Validates trained PokerBot Agent performance."""
    
    def __init__(self, model_path: str, verbose: bool = True):
        """
        Initialize validator.
        
        Args:
            model_path: Path to trained model (without extension)
            verbose: Enable verbose output
        """
        self.model_path = model_path
        self.verbose = verbose
        self.logger = Logger(verbose=verbose)
        
        # Load trained agent
        self.logger.info("Loading trained PokerBot Agent...")
        self.agent = create_agent('pokerbot', name="TrainedPokerBot", use_pretrained=False)
        self.agent.load_models(model_path)
        self.logger.info("[OK] Agent loaded successfully")
        
        # Create baseline agents for comparison
        self.baseline_agents = self._create_baseline_agents()
    
    def _create_baseline_agents(self) -> Dict:
        """Create baseline agents for comparison."""
        return {
            'random': RandomAgent("Random"),
            'fixed': FixedStrategyAgent("Fixed"),
            'cfr': CFRAgent("CFR"),
            'untrained_champion': ChampionAgent(name="UntrainedChampion", use_pretrained=False)
        }
    
    def validate(self, num_hands: int = 200) -> Dict:
        """
        Run comprehensive validation tests.
        
        Args:
            num_hands: Number of hands to play against each opponent
            
        Returns:
            Validation results dictionary
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("CHAMPION AGENT VALIDATION")
        self.logger.info("="*70)
        self.logger.info(f"Model: {self.model_path}")
        self.logger.info(f"Validation hands per opponent: {num_hands}\n")
        
        results = {}
        
        # Test 1: Agent state validation
        self.logger.info("TEST 1: Agent State Validation")
        self.logger.info("-"*70)
        state_valid = self._validate_agent_state()
        results['state_validation'] = state_valid
        
        # Test 2: Performance against baseline agents
        self.logger.info("\nTEST 2: Performance Against Baseline Agents")
        self.logger.info("-"*70)
        baseline_results = self._validate_against_baselines(num_hands)
        results['baseline_performance'] = baseline_results
        
        # Test 3: Consistency check
        self.logger.info("\nTEST 3: Decision Consistency Check")
        self.logger.info("-"*70)
        consistency_results = self._validate_consistency()
        results['consistency'] = consistency_results
        
        # Test 4: Memory and learning validation
        self.logger.info("\nTEST 4: Memory & Learning Validation")
        self.logger.info("-"*70)
        learning_results = self._validate_learning()
        results['learning'] = learning_results
        
        # Generate summary
        self._print_validation_summary(results)
        
        return results
    
    def _validate_agent_state(self) -> Dict:
        """Validate agent internal state."""
        checks = {}
        
        # Check epsilon (should be decayed from initial)
        checks['epsilon_decayed'] = self.agent.epsilon < 0.3
        self.logger.info(f"  Epsilon value: {self.agent.epsilon:.4f} {'[OK]' if checks['epsilon_decayed'] else '[FAIL]'}")
        
        # Check memory size
        checks['has_memory'] = len(self.agent.memory) > 0
        self.logger.info(f"  Memory size: {len(self.agent.memory)} {'[OK]' if checks['has_memory'] else '[FAIL]'}")
        
        # Check CFR training
        checks['cfr_trained'] = self.agent.cfr.iterations > 0
        self.logger.info(f"  CFR iterations: {self.agent.cfr.iterations} {'[OK]' if checks['cfr_trained'] else '[FAIL]'}")
        
        # Check information sets
        checks['has_infosets'] = len(self.agent.cfr.infosets) > 0
        self.logger.info(f"  Information sets: {len(self.agent.cfr.infosets)} {'[OK]' if checks['has_infosets'] else '[FAIL]'}")
        
        # Check model exists
        checks['has_model'] = self.agent.model is not None
        self.logger.info(f"  DQN model: {'Present [OK]' if checks['has_model'] else 'Missing [FAIL]'}")
        
        all_passed = all(checks.values())
        self.logger.info(f"\n  Overall: {'PASSED [OK]' if all_passed else 'FAILED [FAIL]'}")
        
        return checks
    
    def _validate_against_baselines(self, num_hands: int) -> Dict:
        """Validate performance against baseline agents."""
        results = {}
        
        # Set agent to evaluation mode
        self.agent.set_training_mode(False)
        
        for name, opponent in self.baseline_agents.items():
            self.logger.info(f"\n  Testing against {name}...")
            
            wins = 0
            losses = 0
            total_reward = 0
            
            for _ in range(num_hands):
                reward, won = self._play_validation_hand(opponent)
                total_reward += reward
                if won:
                    wins += 1
                else:
                    losses += 1
            
            win_rate = wins / num_hands
            avg_reward = total_reward / num_hands
            
            results[name] = {
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_reward': avg_reward
            }
            
            # Determine if performance is acceptable
            acceptable = win_rate >= 0.45  # Should win at least 45% against decent opponents
            status = "[OK]" if acceptable else "[FAIL]"
            
            self.logger.info(f"    Win Rate: {win_rate:.2%} {status}")
            self.logger.info(f"    Avg Reward: {avg_reward:.2f}")
            self.logger.info(f"    Record: {wins}W - {losses}L")
        
        return results
    
    def _play_validation_hand(self, opponent) -> tuple:
        """
        Play one validation hand.
        
        Args:
            opponent: Opponent agent
            
        Returns:
            Tuple of (reward, won)
        """
        game = GameState(num_players=2)
        game.reset()
        
        agent_idx = 0
        opponent_idx = 1
        
        done = False
        max_actions = 100  # Prevent infinite loops
        action_count = 0
        
        while not done and action_count < max_actions:
            action_count += 1
            
            # Check if hand is complete
            active_players = [p for p in game.players if not p.folded and not p.all_in]
            if len(active_players) <= 1 or game.is_hand_complete():
                done = True
                break
            
            # Alternate turns
            for current_idx in [agent_idx, opponent_idx]:
                player = game.players[current_idx]
                
                if player.folded or player.all_in:
                    continue
                
                # Get action
                hole_cards = player.hand
                community_cards = game.community_cards
                pot = game.pot
                current_bet = game.current_bet - player.current_bet
                player_stack = player.stack
                
                if current_idx == agent_idx:
                    action, raise_amount = self.agent.choose_action(
                        hole_cards, community_cards, pot,
                        current_bet, player_stack, current_bet
                    )
                else:
                    action, raise_amount = opponent.choose_action(
                        hole_cards, community_cards, pot,
                        current_bet, player_stack, current_bet
                    )
                
                # Apply action
                try:
                    game.apply_action(current_idx, action, raise_amount)
                except Exception:
                    done = True
                    break
                
                if game.is_hand_complete():
                    done = True
                    break
            
            # Advance betting round if needed
            if not done:
                active_players = [p for p in game.players if not p.folded and not p.all_in]
                if active_players:
                    bets = [p.current_bet for p in active_players]
                    if len(set(bets)) == 1:
                        try:
                            game.advance_betting_round()
                        except Exception:
                            done = True
        
        # Calculate result
        try:
            winners = game.get_winners()
            won = agent_idx in winners
            
            if won:
                reward = game.pot / len(winners) if winners else 0
            else:
                reward = -game.players[agent_idx].current_bet
        except Exception:
            # If winner determination fails, use stack difference as reward
            won = False
            reward = -game.players[agent_idx].current_bet if game.players[agent_idx].current_bet > 0 else 0
        
        return reward, won
    
    def _validate_consistency(self) -> Dict:
        """Test decision consistency."""
        from src.deepstack.game import Card, Rank, Suit
        self.logger.info("  Testing decision consistency with same game state...")
        # Create a specific game scenario
        hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        community_cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.TEN, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.CLUBS)
        ]
        pot = 100
        current_bet = 20
        player_stack = 1000
        # Get multiple decisions
        decisions = []
        for _ in range(10):
            action, raise_amt = self.agent.choose_action(
                hole_cards, community_cards, pot,
                current_bet, player_stack, current_bet
            )
            decisions.append((action, raise_amt))
        # Check consistency (with epsilon, may vary slightly)
        unique_actions = set([d[0] for d in decisions])
        consistency_score = 1.0 - (len(unique_actions) - 1) / len(decisions)
        self.logger.info(f"    Unique actions: {len(unique_actions)}")
        self.logger.info(f"    Consistency score: {consistency_score:.2%}")
        self.logger.info(f"    Status: {'[OK]' if consistency_score > 0.5 else '[FAIL]'}")
        return {
            'consistency_score': consistency_score,
            'unique_actions': len(unique_actions),
            'passed': consistency_score > 0.5
        }
    
    def _validate_learning(self) -> Dict:
        """Validate that agent has learned."""
        self.logger.info("  Checking learning indicators...")
        
        checks = {}
        
        # Check that epsilon has decayed (agent explored and is now exploiting)
        checks['epsilon_learning'] = self.agent.epsilon < self.agent.epsilon_min + 0.1
        self.logger.info(f"    Epsilon decay: {self.agent.epsilon:.4f} {'[OK]' if checks['epsilon_learning'] else '[FAIL]'}")
        
        # Check memory utilization
        memory_usage = len(self.agent.memory) / self.agent.memory.maxlen
        checks['memory_utilized'] = memory_usage > 0.1
        self.logger.info(f"    Memory usage: {memory_usage:.2%} {'[OK]' if checks['memory_utilized'] else '[FAIL]'}")
        
        # Check CFR convergence (has trained enough)
        checks['cfr_converged'] = self.agent.cfr.iterations >= 50
        self.logger.info(f"    CFR convergence: {self.agent.cfr.iterations} iterations {'[OK]' if checks['cfr_converged'] else '[FAIL]'}")
        
        all_passed = all(checks.values())
        self.logger.info(f"\n    Overall: {'PASSED [OK]' if all_passed else 'FAILED [FAIL]'}")
        
        return checks
    
    def _print_validation_summary(self, results: Dict):
        """Print validation summary."""
        self.logger.info("\n" + "="*70)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("="*70)
        
        # Count passed tests
        total_tests = 0
        passed_tests = 0
        
        # State validation
        if 'state_validation' in results:
            state_checks = results['state_validation']
            total_tests += len(state_checks)
            passed_tests += sum(state_checks.values())
        
        # Baseline performance (consider pass if win rate > 45%)
        if 'baseline_performance' in results:
            for opponent, result in results['baseline_performance'].items():
                total_tests += 1
                if result['win_rate'] >= 0.45:
                    passed_tests += 1
        
        # Consistency
        if 'consistency' in results and results['consistency'].get('passed'):
            passed_tests += 1
            total_tests += 1
        elif 'consistency' in results:
            total_tests += 1
        
        # Learning
        if 'learning' in results:
            learning_checks = results['learning']
            total_tests += len(learning_checks)
            passed_tests += sum(learning_checks.values())
        
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        self.logger.info(f"Tests Passed: {passed_tests}/{total_tests} ({pass_rate:.1%})")
        
        if pass_rate >= 0.8:
            self.logger.info("Status: EXCELLENT [OK][OK][OK]")
        elif pass_rate >= 0.6:
            self.logger.info("Status: GOOD [OK][OK]")
        elif pass_rate >= 0.4:
            self.logger.info("Status: ACCEPTABLE [OK]")
        else:
            self.logger.info("Status: NEEDS IMPROVEMENT ✗")
        
        self.logger.info("="*70)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description='Validate trained Champion Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (without extension, e.g., models/champion_final)'
    )
    
    parser.add_argument(
        '--hands',
        type=int,
        default=200,
        help='Number of validation hands per opponent (default: 200)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(f"{args.model}.cfr") and not os.path.exists(f"{args.model}_metadata.json"):
        print(f"Error: Model not found at {args.model}")
        print(f"Please ensure the model files exist:")
        print(f"  - {args.model}.cfr")
        print(f"  - {args.model}.dqn (optional)")
        print(f"  - {args.model}_metadata.json (optional)")
        return 1
    
    try:
        # Create validator
        validator = TrainingValidator(args.model, verbose=args.verbose)
        
        # Run validation
        results = validator.validate(num_hands=args.hands)
        
        # Save results
        results_path = f"{args.model}_validation.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in value.items()
                    }
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)
        
        print(f"\nValidation results saved to: {results_path}")
        
        return 0
    
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
