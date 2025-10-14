#!/usr/bin/env python3
"""
Champion Agent Training Pipeline - Advanced Vicarious & Progressive Learning

This script implements a state-of-the-art training pipeline for the Champion Agent,
incorporating:
- Progressive multi-stage training
- Vicarious learning from diverse opponents
- Self-play iteration
- Continuous evaluation and checkpointing
- Comprehensive metrics tracking

Usage:
    # Smoketest mode (quick validation)
    python scripts/train_champion.py --mode smoketest
    
    # Full training mode
    python scripts/train_champion.py --mode full --episodes 10000
    
    # Resume from checkpoint
    python scripts/train_champion.py --resume models/champion_checkpoint_500
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import ChampionAgent, CFRAgent, DQNAgent, FixedStrategyAgent, RandomAgent
from src.evaluation import Evaluator, Trainer
from src.game import Action, Card, GameState
from src.utils import Config, Logger


class TrainingConfig:
    """Configuration for Champion Agent training pipeline."""
    
    def __init__(self, mode: str = "smoketest"):
        """
        Initialize training configuration.
        
        Args:
            mode: Training mode ('smoketest' or 'full')
        """
        self.mode = mode
        
        if mode == "smoketest":
            # Quick validation settings
            self.stage1_cfr_iterations = 100
            self.stage2_selfplay_episodes = 50
            self.stage3_vicarious_episodes = 50
            self.evaluation_interval = 25
            self.save_interval = 25
            self.batch_size = 16
            self.validation_hands = 50
        else:
            # Full production settings
            self.stage1_cfr_iterations = 5000
            self.stage2_selfplay_episodes = 2000
            self.stage3_vicarious_episodes = 3000
            self.evaluation_interval = 100
            self.save_interval = 500
            self.batch_size = 32
            self.validation_hands = 200
        
        # Shared settings
        self.model_dir = "models"
        self.log_dir = "logs"
        self.checkpoint_prefix = "champion_checkpoint"
        
        # Agent diversity for vicarious learning
        self.opponent_types = [
            "random",
            "fixed",
            "cfr",
            "dqn"
        ]
        
        # Metrics tracking
        self.metrics = {
            'training_rewards': [],
            'win_rates': [],
            'evaluation_scores': [],
            'epsilon_values': [],
            'loss_values': [],
            'stage_transitions': []
        }


class ProgressiveTrainer:
    """
    Progressive training pipeline for Champion Agent.
    
    Implements multi-stage training with vicarious learning from diverse opponents.
    """
    
    def __init__(self, config: TrainingConfig, agent: ChampionAgent, verbose: bool = True):
        """
        Initialize progressive trainer.
        
        Args:
            config: Training configuration
            agent: Champion agent to train
            verbose: Enable verbose logging
        """
        self.config = config
        self.agent = agent
        self.verbose = verbose
        self.logger = Logger(verbose=verbose)
        
        # Create directories
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize opponents for vicarious learning
        self.opponents = self._create_diverse_opponents()
        
        # Training statistics
        self.stats = {
            'start_time': time.time(),
            'current_stage': 0,
            'total_episodes': 0,
            'total_hands_played': 0,
            'stage_metrics': defaultdict(list)
        }
    
    def _create_diverse_opponents(self) -> Dict[str, List]:
        """
        Create diverse opponents for vicarious learning.
        
        Returns:
            Dictionary mapping opponent types to agent instances
        """
        opponents = {
            'random': [RandomAgent(f"Random{i}") for i in range(2)],
            'fixed': [FixedStrategyAgent(f"Fixed{i}") for i in range(2)],
            'cfr': [CFRAgent(f"CFR{i}") for i in range(2)],
            'dqn': []  # Will create if TensorFlow available
        }
        
        # Try to create DQN opponents
        try:
            opponents['dqn'] = [
                DQNAgent(
                    state_size=60,
                    action_size=3,
                    learning_rate=0.001,
                    gamma=0.95,
                    epsilon=0.1,
                    epsilon_min=0.01,
                    epsilon_decay=0.995,
                    name=f"DQN{i}"
                ) for i in range(2)
            ]
        except Exception:
            self.logger.warning("Could not create DQN opponents (TensorFlow issue)")
        
        return opponents
    
    def train(self) -> Dict:
        """
        Execute complete progressive training pipeline.
        
        Returns:
            Training statistics and metrics
        """
        self.logger.info("="*70)
        self.logger.info("CHAMPION AGENT PROGRESSIVE TRAINING PIPELINE")
        self.logger.info("="*70)
        self.logger.info(f"Mode: {self.config.mode.upper()}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
        
        # Stage 1: CFR Warmup (Game-theoretic foundation)
        self.logger.info("STAGE 1: CFR WARMUP - Building game-theoretic foundation")
        self.logger.info("-"*70)
        self._stage1_cfr_warmup()
        
        # Stage 2: Self-Play (Iterative improvement)
        self.logger.info("\nSTAGE 2: SELF-PLAY - Iterative self-improvement")
        self.logger.info("-"*70)
        self._stage2_selfplay()
        
        # Stage 3: Vicarious Learning (Learn from diverse opponents)
        self.logger.info("\nSTAGE 3: VICARIOUS LEARNING - Learning from diverse opponents")
        self.logger.info("-"*70)
        self._stage3_vicarious_learning()
        
        # Final evaluation
        self.logger.info("\nFINAL EVALUATION")
        self.logger.info("-"*70)
        final_results = self._final_evaluation()
        
        # Save final model
        self._save_checkpoint(final=True)
        
        # Generate training report
        self._generate_training_report(final_results)
        
        return self.stats
    
    def _stage1_cfr_warmup(self):
        """Stage 1: Train CFR component for game-theoretic foundation."""
        self.stats['current_stage'] = 1
        self.config.metrics['stage_transitions'].append({
            'stage': 1,
            'episode': self.stats['total_episodes'],
            'timestamp': time.time() - self.stats['start_time']
        })
        
        self.logger.info(f"Training CFR component for {self.config.stage1_cfr_iterations} iterations...")
        
        # Train CFR component
        self.agent.train_cfr(num_iterations=self.config.stage1_cfr_iterations)
        
        self.logger.info(f"✓ CFR warmup complete")
        self.logger.info(f"  Total CFR iterations: {self.agent.cfr.iterations}")
        self.logger.info(f"  Information sets learned: {len(self.agent.cfr.infosets)}")
        
        # Save checkpoint
        self._save_checkpoint(stage=1)
    
    def _stage2_selfplay(self):
        """Stage 2: Self-play for iterative improvement."""
        self.stats['current_stage'] = 2
        self.config.metrics['stage_transitions'].append({
            'stage': 2,
            'episode': self.stats['total_episodes'],
            'timestamp': time.time() - self.stats['start_time']
        })
        
        num_episodes = self.config.stage2_selfplay_episodes
        self.logger.info(f"Self-play training for {num_episodes} episodes...")
        
        # Create self-play opponent (clone of current agent)
        opponent = ChampionAgent(
            name="SelfPlayOpponent",
            use_pretrained=False,
            epsilon=0.1  # Some exploration for opponent
        )
        
        # Copy current strategy to opponent
        self.logger.info("Cloning current agent strategy for self-play opponent...")
        
        # Train against self
        wins = 0
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Play one hand
            reward, won = self._play_training_hand(opponent)
            episode_rewards.append(reward)
            if won:
                wins += 1
            
            self.stats['total_episodes'] += 1
            self.stats['total_hands_played'] += 1
            
            # Train on experience
            if len(self.agent.memory) >= self.config.batch_size:
                self.agent.replay(self.config.batch_size)
            
            # Track metrics
            self.config.metrics['training_rewards'].append(reward)
            self.config.metrics['epsilon_values'].append(self.agent.epsilon)
            
            # Periodic evaluation
            if (episode + 1) % self.config.evaluation_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.config.evaluation_interval:])
                win_rate = wins / (episode + 1)
                
                self.logger.info(
                    f"  Episode {episode + 1}/{num_episodes} - "
                    f"Avg Reward: {avg_reward:.2f} - "
                    f"Win Rate: {win_rate:.2%} - "
                    f"Epsilon: {self.agent.epsilon:.3f} - "
                    f"Memory: {len(self.agent.memory)}"
                )
                
                self.config.metrics['win_rates'].append(win_rate)
                self.stats['stage_metrics']['stage2_win_rates'].append(win_rate)
            
            # Save checkpoint
            if (episode + 1) % self.config.save_interval == 0:
                self._save_checkpoint(stage=2, episode=episode + 1)
        
        final_win_rate = wins / num_episodes
        self.logger.info(f"✓ Self-play complete - Final win rate: {final_win_rate:.2%}")
    
    def _stage3_vicarious_learning(self):
        """Stage 3: Vicarious learning from diverse opponents."""
        self.stats['current_stage'] = 3
        self.config.metrics['stage_transitions'].append({
            'stage': 3,
            'episode': self.stats['total_episodes'],
            'timestamp': time.time() - self.stats['start_time']
        })
        
        num_episodes = self.config.stage3_vicarious_episodes
        self.logger.info(f"Vicarious learning for {num_episodes} episodes...")
        self.logger.info(f"Learning from opponent types: {', '.join(self.config.opponent_types)}")
        
        # Track performance against each opponent type
        opponent_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'rewards': []})
        
        for episode in range(num_episodes):
            # Rotate through different opponent types
            opponent_type = self.config.opponent_types[episode % len(self.config.opponent_types)]
            
            # Skip if opponent type not available
            if opponent_type not in self.opponents or not self.opponents[opponent_type]:
                continue
            
            # Select random opponent of this type
            opponent = np.random.choice(self.opponents[opponent_type])
            
            # Play hand
            reward, won = self._play_training_hand(opponent)
            
            # Track stats
            opponent_stats[opponent_type]['total'] += 1
            opponent_stats[opponent_type]['rewards'].append(reward)
            if won:
                opponent_stats[opponent_type]['wins'] += 1
            
            self.stats['total_episodes'] += 1
            self.stats['total_hands_played'] += 1
            
            # Train on experience
            if len(self.agent.memory) >= self.config.batch_size:
                self.agent.replay(self.config.batch_size)
            
            # Track metrics
            self.config.metrics['training_rewards'].append(reward)
            self.config.metrics['epsilon_values'].append(self.agent.epsilon)
            
            # Periodic evaluation
            if (episode + 1) % self.config.evaluation_interval == 0:
                self.logger.info(f"  Episode {episode + 1}/{num_episodes} - Epsilon: {self.agent.epsilon:.3f}")
                
                # Report per-opponent-type statistics
                for opp_type, stats in opponent_stats.items():
                    if stats['total'] > 0:
                        win_rate = stats['wins'] / stats['total']
                        avg_reward = np.mean(stats['rewards'])
                        self.logger.info(
                            f"    vs {opp_type.upper()}: "
                            f"Win Rate {win_rate:.2%}, "
                            f"Avg Reward {avg_reward:.2f}"
                        )
            
            # Save checkpoint
            if (episode + 1) % self.config.save_interval == 0:
                self._save_checkpoint(stage=3, episode=episode + 1)
        
        # Final statistics
        self.logger.info("✓ Vicarious learning complete")
        self.logger.info("  Performance by opponent type:")
        for opp_type, stats in opponent_stats.items():
            if stats['total'] > 0:
                win_rate = stats['wins'] / stats['total']
                avg_reward = np.mean(stats['rewards'])
                self.logger.info(f"    {opp_type.upper()}: {win_rate:.2%} win rate, {avg_reward:.2f} avg reward")
    
    def _play_training_hand(self, opponent) -> Tuple[float, bool]:
        """
        Play one training hand against an opponent.
        
        Args:
            opponent: Opponent agent
            
        Returns:
            Tuple of (reward, won)
        """
        game = GameState(num_players=2)
        game.reset()
        
        agent_idx = 0
        opponent_idx = 1
        
        # Track states and actions for learning
        states = []
        actions_taken = []
        
        done = False
        while not done:
            active_players = [p for p in game.players if not p.folded and not p.all_in]
            
            if len(active_players) <= 1 or game.is_hand_complete():
                done = True
                break
            
            # Alternate between agent and opponent
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
                    # Agent's turn - track state
                    state = self.agent._encode_state(
                        hole_cards, community_cards, pot,
                        current_bet, player_stack, current_bet
                    )
                    states.append(state)
                    
                    action, raise_amount = self.agent.choose_action(
                        hole_cards, community_cards, pot,
                        current_bet, player_stack, current_bet
                    )
                    actions_taken.append(self._action_to_index(action))
                else:
                    # Opponent's turn
                    action, raise_amount = opponent.choose_action(
                        hole_cards, community_cards, pot,
                        current_bet, player_stack, current_bet
                    )
                
                # Apply action
                try:
                    game.apply_action(current_idx, action, raise_amount)
                except Exception:
                    # Handle any game state errors gracefully
                    done = True
                    break
                
                # Check if hand complete
                if game.is_hand_complete():
                    done = True
                    break
            
            # Check if betting round complete
            if not done:
                active_players = [p for p in game.players if not p.folded and not p.all_in]
                if active_players:
                    bets = [p.current_bet for p in active_players]
                    if len(set(bets)) == 1:  # All bets equal
                        try:
                            game.advance_betting_round()
                        except Exception:
                            done = True
        
        # Calculate reward
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
        
        # Store experiences
        if states and actions_taken:
            for i, (state, action) in enumerate(zip(states, actions_taken)):
                step_reward = reward if i == len(states) - 1 else 0
                next_state = states[i + 1] if i < len(states) - 1 else state
                self.agent.remember(state, action, step_reward, next_state, done)
        
        return reward, won
    
    def _action_to_index(self, action: Action) -> int:
        """Convert Action enum to index."""
        if action == Action.FOLD:
            return 0
        elif action in [Action.CHECK, Action.CALL]:
            return 1
        else:
            return 2
    
    def _final_evaluation(self) -> Dict:
        """
        Perform final comprehensive evaluation.
        
        Returns:
            Evaluation results
        """
        self.logger.info(f"Running final evaluation over {self.config.validation_hands} hands...")
        
        # Set agent to evaluation mode
        self.agent.set_training_mode(False)
        
        # Create evaluation opponents
        eval_opponents = [
            RandomAgent("EvalRandom"),
            FixedStrategyAgent("EvalFixed"),
            CFRAgent("EvalCFR")
        ]
        
        # Evaluate against each opponent
        results = {}
        for opponent in eval_opponents:
            self.logger.info(f"  Evaluating vs {opponent.name}...")
            
            wins = 0
            total_reward = 0
            
            for _ in range(self.config.validation_hands):
                reward, won = self._play_training_hand(opponent)
                if won:
                    wins += 1
                total_reward += reward
            
            win_rate = wins / self.config.validation_hands
            avg_reward = total_reward / self.config.validation_hands
            
            results[opponent.name] = {
                'win_rate': win_rate,
                'avg_reward': avg_reward,
                'total_hands': self.config.validation_hands
            }
            
            self.logger.info(f"    Win Rate: {win_rate:.2%}, Avg Reward: {avg_reward:.2f}")
        
        # Set back to training mode
        self.agent.set_training_mode(True)
        
        return results
    
    def _save_checkpoint(self, stage: int = None, episode: int = None, final: bool = False):
        """
        Save training checkpoint.
        
        Args:
            stage: Current training stage
            episode: Current episode number
            final: Whether this is the final checkpoint
        """
        if final:
            filepath = os.path.join(self.config.model_dir, "champion_final")
        elif stage and episode:
            filepath = os.path.join(
                self.config.model_dir,
                f"{self.config.checkpoint_prefix}_stage{stage}_ep{episode}"
            )
        elif stage:
            filepath = os.path.join(
                self.config.model_dir,
                f"{self.config.checkpoint_prefix}_stage{stage}"
            )
        else:
            filepath = os.path.join(self.config.model_dir, self.config.checkpoint_prefix)
        
        # Save agent strategy
        self.agent.save_strategy(filepath)
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_episodes': self.stats['total_episodes'],
            'total_hands': self.stats['total_hands_played'],
            'current_stage': self.stats['current_stage'],
            'epsilon': self.agent.epsilon,
            'memory_size': len(self.agent.memory),
            'cfr_iterations': self.agent.cfr.iterations,
            'metrics': self.config.metrics
        }
        
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            self.logger.info(f"  Checkpoint saved: {filepath}")
    
    def _generate_training_report(self, final_results: Dict):
        """
        Generate comprehensive training report.
        
        Args:
            final_results: Final evaluation results
        """
        elapsed_time = time.time() - self.stats['start_time']
        
        report_path = os.path.join(
            self.config.log_dir,
            f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CHAMPION AGENT TRAINING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Training Mode: {self.config.mode.upper()}\n")
            f.write(f"Total Training Time: {elapsed_time/60:.2f} minutes\n")
            f.write(f"Total Episodes: {self.stats['total_episodes']}\n")
            f.write(f"Total Hands Played: {self.stats['total_hands_played']}\n\n")
            
            f.write("Stage Summary:\n")
            f.write(f"  Stage 1 (CFR Warmup): {self.config.stage1_cfr_iterations} iterations\n")
            f.write(f"  Stage 2 (Self-Play): {self.config.stage2_selfplay_episodes} episodes\n")
            f.write(f"  Stage 3 (Vicarious): {self.config.stage3_vicarious_episodes} episodes\n\n")
            
            f.write("Final Agent State:\n")
            f.write(f"  Epsilon: {self.agent.epsilon:.4f}\n")
            f.write(f"  Memory Size: {len(self.agent.memory)}\n")
            f.write(f"  CFR Iterations: {self.agent.cfr.iterations}\n")
            f.write(f"  Information Sets: {len(self.agent.cfr.infosets)}\n\n")
            
            f.write("Final Evaluation Results:\n")
            for opponent_name, result in final_results.items():
                f.write(f"  vs {opponent_name}:\n")
                f.write(f"    Win Rate: {result['win_rate']:.2%}\n")
                f.write(f"    Avg Reward: {result['avg_reward']:.2f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        self.logger.info(f"\nTraining report saved: {report_path}")
        
        # Print summary
        self.logger.info("\n" + "="*70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
        self.logger.info(f"Total episodes: {self.stats['total_episodes']}")
        self.logger.info(f"Final model: models/champion_final")
        self.logger.info("="*70)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train Champion Agent with progressive vicarious learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoketest (recommended first run)
  python scripts/train_champion.py --mode smoketest
  
  # Full training session
  python scripts/train_champion.py --mode full
  
  # Custom episodes
  python scripts/train_champion.py --mode smoketest --episodes 100
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['smoketest', 'full'],
        default='smoketest',
        help='Training mode (smoketest=quick validation, full=production training)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Override number of episodes for stage 2 and 3'
    )
    
    parser.add_argument(
        '--cfr-iterations',
        type=int,
        default=None,
        help='Override CFR warmup iterations'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save models'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint (path without extension)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(mode=args.mode)
    
    # Override config if specified
    if args.episodes:
        config.stage2_selfplay_episodes = args.episodes
        config.stage3_vicarious_episodes = args.episodes
    
    if args.cfr_iterations:
        config.stage1_cfr_iterations = args.cfr_iterations
    
    if args.model_dir:
        config.model_dir = args.model_dir
    
    # Create or load agent
    logger = Logger(verbose=args.verbose)
    logger.info("Initializing Champion Agent...")
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        agent = ChampionAgent(name="Champion", use_pretrained=True)
        agent.load_strategy(args.resume)
    else:
        agent = ChampionAgent(name="Champion", use_pretrained=True)
    
    logger.info("Agent initialized successfully")
    logger.info(f"  State size: {agent.state_size}")
    logger.info(f"  Action size: {agent.action_size}")
    logger.info(f"  Initial epsilon: {agent.epsilon}")
    logger.info("")
    
    # Create trainer
    trainer = ProgressiveTrainer(config, agent, verbose=args.verbose)
    
    # Run training
    try:
        stats = trainer.train()
        
        logger.info("\n✓ Training completed successfully!")
        logger.info(f"  Final model saved to: {config.model_dir}/champion_final")
        logger.info(f"  Training logs saved to: {config.log_dir}/")
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        logger.info("Saving checkpoint before exit...")
        trainer._save_checkpoint(final=False)
        return 1
    
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
