#!/usr/bin/env python3
"""Champion training pipeline (single entrypoint).

Usage examples:
    - Smoketest:        python scripts/train.py --mode smoketest
    - Standard:         python scripts/train.py --mode standard
    - Full/production:  python scripts/train.py --mode full --episodes 10000
"""

import sys
import os
# Ensure src is in sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
from src.utils import Logger

# Champion pipeline imports
from typing import Dict
import os
import json
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
import random

from src.agents import ChampionAgent, CFRAgent, FixedStrategyAgent, RandomAgent
from src.deepstack.game import Card, GameState, Action
from src.utils.data_validation import validate_deepstacked_samples, validate_equity_table


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train the championship-level agent (progressive pipeline)')
    parser.add_argument('--mode', type=str, choices=['smoketest', 'standard', 'production', 'full'], default='smoketest',
                       help='Training mode selects configuration profile from scripts/config')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override number of episodes for stage 2 and 3')
    parser.add_argument('--cfr-iterations', type=int, default=None,
                       help='Override CFR warmup iterations')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models and versions')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size used during replay')
    parser.add_argument('--optimize-export', action='store_true',
                       help='After training, attempt prune/quantize export (requires TensorFlow)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--report', action='store_true',
                       help='Generate analysis plots and metrics after training')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger(verbose=args.verbose)
    logger.info("Starting championship training pipeline")

    # Deterministic seeds
    try:
        np.random.seed(args.seed)
        random.seed(args.seed)
    except Exception:
        pass

    # Champion pipeline (consolidated)
    # Minimal inline versions of config and trainer to avoid cross-file duplication
    import torch
    class TrainingConfig:
        def __init__(self, mode: str = 'smoketest'):
            self.mode = mode
            config_path = f"scripts/config/{mode}.json"
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            self.stage1_cfr_iterations = cfg["stage1_cfr_iterations"]
            self.stage2_selfplay_episodes = cfg["stage2_selfplay_episodes"]
            self.stage3_vicarious_episodes = cfg["stage3_vicarious_episodes"]
            self.evaluation_interval = cfg["evaluation_interval"]
            self.save_interval = cfg["save_interval"]
            self.batch_size = cfg["batch_size"]
            self.validation_hands = cfg["validation_hands"]

            self.samples_path = cfg.get("samples_path", "data/deepstacked_training/samples/train_samples")
            self.test_samples_path = cfg.get("test_samples_path", "data/deepstacked_training/samples/test_samples")
            self.equity_path = cfg.get("equity_path", "data/equity_tables/preflop_equity.json")

            self.model_dir = "models"
            self.versions_dir = "models/versions"
            self.checkpoints_dir = "models/checkpoints"
            self.log_dir = cfg.get("log_dir", "logs")
            self.report_dir = cfg.get("report_dir", "models/reports")
            self.checkpoint_prefix = "champion_checkpoint"

            self.opponent_types = ["random", "fixed", "cfr", "dqn", "meta"]
            self.metrics = {
                'training_rewards': [],
                'win_rates': [],
                'evaluation_scores': [],
                'epsilon_values': [],
                'loss_values': [],
                'stage_transitions': [],
                'opponent_difficulty': [],
                'episode_reward_mean': [],
            }

            # Explicitly load and log training/validation samples
            self.train_inputs = self._load_and_log_tensor('train_inputs.pt')
            self.train_mask = self._load_and_log_tensor('train_mask.pt')
            self.train_targets = self._load_and_log_tensor('train_targets.pt')
            self.valid_inputs = self._load_and_log_tensor('valid_inputs.pt')
            self.valid_mask = self._load_and_log_tensor('valid_mask.pt')
            self.valid_targets = self._load_and_log_tensor('valid_targets.pt')

        def _load_and_log_tensor(self, fname):
            fpath = os.path.join(self.samples_path, fname)
            if not os.path.isfile(fpath):
                print(f"[SAMPLES] Missing file: {fpath}")
                return None
            try:
                tensor = torch.load(fpath)
                print(f"[SAMPLES] Loaded {fname}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, sum={tensor.sum().item()}")
                return tensor
            except Exception as e:
                print(f"[SAMPLES] Error loading {fname}: {e}")
                return None

    class ProgressiveTrainer:
        def __init__(self, config: TrainingConfig, agent: ChampionAgent, verbose: bool = True):
            self.config = config
            self.agent = agent
            self.verbose = verbose
            self.logger = Logger(verbose=verbose)
            os.makedirs(config.model_dir, exist_ok=True)
            os.makedirs(config.versions_dir, exist_ok=True)
            os.makedirs(config.checkpoints_dir, exist_ok=True)
            os.makedirs(config.log_dir, exist_ok=True)
            os.makedirs(config.report_dir, exist_ok=True)

            self.opponents = self._create_diverse_opponents()
            self.stats = {
                'start_time': time.time(),
                'current_stage': 0,
                'total_episodes': 0,
                'total_hands_played': 0,
                'stage_metrics': defaultdict(list),
            }
            self.last_eval_rewards: list[float] = []

        def _create_diverse_opponents(self):
            return {
                'random': [RandomAgent(f"Random{i}") for i in range(2)],
                'fixed': [FixedStrategyAgent(f"Fixed{i}") for i in range(2)],
                'cfr': [CFRAgent(f"CFR{i}") for i in range(2)],
            }

        def train(self):
            self._stage0_validate_data()
            self._stage1_cfr_warmup()
            self._save_checkpoint(stage=1)
            self._stage2_selfplay()
            self._save_checkpoint(stage=2)
            self._stage3_vicarious_learning()
            final = self._final_evaluation()
            self._save_checkpoint(final=True)
            comp = self._compare_against_previous_best(num_hands=self.config.validation_hands)
            self._promote_to_best_if_better(comp)
            # Persist metrics for post-analysis
            self._save_metrics()
            # Optional export/optimize
            if getattr(args, 'optimize_export', False):
                self._optimize_export()
            return self.stats

        def _stage0_validate_data(self):
            samples = validate_deepstacked_samples(self.config.samples_path)
            equity = validate_equity_table(self.config.equity_path)
            if samples.get('errors'):
                raise ValueError('DeepStacked training samples failed validation')
            if equity.get('error') or not equity.get('ok', False):
                raise ValueError('Equity table validation failed')

        def _stage1_cfr_warmup(self):
            self.agent.train_cfr(num_iterations=self.config.stage1_cfr_iterations)

        def _stage2_selfplay(self):
            opponent = ChampionAgent(name="SelfPlayOpponent", use_pretrained=False, epsilon=0.1)
            wins = 0
            recent_rewards: list[float] = []
            for ep in range(self.config.stage2_selfplay_episodes):
                reward, won = self._play_training_hand(opponent)
                if won:
                    wins += 1
                # Train on experience
                bs = self.config.batch_size
                if len(self.agent.memory) >= bs:
                    self.agent.replay(bs)
                # Metrics
                self.config.metrics['training_rewards'].append(float(reward))
                self.config.metrics['epsilon_values'].append(float(getattr(self.agent, 'epsilon', 0.0)))
                recent_rewards.append(reward)
                self.stats['total_episodes'] += 1
                self.stats['total_hands_played'] += 1
                # Periodic evaluation/logging
                if (ep + 1) % self.config.evaluation_interval == 0:
                    avg_r = float(np.mean(recent_rewards[-self.config.evaluation_interval:])) if recent_rewards else 0.0
                    win_rate = wins / (ep + 1)
                    self.logger.info(
                        f"[Stage2] Ep {ep+1}/{self.config.stage2_selfplay_episodes} | AvgR {avg_r:.2f} | Win {win_rate:.2%} | Eps {getattr(self.agent, 'epsilon', 0.0):.3f} | Mem {len(self.agent.memory)}"
                    )
                    self.config.metrics['episode_reward_mean'].append(avg_r)
                    self.config.metrics['win_rates'].append(float(win_rate))
                if (ep + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(stage=2)

        def _stage3_vicarious_learning(self):
            eval_opps = [RandomAgent("EvalRandom"), FixedStrategyAgent("EvalFixed"), CFRAgent("EvalCFR")]
            wins = defaultdict(int)
            recent_rewards: list[float] = []
            # simple curriculum: round-robin over eval opponents
            for ep in range(self.config.stage3_vicarious_episodes):
                opp = eval_opps[ep % len(eval_opps)]
                reward, won = self._play_training_hand(opp)
                wins[opp.name] += int(won)
                bs = self.config.batch_size
                if len(self.agent.memory) >= bs:
                    # a little extra optimization step now
                    for _ in range(2):
                        self.agent.replay(bs)
                # Metrics
                self.config.metrics['training_rewards'].append(float(reward))
                self.config.metrics['epsilon_values'].append(float(getattr(self.agent, 'epsilon', 0.0)))
                recent_rewards.append(reward)
                self.stats['total_episodes'] += 1
                self.stats['total_hands_played'] += 1
                if (ep + 1) % self.config.evaluation_interval == 0:
                    avg_r = float(np.mean(recent_rewards[-self.config.evaluation_interval:])) if recent_rewards else 0.0
                    overall_wins = sum(wins.values())
                    win_rate = overall_wins / (ep + 1)
                    self.logger.info(
                        f"[Stage3] Ep {ep+1}/{self.config.stage3_vicarious_episodes} | AvgR {avg_r:.2f} | Win {win_rate:.2%} | Eps {getattr(self.agent, 'epsilon', 0.0):.3f} | Mem {len(self.agent.memory)}"
                    )
                    self.config.metrics['episode_reward_mean'].append(avg_r)
                    self.config.metrics['win_rates'].append(float(win_rate))
                if (ep + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(stage=3)

        def _play_training_hand(self, opponent) -> tuple[float, bool]:
            game = GameState(num_players=2)
            game.reset()
            agent_idx, opp_idx = 0, 1
            done = False
            max_actions = 200
            actions = 0
            while not done:
                if game.is_hand_complete():
                    break
                for current in [agent_idx, opp_idx]:
                    player = game.players[current]
                    if player.folded or player.all_in:
                        continue
                    hole_cards = player.hand
                    community_cards = game.community_cards
                    pot = game.pot
                    current_bet = game.current_bet - player.current_bet
                    player_stack = player.stack
                    if current == agent_idx:
                        state = self.agent._encode_state(hole_cards, community_cards, pot, current_bet, player_stack, current_bet)
                        action, raise_amount = self.agent.choose_action(hole_cards, community_cards, pot, current_bet, player_stack, current_bet)
                        self.agent.remember(state, 1 if action in (Action.CALL, Action.CHECK) else (0 if action == Action.FOLD else 2), 0.0, state, False)
                    else:
                        action, raise_amount = opponent.choose_action(hole_cards, community_cards, pot, current_bet, player_stack, current_bet)
                    try:
                        game.apply_action(current, action, raise_amount)
                    except Exception:
                        done = True
                        break
                    if game.is_hand_complete():
                        done = True
                        break
                    actions += 1
                    if actions >= max_actions:
                        done = True
                        break
                if not done:
                    active = [p for p in game.players if not p.folded and not p.all_in]
                    if active:
                        bets = [p.current_bet for p in active]
                        if len(set(bets)) == 1:
                            try:
                                game.advance_betting_round()
                            except Exception:
                                done = True
            try:
                winners = game.get_winners()
                won = (0 in winners)
                reward = game.pot / len(winners) if won else -game.players[0].current_bet
            except Exception:
                won = False
                reward = -game.players[0].current_bet if game.players[0].current_bet > 0 else 0
            return reward, won

        def _final_evaluation(self) -> Dict:
            self.agent.set_training_mode(False)
            eval_opponents = [RandomAgent("EvalRandom"), FixedStrategyAgent("EvalFixed"), CFRAgent("EvalCFR")]
            results = {}
            for opp in eval_opponents:
                wins, total_reward = 0, 0
                for _ in range(self.config.validation_hands):
                    reward, won = self._play_training_hand(opp)
                    wins += int(won)
                    total_reward += reward
                results[opp.name] = {
                    'win_rate': wins / self.config.validation_hands,
                    'avg_reward': total_reward / self.config.validation_hands,
                    'total_hands': self.config.validation_hands,
                }
            self.agent.set_training_mode(True)
            return results

        def _compare_against_previous_best(self, num_hands: int = 100) -> Dict:
            best_path = os.path.join(self.config.versions_dir, "champion_best")
            if not os.path.exists(f"{best_path}.cfr"):
                wins, total_reward = 0, 0
                for _ in range(num_hands):
                    reward, won = self._play_comparison_hand(self.agent)
                    wins += int(won)
                    total_reward += reward
                return {
                    'previous_best_exists': False,
                    'is_better': True,
                    'win_rate': wins / num_hands,
                    'avg_reward': total_reward / num_hands,
                    'total_hands': num_hands,
                }
            try:
                previous_best = ChampionAgent(name="PreviousBest", use_pretrained=True)
                previous_best.load_strategy(best_path)
            except Exception as e:
                return {'previous_best_exists': False, 'is_better': True, 'error': str(e)}
            original_epsilon = getattr(self.agent, 'epsilon', 0.0)
            self.agent.set_training_mode(False)
            previous_best.set_training_mode(False)
            wins, total_reward = 0, 0
            for _ in range(num_hands):
                reward, won = self._play_comparison_hand(previous_best)
                wins += int(won)
                total_reward += reward
            self.agent.epsilon = original_epsilon
            self.agent.set_training_mode(True)
            win_rate = wins / num_hands
            return {
                'previous_best_exists': True,
                'is_better': win_rate > 0.5,
                'win_rate': win_rate,
                'avg_reward': total_reward / num_hands,
                'total_hands': num_hands,
            }

        def _play_comparison_hand(self, opponent) -> tuple:
            game = GameState(num_players=2, small_blind=10, big_blind=20, starting_stack=1000)
            agent_idx, opponent_idx = 0, 1
            done = False
            while not done:
                if game.is_hand_complete():
                    done = True
                    break
                for current_idx in [agent_idx, opponent_idx]:
                    player = game.players[current_idx]
                    if player.folded or player.all_in:
                        continue
                    hole_cards = player.hand
                    community_cards = game.community_cards
                    pot = game.pot
                    current_bet = game.current_bet - player.current_bet
                    player_stack = player.stack
                    if current_idx == agent_idx:
                        action, raise_amount = self.agent.choose_action(hole_cards, community_cards, pot, current_bet, player_stack, current_bet)
                    else:
                        action, raise_amount = opponent.choose_action(hole_cards, community_cards, pot, current_bet, player_stack, current_bet)
                    try:
                        game.apply_action(current_idx, action, raise_amount)
                    except Exception:
                        done = True
                        break
                    if game.is_hand_complete():
                        done = True
                        break
                if not done:
                    active = [p for p in game.players if not p.folded and not p.all_in]
                    if active:
                        bets = [p.current_bet for p in active]
                        if len(set(bets)) == 1:
                            try:
                                game.advance_betting_round()
                            except Exception:
                                done = True
            try:
                winners = game.get_winners()
                won = agent_idx in winners
                reward = game.pot / len(winners) if won else -game.players[agent_idx].current_bet
            except Exception:
                won = False
                reward = -game.players[agent_idx].current_bet if game.players[agent_idx].current_bet > 0 else 0
            return reward, won

        def _save_checkpoint(self, final: bool = False, stage: int | None = None):
            if final:
                filepath = os.path.join(self.config.versions_dir, "champion_current")
            elif stage is not None:
                filepath = os.path.join(self.config.checkpoints_dir, f"{self.config.checkpoint_prefix}")
            else:
                filepath = os.path.join(self.config.checkpoints_dir, self.config.checkpoint_prefix)
            try:
                self.agent.save_strategy(filepath)
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint at {filepath}: {e}")

        def _save_metrics(self):
            try:
                out = os.path.join(self.config.log_dir, 'training_metrics.json')
                with open(out, 'w') as f:
                    json.dump(self.config.metrics, f, indent=2)
                self.logger.info(f"Saved metrics to {out}")
            except Exception as e:
                self.logger.warning(f"Failed to save training metrics: {e}")

        def _optimize_export(self):
            # Try to load and optimize the final keras model if available
            try:
                from src.utils.model_optimization import prune_model, strip_pruning, quantize_model, save_tflite
                import tensorflow as tf
            except Exception:
                self.logger.warning("TensorFlow or optimization utils not available; skipping optimize-export")
                return
            final_path = os.path.join(self.config.versions_dir, "champion_current")
            try:
                model = tf.keras.models.load_model(f"{final_path}.keras")
            except Exception:
                self.logger.info("No Keras model found to optimize; skipping")
                return
            try:
                pruned = prune_model(model, final_sparsity=0.5, end_step=400)
                pruned = strip_pruning(pruned)
                pruned.save(f"{final_path}_pruned.keras")
                tflite = quantize_model(pruned)
                save_tflite(tflite, f"{final_path}.tflite")
                self.logger.info("Optimization export complete (pruned .keras and .tflite saved)")
            except Exception as e:
                self.logger.warning(f"Optimization export failed: {e}")

    # Build champion config
    c_cfg = TrainingConfig(mode=args.mode)
    if args.episodes:
        c_cfg.stage2_selfplay_episodes = args.episodes
        c_cfg.stage3_vicarious_episodes = args.episodes
    if args.cfr_iterations:
        c_cfg.stage1_cfr_iterations = args.cfr_iterations
    if args.model_dir:
        c_cfg.model_dir = args.model_dir
        c_cfg.versions_dir = os.path.join(args.model_dir, 'versions')
        c_cfg.checkpoints_dir = os.path.join(args.model_dir, 'checkpoints')
    if args.batch_size:
        c_cfg.batch_size = args.batch_size

    # Initialize champion agent (prefer resume if available)
    logger = Logger(verbose=args.verbose)
    latest_model_path = os.path.join(c_cfg.versions_dir, "champion_best")
    if os.path.exists(f"{latest_model_path}.cfr") or os.path.exists(f"{latest_model_path}.keras"):
        logger.info(f"Resuming from latest champion model: {latest_model_path}")
        agent = ChampionAgent(name="Champion", use_pretrained=True, use_deepstack=True)
        agent.load_strategy(latest_model_path)
    else:
        logger.info("Initializing Champion Agent with pre-trained models...")
        agent = ChampionAgent(name="Champion", use_pretrained=True, use_deepstack=True)

    trainer = ProgressiveTrainer(c_cfg, agent, verbose=args.verbose)
    try:
        trainer.train()
        logger.info("Champion training completed successfully")
        if args.report:
            try:
                from scripts.run_analysis_report import run_analysis_report
                run_analysis_report(samples_dir=c_cfg.samples_path, report_dir=c_cfg.report_dir)
                logger.info("Post-training analysis report generated")
            except Exception as e:
                logger.warning(f"Post-training analysis report failed: {e}")
        return
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
