#!/usr/bin/env python3
"""Champion training pipeline (single entrypoint).

Usage examples:
    - Smoketest:        python fixed_train.py --mode smoketest
    - Standard:         python fixed_train.py --mode standard
    - Full/production:  python fixed_train.py --mode full --episodes 10000
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Load .env for PYTHONPATH and other env vars
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath and pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"[DEBUG] PYTHONPATH from .env: {pythonpath}")
print(f"[DEBUG] src_path resolved: {src_path}")
print(f"[DEBUG] sys.path: {sys.path}")

from utils import Logger
from agents import create_agent, PokerBotAgent
from deepstack.game import Card, GameState, Action
from utils.data_validation import validate_deepstacked_samples, validate_equity_table
from agents.random_agent import RandomAgent
from agents.fixed_strategy_agent import FixedStrategyAgent
from agents.cfr_agent import CFRAgent
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
        self.samples_path = cfg.get("samples_path", r"C:\Users\AMD\pokerbot\src\train_samples")
        self.test_samples_path = cfg.get("test_samples_path", "data/deepstacked_training/samples/test_samples")
        self.equity_path = cfg.get("equity_path", "data/equity_tables/preflop_equity.json")
        self.model_dir = "models"
        self.versions_dir = "models/versions"
        self.checkpoints_dir = "models/checkpoints"
        self.log_dir = cfg.get("log_dir", "logs")
        self.report_dir = cfg.get("report_dir", "models/reports")
        self.checkpoint_prefix = "pokerbot"
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
    def __init__(self, config: TrainingConfig, agent, agent_type: str = "pokerbot", verbose: bool = True):
        self.config = config
        self.agent = agent
        self.agent_type = agent_type
        self.verbose = verbose
        self.logger = Logger(verbose=verbose)
        self.checkpoints_dir = "models/checkpoints"
        self.log_dir = self.config.log_dir
        self.report_dir = self.config.report_dir
        self.checkpoint_prefix = "pokerbot"
        os.makedirs(self.config.report_dir, exist_ok=True)
        self.opponents = self._create_diverse_opponents()
        self.stats = {
            'start_time': time.time(),
            'current_stage': 0,
            'total_episodes': 0,
            'total_hands_played': 0,
            'stage_metrics': defaultdict(list),
        }
        self.last_eval_rewards = []
    def _promote_to_best_if_better(self, comp_result):
        import shutil
        if comp_result.get('is_better', False):
            self.logger.info("Promoting current model to best (champion_best)")
            src_base = os.path.join(self.config.versions_dir, "champion_current")
            dst_base = os.path.join(self.config.versions_dir, "champion_best")
            for ext in [".pt", ".keras", ".cfr", ".json"]:
                src = f"{src_base}{ext}"
                dst = f"{dst_base}{ext}"
                if os.path.exists(src):
                    try:
                        shutil.copy2(src, dst)
                        self.logger.info(f"Promoted {src} -> {dst}")
                    except Exception as e:
                        self.logger.warning(f"Failed to promote {src} to {dst}: {e}")
        else:
            self.logger.info("Current model did not outperform previous best; not promoted.")
    def _create_diverse_opponents(self):
        return {
            'random': [RandomAgent(f"Random{i}") for i in range(2)],
            'fixed': [FixedStrategyAgent(f"Fixed{i}") for i in range(2)],
            'cfr': [CFRAgent(f"CFR{i}") for i in range(2)],
        }
    def run(self):
        self.logger.info(f"Starting championship training pipeline with {self.agent_type} agent")
        self._stage1_cfr_warmup()
        self._stage2_selfplay()
        self._stage3_vicarious_learning()
        # Only save/export if not smoketest
        if self.config.mode != "smoketest":
            self._save_checkpoint()
            self.logger.info("Training pipeline complete.")
            self._export_report()
        else:
            self.logger.info("Smoketest complete. No model files saved/exported.")
    def _export_report(self):
        report_dir = os.path.join("models", "reports")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"training_report_{timestamp}.txt")
        with open(report_path, "w") as f:
            f.write("="*70 + "\n")
            f.write(f"POKERBOT AGENT TRAINING REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Agent Type: {self.agent_type}\n")
            f.write(f"Training Mode: {self.config.mode.upper()}\n")
            f.write(f"Episodes: {self.config.stage2_selfplay_episodes}\n")
            f.write(f"Vicarious Episodes: {self.config.stage3_vicarious_episodes}\n")
            f.write(f"Total Hands Played: {self.stats.get('total_hands_played', 0)}\n")
            f.write(f"Total Episodes: {self.stats.get('total_episodes', 0)}\n")
            f.write(f"Batch Size: {self.config.batch_size}\n")
            f.write(f"Evaluation Interval: {self.config.evaluation_interval}\n")
            f.write("\n---\n")
            f.write("STAGE SUMMARY\n")
            f.write(f"  Stage 1 (CFR Warmup): {self.config.stage1_cfr_iterations} iterations\n")
            f.write(f"  Stage 2 (Self-Play): {self.config.stage2_selfplay_episodes} episodes\n")
            f.write(f"  Stage 3 (Vicarious): {self.config.stage3_vicarious_episodes} episodes\n")
            f.write("\n---\n")
            f.write("PROGRESS & METRICS\n")
            win_rates = self.config.metrics.get('win_rates', [])
            episode_reward_mean = self.config.metrics.get('episode_reward_mean', [])
            training_rewards = self.config.metrics.get('training_rewards', [])
            epsilon_values = self.config.metrics.get('epsilon_values', [])
            f.write(f"Win Rates: {win_rates}\n")
            f.write(f"Episode Reward Mean: {episode_reward_mean}\n")
            f.write(f"Training Rewards: {training_rewards}\n")
            f.write(f"Epsilon Values: {epsilon_values}\n")
            if hasattr(self.agent, 'memory'):
                f.write(f"Memory Size: {len(self.agent.memory)}\n")
            f.write("\n---\n")
            f.write("VICARIOUS LEARNING PERFORMANCE\n")
            f.write("| Opponent   | Win Rate | Avg Reward | Games |\n")
            f.write("|------------|----------|------------|-------|\n")
            vicarious_stats = self.stats.get('vicarious_opponent_stats', {})
            for opp, stats in vicarious_stats.items():
                f.write(f"| {opp:10} | {stats['win_rate']*100:8.2f}% | {stats['avg_reward']:10.2f} | {stats['games']:5d} |\n")
            f.write("\n---\n")
            f.write("MODEL COMPARISON VS PREVIOUS BEST\n")
            comp_result = self._compare_against_previous_best(num_hands=100)
            if comp_result.get('previous_best_exists', False):
                f.write(f"Win Rate: {comp_result['win_rate']*100:.2f}%\n")
                f.write(f"Avg Reward: {comp_result['avg_reward']:.2f}\n")
                f.write(f"Result: [{'PASS' if comp_result['is_better'] else 'FAIL'}] {'Current model is better' if comp_result['is_better'] else 'Previous model is still better'}\n")
            else:
                f.write("No previous best model found. Current model set as best.\n")
            f.write("\n---\n")
            f.write("NOTES\n")
            f.write("- All metrics are computed at the end of training.\n")
            f.write("- For full per-opponent breakdown, see logs or implement aggregation.\n")
            f.write("- For advanced analysis, export metrics to CSV or visualization tools.\n")
        print(f"[REPORT] Training report saved to {report_path}")
        comp_result = self._compare_against_previous_best(num_hands=100)
        self._promote_to_best_if_better(comp_result)
        self._save_metrics()
    def _stage1_cfr_warmup(self):
        self.agent.train_cfr(num_iterations=self.config.stage1_cfr_iterations)
    def _stage2_selfplay(self):
        opponent = PokerBotAgent(name="SelfPlayOpponent", use_pretrained=False, epsilon=0.1)
        wins = 0
        recent_rewards = []
        for ep in range(self.config.stage2_selfplay_episodes):
            print(f"[PROGRESS] Self-play episode {ep+1}/{self.config.stage2_selfplay_episodes}")
            reward, won = self._play_training_hand(opponent, episode=ep+1)
            if won:
                wins += 1
            bs = self.config.batch_size
            if len(self.agent.memory) >= bs:
                self.agent.replay(bs)
            self.config.metrics['training_rewards'].append(float(reward))
            self.config.metrics['epsilon_values'].append(float(getattr(self.agent, 'epsilon', 0.0)))
            recent_rewards.append(reward)
            self.stats['total_episodes'] += 1
            self.stats['total_hands_played'] += 1
            print(f"[SUMMARY] Ep {ep+1}: Reward={reward:.2f}, Win={won}, Mem={len(self.agent.memory)}")
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
        rewards = defaultdict(list)
        recent_rewards = []
        for ep in range(self.config.stage3_vicarious_episodes):
            print(f"[PROGRESS] Vicarious episode {ep+1}/{self.config.stage3_vicarious_episodes}")
            opp = eval_opps[ep % len(eval_opps)]
            reward, won = self._play_training_hand(opp, episode=ep+1)
            wins[opp.name] += int(won)
            rewards[opp.name].append(reward)
            bs = self.config.batch_size
            if len(self.agent.memory) >= bs:
                for _ in range(2):
                    self.agent.replay(bs)
            self.config.metrics['training_rewards'].append(float(reward))
            self.config.metrics['epsilon_values'].append(float(getattr(self.agent, 'epsilon', 0.0)))
            recent_rewards.append(reward)
            self.stats['total_episodes'] += 1
            self.stats['total_hands_played'] += 1
            print(f"[SUMMARY] Ep {ep+1}: Reward={reward:.2f}, Win={won}, Mem={len(self.agent.memory)}")
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
        # Save per-opponent stats for report
        self.stats['vicarious_opponent_stats'] = {}
        for opp in eval_opps:
            name = opp.name
            total = wins[name]
            count = len(rewards[name])
            avg_reward = float(np.mean(rewards[name])) if rewards[name] else 0.0
            win_rate = total / count if count > 0 else 0.0
            self.stats['vicarious_opponent_stats'][name] = {
                'win_rate': win_rate,
                'avg_reward': avg_reward,
                'games': count
            }
    def _play_training_hand(self, opponent, episode=None):
        game = GameState(num_players=2)
        game.reset()
        # Reset lookahead build flag for agent and opponent
        if hasattr(self.agent, 'start_new_hand'):
            self.agent.start_new_hand()
        if hasattr(opponent, 'start_new_hand'):
            opponent.start_new_hand()
        agent_idx, opp_idx = 0, 1
        done = False
        max_actions = 200
        actions = 0
        # Build lookahead tree once per hand
        print(f"[Lookahead] Built lookahead tree, depth=7 (hand {episode})")
        while not done:
            print(f"[DEBUG] Loop start: actions={actions}, episode={episode}")
            print(f"[DEBUG] is_hand_complete={game.is_hand_complete()}")
            for idx, p in enumerate(game.players):
                print(f"[DEBUG] Player {idx}: folded={p.folded}, all_in={p.all_in}, stack={p.stack}, current_bet={p.current_bet}")
            if game.is_hand_complete():
                print(f"[DEBUG] Hand complete detected, breaking loop.")
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
                print(f"[DEBUG] Player {current} action: folded={player.folded}, all_in={player.all_in}, stack={player_stack}, current_bet={player.current_bet}")
                if current == agent_idx:
                    state = self.agent._encode_dqn_state(hole_cards, community_cards, pot, current_bet, player_stack, current_bet)
                    action, raise_amount = self.agent.choose_action(hole_cards, community_cards, pot, current_bet, player_stack, current_bet)
                    print(f"[DEBUG] Agent action: {action}, raise_amount={raise_amount}")
                    self.agent.remember(state, 1 if action in (Action.CALL, Action.CHECK) else (0 if action == Action.FOLD else 2), 0.0, state, False)
                else:
                    action, raise_amount = opponent.choose_action(hole_cards, community_cards, pot, current_bet, player_stack, current_bet)
                    print(f"[DEBUG] Opponent action: {action}, raise_amount={raise_amount}")
                try:
                    game.apply_action(current, action, raise_amount)
                    print(f"[DEBUG] Applied action for player {current}")
                except Exception as e:
                    print(f"[ERROR] Exception in apply_action: {e}")
                    done = True
                    break
                print(f"[DEBUG] is_hand_complete after action={game.is_hand_complete()}")
                if game.is_hand_complete():
                    print(f"[DEBUG] Hand complete detected after action, breaking loop.")
                    done = True
                    break
                actions += 1
                if actions % 10 == 0:
                    print(f"[PROGRESS] Episode {episode} Hand {actions}")
                if actions >= max_actions:
                    print(f"[WARNING] Max actions ({max_actions}) reached in episode {episode}. Possible infinite loop.")
                    done = True
                    break
            if not done:
                active = [p for p in game.players if not p.folded and not p.all_in]
                print(f"[DEBUG] Active players: {[i for i, p in enumerate(game.players) if not p.folded and not p.all_in]}")
                # If no active players remain, forcibly terminate hand to prevent infinite loop
                if not active:
                    print(f"[FIX] No active players remain, forcing hand complete and breaking loop.")
                    done = True
                    break
                if active:
                    bets = [p.current_bet for p in active]
                    print(f"[DEBUG] Active bets: {bets}")
                    if len(set(bets)) == 1:
                        try:
                            print(f"[DEBUG] Advancing betting round.")
                            game.advance_betting_round()
                        except Exception as e:
                            print(f"[ERROR] Exception in advance_betting_round: {e}")
                            done = True
        try:
            winners = game.get_winners()
            print(f"[DEBUG] Winners: {winners}")
            won = (0 in winners)
            reward = game.pot / len(winners) if won else -game.players[0].current_bet
        except Exception as e:
            print(f"[ERROR] Exception in get_winners: {e}")
            won = False
            reward = -game.players[0].current_bet if game.players[0].current_bet > 0 else 0
        return reward, won
    def _compare_against_previous_best(self, num_hands=100):
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
            previous_best = PokerBotAgent(name="PreviousBest", use_pretrained=True)
            previous_best.load_models(best_path)
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
    def _play_comparison_hand(self, opponent):
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
    def _save_checkpoint(self, final=False, stage=None):
        if final:
            filepath = os.path.join(self.config.versions_dir, "champion_current")
        elif stage is not None:
            filepath = os.path.join(self.config.checkpoints_dir, f"{self.config.checkpoint_prefix}")
        else:
            filepath = os.path.join(self.config.checkpoints_dir, self.config.checkpoint_prefix)
        try:
            self.agent.save_models(filepath)
            # Save stats to stats file in checkpoints dir
            stats_path = os.path.join(self.config.checkpoints_dir, "pokerbot_stats.json")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f, indent=2)
            self.logger.info(f"Saved stats to {stats_path}")
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

def main():
    parser = argparse.ArgumentParser(description='Train the championship-level agent (progressive pipeline)')
    parser.add_argument('--mode', type=str, choices=['smoketest', 'standard', 'intermediate', 'production', 'full'], default='smoketest', help='Training mode selects configuration profile from scripts/config')
    parser.add_argument('--agent-type', type=str, choices=['champion', 'pokerbot'], default='pokerbot', help='Agent type to train (champion or pokerbot - default: pokerbot)')
    parser.add_argument('--episodes', type=int, default=None, help='Override number of episodes for stage 2 and 3')
    parser.add_argument('--cfr-iterations', type=int, default=None, help='Override CFR warmup iterations')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models and versions')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size used during replay')
    parser.add_argument('--optimize-export', action='store_true', help='After training, attempt prune/quantize export (requires TensorFlow)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--report', action='store_true', help='Generate analysis plots and metrics after training')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    args = parser.parse_args()
    logger = Logger(verbose=args.verbose)
    logger.info(f"Starting championship training pipeline with {args.agent_type} agent")
    try:
        np.random.seed(args.seed)
        random.seed(args.seed)
    except Exception:
        pass
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
    versions_dir = c_cfg.versions_dir
    final_pt_path = os.path.join(versions_dir, "final_model.pt")
    best_pt_path = os.path.join(versions_dir, "best_model.pt")
    legacy_best_path = os.path.join(versions_dir, "champion_best")
    logger.info("Using unified PokerBot agent")
    agent = create_agent('pokerbot', name="PokerBot", use_pretrained=True)
    if os.path.exists(final_pt_path):
        logger.info(f"Resuming from final model: {final_pt_path}")
        agent.load_models(final_pt_path)
    elif os.path.exists(best_pt_path):
        logger.info(f"Resuming from best model: {best_pt_path}")
        agent.load_models(best_pt_path)
    elif os.path.exists(f"{legacy_best_path}.cfr") or os.path.exists(f"{legacy_best_path}.keras"):
        logger.info(f"Resuming from legacy model: {legacy_best_path}")
        agent.load_models(legacy_best_path)
    else:
        logger.info("Initializing PokerBot Agent with pre-trained models...")
    trainer = ProgressiveTrainer(c_cfg, agent, agent_type=args.agent_type, verbose=args.verbose)
    try:
        trainer.run()
        logger.info("Champion training completed successfully")
        if args.report:
            try:
                from scripts.run_analysis_report import run_analysis_report
                run_analysis_report(samples_dir=c_cfg.samples_path, report_dir=c_cfg.report_dir)
                logger.info("Post-training analysis report generated")
            except Exception as e:
                logger.warning(f"Post-training analysis report failed: {e}")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
