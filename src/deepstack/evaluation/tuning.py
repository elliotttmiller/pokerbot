"""Hyperparameter tuning utilities using Optuna (with optional Ray Tune).

This module exposes study runners to tune both baseline DQN agents and the
unified PokerBot agent. Objectives are intentionally lightweight so they
can be invoked from CI or smoketests without running the full training
pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from src.agents import create_agent
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from ..evaluation.trainer import UnifiedTrainer
from ..game import Action, GameState

# ---------------------------------------------------------------------------
# DQN objective (legacy support)
# ---------------------------------------------------------------------------


@dataclass
class TuningConfig:
    episodes: int = 300
    batch_size: int = 32
    study_name: str = "dqn_tuning"
    storage: Optional[str] = None  # e.g. "sqlite:///optuna.db"
    direction: str = "maximize"


def objective(trial, episodes: int, batch_size: int) -> float:
    """Optuna objective for DQNAgent.

    Returns average reward over last 50 episodes.
    """
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.85, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.05, 1.0)
    epsilon_min = trial.suggest_float("epsilon_min", 0.001, 0.1)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.90, 0.9999)
    mem_size = trial.suggest_int("memory_size", 1000, 20000, log=True)

    # Create agent with sampled params
    agent = DQNAgent(
        state_size=60,
        action_size=3,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        memory_size=mem_size,
    )

    # Train and evaluate
    trainer = UnifiedTrainer(agent, training_mode='dqn')
    rewards = []
    for ep in range(episodes):
        r = trainer._play_training_hand(ep)
        rewards.append(r)
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)

    avg_last = float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards))
    return avg_last

# ---------------------------------------------------------------------------
# PokerBot agent objective
# ---------------------------------------------------------------------------


@dataclass
class PokerBotTuningConfig:
    """Configuration for tuning PokerBot agent ensemble weights."""

    hands: int = 30
    batch_size: int = 16
    cfr_warmup: int = 200
    study_name: str = "pokerbot_tuning"
    storage: Optional[str] = None
    direction: str = "maximize"
    use_deepstack: bool = True


def _play_quick_hand(agent, opponent: RandomAgent) -> float:
    """Simulate a single hand for tuning purposes (lightweight)."""

    game = GameState(num_players=2)
    game.reset()

    agent_idx = 0
    opponent_idx = 1

    states = []
    actions = []

    done = False
    while not done:
        for current_idx in [agent_idx, opponent_idx]:
            player = game.players[current_idx]
            if player.folded or player.all_in:
                continue

            hole_cards = player.hand
            community_cards = game.community_cards
            pot = game.pot
            current_bet = game.current_bet - player.current_bet
            stack = player.stack

            if current_idx == agent_idx:
                # Handle different agent types
                if hasattr(agent, '_encode_dqn_state'):
                    state = agent._encode_dqn_state(hole_cards, community_cards, pot, current_bet, stack, current_bet)
                elif hasattr(agent, '_encode_state'):
                    state = agent._encode_state(hole_cards, community_cards, pot, current_bet, stack, current_bet)
                else:
                    state = np.zeros(120)
                
                states.append(state)
                action, raise_amt = agent.choose_action(hole_cards, community_cards, pot, current_bet, stack, current_bet)
                if action == Action.FOLD:
                    actions.append(0)
                elif action in (Action.CALL, Action.CHECK):
                    actions.append(1)
                else:
                    actions.append(2)
            else:
                action, raise_amt = opponent.choose_action(hole_cards, community_cards, pot, current_bet, stack, current_bet)

            try:
                game.apply_action(current_idx, action, raise_amt)
            except Exception:
                done = True
                break

            if game.is_hand_complete():
                done = True
                break

        if not done:
            active_players = [p for p in game.players if not p.folded and not p.all_in]
            if active_players:
                bets = [p.current_bet for p in active_players]
                if len(set(bets)) == 1:
                    try:
                        game.advance_betting_round()
                    except Exception:
                        done = True

    try:
        winners = game.get_winners()
        won = agent_idx in winners
        reward = game.pot / len(winners) if won and winners else -game.players[agent_idx].current_bet
    except Exception:
        won = False
        reward = -game.players[agent_idx].current_bet

    if states and actions:
        for i, (state, action_idx) in enumerate(zip(states, actions)):
            step_reward = reward if i == len(states) - 1 else 0.0
            next_state = states[i + 1] if i < len(states) - 1 else state
            agent.remember(state, action_idx, step_reward, next_state, True)

    return float(reward)


def pokerbot_objective(trial, cfg: PokerBotTuningConfig) -> float:
    """Optuna objective targeting PokerBot agent ensemble weights and params."""

    cfr_weight = trial.suggest_float("cfr_weight", 0.2, 0.6)
    dqn_weight = trial.suggest_float("dqn_weight", 0.2, 0.6)
    deepstack_weight = trial.suggest_float("deepstack_weight", 0.1, 0.5)
    epsilon = trial.suggest_float("epsilon", 0.05, 0.4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    cfr_iterations = trial.suggest_int("cfr_iterations", 150, 1200, step=50)

    agent = create_agent(
        'pokerbot',
        use_pretrained=False,
        cfr_weight=cfr_weight,
        dqn_weight=dqn_weight,
        deepstack_weight=deepstack_weight,
        epsilon=epsilon,
        learning_rate=learning_rate,
        gamma=gamma,
        use_deepstack=cfg.use_deepstack,
    )

    if cfr_iterations > 0:
        try:
            agent.train_cfr(num_iterations=cfr_iterations)
        except Exception:
            pass

    opponent = RandomAgent("TuningOpponent")

    rewards = []
    for _ in range(cfg.hands):
        reward = _play_quick_hand(agent, opponent)
        rewards.append(reward)
        if hasattr(agent, 'memory') and hasattr(agent, 'replay'):
            if len(agent.memory) >= cfg.batch_size:
                agent.replay(cfg.batch_size)

    return float(np.mean(rewards)) if rewards else 0.0


def run_optuna_study(
    n_trials: int = 20,
    config: Optional[Union[TuningConfig, PokerBotTuningConfig]] = None,
    target: str = "dqn",
):
    import optuna

    if target == "pokerbot":
        cfg = config or PokerBotTuningConfig()
    else:
        cfg = config or TuningConfig()

    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=cfg.storage,
        direction=cfg.direction,
        sampler=sampler,
        load_if_exists=True,
    )

    if target == "pokerbot":
        study.optimize(lambda t: pokerbot_objective(t, cfg), n_trials=n_trials)
    else:
        study.optimize(lambda t: objective(t, cfg.episodes, cfg.batch_size), n_trials=n_trials)
    return study
