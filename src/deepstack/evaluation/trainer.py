"""Unified Training Framework for Poker Agents.

This unified trainer supports multiple training paradigms:
- DQN/Deep Q-Network training through self-play
- CFR/CFR+ training through regret minimization
- DeepStack neural network training
- Distributed CFR training across multiple CPU cores

The trainer automatically detects the agent type and uses the appropriate
training method.
"""

import os
from typing import List, Optional, Dict, Any
import multiprocessing as mp

import numpy as np

from src.agents import BaseAgent
from ..game import Action, BettingRound, GameState


class UnifiedTrainer:
    """Unified trainer supporting DQN, CFR, DeepStack, and distributed training."""
    
    def __init__(self, agent: BaseAgent, opponent: Optional[BaseAgent] = None,
                 training_mode: str = 'auto', n_workers: Optional[int] = None):
        """
        Initialize unified trainer.
        
        Args:
            agent: Agent to train
            opponent: Opponent agent (defaults to random if not specified)
            training_mode: Training mode ('auto', 'dqn', 'cfr', 'deepstack', 'distributed')
            n_workers: Number of worker processes for distributed training
        """
        self.agent = agent
        self.opponent = opponent
        self.training_mode = training_mode
        self.n_workers = n_workers or mp.cpu_count()
        
        if self.opponent is None:
            from src.agents import RandomAgent
            self.opponent = RandomAgent("Opponent")
        
        # Auto-detect training mode if needed
        if self.training_mode == 'auto':
            self.training_mode = self._detect_training_mode()
    
    def _detect_training_mode(self) -> str:
        """Detect appropriate training mode based on agent type."""
        agent_class = self.agent.__class__.__name__
        
        if 'DQN' in agent_class:
            return 'dqn'
        elif 'CFR' in agent_class:
            return 'cfr'
        elif hasattr(self.agent, 'use_dqn') and self.agent.use_dqn:
            return 'dqn'
        elif hasattr(self.agent, 'use_cfr') and self.agent.use_cfr:
            return 'cfr'
        elif hasattr(self.agent, 'dqn_model') and self.agent.dqn_model is not None:
            return 'dqn'
        else:
            return 'dqn'  # Default to DQN
    
    def train(self, num_episodes: int = 1000, 
             batch_size: int = 32,
             save_interval: int = 100,
             save_dir: str = "models",
             verbose: bool = True,
             **kwargs):
        """
        Train agent using appropriate method.
        
        Args:
            num_episodes: Number of training episodes
            batch_size: Batch size for experience replay
            save_interval: Save model every N episodes
            save_dir: Directory to save models
            verbose: Print training progress
            **kwargs: Additional training parameters
        """
        if self.training_mode == 'dqn':
            return self._train_dqn(num_episodes, batch_size, save_interval, save_dir, verbose)
        elif self.training_mode == 'cfr':
            return self._train_cfr(num_episodes, save_dir, verbose)
        elif self.training_mode == 'distributed':
            return self._train_distributed_cfr(num_episodes, save_dir, verbose)
        else:
            return self._train_dqn(num_episodes, batch_size, save_interval, save_dir, verbose)
    
    def _train_dqn(self, num_episodes: int, batch_size: int, save_interval: int,
                   save_dir: str, verbose: bool):
        """Train agent using DQN through self-play."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Training statistics
        total_rewards = []
        wins = 0
        
        for episode in range(num_episodes):
            # Play one hand
            reward = self._play_training_hand(episode)
            total_rewards.append(reward)
            
            if reward > 0:
                wins += 1
            
            # Train agent (DQN replay)
            if hasattr(self.agent, 'replay') and hasattr(self.agent, 'memory'):
                if len(self.agent.memory) > batch_size:
                    self.agent.replay(batch_size)
            
            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                win_rate = wins / (episode + 1)
                epsilon = getattr(self.agent, 'epsilon', 0.0)
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Win Rate: {win_rate:.2%} - "
                      f"Epsilon: {epsilon:.3f}")
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                self._save_agent(save_dir, f"episode_{episode + 1}")
                if verbose:
                    print(f"Model saved at episode {episode + 1}")
        
        # Save final model
        self._save_agent(save_dir, "final")
        if verbose:
            print(f"Training complete. Final model saved to {save_dir}")
    
    def _train_cfr(self, num_iterations: int, save_dir: str, verbose: bool):
        """Train agent using CFR."""
        os.makedirs(save_dir, exist_ok=True)
        
        if hasattr(self.agent, 'train_cfr'):
            # PokerBotAgent with CFR component
            self.agent.train_cfr(num_iterations)
        elif hasattr(self.agent, 'train'):
            # CFRAgent
            self.agent.train(num_iterations)
        else:
            print("Warning: Agent does not support CFR training")
            return
        
        # Save strategy
        self._save_agent(save_dir, "cfr_final")
        if verbose:
            print(f"CFR training complete. Strategy saved to {save_dir}")
    
    def _train_distributed_cfr(self, num_iterations: int, save_dir: str, verbose: bool):
        """Train agent using distributed CFR (multiprocessing)."""
        # Implementation for distributed CFR training
        # This would use multiprocessing to speed up CFR iterations
        print("Distributed CFR training not yet fully implemented")
        # Fall back to regular CFR for now
        return self._train_cfr(num_iterations, save_dir, verbose)
    
    def _save_agent(self, save_dir: str, suffix: str):
        """Save agent models."""
        if hasattr(self.agent, 'save_models'):
            # PokerBotAgent
            self.agent.save_models(os.path.join(save_dir, suffix))
        elif hasattr(self.agent, 'save_model'):
            # DQNAgent
            model_path = os.path.join(save_dir, f"model_{suffix}.h5")
            self.agent.save_model(model_path)
        elif hasattr(self.agent, 'save_strategy'):
            # CFRAgent
            self.agent.save_strategy(os.path.join(save_dir, suffix))
    
    def _play_training_hand(self, episode_num: int) -> float:
        """
        Play one training hand.
        
        Args:
            episode_num: Current episode number
        
        Returns:
            Total reward for the hand
        """
        game = GameState(num_players=2)
        game.reset()
        
        total_reward = 0
        done = False
        agent_idx = 0
        opponent_idx = 1
        
        # Track states and actions for learning
        states = []
        actions_taken = []
        
        while not done:
            current_player_idx = agent_idx if len(states) % 2 == 0 else opponent_idx
            player = game.players[current_player_idx]
            
            if player.folded or player.all_in:
                # Skip if player has folded or is all-in
                if game.betting_round == game.betting_round.RIVER:
                    done = True
                else:
                    game.advance_betting_round()
                continue
            
            # Get current state
            hole_cards = player.hand
            community_cards = game.community_cards
            pot = game.pot
            current_bet = game.current_bet - player.current_bet
            player_stack = player.stack
            
            # Choose action
            if current_player_idx == agent_idx:
                # Agent's turn
                if hasattr(self.agent, '_encode_dqn_state'):
                    state = self.agent._encode_dqn_state(
                        hole_cards, community_cards, pot,
                        current_bet, player_stack, current_bet
                    )
                elif hasattr(self.agent, '_encode_state'):
                    state = self.agent._encode_state(
                        hole_cards, community_cards, pot,
                        current_bet, player_stack, current_bet
                    )
                else:
                    state = np.zeros(120)  # Default state
                
                states.append(state)
                
                action, raise_amount = self.agent.choose_action(
                    hole_cards, community_cards, pot,
                    current_bet, player_stack, current_bet
                )
                actions_taken.append(self._action_to_index(action))
            else:
                # Opponent's turn
                action, raise_amount = self.opponent.choose_action(
                    hole_cards, community_cards, pot,
                    current_bet, player_stack, current_bet
                )
            
            # Apply action
            game.apply_action(current_player_idx, action, raise_amount)
            
            # Check if hand is complete
            if game.is_hand_complete():
                done = True
                
                # Calculate reward
                winners = game.get_winners()
                if agent_idx in winners:
                    reward = game.pot / len(winners)
                else:
                    reward = -player.current_bet
                
                total_reward += reward
                
                # Store final experiences
                if states and hasattr(self.agent, 'remember'):
                    final_state = states[-1]
                    for i, (state, action) in enumerate(zip(states, actions_taken)):
                        # Calculate step reward
                        step_reward = reward if i == len(states) - 1 else 0
                        next_state = states[i + 1] if i < len(states) - 1 else final_state
                        self.agent.remember(state, action, step_reward, next_state, done)
            
            # Advance betting round if needed
            elif not game.is_hand_complete():
                active_players = [p for p in game.players if not p.folded and not p.all_in]
                bets = [p.current_bet for p in active_players]
                
                if len(set(bets)) == 1:  # All equal bets
                    if game.betting_round != BettingRound.RIVER:
                        game.advance_betting_round()
                    else:
                        done = True
        
        return total_reward
    
    def _action_to_index(self, action: Action) -> int:
        """Convert Action enum to index."""
        if action == Action.FOLD:
            return 0
        elif action in [Action.CHECK, Action.CALL]:
            return 1
        else:
            return 2


# Backward compatibility - keep Trainer class as alias
Trainer = UnifiedTrainer
