"""Training framework for poker agents."""

import os
from typing import List, Optional

import numpy as np

from src.agents import BaseAgent, DQNAgent
from ..game import Action, BettingRound, GameState


class Trainer:
    """Trains poker agents through simulated games."""
    
    def __init__(self, agent: BaseAgent, opponent: Optional[BaseAgent] = None):
        """
        Initialize trainer.
        
        Args:
            agent: Agent to train
            opponent: Opponent agent (defaults to random if not specified)
        """
        self.agent = agent
        self.opponent = opponent
        
        if self.opponent is None:
            from src.agents import RandomAgent
            self.opponent = RandomAgent("Opponent")
    
    def train(self, num_episodes: int = 1000, 
             batch_size: int = 32,
             save_interval: int = 100,
             save_dir: str = "models",
             verbose: bool = True):
        """
        Train agent through self-play.
        
        Args:
            num_episodes: Number of training episodes
            batch_size: Batch size for experience replay
            save_interval: Save model every N episodes
            save_dir: Directory to save models
            verbose: Print training progress
        """
        if not isinstance(self.agent, DQNAgent):
            print("Warning: Only DQN agents can be trained")
            return
        
        # Create save directory
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
            
            # Train agent
            if len(self.agent.memory) > batch_size:
                self.agent.replay(batch_size)
            
            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                win_rate = wins / (episode + 1)
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Win Rate: {win_rate:.2%} - "
                      f"Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                model_path = os.path.join(save_dir, f"model_episode_{episode + 1}.h5")
                self.agent.save_model(model_path)
                if verbose:
                    print(f"Model saved to {model_path}")
        
        # Save final model
        final_path = os.path.join(save_dir, "model_final.h5")
        self.agent.save_model(final_path)
        if verbose:
            print(f"Training complete. Final model saved to {final_path}")
    
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
                if states:
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
