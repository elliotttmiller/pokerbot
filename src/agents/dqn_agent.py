"""Deep Q-Network agent for poker."""

import random
from collections import deque
from typing import List, Optional

import numpy as np

from ..game import Action, Card
from .base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Deep Q-Network reinforcement learning agent."""
    
    def __init__(self, 
                 state_size: int = 60,
                 action_size: int = 3,
                 name: str = "DQNAgent",
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 2000):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state representation
            action_size: Number of possible actions
            name: Agent name
            learning_rate: Learning rate for neural network
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of replay memory
        """
        super().__init__(name)
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Neural network model
        self.model = None
        self._build_model()
        
        # Track training
        self.training_mode = True
    
    def _build_model(self):
        """Build the neural network model."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Input
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential()
            model.add(Input(shape=(self.state_size,)))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            
            self.model = model
        except ImportError:
            print("Warning: TensorFlow not available. DQN agent will use random actions.")
            self.model = None
    
    def _encode_state(self,
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> np.ndarray:
        """
        Encode game state as a feature vector.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot: Current pot
            current_bet: Current bet to call
            player_stack: Player's stack
            opponent_bet: Opponent's bet
        
        Returns:
            NumPy array representing state
        """
        # Initialize state vector
        state = np.zeros(self.state_size)
        
        # Encode hole cards (2 cards * 2 features = 4)
        for i, card in enumerate(hole_cards[:2]):
            state[i * 2] = card.rank / 12.0  # Normalize rank
            state[i * 2 + 1] = card.suit / 3.0  # Normalize suit
        
        # Encode community cards (5 cards * 2 features = 10)
        for i, card in enumerate(community_cards[:5]):
            idx = 4 + i * 2
            state[idx] = card.rank / 12.0
            state[idx + 1] = card.suit / 3.0
        
        # Encode game state (normalized values)
        idx = 14
        state[idx] = min(pot / 1000.0, 1.0)  # Pot size
        state[idx + 1] = min(current_bet / 500.0, 1.0)  # Current bet
        state[idx + 2] = min(player_stack / 1000.0, 1.0)  # Player stack
        state[idx + 3] = min(opponent_bet / 500.0, 1.0)  # Opponent bet
        state[idx + 4] = len(community_cards) / 5.0  # Betting round indicator
        
        return state
    
    def choose_action(self,
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> tuple[Action, int]:
        """
        Choose action using DQN or exploration.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot: Current pot
            current_bet: Current bet to call
            player_stack: Player's stack
            opponent_bet: Opponent's bet
        
        Returns:
            Tuple of (Action, raise_amount)
        """
        # Encode state
        state = self._encode_state(hole_cards, community_cards, pot, 
                                   current_bet, player_stack, opponent_bet)
        
        # Exploration vs exploitation
        if self.training_mode and self.model and np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        elif self.model:
            # Use model to predict best action
            q_values = self.model.predict(np.array([state]), verbose=0)
            action_idx = np.argmax(q_values[0])
        else:
            # Fallback to random if no model
            action_idx = random.randrange(self.action_size)
        
        # Map action index to Action enum
        if action_idx == 0:
            return Action.FOLD, 0
        elif action_idx == 1:
            if current_bet == 0:
                return Action.CHECK, 0
            else:
                return Action.CALL, 0
        else:  # action_idx == 2
            # Raise: use percentage of pot
            raise_amount = max(int(pot * 0.5), 20)
            raise_amount = min(raise_amount, player_stack)
            return Action.RAISE, raise_amount
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """
        Train model on a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        """
        if not self.model or len(self.memory) < batch_size:
            return
        
        # Sample random batch
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare training data
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Q-learning update
                next_q_values = self.model.predict(np.array([next_state]), verbose=0)
                target = reward + self.gamma * np.amax(next_q_values[0])
            
            # Get current Q-values
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            
            states.append(state)
            targets.append(target_f[0])
        
        # Train model
        self.model.fit(np.array(states), np.array(targets), 
                      epochs=1, verbose=0, batch_size=batch_size)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """Save model to file."""
        if self.model:
            self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        if self.model:
            self.model.load_weights(filepath)
    
    def set_training_mode(self, training: bool):
        """Set whether agent is in training mode."""
        self.training_mode = training
        if not training:
            self.epsilon = 0.0  # No exploration during evaluation
