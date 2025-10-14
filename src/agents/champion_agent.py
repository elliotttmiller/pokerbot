"""Champion Agent - Unified CFR + DQN hybrid with pre-trained model integration.

This agent combines the best of multiple approaches:
- CFR (Counterfactual Regret Minimization) for game-theoretic play
- DQN (Deep Q-Network) for learned patterns and adaptability  
- Pre-trained DeepStack models for champion-level value estimation
- Preflop equity tables for optimal early-game decisions

The result is a superior, championship-level poker agent that starts with
advanced pre-trained knowledge rather than from scratch.
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..game import Action, Card, HandEvaluator
from ..utils.model_loader import ModelLoader, TrainingDataManager
from .base_agent import BaseAgent
from .cfr_agent import CFRAgent, InfoSet


class ChampionAgent(BaseAgent):
    """
    Champion-level hybrid poker agent combining CFR, DQN, and pre-trained models.
    
    This agent unifies:
    1. CFR strategy computation for game-theoretic optimal play
    2. DQN neural network for pattern learning and adaptation
    3. Pre-trained DeepStack models for value estimation
    4. Preflop equity tables for informed early decisions
    
    The hybrid approach uses ensemble decision-making to leverage the strengths
    of each component.
    """
    
    def __init__(self,
                 name: str = "ChampionAgent",
                 state_size: int = 60,
                 action_size: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 0.3,  # Lower initial exploration
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 2000,
                 use_pretrained: bool = True,
                 cfr_weight: float = 0.4,
                 dqn_weight: float = 0.4,
                 equity_weight: float = 0.2):
        """
        Initialize Champion Agent.
        
        Args:
            name: Agent name
            state_size: Size of DQN state representation
            action_size: Number of possible actions
            learning_rate: Learning rate for DQN
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate (lower than pure DQN)
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of replay memory
            use_pretrained: Whether to load pre-trained models
            cfr_weight: Weight for CFR strategy in ensemble (0-1)
            dqn_weight: Weight for DQN strategy in ensemble (0-1)
            equity_weight: Weight for equity-based decisions (0-1)
        """
        super().__init__(name)
        
        # Initialize CFR component
        self.cfr = CFRAgent(name=f"{name}_CFR")
        
        # DQN components
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Neural network model
        self.model = None
        self._build_model()
        
        # Pre-trained models and data
        self.use_pretrained = use_pretrained
        self.model_loader = None
        self.data_manager = None
        
        if use_pretrained:
            self._load_pretrained_models()
        
        # Ensemble weights
        self.cfr_weight = cfr_weight
        self.dqn_weight = dqn_weight
        self.equity_weight = equity_weight
        self._normalize_weights()
        
        # Training mode
        self.training_mode = True
        
        print(f"🏆 {name} initialized with champion pre-trained knowledge!")
        print(f"   Strategy weights: CFR={self.cfr_weight:.2f}, "
              f"DQN={self.dqn_weight:.2f}, Equity={self.equity_weight:.2f}")
    
    def _normalize_weights(self):
        """Normalize ensemble weights to sum to 1.0."""
        total = self.cfr_weight + self.dqn_weight + self.equity_weight
        if total > 0:
            self.cfr_weight /= total
            self.dqn_weight /= total
            self.equity_weight /= total
    
    def _build_model(self):
        """Build the DQN neural network model."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Input, Dropout
            from tensorflow.keras.optimizers import Adam
            
            # Enhanced architecture for champion agent
            model = Sequential()
            model.add(Input(shape=(self.state_size,)))
            model.add(Dense(256, activation='relu'))  # Larger first layer
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            
            self.model = model
        except ImportError:
            print("Warning: TensorFlow not available. Champion agent will use CFR + equity only.")
            self.model = None
    
    def _load_pretrained_models(self):
        """Load pre-trained champion models and data."""
        try:
            self.model_loader = ModelLoader()
            self.data_manager = TrainingDataManager()
            
            # Try to load DeepStack model info
            try:
                deepstack_info = self.model_loader.load_deepstack_model(use_gpu=False)
                print(f"   ✓ Loaded DeepStack model ({deepstack_info['size_mb']:.1f} MB)")
            except FileNotFoundError:
                print("   ⚠ DeepStack models not found (continuing without)")
            
            # Load preflop equity tables
            try:
                equity_data = self.data_manager.load_preflop_equity()
                print(f"   ✓ Loaded preflop equity table ({len(equity_data)} hands)")
            except FileNotFoundError:
                print("   ⚠ Equity tables not found (continuing without)")
                
        except Exception as e:
            print(f"   ⚠ Error loading pre-trained models: {e}")
            self.model_loader = None
            self.data_manager = None
    
    def _encode_state(self,
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> np.ndarray:
        """
        Encode game state as a feature vector for DQN.
        
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
        state = np.zeros(self.state_size)
        
        # Encode hole cards (2 cards * 2 features = 4)
        for i, card in enumerate(hole_cards[:2]):
            state[i * 2] = card.rank / 12.0
            state[i * 2 + 1] = card.suit / 3.0
        
        # Encode community cards (5 cards * 2 features = 10)
        for i, card in enumerate(community_cards[:5]):
            idx = 4 + i * 2
            state[idx] = card.rank / 12.0
            state[idx + 1] = card.suit / 3.0
        
        # Encode game state
        idx = 14
        state[idx] = min(pot / 1000.0, 1.0)
        state[idx + 1] = min(current_bet / 500.0, 1.0)
        state[idx + 2] = min(player_stack / 1000.0, 1.0)
        state[idx + 3] = min(opponent_bet / 500.0, 1.0)
        state[idx + 4] = len(community_cards) / 5.0
        
        # Add equity information if available
        if self.data_manager and len(community_cards) == 0:
            hand_notation = self._get_hand_notation(hole_cards)
            equity = self.data_manager.get_hand_equity(hand_notation)
            if equity is not None:
                state[idx + 5] = equity
        
        return state
    
    def _get_hand_notation(self, hole_cards: List[Card]) -> str:
        """
        Convert hole cards to standard hand notation (e.g., 'KAS', 'QQ', '27O').
        
        Note: The equity table uses a specific format where:
        - Pairs: Use the rank twice (e.g., 'AA', 'KK')
        - Non-pairs: LOWER rank comes FIRST (e.g., 'KAS' not 'AKS', '27O' not '72O')
        
        Args:
            hole_cards: Two hole cards
            
        Returns:
            Hand notation string matching equity table format
        """
        if len(hole_cards) < 2:
            return ""
        
        card1, card2 = hole_cards[0], hole_cards[1]
        
        # Map rank to notation
        rank_map = {
            12: 'A', 11: 'K', 10: 'Q', 9: 'J', 8: 'T',
            7: '9', 6: '8', 5: '7', 4: '6', 3: '5',
            2: '4', 1: '3', 0: '2'
        }
        
        rank1 = rank_map.get(card1.rank, str(card1.rank))
        rank2 = rank_map.get(card2.rank, str(card2.rank))
        
        # Determine if suited or offsuit
        if card1.rank == card2.rank:
            # Pocket pair - order doesn't matter, use consistent format
            return f"{rank1}{rank2}"
        else:
            # Non-pair: ALWAYS use lower rank first (equity table format)
            if card1.rank < card2.rank:
                # card1 is lower
                lower_rank, higher_rank = rank1, rank2
                suited = card1.suit == card2.suit
            else:
                # card2 is lower
                lower_rank, higher_rank = rank2, rank1
                suited = card1.suit == card2.suit
            
            notation = f"{lower_rank}{higher_rank}"
            return notation + ('S' if suited else 'O')
    
    def _get_equity_based_action(self,
                                 hole_cards: List[Card],
                                 community_cards: List[Card],
                                 pot: int,
                                 current_bet: int,
                                 player_stack: int) -> Tuple[Optional[Action], int]:
        """
        Get action based on preflop equity tables.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot: Current pot
            current_bet: Current bet to call
            player_stack: Player's stack
            
        Returns:
            Tuple of (Action, raise_amount) or (None, 0) if not applicable
        """
        # Only use equity tables preflop
        if len(community_cards) > 0 or not self.data_manager:
            return None, 0
        
        hand_notation = self._get_hand_notation(hole_cards)
        equity = self.data_manager.get_hand_equity(hand_notation)
        
        if equity is None:
            return None, 0
        
        # Make decisions based on equity
        if equity > 0.65:  # Premium hands (top ~13%)
            # Strong hand - raise aggressively
            raise_amount = min(int(pot * 0.75), player_stack)
            return Action.RAISE, max(raise_amount, 20)
        elif equity > 0.55:  # Good hands (top ~25%)
            # Good hand - raise or call
            if current_bet < pot * 0.3:
                raise_amount = min(int(pot * 0.5), player_stack)
                return Action.RAISE, max(raise_amount, 20)
            else:
                return Action.CALL, 0
        elif equity > 0.45:  # Playable hands (top ~40%)
            # Decent hand - call reasonable bets
            if current_bet < pot * 0.5:
                return Action.CALL if current_bet > 0 else Action.CHECK, 0
            else:
                return Action.FOLD, 0
        else:
            # Weak hand - fold to bets, check otherwise
            if current_bet > 0:
                return Action.FOLD, 0
            else:
                return Action.CHECK, 0
    
    def choose_action(self,
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> Tuple[Action, int]:
        """
        Choose action using ensemble of CFR, DQN, and equity-based strategies.
        
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
        # Get votes from each component
        votes = {}
        
        # 1. CFR vote
        try:
            cfr_action, cfr_raise = self.cfr.choose_action(
                hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
            )
            votes['cfr'] = (cfr_action, cfr_raise, self.cfr_weight)
        except Exception:
            votes['cfr'] = None
        
        # 2. DQN vote
        if self.model:
            try:
                state = self._encode_state(
                    hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
                )
                
                # Exploration vs exploitation
                if self.training_mode and np.random.rand() <= self.epsilon:
                    action_idx = random.randrange(self.action_size)
                else:
                    q_values = self.model.predict(np.array([state]), verbose=0)
                    action_idx = np.argmax(q_values[0])
                
                # Map action index to Action enum
                if action_idx == 0:
                    dqn_action, dqn_raise = Action.FOLD, 0
                elif action_idx == 1:
                    if current_bet == 0:
                        dqn_action, dqn_raise = Action.CHECK, 0
                    else:
                        dqn_action, dqn_raise = Action.CALL, 0
                else:
                    dqn_raise = max(int(pot * 0.5), 20)
                    dqn_raise = min(dqn_raise, player_stack)
                    dqn_action, dqn_raise = Action.RAISE, dqn_raise
                
                votes['dqn'] = (dqn_action, dqn_raise, self.dqn_weight)
            except Exception:
                votes['dqn'] = None
        else:
            votes['dqn'] = None
        
        # 3. Equity-based vote
        equity_action, equity_raise = self._get_equity_based_action(
            hole_cards, community_cards, pot, current_bet, player_stack
        )
        if equity_action is not None:
            votes['equity'] = (equity_action, equity_raise, self.equity_weight)
        else:
            votes['equity'] = None
        
        # Ensemble decision making
        return self._ensemble_decision(votes, current_bet, pot, player_stack)
    
    def _ensemble_decision(self,
                          votes: Dict,
                          current_bet: int,
                          pot: int,
                          player_stack: int) -> Tuple[Action, int]:
        """
        Make final decision based on ensemble voting.
        
        Args:
            votes: Dictionary of votes from each component
            current_bet: Current bet to call
            pot: Current pot size
            player_stack: Player's remaining stack
            
        Returns:
            Tuple of (Action, raise_amount)
        """
        # Filter out None votes
        valid_votes = {k: v for k, v in votes.items() if v is not None}
        
        if not valid_votes:
            # Fallback to conservative play
            if current_bet == 0:
                return Action.CHECK, 0
            elif current_bet < pot * 0.3:
                return Action.CALL, 0
            else:
                return Action.FOLD, 0
        
        # Weighted voting for action type
        action_scores = {Action.FOLD: 0.0, Action.CHECK: 0.0, 
                        Action.CALL: 0.0, Action.RAISE: 0.0}
        raise_amounts = []
        
        for component, (action, raise_amt, weight) in valid_votes.items():
            action_scores[action] += weight
            if action == Action.RAISE:
                raise_amounts.append(raise_amt)
        
        # Choose action with highest score
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate raise amount if needed
        if best_action == Action.RAISE and raise_amounts:
            # Use median of suggested raise amounts
            raise_amount = int(np.median(raise_amounts))
            raise_amount = min(raise_amount, player_stack)
            raise_amount = max(raise_amount, 20)  # Minimum raise
        else:
            raise_amount = 0
        
        return best_action, raise_amount
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        """
        Store experience in replay memory for DQN training.
        
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
        Train DQN model on a batch of experiences.
        
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
                next_q_values = self.model.predict(np.array([next_state]), verbose=0)
                target = reward + self.gamma * np.amax(next_q_values[0])
            
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
    
    def train_cfr(self, num_iterations: int = 1000):
        """
        Train CFR component through self-play.
        
        Args:
            num_iterations: Number of CFR training iterations
        """
        print(f"Training CFR component for {num_iterations} iterations...")
        self.cfr.train(num_iterations)
        print(f"✓ CFR training complete ({self.cfr.iterations} total iterations)")
    
    def save_strategy(self, filepath_prefix: str):
        """
        Save champion agent strategy to files.
        
        Args:
            filepath_prefix: Prefix for save files (will create .cfr and .dqn files)
        """
        # Save CFR strategy
        self.cfr.save_strategy(f"{filepath_prefix}.cfr")
        print(f"✓ CFR strategy saved to {filepath_prefix}.cfr")
        
        # Save DQN model
        if self.model:
            self.model.save(f"{filepath_prefix}.dqn")
            print(f"✓ DQN model saved to {filepath_prefix}.dqn")
    
    def load_strategy(self, filepath_prefix: str):
        """
        Load champion agent strategy from files.
        
        Args:
            filepath_prefix: Prefix for load files
        """
        # Load CFR strategy
        try:
            self.cfr.load_strategy(f"{filepath_prefix}.cfr")
            print(f"✓ CFR strategy loaded from {filepath_prefix}.cfr")
        except FileNotFoundError:
            print(f"⚠ CFR strategy file not found: {filepath_prefix}.cfr")
        
        # Load DQN model
        if self.model:
            try:
                self.model.load_weights(f"{filepath_prefix}.dqn")
                print(f"✓ DQN model loaded from {filepath_prefix}.dqn")
            except Exception:
                print(f"⚠ DQN model file not found: {filepath_prefix}.dqn")
    
    def set_training_mode(self, training: bool):
        """
        Set whether agent is in training mode.
        
        Args:
            training: Whether to enable training mode
        """
        self.training_mode = training
        if not training:
            self.epsilon = 0.0  # No exploration during evaluation
    
    def reset(self):
        """Reset agent state for a new hand."""
        pass
    
    def observe_result(self, won: bool, amount: int):
        """
        Observe the result of a hand.
        
        Args:
            won: Whether the agent won the hand
            amount: Amount won or lost
        """
        pass
