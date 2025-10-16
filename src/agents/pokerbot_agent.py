"""
PokerBot Agent - Unified world-class championship poker AI.

This is the consolidated, modular agent combining the best features from:
- ChampionAgent: CFR + DQN + DeepStack + Equity Tables
- EliteUnifiedAgent: DeepStack continual re-solving + CFR+ + Opponent Modeling

Architecture:
- Modular component system with configurable features
- Ensemble decision-making with weighted voting
- CFR/CFR+ for game-theoretic optimal play
- DQN for pattern recognition and adaptation
- DeepStack continual re-solving for dynamic gameplay
- Opponent modeling for exploitation
- Pre-trained models and equity tables
"""

import json
import os
import pickle
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Lazy imports for optional dependencies
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from deepstack.game.game_state import Action
from deepstack.game.card import Card
from .base_agent import BaseAgent


class PokerBotAgent(BaseAgent):
    """
    Unified world-class poker agent combining multiple AI techniques.
    
    Features:
    - CFR/CFR+ for game-theoretic play
    - DQN for learned patterns
    - DeepStack continual re-solving
    - Opponent modeling
    - Pre-trained knowledge
    - Modular and configurable
    """
    
    def __init__(self,
                 name: str = "PokerBot",
                 
                 # Component toggles
                 use_cfr: bool = True,
                 use_cfr_plus: bool = True,
                 use_dqn: bool = True,
                 use_deepstack: bool = True,
                 use_opponent_modeling: bool = True,
                 use_pretrained: bool = True,
                 
                 # CFR configuration
                 cfr_iterations: int = 1000,
                 cfr_skip_iterations: int = 500,
                 enable_pruning: bool = True,
                 pruning_threshold: float = 0.01,
                 
                 # DQN configuration
                 state_size: int = 120,
                 action_size: int = 5,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995,
                 memory_size: int = 50000,
                 
                 # DeepStack configuration
                 lookahead_depth: int = 3,
                 
                 # Ensemble weights
                 cfr_weight: float = 0.4,
                 dqn_weight: float = 0.3,
                 deepstack_weight: float = 0.3):
        """
        Initialize PokerBot Agent.
        
        Args:
            name: Agent identifier
            use_cfr: Enable CFR component
            use_cfr_plus: Enable CFR+ enhancements
            use_dqn: Enable DQN component
            use_deepstack: Enable DeepStack component
            use_opponent_modeling: Enable opponent modeling
            use_pretrained: Load pre-trained models
            cfr_iterations: CFR iterations per training cycle
            cfr_skip_iterations: Warmup iterations
            enable_pruning: Enable action pruning in CFR+
            pruning_threshold: Threshold for pruning
            state_size: DQN state vector size
            action_size: Number of discrete actions
            learning_rate: DQN learning rate
            gamma: DQN discount factor
            epsilon: DQN exploration rate
            epsilon_min: Minimum exploration
            epsilon_decay: Exploration decay rate
            memory_size: DQN replay buffer size
            lookahead_depth: DeepStack lookahead depth
            cfr_weight: CFR ensemble weight
            dqn_weight: DQN ensemble weight
            deepstack_weight: DeepStack ensemble weight
        """
        super().__init__(name)
        
        # Configuration
        self.use_cfr = use_cfr
        self.use_cfr_plus = use_cfr_plus
        self.use_dqn = use_dqn
        self.use_deepstack = use_deepstack
        self.use_opponent_modeling = use_opponent_modeling
        self.use_pretrained = use_pretrained
        
        # Normalize ensemble weights
        total_weight = cfr_weight + dqn_weight + deepstack_weight
        self.cfr_weight = cfr_weight / total_weight if total_weight > 0 else 0.33
        self.dqn_weight = dqn_weight / total_weight if total_weight > 0 else 0.33
        self.deepstack_weight = deepstack_weight / total_weight if total_weight > 0 else 0.34
        
        # Initialize components
        self.cfr_component = None
        self.dqn_component = None
        self.deepstack_component = None
        self.opponent_model = None
        
        if use_cfr:
            self._init_cfr_component(cfr_iterations, cfr_skip_iterations, 
                                    enable_pruning, pruning_threshold)
        
        if use_dqn:
            self._init_dqn_component(state_size, action_size, learning_rate,
                                    gamma, epsilon, epsilon_min, epsilon_decay,
                                    memory_size)
        
        if use_deepstack:
            self._init_deepstack_component(lookahead_depth)
        
        if use_opponent_modeling:
            self._init_opponent_modeling()
        
        if use_pretrained:
            self._load_pretrained_models()
        
        # State tracking
        self.training_mode = True
        self.stats = {
            'hands_played': 0,
            'decisions_made': 0,
            'cfr_decisions': 0,
            'dqn_decisions': 0,
            'deepstack_decisions': 0
        }
        
        print(f"[PokerBot] {name} initialized successfully")
        print(f"  Components: CFR={use_cfr}, CFR+={use_cfr_plus}, DQN={use_dqn}, DeepStack={use_deepstack}")
        print(f"  Weights: CFR={self.cfr_weight:.2f}, DQN={self.dqn_weight:.2f}, DeepStack={self.deepstack_weight:.2f}")
    
    def _init_cfr_component(self, iterations: int, skip_iterations: int,
                           enable_pruning: bool, threshold: float):
        """Initialize CFR/CFR+ component."""
        try:
            from .cfr_agent import CFRAgent
            
            self.cfr_component = CFRAgent(name=f"{self.name}_CFR")
            self.cfr_iterations = iterations
            self.cfr_skip_iterations = skip_iterations
            self.enable_pruning = enable_pruning
            self.pruning_threshold = threshold
            
            if self.use_cfr_plus:
                # Enable CFR+ enhancements if available
                if hasattr(self.cfr_component, 'enable_cfr_plus'):
                    self.cfr_component.enable_cfr_plus()
                    print(f"  [OK] CFR+ enhancements enabled")
        except Exception as e:
            print(f"  [WARN] CFR component initialization failed: {e}")
            self.cfr_component = None
            self.use_cfr = False
    
    def _init_dqn_component(self, state_size: int, action_size: int,
                           learning_rate: float, gamma: float, epsilon: float,
                           epsilon_min: float, epsilon_decay: float,
                           memory_size: int):
        """Initialize DQN component."""
        if not TENSORFLOW_AVAILABLE:
            print(f"  [WARN] TensorFlow not available, DQN disabled")
            self.use_dqn = False
            return
        
        try:
            self.state_size = state_size
            self.action_size = action_size
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            
            # Experience replay
            self.memory = deque(maxlen=memory_size)
            
            # Build DQN model
            self.dqn_model = self._build_dqn_model()
            self.target_model = self._build_dqn_model()
            self.training_step = 0
            
            print(f"  [OK] DQN component initialized")
        except Exception as e:
            print(f"  [WARN] DQN component initialization failed: {e}")
            self.dqn_model = None
            self.use_dqn = False
    
    def _build_dqn_model(self):
        """Build DQN neural network."""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
    
    def _init_deepstack_component(self, lookahead_depth: int):
        """Initialize DeepStack component."""
        try:
            from .lookahead_wrapper import DeepStackLookahead
            
            self.deepstack_component = DeepStackLookahead()
            self.lookahead_depth = lookahead_depth
            
            # Initialize DeepStack value network if available
            if TORCH_AVAILABLE:
                self._init_value_network()
            
            print(f"  [OK] DeepStack component initialized")
        except Exception as e:
            print(f"  [WARN] DeepStack component initialization failed: {e}")
            self.deepstack_component = None
            self.use_deepstack = False
    
    def _init_value_network(self):
        """Initialize DeepStack value network."""
        try:
            class SimpleValueNetwork(nn.Module):
                def __init__(self, bucket_count=169):
                    super().__init__()
                    self.bucket_count = bucket_count
                    input_size = 2 * bucket_count + 1
                    output_size = 2 * bucket_count
                    
                    self.network = nn.Sequential(
                        nn.Linear(input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, 512),
                        nn.ReLU(),
                        nn.Linear(512, output_size)
                    )
                
                def forward(self, x):
                    return self.network(x)
            
            self.value_network = SimpleValueNetwork()
        except Exception as e:
            print(f"  [WARN] Value network initialization failed: {e}")
            self.value_network = None
    
    def _init_opponent_modeling(self):
        """Initialize opponent modeling component."""
        try:
            from .opponent_model import OpponentModel
            
            self.opponent_model = OpponentModel()
            print(f"  [OK] Opponent modeling enabled")
        except Exception as e:
            print(f"  [WARN] Opponent modeling initialization failed: {e}")
            self.opponent_model = None
            self.use_opponent_modeling = False
    
    def _load_pretrained_models(self):
        """Load pre-trained models and data."""
        try:
            from src.utils.model_loader import ModelLoader, TrainingDataManager
            
            self.model_loader = ModelLoader()
            self.data_manager = TrainingDataManager()
            
            # Try loading components
            try:
                deepstack_info = self.model_loader.load_deepstack_model(use_gpu=False)
                print(f"  [OK] Loaded DeepStack model ({deepstack_info.get('size_mb', 0):.1f} MB)")
            except:
                pass
            
            try:
                equity_data = self.data_manager.load_preflop_equity()
                print(f"  [OK] Loaded preflop equity table ({len(equity_data)} hands)")
            except:
                pass
        except Exception as e:
            print(f"  [WARN] Pre-trained models not loaded: {e}")
            self.model_loader = None
            self.data_manager = None
    
    def choose_action(self,
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> Tuple[Action, int]:
        """
        Choose action using ensemble of all enabled components.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot: Current pot size
            current_bet: Current bet to call
            player_stack: Player's remaining stack
            opponent_bet: Opponent's bet
        
        Returns:
            Tuple of (Action, raise_amount)
        """
        self.stats['decisions_made'] += 1
        
        # Collect decisions from enabled components
        decisions = {}
        weights = {}
        
        # CFR decision
        if self.use_cfr and self.cfr_component:
            try:
                cfr_action, cfr_amount = self.cfr_component.choose_action(
                    hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
                )
                decisions['cfr'] = (cfr_action, cfr_amount)
                weights['cfr'] = self.cfr_weight
                self.stats['cfr_decisions'] += 1
            except Exception as e:
                pass
        
        # DQN decision
        if self.use_dqn and self.dqn_model:
            try:
                dqn_action, dqn_amount = self._dqn_decision(
                    hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
                )
                decisions['dqn'] = (dqn_action, dqn_amount)
                weights['dqn'] = self.dqn_weight
                self.stats['dqn_decisions'] += 1
            except Exception as e:
                pass
        
        # DeepStack decision
        if self.use_deepstack and self.deepstack_component:
            try:
                ds_action, ds_amount = self._deepstack_decision(
                    hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
                )
                decisions['deepstack'] = (ds_action, ds_amount)
                weights['deepstack'] = self.deepstack_weight
                self.stats['deepstack_decisions'] += 1
            except Exception as e:
                pass
        
        # Ensemble decision
        final_action, final_amount = self._ensemble_decision(
            decisions, weights, current_bet, pot, player_stack
        )
        
        # Apply opponent modeling adjustment
        if self.use_opponent_modeling and self.opponent_model:
            final_action, final_amount = self._apply_opponent_adjustment(
                final_action, final_amount, hole_cards, community_cards, pot, current_bet
            )
        
        return final_action, final_amount
    
    def _dqn_decision(self, hole_cards: List[Card], community_cards: List[Card],
                     pot: int, current_bet: int, player_stack: int,
                     opponent_bet: int) -> Tuple[Action, int]:
        """Get decision from DQN component."""
        # Encode state
        state = self._encode_dqn_state(hole_cards, community_cards, pot,
                                      current_bet, player_stack, opponent_bet)
        
        # Epsilon-greedy action selection
        if self.training_mode and np.random.random() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            q_values = self.dqn_model.predict(state.reshape(1, -1), verbose=0)
            action_idx = np.argmax(q_values[0])
        
        # Convert to poker action
        return self._convert_dqn_action(action_idx, current_bet, pot, player_stack)
    
    def _deepstack_decision(self, hole_cards: List[Card], community_cards: List[Card],
                          pot: int, current_bet: int, player_stack: int,
                          opponent_bet: int) -> Tuple[Action, int]:
        """Get decision from DeepStack component."""
        # Simplified DeepStack decision - in production would use full re-solving
        # For now, use lookahead wrapper
        try:
            self.deepstack_component.build_lookahead(0, 1, community_cards)
            player_range = np.ones(self.action_size) / self.action_size
            opponent_range = np.ones(self.action_size) / self.action_size
            self.deepstack_component.resolve_first_node(player_range, opponent_range)
            results = self.deepstack_component.get_results()
            
            # Sample action from strategy
            strategy = results.get('strategy', {})
            if strategy:
                # Use strategy to select action
                return self._sample_from_strategy(strategy, current_bet, pot, player_stack)
        except:
            pass
        
        # Fallback
        return (Action.CALL if current_bet > 0 else Action.CHECK, 0)
    
    def _ensemble_decision(self, decisions: Dict, weights: Dict,
                          current_bet: int, pot: int,
                          player_stack: int) -> Tuple[Action, int]:
        """Combine decisions using weighted voting."""
        if not decisions:
            # Fallback to conservative play
            return (Action.CALL if current_bet > 0 else Action.CHECK, 0)
        
        # Weighted voting for action type
        action_votes = defaultdict(float)
        amount_by_action = defaultdict(list)
        
        for component, (action, amount) in decisions.items():
            weight = weights.get(component, 1.0)
            action_votes[action] += weight
            amount_by_action[action].append((amount, weight))
        
        # Select best action
        best_action = max(action_votes.keys(), key=lambda a: action_votes[a])
        
        # Calculate weighted average amount
        if best_action in amount_by_action and amount_by_action[best_action]:
            amounts_and_weights = amount_by_action[best_action]
            total_weight = sum(w for _, w in amounts_and_weights)
            if total_weight > 0:
                avg_amount = sum(amt * w for amt, w in amounts_and_weights) / total_weight
                final_amount = int(avg_amount)
            else:
                final_amount = amounts_and_weights[0][0]
        else:
            final_amount = 0
        
        # Ensure amount is valid
        final_amount = min(max(final_amount, 0), player_stack)
        if best_action == Action.RAISE:
            final_amount = max(final_amount, 20)  # Minimum raise
        
        return best_action, final_amount
    
    def _apply_opponent_adjustment(self, action: Action, amount: int,
                                  hole_cards: List[Card], community_cards: List[Card],
                                  pot: int, current_bet: int) -> Tuple[Action, int]:
        """Apply opponent modeling adjustments."""
        try:
            aggression = self.opponent_model.get_aggression("opponent")
            
            # Adjust against aggressive opponents
            if aggression > 1.5:
                if action == Action.RAISE:
                    amount = int(amount * 0.8)
            
            # Adjust against passive opponents
            elif aggression < 0.5:
                if action == Action.CALL:
                    hand_strength = self._get_hand_strength(hole_cards, community_cards)
                    if hand_strength > 0.6:
                        action = Action.RAISE
                        amount = max(int(pot * 0.5), 20)
        except:
            pass
        
        return action, amount
    
    def _encode_dqn_state(self, hole_cards: List[Card], community_cards: List[Card],
                         pot: int, current_bet: int, player_stack: int,
                         opponent_bet: int) -> np.ndarray:
        """Encode game state for DQN."""
        state = np.zeros(self.state_size)
        
        # Hole cards
        if len(hole_cards) >= 2:
            state[0] = hole_cards[0].rank / 12.0
            state[1] = hole_cards[0].suit / 3.0
            state[2] = hole_cards[1].rank / 12.0
            state[3] = hole_cards[1].suit / 3.0
        
        # Community cards
        for i, card in enumerate(community_cards[:5]):
            idx = 4 + i * 2
            if idx + 1 < self.state_size:
                state[idx] = card.rank / 12.0
                state[idx + 1] = card.suit / 3.0
        
        # Game state
        idx = 14
        if idx + 5 < self.state_size:
            state[idx] = min(pot / 10000.0, 1.0)
            state[idx + 1] = min(current_bet / 5000.0, 1.0)
            state[idx + 2] = min(player_stack / 20000.0, 1.0)
            state[idx + 3] = min(opponent_bet / 5000.0, 1.0)
            state[idx + 4] = len(community_cards) / 5.0
            state[idx + 5] = self._get_hand_strength(hole_cards, community_cards)
        
        return state
    
    def _get_hand_strength(self, hole_cards: List[Card], community_cards: List[Card]) -> float:
        """Calculate normalized hand strength."""
        if not hole_cards or len(hole_cards) < 2:
            return 0.0
        
        # Simple heuristic
        high_card = max(hole_cards[0].rank, hole_cards[1].rank)
        strength = high_card / 12.0
        
        # Pair bonus
        if hole_cards[0].rank == hole_cards[1].rank:
            strength += 0.3
        
        # Suited bonus
        if hole_cards[0].suit == hole_cards[1].suit:
            strength += 0.1
        
        return min(strength, 1.0)
    
    def _convert_dqn_action(self, action_idx: int, current_bet: int,
                          pot: int, stack: int) -> Tuple[Action, int]:
        """Convert DQN action index to poker action."""
        if action_idx == 0:
            return Action.FOLD, 0
        elif action_idx == 1:
            return Action.CHECK if current_bet == 0 else Action.CALL, 0
        elif action_idx == 2:
            amount = min(int(pot * 0.5), stack)
            return Action.RAISE, max(amount, 20)
        elif action_idx == 3:
            amount = min(int(pot * 1.0), stack)
            return Action.RAISE, max(amount, 20)
        else:
            amount = min(stack, int(pot * 2.0))
            return Action.RAISE, max(amount, 20)
    
    def _sample_from_strategy(self, strategy: Dict, current_bet: int,
                             pot: int, stack: int) -> Tuple[Action, int]:
        """Sample action from strategy distribution."""
        # Simplified sampling
        return (Action.CALL if current_bet > 0 else Action.CHECK, 0)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        """Store experience for DQN training."""
        if self.use_dqn:
            self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """Train DQN on batch of experiences."""
        if not self.use_dqn or not self.dqn_model or len(self.memory) < batch_size:
            return
        
        try:
            # Sample batch
            batch = random.sample(self.memory, batch_size)
            
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            dones = np.array([e[4] for e in batch])
            
            # Compute targets
            current_q = self.dqn_model.predict(states, verbose=0)
            next_q = self.target_model.predict(next_states, verbose=0)
            
            targets = current_q.copy()
            for i in range(batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
            
            # Train
            self.dqn_model.fit(states, targets, epochs=1, verbose=0)
            
            # Update target network
            self.training_step += 1
            if self.training_step % 1000 == 0:
                self.target_model.set_weights(self.dqn_model.get_weights())
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        except Exception as e:
            print(f"[DQN] Training error: {e}")
    
    def train_cfr(self, num_iterations: int = 1000):
        """Train CFR component."""
        if self.use_cfr and self.cfr_component:
            print(f"Training CFR component for {num_iterations} iterations...")
            self.cfr_component.train(num_iterations)
            print(f"[OK] CFR training complete")
    
    def save_models(self, directory: str):
        """Save all component models."""
        os.makedirs(directory, exist_ok=True)
        
        # Save DQN
        if self.use_dqn and self.dqn_model:
            try:
                dqn_path = os.path.join(directory, "pokerbot_dqn.keras")
                self.dqn_model.save(dqn_path)
                print(f"[OK] DQN model saved to {dqn_path}")
            except Exception as e:
                print(f"[WARN] DQN save failed: {e}")
        
        # Save CFR
        if self.use_cfr and self.cfr_component:
            try:
                cfr_path = os.path.join(directory, "pokerbot_cfr.pkl")
                self.cfr_component.save_strategy(cfr_path)
                print(f"[OK] CFR strategy saved to {cfr_path}")
            except Exception as e:
                print(f"[WARN] CFR save failed: {e}")
        
        # Save stats
        stats_path = os.path.join(directory, "pokerbot_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"[PokerBot] Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load all component models."""
        # Load DQN
        if self.use_dqn and self.dqn_model:
            try:
                dqn_path = os.path.join(directory, "pokerbot_dqn.keras")
                if os.path.exists(dqn_path):
                    self.dqn_model = tf.keras.models.load_model(dqn_path)
                    self.target_model.set_weights(self.dqn_model.get_weights())
                    print(f"[OK] DQN model loaded")
            except Exception as e:
                print(f"[WARN] DQN load failed: {e}")
        
        # Load CFR
        if self.use_cfr and self.cfr_component:
            try:
                cfr_path = os.path.join(directory, "pokerbot_cfr.pkl")
                if os.path.exists(cfr_path):
                    self.cfr_component.load_strategy(cfr_path)
                    print(f"[OK] CFR strategy loaded")
            except Exception as e:
                print(f"[WARN] CFR load failed: {e}")
        
        # Load stats
        stats_path = os.path.join(directory, "pokerbot_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
        
        print(f"[PokerBot] Models loaded from {directory}")
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training_mode = training
        if not training:
            self.epsilon = 0.0
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return self.stats.copy()
    
    def reset(self):
        """Reset agent state for new hand."""
        pass
    
    def observe_result(self, won: bool, amount: int):
        """Observe hand result."""
        self.stats['hands_played'] += 1
