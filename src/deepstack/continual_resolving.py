"""
Continual Resolving: DeepStack's main algorithm for real-time game play.

Performs continual re-solving during gameplay, tracking player ranges and 
opponent counterfactual values to enable re-solving at each decision point.

Ported from the original DeepStack continual_resolving.lua module.
"""
import numpy as np
from typing import Dict, Optional

from src.deepstack.resolving import Resolving
from src.deepstack.card_abstraction import CardAbstraction
from src.deepstack.card_tools import get_card_tools


class ContinualResolving:
    """
    Main DeepStack algorithm for continual re-solving during gameplay.
    
    Tracks game state and performs depth-limited re-solving at each decision point,
    maintaining consistency of player ranges and opponent counterfactual values.
    """
    
    def __init__(self, game_variant: str = 'leduc', stack_size: int = 1200, ante: int = 100):
        """
        Initialize continual resolving.
        
        Args:
            game_variant: 'leduc' or 'holdem'
            stack_size: Starting stack size
            ante: Ante amount
        """
        self.game_variant = game_variant
        self.stack_size = stack_size
        self.ante = ante
        
        # Game state tracking
        self.decision_id = 0
        self.position = None  # 1 or 2 (P1 or P2)
        self.hand_id = None
        self.last_node = None
        self.last_bet = None
        
        # Range and CFV tracking
        self.current_player_range = None
        self.current_opponent_cfvs_bound = None
        self.starting_player_range = None
        self.starting_cfvs_p1 = None
        
        # Components
        self.card_abstraction = CardAbstraction()
        self.card_tools = get_card_tools(game_variant)
        self.first_node_resolving = None
        self.resolving = None
        
        # Initialize with first node solve
        self._resolve_first_node()
    
    def _resolve_first_node(self):
        """
        Solve depth-limited lookahead from first node to get opponent CFVs.
        
        This provides the starting CFVs for P1 that will be used when P2 acts first.
        """
        # Create first node parameters
        first_node = {
            'board': [],
            'street': 0,  # Preflop
            'current_player': 1,  # P1
            'bets': [self.ante, self.ante]
        }
        
        # Create uniform starting ranges
        player_range = self.card_tools.get_uniform_range(first_node['board'])
        opponent_range = self.card_tools.get_uniform_range(first_node['board'])
        num_hands = player_range.size
        
        self.starting_player_range = player_range.copy()
        
        # Create and resolve first node
        self.first_node_resolving = Resolving(
            num_hands=num_hands, 
            game_variant=self.game_variant
        )
        
        self.first_node_resolving.resolve_first_node(
            first_node, player_range, opponent_range
        )
        
        # Store initial CFVs for P1
        self.starting_cfvs_p1 = self.first_node_resolving.get_root_cfv()
        
        print(f"[ContinualResolving] First node resolved, starting CFVs computed")
    
    def start_new_hand(self, state: Dict):
        """
        Initialize for a new hand.
        
        Args:
            state: Game state dict with keys: position, hand_id, etc.
        """
        self.last_node = None
        self.decision_id = 0
        self.position = state.get('position', 1)
        self.hand_id = state.get('hand_id', 0)
        
        print(f"[ContinualResolving] Starting new hand {self.hand_id}, position {self.position}")
    
    def compute_action(self, node: Dict, state: Dict) -> Dict:
        """
        Re-solve node and choose the re-solving player's action.
        
        Args:
            node: Current game node with keys: street, bets, board, current_player, etc.
            state: Current game state
            
        Returns:
            Action dict with keys: action, raise_amount
        """
        # Re-solve at current node
        self._resolve_node(node, state)
        
        # Sample action from strategy
        sampled_bet = self._sample_bet(node, state)
        
        # Update tracking
        self.decision_id += 1
        self.last_bet = sampled_bet
        self.last_node = node
        
        # Convert to action format
        action = self._bet_to_action(node, sampled_bet)
        
        print(f"[ContinualResolving] Decision {self.decision_id}: {action}")
        return action
    
    def _resolve_node(self, node: Dict, state: Dict):
        """
        Re-solve current node using appropriate method.
        
        Args:
            node: Current game node
            state: Current game state
        """
        # First node for P1 - use pre-computed strategy
        if self.decision_id == 0 and self.position == 1:
            self.current_player_range = self.starting_player_range.copy()
            self.resolving = self.first_node_resolving
            
        # Other nodes - update invariant and re-solve
        else:
            assert not node.get('terminal', False)
            assert node.get('current_player') == self.position
            
            # Update ranges and CFVs based on game history
            self._update_invariant(node, state)
            
            # Create new resolving instance and solve
            num_hands = int(self.current_player_range.size)
            self.resolving = Resolving(
                num_hands=num_hands,
                game_variant=self.game_variant
            )
            
            self.resolving.resolve(
                node, 
                self.current_player_range, 
                self.current_opponent_cfvs_bound
            )
    
    def _update_invariant(self, node: Dict, state: Dict):
        """
        Update player range and opponent CFVs to be consistent with actions
        since the last re-solved state.
        
        Args:
            node: Current game node
            state: Current game state
        """
        # Street has changed - handle transition
        if self.last_node and self.last_node.get('street') != node.get('street'):
            assert self.last_node['street'] + 1 == node['street']
            
            # Get opponent CFVs for new street
            self.current_opponent_cfvs_bound = self.resolving.get_chance_action_cfv(
                self.last_bet, node.get('board', [])
            )
            
            # Normalize player range for new board
            self.current_player_range = self.card_tools.normalize_range(
                node.get('board', []), self.current_player_range
            )
            
        # First decision for P2
        elif self.decision_id == 0:
            assert self.position == 2
            assert node.get('street', 0) == 0  # Preflop
            
            self.current_player_range = self.starting_player_range.copy()
            self.current_opponent_cfvs_bound = self.starting_cfvs_p1.copy()
            
        # Same street continuation
        else:
            assert self.last_node['street'] == node['street']
            # Range and CFVs remain the same within street
    
    def _sample_bet(self, node: Dict, state: Dict) -> int:
        """
        Sample action from computed strategy.
        
        Args:
            node: Current game node
            state: Current game state
            
        Returns:
            Bet index/amount
        """
        # Get possible bets from resolving
        possible_bets = self.resolving.get_possible_actions()
        
        if not possible_bets:
            return 0  # Default to first action
        
        # Get strategy for sampling
        root_strategy = self.resolving.get_root_strategy()

        if root_strategy is None or len(root_strategy) == 0:
            return 0

        # Ensure proper normalization to avoid numerical issues
        probs = np.clip(root_strategy, 0.0, 1.0)
        total = probs.sum()
        if total <= 0:
            return 0
        probs = probs / total

        return int(np.random.choice(len(probs), p=probs))
    
    def _bet_to_action(self, node: Dict, sampled_bet: int) -> Dict:
        """
        Convert sampled bet to action format.
        
        Args:
            node: Current game node
            sampled_bet: Sampled bet index/amount
            
        Returns:
            Action dict with 'action' and 'raise_amount' keys
        """
        possible_actions = self.resolving.get_possible_actions()
        
        if not possible_actions or sampled_bet >= len(possible_actions):
            # Default to check/call
            current_bet = node.get('bets', [0, 0])[0] - node.get('bets', [0, 0])[1]
            if current_bet > 0:
                return {'action': 'call', 'raise_amount': 0}
            else:
                return {'action': 'check', 'raise_amount': 0}
        
        action_name = possible_actions[sampled_bet]
        
        # Convert action name to ACPC format
        if 'fold' in action_name.lower():
            return {'action': 'fold', 'raise_amount': 0}
        elif 'check' in action_name.lower():
            return {'action': 'check', 'raise_amount': 0}
        elif 'call' in action_name.lower():
            return {'action': 'call', 'raise_amount': 0}
        elif 'raise' in action_name.lower() or 'bet' in action_name.lower():
            # Calculate raise amount (simplified)
            pot = sum(node.get('bets', [0, 0]))
            raise_amount = max(int(pot * 0.75), 20)
            return {'action': 'raise', 'raise_amount': raise_amount}
        else:
            return {'action': 'check', 'raise_amount': 0}
    
    def get_current_strategy(self) -> Optional[Dict]:
        """Get current computed strategy."""
        if self.resolving:
            return self.resolving.get_strategy()
        return None
    
    def get_current_cfvs(self) -> Optional[np.ndarray]:
        """Get current counterfactual values."""
        if self.resolving:
            return self.resolving.get_root_cfv()
        return None
    
    def get_player_range(self) -> Optional[np.ndarray]:
        """Get current player range."""
        return self.current_player_range
    
    def update_player_range(self, new_range: np.ndarray):
        """Update player range (for external range tracking)."""
        self.current_player_range = new_range.copy()
    
    def update_opponent_cfvs(self, new_cfvs: np.ndarray):
        """Update opponent CFVs (for external CFV tracking)."""
        self.current_opponent_cfvs_bound = new_cfvs.copy()
