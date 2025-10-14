"""
Game State Representation for No-Limit Texas Hold'em
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class Suit(Enum):
    """Card suits"""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(Enum):
    """Card ranks"""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass(frozen=True)
class Card:
    """Immutable playing card"""
    rank: Rank
    suit: Suit
    
    def __str__(self) -> str:
        rank_str = {
            Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
            Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9',
            Rank.TEN: 'T', Rank.JACK: 'J', Rank.QUEEN: 'Q', 
            Rank.KING: 'K', Rank.ACE: 'A'
        }
        suit_str = {Suit.CLUBS: '♣', Suit.DIAMONDS: '♦', 
                    Suit.HEARTS: '♥', Suit.SPADES: '♠'}
        return f"{rank_str[self.rank]}{suit_str[self.suit]}"
    
    def to_index(self) -> int:
        """Convert card to unique index (0-51)"""
        return (self.rank.value - 2) * 4 + self.suit.value


class ActionType(Enum):
    """Possible poker actions"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5


@dataclass
class Action:
    """Poker action with amount"""
    action_type: ActionType
    amount: float = 0.0
    
    def __str__(self) -> str:
        if self.action_type in (ActionType.FOLD, ActionType.CHECK):
            return self.action_type.name
        return f"{self.action_type.name}({self.amount:.1f})"


@dataclass
class PlayerState:
    """State of a single player"""
    player_id: int
    stack: float
    current_bet: float
    hole_cards: Optional[List[Card]] = None
    is_active: bool = True
    is_all_in: bool = False
    total_invested: float = 0.0
    
    def __post_init__(self):
        if self.hole_cards is None:
            self.hole_cards = []


class Street(Enum):
    """Betting streets"""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass
class GameState:
    """Complete state of a poker game"""
    num_players: int
    small_blind: float
    big_blind: float
    ante: float = 0.0
    
    # Players
    players: List[PlayerState] = field(default_factory=list)
    button_position: int = 0
    current_player: int = 0
    
    # Cards
    community_cards: List[Card] = field(default_factory=list)
    deck: List[Card] = field(default_factory=list)
    
    # Betting
    pot: float = 0.0
    current_bet: float = 0.0
    min_raise: float = 0.0
    street: Street = Street.PREFLOP
    
    # History
    betting_history: List[Tuple[int, Action]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize game state"""
        if not self.players:
            self.players = [
                PlayerState(player_id=i, stack=100.0, current_bet=0.0)
                for i in range(self.num_players)
            ]
        
        if not self.deck:
            self._initialize_deck()
    
    def _initialize_deck(self):
        """Create a standard 52-card deck"""
        self.deck = [
            Card(rank=rank, suit=suit)
            for rank in Rank
            for suit in Suit
        ]
    
    def get_active_players(self) -> List[PlayerState]:
        """Get list of active players"""
        return [p for p in self.players if p.is_active]
    
    def get_current_player_state(self) -> PlayerState:
        """Get the state of the current player"""
        return self.players[self.current_player]
    
    def get_legal_actions(self) -> List[Action]:
        """Get all legal actions for current player"""
        player = self.get_current_player_state()
        actions = []
        
        # Can always fold (unless already checked)
        if self.current_bet > player.current_bet:
            actions.append(Action(ActionType.FOLD))
        
        # Check if no bet to call
        if self.current_bet == player.current_bet:
            actions.append(Action(ActionType.CHECK))
        
        # Call if there's a bet
        if self.current_bet > player.current_bet:
            call_amount = min(self.current_bet - player.current_bet, player.stack)
            if call_amount > 0:
                actions.append(Action(ActionType.CALL, call_amount))
        
        # Bet if no current bet
        if self.current_bet == 0:
            min_bet = self.big_blind
            if player.stack > min_bet:
                actions.append(Action(ActionType.BET, min_bet))
                # Add some standard bet sizes
                for size in [0.5, 0.75, 1.0]:
                    bet_amount = min(size * self.pot, player.stack)
                    if bet_amount >= min_bet:
                        actions.append(Action(ActionType.BET, bet_amount))
        
        # Raise if there's a bet
        if self.current_bet > 0:
            min_raise_amount = self.current_bet + self.min_raise
            if player.stack > min_raise_amount - player.current_bet:
                raise_amount = min_raise_amount - player.current_bet
                actions.append(Action(ActionType.RAISE, raise_amount))
                # Add some standard raise sizes
                for size in [2.0, 3.0]:
                    raise_total = self.current_bet * size
                    raise_amount = min(raise_total - player.current_bet, player.stack)
                    if raise_amount >= min_raise_amount - player.current_bet:
                        actions.append(Action(ActionType.RAISE, raise_amount))
        
        # All-in is always available if stack > 0
        if player.stack > 0:
            actions.append(Action(ActionType.ALL_IN, player.stack))
        
        return actions
    
    def to_observation(self) -> np.ndarray:
        """
        Convert game state to neural network observation
        Returns normalized vector representation
        """
        obs = []
        
        # Current player's hole cards (one-hot encoded, 2 cards * 52 possibilities)
        current_player = self.players[self.current_player]
        for card in current_player.hole_cards or []:
            card_vec = np.zeros(52)
            card_vec[card.to_index()] = 1.0
            obs.append(card_vec)
        # Pad if fewer than 2 cards
        while len(current_player.hole_cards or []) < 2:
            obs.append(np.zeros(52))
        
        # Community cards (one-hot encoded, up to 5 cards)
        for card in self.community_cards:
            card_vec = np.zeros(52)
            card_vec[card.to_index()] = 1.0
            obs.append(card_vec)
        # Pad remaining community cards
        while len(self.community_cards) < 5:
            obs.append(np.zeros(52))
        
        # Pot size (normalized)
        obs.append(np.array([self.pot / 100.0]))
        
        # Stack sizes (normalized)
        for player in self.players:
            obs.append(np.array([player.stack / 100.0]))
        
        # Current bets (normalized)
        for player in self.players:
            obs.append(np.array([player.current_bet / 100.0]))
        
        # Position (one-hot)
        position_vec = np.zeros(self.num_players)
        position_vec[self.current_player] = 1.0
        obs.append(position_vec)
        
        # Street (one-hot)
        street_vec = np.zeros(4)
        street_vec[self.street.value] = 1.0
        obs.append(street_vec)
        
        # Concatenate all features
        return np.concatenate([arr.flatten() for arr in obs])
    
    def clone(self) -> 'GameState':
        """Create a deep copy of the game state"""
        from copy import deepcopy
        return deepcopy(self)
