"""
Core Data Models for Poker Bot System.

Defines the universal language of the system using Pydantic models for
runtime data validation. These models form the Data Bridge between
the Perception Engine (Croupier) and Decision Engine (Strategist).

Key Models:
- Card: Represents a playing card
- GameState: Complete game state from VLM perception
- Action: Decision from the strategist
- PlayerState: Player-specific state information

References:
- DeepStack paper: Game state representation
- Pydantic: Runtime validation for system reliability
"""

from enum import IntEnum, Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class Suit(IntEnum):
    """Card suits."""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    """Card ranks (2-14 where 14=Ace)."""
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


class ActionType(str, Enum):
    """Possible player actions."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class BettingRound(str, Enum):
    """Betting rounds in Texas Hold'em."""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class Card(BaseModel):
    """
    Represents a playing card.
    
    Attributes:
        rank: Card rank (2-14, where 14=Ace)
        suit: Card suit (0-3: clubs, diamonds, hearts, spades)
    """
    rank: int = Field(..., ge=2, le=14, description="Card rank (2-14)")
    suit: int = Field(..., ge=0, le=3, description="Card suit (0-3)")
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card':
        """
        Parse card from string notation (e.g., 'As', 'Kh', 'Td').
        
        Args:
            card_str: Card string like 'As' (Ace of spades)
            
        Returns:
            Card instance
        """
        if len(card_str) < 2:
            raise ValueError(f"Invalid card string: {card_str}")
        
        rank_char = card_str[:-1].upper()
        suit_char = card_str[-1].lower()
        
        # Parse rank
        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        if rank_char not in rank_map:
            raise ValueError(f"Invalid rank: {rank_char}")
        rank = rank_map[rank_char]
        
        # Parse suit
        suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
        if suit_char not in suit_map:
            raise ValueError(f"Invalid suit: {suit_char}")
        suit = suit_map[suit_char]
        
        return cls(rank=rank, suit=suit)
    
    def to_string(self) -> str:
        """Convert to string notation (e.g., 'As', 'Kh')."""
        rank_chars = {
            2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
        }
        suit_chars = {0: 'c', 1: 'd', 2: 'h', 3: 's'}
        return f"{rank_chars[self.rank]}{suit_chars[self.suit]}"
    
    def to_index(self) -> int:
        """Convert to 0-51 card index."""
        return (self.rank - 2) * 4 + self.suit
    
    @classmethod
    def from_index(cls, index: int) -> 'Card':
        """Create card from 0-51 index."""
        rank = (index // 4) + 2
        suit = index % 4
        return cls(rank=rank, suit=suit)
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self) -> str:
        return f"Card({self.to_string()})"
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False


class PlayerState(BaseModel):
    """
    Player-specific state information.
    
    Attributes:
        stack: Player's remaining chips
        current_bet: Player's bet in current round
        is_active: Whether player is still in the hand
        is_all_in: Whether player is all-in
        position: Player's position at table
    """
    stack: int = Field(default=1000, ge=0, description="Player's chip stack")
    current_bet: int = Field(default=0, ge=0, description="Current bet in round")
    is_active: bool = Field(default=True, description="Still in the hand")
    is_all_in: bool = Field(default=False, description="Is all-in")
    position: Optional[str] = Field(default=None, description="Table position")


class GameState(BaseModel):
    """
    Complete game state from VLM perception.
    
    This is the Data Bridge between the Perception Engine (Croupier)
    and Decision Engine (Strategist). The Croupier produces this,
    and the Strategist consumes it.
    
    Attributes:
        hole_cards: Player's private cards
        community_cards: Board cards (0 preflop, 3 flop, 4 turn, 5 river)
        pot_size: Total chips in pot
        current_bet: Amount to call
        player: Player's state
        opponent: Opponent's state  
        street: Current betting round
        action_required: Whether player must act
        available_actions: List of legal actions
        confidence: Detection confidence (0-1)
        raw_perception: Raw VLM output for debugging
    """
    hole_cards: List[Card] = Field(default_factory=list, description="Player's hole cards")
    community_cards: List[Card] = Field(default_factory=list, description="Community cards")
    pot_size: int = Field(default=0, ge=0, description="Total pot")
    current_bet: int = Field(default=0, ge=0, description="Amount to call")
    player: PlayerState = Field(default_factory=PlayerState, description="Player state")
    opponent: PlayerState = Field(default_factory=PlayerState, description="Opponent state")
    street: BettingRound = Field(default=BettingRound.PREFLOP, description="Current street")
    action_required: bool = Field(default=True, description="Must player act")
    available_actions: List[ActionType] = Field(
        default_factory=lambda: [ActionType.FOLD, ActionType.CALL, ActionType.RAISE],
        description="Legal actions"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    raw_perception: Optional[str] = Field(default=None, description="Raw VLM output")
    
    @field_validator('hole_cards')
    @classmethod
    def validate_hole_cards(cls, v: List[Card]) -> List[Card]:
        """Validate hole cards (should be 0 or 2)."""
        if len(v) not in [0, 2]:
            raise ValueError(f"Expected 0 or 2 hole cards, got {len(v)}")
        return v
    
    @field_validator('community_cards')
    @classmethod
    def validate_community_cards(cls, v: List[Card]) -> List[Card]:
        """Validate community cards (should be 0, 3, 4, or 5)."""
        if len(v) not in [0, 3, 4, 5]:
            raise ValueError(f"Expected 0, 3, 4, or 5 community cards, got {len(v)}")
        return v
    
    def get_all_cards(self) -> List[Card]:
        """Get all visible cards."""
        return self.hole_cards + self.community_cards
    
    def get_state_vector(self) -> List[float]:
        """
        Convert to numeric vector for neural network input.
        
        Returns:
            Normalized state vector
        """
        vector = []
        
        # Hole cards (one-hot, 52 dims per card = 104 dims)
        for i in range(2):
            card_vec = [0.0] * 52
            if i < len(self.hole_cards):
                card_vec[self.hole_cards[i].to_index()] = 1.0
            vector.extend(card_vec)
        
        # Community cards (one-hot, 52 dims per card = 260 dims)
        for i in range(5):
            card_vec = [0.0] * 52
            if i < len(self.community_cards):
                card_vec[self.community_cards[i].to_index()] = 1.0
            vector.extend(card_vec)
        
        # Numeric features (normalized)
        stack_norm = 10000.0  # Max stack for normalization
        vector.append(self.pot_size / stack_norm)
        vector.append(self.current_bet / stack_norm)
        vector.append(self.player.stack / stack_norm)
        vector.append(self.opponent.stack / stack_norm)
        vector.append(float(self.player.is_all_in))
        vector.append(float(self.opponent.is_all_in))
        
        # Street (one-hot, 4 dims)
        street_vec = [0.0] * 4
        street_idx = list(BettingRound).index(self.street)
        street_vec[street_idx] = 1.0
        vector.extend(street_vec)
        
        return vector


class Action(BaseModel):
    """
    Decision from the Strategist.
    
    This is the output of the Decision Engine that gets sent
    to the Action Executor for GUI interaction.
    
    Attributes:
        action_type: Type of action (fold, check, call, bet, raise, all_in)
        amount: Bet/raise amount (0 for fold/check/call)
        confidence: Decision confidence (0-1)
        expected_value: Estimated EV of action
        reasoning: Human-readable explanation
    """
    action_type: ActionType = Field(..., description="Action type")
    amount: int = Field(default=0, ge=0, description="Bet/raise amount")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence")
    expected_value: float = Field(default=0.0, description="Expected value")
    reasoning: Optional[str] = Field(default=None, description="Explanation")
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: int, info) -> int:
        """Validate amount based on action type."""
        # Amount should be 0 for fold/check/call
        action_type = info.data.get('action_type')
        if action_type in [ActionType.FOLD, ActionType.CHECK, ActionType.CALL]:
            return 0
        return v
    
    def to_command(self) -> str:
        """Convert to command string for action executor."""
        if self.action_type == ActionType.FOLD:
            return "fold"
        elif self.action_type == ActionType.CHECK:
            return "check"
        elif self.action_type == ActionType.CALL:
            return "call"
        elif self.action_type == ActionType.BET:
            return f"bet {self.amount}"
        elif self.action_type == ActionType.RAISE:
            return f"raise {self.amount}"
        elif self.action_type == ActionType.ALL_IN:
            return "all_in"
        return "fold"
    
    def __str__(self) -> str:
        if self.amount > 0:
            return f"{self.action_type.value.upper()} {self.amount}"
        return self.action_type.value.upper()


__all__ = [
    'Suit',
    'Rank', 
    'ActionType',
    'BettingRound',
    'Card',
    'PlayerState',
    'GameState',
    'Action'
]
