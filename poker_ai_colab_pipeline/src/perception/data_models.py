"""
Data Models for Perception Pipeline

Pydantic-validated game state models compatible with DeepStack input format.
"""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class BettingRound(str, Enum):
    """Poker betting rounds."""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class Card(BaseModel):
    """
    Represents a playing card.
    
    Attributes:
        rank: Card rank (2-9, T, J, Q, K, A)
        suit: Card suit (s=spades, h=hearts, d=diamonds, c=clubs)
    """
    rank: str
    suit: str
    confidence: float = 1.0
    
    @field_validator('rank')
    @classmethod
    def validate_rank(cls, v: str) -> str:
        valid_ranks = '23456789TJQKA'
        v = v.upper()
        if v == '10':
            v = 'T'
        if v not in valid_ranks:
            raise ValueError(f"Invalid rank: {v}. Must be one of {valid_ranks}")
        return v
    
    @field_validator('suit')
    @classmethod
    def validate_suit(cls, v: str) -> str:
        valid_suits = 'shdc'
        v = v.lower()
        if v not in valid_suits:
            raise ValueError(f"Invalid suit: {v}. Must be one of {valid_suits}")
        return v
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card':
        """Parse card from string notation (e.g., 'As', 'Kh', 'Td')."""
        if len(card_str) < 2:
            raise ValueError(f"Invalid card string: {card_str}")
        rank = card_str[:-1]
        suit = card_str[-1]
        return cls(rank=rank, suit=suit)
    
    def to_string(self) -> str:
        """Convert to string notation."""
        return f"{self.rank}{self.suit}"
    
    def to_index(self) -> int:
        """Convert to 0-51 index for DeepStack compatibility."""
        rank_idx = '23456789TJQKA'.index(self.rank)
        suit_idx = 'shdc'.index(self.suit)
        return rank_idx * 4 + suit_idx
    
    @classmethod
    def from_index(cls, index: int) -> 'Card':
        """Create card from 0-51 index."""
        rank = '23456789TJQKA'[index // 4]
        suit = 'shdc'[index % 4]
        return cls(rank=rank, suit=suit)
    
    def __str__(self) -> str:
        return self.to_string()


class GameState(BaseModel):
    """
    Complete poker game state for DeepStack solver input.
    
    This model is validated and directly compatible with the DeepStack
    decision engine's expected input format.
    
    Attributes:
        hole_cards: Player's two private cards
        community_cards: Board cards (0-5 depending on street)
        pot_size: Current pot amount in chips
        current_bet: Amount to call
        player_stack: Player's remaining chips
        opponent_stack: Opponent's remaining chips
        street: Current betting round
        action_required: Whether player needs to act
        available_actions: List of legal actions
        confidence: Overall detection confidence (0-1)
    """
    hole_cards: List[Card] = Field(default_factory=list)
    community_cards: List[Card] = Field(default_factory=list)
    pot_size: int = Field(default=0, ge=0)
    current_bet: int = Field(default=0, ge=0)
    player_stack: int = Field(default=1000, ge=0)
    opponent_stack: int = Field(default=1000, ge=0)
    street: BettingRound = Field(default=BettingRound.PREFLOP)
    action_required: bool = Field(default=True)
    available_actions: List[str] = Field(default_factory=lambda: ['fold', 'call', 'raise'])
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    raw_response: Optional[str] = Field(default=None, exclude=True)
    
    @field_validator('hole_cards')
    @classmethod
    def validate_hole_cards(cls, v: List[Card]) -> List[Card]:
        if len(v) > 2:
            raise ValueError(f"Maximum 2 hole cards allowed, got {len(v)}")
        return v
    
    @field_validator('community_cards')
    @classmethod
    def validate_community_cards(cls, v: List[Card]) -> List[Card]:
        if len(v) > 5:
            raise ValueError(f"Maximum 5 community cards allowed, got {len(v)}")
        return v
    
    def to_deepstack_format(self) -> dict:
        """
        Convert to DeepStack-compatible node_params format.
        
        Returns:
            Dictionary with keys: street, bets, current_player, board, bet_sizing
        """
        # Convert street to numeric (0=preflop, 1=flop, 2=turn, 3=river)
        street_map = {
            BettingRound.PREFLOP: 0,
            BettingRound.FLOP: 1,
            BettingRound.TURN: 2,
            BettingRound.RIVER: 3
        }
        
        # Convert board cards to indices
        board = [card.to_index() for card in self.community_cards]
        
        return {
            'street': street_map[self.street],
            'bets': [self.current_bet, self.current_bet],  # Both players' contributions
            'current_player': 1,  # Player to act (1 = hero)
            'board': board,
            'bet_sizing': [1.0],  # Pot-relative bet sizing
            'pot': self.pot_size,
            'player_stack': self.player_stack,
            'opponent_stack': self.opponent_stack
        }
    
    def get_hole_card_indices(self) -> List[int]:
        """Get hole card indices for DeepStack range computation."""
        return [card.to_index() for card in self.hole_cards]
    
    def validate_street_cards(self) -> bool:
        """Validate community card count matches street."""
        expected = {
            BettingRound.PREFLOP: 0,
            BettingRound.FLOP: 3,
            BettingRound.TURN: 4,
            BettingRound.RIVER: 5
        }
        return len(self.community_cards) == expected[self.street]
    
    class Config:
        use_enum_values = True
