"""Card representation and utilities for poker game."""

from enum import IntEnum
from typing import List


class Suit(IntEnum):
    """Card suits with integer values."""
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3


class Rank(IntEnum):
    """Card ranks with integer values."""
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


SUIT_SYMBOLS = {
    Suit.SPADES: '♠',
    Suit.HEARTS: '♥',
    Suit.DIAMONDS: '♦',
    Suit.CLUBS: '♣'
}

RANK_SYMBOLS = {
    Rank.TWO: '2',
    Rank.THREE: '3',
    Rank.FOUR: '4',
    Rank.FIVE: '5',
    Rank.SIX: '6',
    Rank.SEVEN: '7',
    Rank.EIGHT: '8',
    Rank.NINE: '9',
    Rank.TEN: 'T',
    Rank.JACK: 'J',
    Rank.QUEEN: 'Q',
    Rank.KING: 'K',
    Rank.ACE: 'A'
}

RANK_FROM_STRING = {v: k for k, v in RANK_SYMBOLS.items()}
RANK_FROM_STRING['10'] = Rank.TEN  # Handle '10' specially

SUIT_FROM_STRING = {
    'S': Suit.SPADES,
    '♠': Suit.SPADES,
    'H': Suit.HEARTS,
    '♥': Suit.HEARTS,
    'D': Suit.DIAMONDS,
    '♦': Suit.DIAMONDS,
    'C': Suit.CLUBS,
    '♣': Suit.CLUBS
}


class Card:
    """Represents a playing card."""
    
    def __init__(self, rank: Rank, suit: Suit):
        """Initialize a card with rank and suit."""
        self.rank = rank
        self.suit = suit
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card':
        """
        Create a card from string representation.
        
        Args:
            card_str: String like "A-S", "10-H", "K-D"
        
        Returns:
            Card object
        """
        parts = card_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid card string format: {card_str}")
        
        rank_str, suit_str = parts
        rank = RANK_FROM_STRING.get(rank_str.upper())
        suit = SUIT_FROM_STRING.get(suit_str.upper())
        
        if rank is None:
            raise ValueError(f"Invalid rank: {rank_str}")
        if suit is None:
            raise ValueError(f"Invalid suit: {suit_str}")
        
        return cls(rank, suit)
    
    @classmethod
    def from_index(cls, index: int) -> 'Card':
        """
        Create a card from index (0-51).
        
        Args:
            index: Integer from 0 to 51
        
        Returns:
            Card object
        """
        if not 0 <= index < 52:
            raise ValueError(f"Card index must be 0-51, got {index}")
        
        rank = Rank(index % 13)
        suit = Suit(index // 13)
        return cls(rank, suit)
    
    def to_index(self) -> int:
        """Convert card to index (0-51)."""
        return int(self.rank) + int(self.suit) * 13
    
    def __str__(self) -> str:
        """String representation of card."""
        return f"{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Card({RANK_SYMBOLS[self.rank]}, {SUIT_SYMBOLS[self.suit]})"
    
    def __eq__(self, other) -> bool:
        """Check if two cards are equal."""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        """Hash function for use in sets and dicts."""
        return hash((self.rank, self.suit))


class Deck:
    """Represents a deck of cards."""
    
    def __init__(self):
        """Initialize a standard 52-card deck."""
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self):
        """Reset deck to all 52 cards."""
        self.cards = []
        for suit in Suit:
            for rank in Rank:
                self.cards.append(Card(rank, suit))
    
    def shuffle(self):
        """Shuffle the deck."""
        import random
        random.shuffle(self.cards)
    
    def deal(self, num_cards: int = 1) -> List[Card]:
        """
        Deal cards from the deck.
        
        Args:
            num_cards: Number of cards to deal
        
        Returns:
            List of dealt cards
        """
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot deal {num_cards} cards, only {len(self.cards)} remaining")
        
        dealt = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt
    
    def __len__(self) -> int:
        """Return number of cards remaining in deck."""
        return len(self.cards)
