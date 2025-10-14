"""Base agent interface for poker bots."""

from abc import ABC, abstractmethod
from typing import List

from ..game import Action, Card


class BaseAgent(ABC):
    """Abstract base class for poker agents."""
    
    def __init__(self, name: str = "BaseAgent"):
        """
        Initialize the agent.
        
        Args:
            name: Name of the agent
        """
        self.name = name
    
    @abstractmethod
    def choose_action(self, 
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> tuple[Action, int]:
        """
        Choose an action based on game state.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards on the table
            pot: Current pot size
            current_bet: Current bet amount player must match
            player_stack: Player's remaining stack
            opponent_bet: Opponent's current bet
        
        Returns:
            Tuple of (Action, raise_amount)
        """
        pass
    
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
