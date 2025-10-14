"""Random agent that makes random decisions."""

import random
from typing import List

from ..game import Action, Card
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that makes random decisions."""
    
    def __init__(self, name: str = "RandomAgent"):
        """Initialize random agent."""
        super().__init__(name)
    
    def choose_action(self,
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> tuple[Action, int]:
        """
        Choose a random action.
        
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
        # Determine valid actions
        valid_actions = []
        
        # Can always fold
        valid_actions.append(Action.FOLD)
        
        # Can check if no bet to call
        if current_bet == 0:
            valid_actions.append(Action.CHECK)
        
        # Can call if there's a bet and we have chips
        if current_bet > 0 and player_stack >= current_bet:
            valid_actions.append(Action.CALL)
        
        # Can raise if we have enough chips
        min_raise = max(current_bet * 2, 20)
        if player_stack >= min_raise:
            valid_actions.append(Action.RAISE)
        
        # Choose random action
        action = random.choice(valid_actions)
        
        # If raising, choose random raise amount
        raise_amount = 0
        if action == Action.RAISE:
            min_raise = max(current_bet, 20)
            max_raise = player_stack
            raise_amount = random.randint(min_raise, min(max_raise, min_raise * 5))
        
        return action, raise_amount
