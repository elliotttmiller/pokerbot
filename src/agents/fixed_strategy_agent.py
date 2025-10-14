"""Fixed strategy agent using GTO-based heuristics."""

from typing import List

from ..game import Action, Card
from src.deepstack.hand_evaluator import HandEvaluator
from .base_agent import BaseAgent


class FixedStrategyAgent(BaseAgent):
    """Agent using fixed GTO-inspired strategy."""
    
    def __init__(self, name: str = "FixedStrategy"):
        """Initialize fixed strategy agent."""
        super().__init__(name)
    
    def choose_action(self,
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> tuple[Action, int]:
        """
        Choose action based on fixed strategy.
        
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
        # Combine cards for evaluation
        all_cards = hole_cards + community_cards
        
        # Calculate hand strength
        hand_strength = HandEvaluator.calculate_hand_strength(all_cards)
        
        # Calculate pot odds
        if current_bet > 0 and pot > 0:
            pot_odds = current_bet / (pot + current_bet)
        else:
            pot_odds = 0
        
        # Estimate win probability based on hand strength
        # Scale hand strength (0-9) to probability (0-1)
        if len(community_cards) >= 5:
            # Post-river: use actual hand rank
            win_probability = hand_strength / 9.0
        elif len(community_cards) >= 3:
            # Post-flop: use hand strength with some uncertainty
            win_probability = hand_strength / 10.0
        else:
            # Pre-flop: use hole card strength
            win_probability = hand_strength
        
        # Decision logic
        if not community_cards:
            # Pre-flop strategy
            return self._preflop_strategy(hand_strength, current_bet, pot, 
                                         player_stack, win_probability, pot_odds)
        else:
            # Post-flop strategy
            return self._postflop_strategy(hand_strength, current_bet, pot,
                                          player_stack, win_probability, pot_odds)
    
    def _preflop_strategy(self, hand_strength: float, current_bet: int, 
                         pot: int, player_stack: int,
                         win_probability: float, pot_odds: float) -> tuple[Action, int]:
        """Pre-flop decision strategy."""
        # Fold weak hands
        if hand_strength < 0.35:
            return Action.FOLD, 0
        
        # With a bet to call
        if current_bet > 0:
            # Strong hands: raise
            if hand_strength >= 0.75:
                raise_amount = min(current_bet * 3, player_stack // 4)
                return Action.RAISE, raise_amount
            
            # Good hands: call if pot odds are favorable
            if hand_strength >= 0.5:
                if win_probability > pot_odds * 0.7:
                    return Action.CALL, 0
                else:
                    return Action.FOLD, 0
            
            # Marginal hands: fold if bet is too high
            if current_bet > pot * 0.5:
                return Action.FOLD, 0
            
            # Otherwise call
            return Action.CALL, 0
        
        # No bet: check or raise with strong hands
        if hand_strength >= 0.7:
            raise_amount = min(pot // 2, player_stack // 5)
            return Action.RAISE, max(raise_amount, 20)
        
        return Action.CHECK, 0
    
    def _postflop_strategy(self, hand_strength: float, current_bet: int,
                          pot: int, player_stack: int,
                          win_probability: float, pot_odds: float) -> tuple[Action, int]:
        """Post-flop decision strategy."""
        # With a bet to call
        if current_bet > 0:
            # Very strong hands: raise
            if hand_strength >= 6.0:  # Full house or better
                raise_amount = min(pot, player_stack // 3)
                return Action.RAISE, raise_amount
            
            # Strong hands: call/raise based on pot odds
            if hand_strength >= 4.0:  # Straight or better
                if win_probability > pot_odds:
                    return Action.CALL, 0
                else:
                    raise_amount = min(pot // 2, player_stack // 4)
                    return Action.RAISE, raise_amount
            
            # Medium hands: call if pot odds are good
            if hand_strength >= 2.0:  # Pair or better
                if win_probability > pot_odds * 1.2:
                    return Action.CALL, 0
                else:
                    return Action.FOLD, 0
            
            # Weak hands: fold
            return Action.FOLD, 0
        
        # No bet to call
        else:
            # Strong hands: bet
            if hand_strength >= 3.0:  # Three of a kind or better
                raise_amount = min(pot * 0.6, player_stack // 3)
                return Action.RAISE, max(raise_amount, 20)
            
            # Medium hands: check
            if hand_strength >= 1.0:  # Pair
                if win_probability > 0.5:
                    raise_amount = min(pot * 0.4, player_stack // 4)
                    return Action.RAISE, max(raise_amount, 20)
                return Action.CHECK, 0
            
            # Weak hands: check
            return Action.CHECK, 0
