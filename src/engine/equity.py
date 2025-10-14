"""
Equity Calculator - Monte Carlo simulation for hand strength
"""
import random
from typing import List, Dict
from .game_state import Card, Rank, Suit
from .hand_evaluator import HandEvaluator


class EquityCalculator:
    """Calculate hand equity using Monte Carlo simulation"""
    
    @staticmethod
    def calculate_equity(
        hole_cards: List[Card],
        community_cards: List[Card],
        num_opponents: int = 1,
        num_simulations: int = 1000
    ) -> float:
        """
        Calculate equity (win probability) of a hand
        
        Args:
            hole_cards: Our hole cards
            community_cards: Current community cards
            num_opponents: Number of opponents
            num_simulations: Number of Monte Carlo simulations
        
        Returns:
            Equity as a probability (0.0 to 1.0)
        """
        if not hole_cards:
            return 0.0
        
        wins = 0
        ties = 0
        
        # Create remaining deck
        used_cards = set(hole_cards + community_cards)
        deck = [
            Card(rank=rank, suit=suit)
            for rank in Rank
            for suit in Suit
            if Card(rank=rank, suit=suit) not in used_cards
        ]
        
        for _ in range(num_simulations):
            # Shuffle deck
            random.shuffle(deck)
            
            # Deal remaining community cards
            sim_community = community_cards.copy()
            cards_needed = 5 - len(community_cards)
            sim_community.extend(deck[:cards_needed])
            
            # Deal opponent hole cards
            opponent_hands = []
            card_idx = cards_needed
            for _ in range(num_opponents):
                opponent_hole = [deck[card_idx], deck[card_idx + 1]]
                opponent_hands.append(opponent_hole)
                card_idx += 2
            
            # Evaluate our hand
            our_hand = HandEvaluator.best_hand(hole_cards, sim_community)
            
            # Compare with opponents
            we_win = True
            tie = False
            
            for opp_hole in opponent_hands:
                opp_hand = HandEvaluator.best_hand(opp_hole, sim_community)
                
                comparison = HandEvaluator._compare_hands(
                    (our_hand[0], our_hand[1]),
                    (opp_hand[0], opp_hand[1])
                )
                
                if comparison < 0:
                    we_win = False
                    break
                elif comparison == 0:
                    tie = True
            
            if we_win:
                if tie:
                    ties += 1
                else:
                    wins += 1
        
        # Equity = (wins + ties/2) / total
        equity = (wins + ties * 0.5) / num_simulations
        return equity
    
    @staticmethod
    def get_hand_strength(
        hole_cards: List[Card],
        community_cards: List[Card],
        num_opponents: int = 1
    ) -> str:
        """
        Get a categorical assessment of hand strength
        
        Returns: "very_strong", "strong", "medium", "weak", or "very_weak"
        """
        equity = EquityCalculator.calculate_equity(
            hole_cards, community_cards, num_opponents, num_simulations=500
        )
        
        if equity >= 0.75:
            return "very_strong"
        elif equity >= 0.60:
            return "strong"
        elif equity >= 0.45:
            return "medium"
        elif equity >= 0.30:
            return "weak"
        else:
            return "very_weak"
