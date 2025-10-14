"""
Hand Evaluator for Poker
Evaluates 5-card poker hands and determines winner
"""
from enum import Enum
from typing import List, Tuple
from collections import Counter
from .game_state import Card, Rank, Suit


class HandRank(Enum):
    """Poker hand rankings (higher is better)"""
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


class HandEvaluator:
    """Evaluates poker hands"""
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """
        Evaluate a 5-card poker hand
        Returns (hand_rank, tiebreakers) where tiebreakers is sorted list of card values
        """
        if len(cards) != 5:
            raise ValueError(f"Must evaluate exactly 5 cards, got {len(cards)}")
        
        ranks = sorted([card.rank.value for card in cards], reverse=True)
        suits = [card.suit for card in cards]
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        is_straight = HandEvaluator._is_straight(ranks)
        
        # Special case: A-2-3-4-5 straight (wheel)
        if ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            ranks = [5, 4, 3, 2, 1]  # Ace is low in wheel
        
        # Count rank frequencies
        rank_counts = Counter(ranks)
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True)
        
        # Determine hand rank
        if is_straight and is_flush:
            if ranks[0] == 14:  # Ace high straight flush
                return HandRank.ROYAL_FLUSH, ranks
            return HandRank.STRAIGHT_FLUSH, ranks
        
        if counts == [4, 1]:
            return HandRank.FOUR_OF_A_KIND, unique_ranks
        
        if counts == [3, 2]:
            return HandRank.FULL_HOUSE, unique_ranks
        
        if is_flush:
            return HandRank.FLUSH, ranks
        
        if is_straight:
            return HandRank.STRAIGHT, ranks
        
        if counts == [3, 1, 1]:
            return HandRank.THREE_OF_A_KIND, unique_ranks
        
        if counts == [2, 2, 1]:
            return HandRank.TWO_PAIR, unique_ranks
        
        if counts == [2, 1, 1, 1]:
            return HandRank.PAIR, unique_ranks
        
        return HandRank.HIGH_CARD, ranks
    
    @staticmethod
    def _is_straight(ranks: List[int]) -> bool:
        """Check if ranks form a straight"""
        if len(ranks) != 5:
            return False
        return ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5
    
    @staticmethod
    def best_hand(hole_cards: List[Card], community_cards: List[Card]) -> Tuple[HandRank, List[int], List[Card]]:
        """
        Find the best 5-card hand from hole cards + community cards
        Returns (hand_rank, tiebreakers, best_5_cards)
        """
        from itertools import combinations
        
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate")
        
        best_rank = None
        best_tiebreakers = None
        best_cards = None
        
        # Try all 5-card combinations
        for combo in combinations(all_cards, 5):
            rank, tiebreakers = HandEvaluator.evaluate_hand(list(combo))
            
            if best_rank is None or HandEvaluator._compare_hands(
                (rank, tiebreakers), (best_rank, best_tiebreakers)
            ) > 0:
                best_rank = rank
                best_tiebreakers = tiebreakers
                best_cards = list(combo)
        
        return best_rank, best_tiebreakers, best_cards
    
    @staticmethod
    def _compare_hands(hand1: Tuple[HandRank, List[int]], 
                       hand2: Tuple[HandRank, List[int]]) -> int:
        """
        Compare two hands
        Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        rank1, tie1 = hand1
        rank2, tie2 = hand2
        
        if rank1.value > rank2.value:
            return 1
        elif rank1.value < rank2.value:
            return -1
        
        # Same rank, compare tiebreakers
        for t1, t2 in zip(tie1, tie2):
            if t1 > t2:
                return 1
            elif t1 < t2:
                return -1
        
        return 0  # Complete tie
    
    @staticmethod
    def compare_hands_full(hole1: List[Card], hole2: List[Card], 
                          community: List[Card]) -> int:
        """
        Compare two players' hands given community cards
        Returns: 1 if player1 wins, -1 if player2 wins, 0 if tie
        """
        hand1 = HandEvaluator.best_hand(hole1, community)
        hand2 = HandEvaluator.best_hand(hole2, community)
        
        return HandEvaluator._compare_hands(
            (hand1[0], hand1[1]),
            (hand2[0], hand2[1])
        )
