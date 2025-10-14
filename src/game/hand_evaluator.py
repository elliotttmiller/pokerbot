"""Hand evaluation for poker hands."""

from collections import Counter
from enum import IntEnum
from typing import List, Tuple

from .card import Card, Rank


class HandRank(IntEnum):
    """Poker hand rankings from weakest to strongest."""
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


class HandEvaluator:
    """Evaluates poker hands and determines winners."""
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """
        Evaluate a poker hand and return its rank and tiebreaker values.
        
        Args:
            cards: List of 5-7 cards to evaluate
        
        Returns:
            Tuple of (HandRank, tiebreaker_values)
        """
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate a hand")
        
        # For 7 cards, evaluate all 5-card combinations and return the best
        if len(cards) > 5:
            from itertools import combinations
            best_rank = HandRank.HIGH_CARD
            best_tiebreakers = []
            
            for combo in combinations(cards, 5):
                rank, tiebreakers = HandEvaluator._evaluate_five_cards(list(combo))
                if rank > best_rank or (rank == best_rank and tiebreakers > best_tiebreakers):
                    best_rank = rank
                    best_tiebreakers = tiebreakers
            
            return best_rank, best_tiebreakers
        
        return HandEvaluator._evaluate_five_cards(cards)
    
    @staticmethod
    def _evaluate_five_cards(cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """Evaluate exactly 5 cards."""
        ranks = sorted([card.rank for card in cards], reverse=True)
        suits = [card.suit for card in cards]
        rank_counts = Counter(ranks)
        
        is_flush = len(set(suits)) == 1
        is_straight = HandEvaluator._is_straight(ranks)
        
        # Check for royal flush
        if is_flush and is_straight and ranks[0] == Rank.ACE:
            return HandRank.ROYAL_FLUSH, ranks
        
        # Check for straight flush
        if is_flush and is_straight:
            return HandRank.STRAIGHT_FLUSH, ranks
        
        # Check for four of a kind
        if 4 in rank_counts.values():
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r in ranks if r != quad_rank][0]
            return HandRank.FOUR_OF_A_KIND, [quad_rank, kicker]
        
        # Check for full house
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trips_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return HandRank.FULL_HOUSE, [trips_rank, pair_rank]
        
        # Check for flush
        if is_flush:
            return HandRank.FLUSH, ranks
        
        # Check for straight
        if is_straight:
            return HandRank.STRAIGHT, ranks
        
        # Check for three of a kind
        if 3 in rank_counts.values():
            trips_rank = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r in ranks if r != trips_rank], reverse=True)
            return HandRank.THREE_OF_A_KIND, [trips_rank] + kickers
        
        # Check for two pair
        pairs = [r for r, c in rank_counts.items() if c == 2]
        if len(pairs) == 2:
            pairs = sorted(pairs, reverse=True)
            kicker = [r for r in ranks if r not in pairs][0]
            return HandRank.TWO_PAIR, pairs + [kicker]
        
        # Check for one pair
        if len(pairs) == 1:
            pair_rank = pairs[0]
            kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
            return HandRank.ONE_PAIR, [pair_rank] + kickers
        
        # High card
        return HandRank.HIGH_CARD, ranks
    
    @staticmethod
    def _is_straight(ranks: List[int]) -> bool:
        """Check if ranks form a straight."""
        sorted_ranks = sorted(set(ranks))
        
        # Check for wheel (A-2-3-4-5)
        if sorted_ranks == [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.ACE]:
            return True
        
        # Check for regular straight
        if len(sorted_ranks) == 5:
            return sorted_ranks[-1] - sorted_ranks[0] == 4
        
        return False
    
    @staticmethod
    def calculate_hand_strength(cards: List[Card]) -> float:
        """
        Calculate a simple hand strength value (0-9+).
        
        Args:
            cards: List of cards to evaluate
        
        Returns:
            Float representing hand strength
        """
        if len(cards) < 2:
            return 0.0
        
        if len(cards) < 5:
            # Pre-flop or incomplete hand - evaluate hole cards
            return HandEvaluator._evaluate_hole_cards(cards)
        
        rank, _ = HandEvaluator.evaluate_hand(cards)
        return float(rank)
    
    @staticmethod
    def _evaluate_hole_cards(hole_cards: List[Card]) -> float:
        """Evaluate strength of hole cards (2 cards)."""
        if len(hole_cards) != 2:
            return 0.0
        
        card1, card2 = hole_cards
        high_rank = max(card1.rank, card2.rank)
        low_rank = min(card1.rank, card2.rank)
        
        # Base value on high card
        base_value = high_rank / 12.0  # Normalize to 0-1
        
        # Bonus for pairs
        if card1.rank == card2.rank:
            return 0.9 + (high_rank / 120.0)
        
        # Bonus for suited cards
        suited_bonus = 0.1 if card1.suit == card2.suit else 0.0
        
        # Bonus for connected cards
        connected_bonus = 0.05 if abs(high_rank - low_rank) <= 4 else 0.0
        
        return base_value + suited_bonus + connected_bonus
    
    @staticmethod
    def compare_hands(cards1: List[Card], cards2: List[Card]) -> int:
        """
        Compare two hands.
        
        Args:
            cards1: First hand
            cards2: Second hand
        
        Returns:
            1 if cards1 wins, -1 if cards2 wins, 0 if tie
        """
        rank1, tie1 = HandEvaluator.evaluate_hand(cards1)
        rank2, tie2 = HandEvaluator.evaluate_hand(cards2)
        
        if rank1 > rank2:
            return 1
        elif rank1 < rank2:
            return -1
        
        # Same rank, compare tiebreakers
        for t1, t2 in zip(tie1, tie2):
            if t1 > t2:
                return 1
            elif t1 < t2:
                return -1
        
        return 0
