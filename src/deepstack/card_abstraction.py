"""Card abstraction and information set clustering.

Reduces the massive poker game tree to tractable sizes through:
- Hand strength bucketing
- Strategic similarity clustering
- Lookup tables for fast abstraction
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pickle
from pathlib import Path

from ..game import Card, HandEvaluator


class CardAbstraction:
    """
    Groups strategically similar card combinations together.
    
    Reduces 56+ billion information sets to manageable size through:
    - Hand strength percentile bucketing
    - Potential for improvement (EHS)
    - Board texture analysis
    """
    
    def __init__(self, 
                 n_buckets_preflop: int = 10,
                 n_buckets_flop: int = 50,
                 n_buckets_turn: int = 50,
                 n_buckets_river: int = 50,
                 lut_path: Optional[Path] = None):
        """
        Initialize card abstraction.
        
        Args:
            n_buckets_preflop: Number of buckets for preflop
            n_buckets_flop: Number of buckets for flop
            n_buckets_turn: Number of buckets for turn
            n_buckets_river: Number of buckets for river
            lut_path: Path to pre-computed lookup table
        """
        self.n_buckets = {
            'preflop': n_buckets_preflop,
            'flop': n_buckets_flop,
            'turn': n_buckets_turn,
            'river': n_buckets_river
        }
        
        self.card_info_lut = {}
        self.hand_evaluator = HandEvaluator()
        
        if lut_path and lut_path.exists():
            self.load_lut(lut_path)
    
    def get_bucket(self, 
                   hole_cards: List[Card],
                   community_cards: List[Card]) -> int:
        """
        Get abstraction bucket for given cards.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards on board
        
        Returns:
            Bucket number (0 to n_buckets-1)
        """
        # Determine betting round
        n_community = len(community_cards)
        if n_community == 0:
            round_name = 'preflop'
        elif n_community == 3:
            round_name = 'flop'
        elif n_community == 4:
            round_name = 'turn'
        else:
            round_name = 'river'
        
        # Create cache key
        cards_key = self._cards_to_key(hole_cards, community_cards)
        
        # Check cache
        if cards_key in self.card_info_lut:
            return self.card_info_lut[cards_key]
        
        # Calculate hand strength
        if round_name == 'preflop':
            bucket = self._get_preflop_bucket(hole_cards)
        else:
            bucket = self._get_postflop_bucket(
                hole_cards, community_cards, round_name
            )
        
        # Cache result
        self.card_info_lut[cards_key] = bucket
        
        return bucket
    
    def _get_preflop_bucket(self, hole_cards: List[Card]) -> int:
        """
        Get preflop bucket based on hand strength.
        
        Uses Chen formula or similar hand strength metric.
        
        Args:
            hole_cards: Two hole cards
        
        Returns:
            Bucket number
        """
        if len(hole_cards) < 2:
            return 0
        
        card1, card2 = hole_cards[0], hole_cards[1]
        
        # Calculate hand strength score
        # Higher ranks = better
        high_rank = max(card1.rank, card2.rank)
        low_rank = min(card1.rank, card2.rank)
        
        # Base score from high card
        score = high_rank * 2
        
        # Bonus for pairs
        if card1.rank == card2.rank:
            score += 20
        
        # Bonus for suited
        if card1.suit == card2.suit:
            score += 4
        
        # Bonus for connected
        if abs(card1.rank - card2.rank) == 1:
            score += 2
        elif abs(card1.rank - card2.rank) <= 2:
            score += 1
        
        # Normalize to bucket range
        # Score range is roughly 0-50
        max_score = 50
        bucket = int((score / max_score) * self.n_buckets['preflop'])
        bucket = min(bucket, self.n_buckets['preflop'] - 1)
        
        return bucket
    
    def _get_postflop_bucket(self,
                            hole_cards: List[Card],
                            community_cards: List[Card],
                            round_name: str) -> int:
        """
        Get postflop bucket based on hand strength and potential.
        
        Uses Expected Hand Strength (EHS) and potential.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            round_name: 'flop', 'turn', or 'river'
        
        Returns:
            Bucket number
        """
        # Combine cards
        all_cards = hole_cards + community_cards
        
        # Calculate current hand strength
        hand_result = self.hand_evaluator.evaluate_hand(all_cards)
        
        # Handle tuple return (rank, description) or just rank
        if isinstance(hand_result, tuple):
            hand_rank = hand_result[0]
        else:
            hand_rank = hand_result
        
        # Normalize hand rank to 0-1
        # Lower rank is better (royal flush = 1)
        # Worst hand is around 7462
        strength = 1.0 - (hand_rank / 7462.0)
        
        # Add potential for improvement (simplified)
        if round_name in ['flop', 'turn']:
            # Check for draws
            potential = self._calculate_potential(hole_cards, community_cards)
            strength = 0.7 * strength + 0.3 * potential
        
        # Map to bucket
        bucket = int(strength * self.n_buckets[round_name])
        bucket = min(bucket, self.n_buckets[round_name] - 1)
        
        return bucket
    
    def _calculate_potential(self,
                           hole_cards: List[Card],
                           community_cards: List[Card]) -> float:
        """
        Calculate potential for improvement.
        
        Simplified version - checks for flush/straight draws.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
        
        Returns:
            Potential score (0-1)
        """
        all_cards = hole_cards + community_cards
        
        # Check for flush draw
        suit_counts = {}
        for card in all_cards:
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        has_flush_draw = max_suit_count == 4
        
        # Check for straight draw (simplified)
        ranks = sorted([card.rank for card in all_cards])
        has_straight_draw = False
        for i in range(len(ranks) - 3):
            if ranks[i+3] - ranks[i] <= 4:
                has_straight_draw = True
                break
        
        # Calculate potential
        potential = 0.0
        if has_flush_draw:
            potential += 0.35
        if has_straight_draw:
            potential += 0.25
        
        return min(potential, 1.0)
    
    def _cards_to_key(self, 
                     hole_cards: List[Card],
                     community_cards: List[Card]) -> str:
        """
        Convert cards to cache key.
        
        Args:
            hole_cards: Hole cards
            community_cards: Community cards
        
        Returns:
            Cache key string
        """
        hole_str = ''.join(sorted([f"{c.rank}-{c.suit}" for c in hole_cards]))
        comm_str = ''.join(sorted([f"{c.rank}-{c.suit}" for c in community_cards]))
        return f"{hole_str}|{comm_str}"
    
    def save_lut(self, filepath: Path):
        """
        Save lookup table to file.
        
        Args:
            filepath: Path to save file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.card_info_lut, f)
    
    def load_lut(self, filepath: Path):
        """
        Load lookup table from file.
        
        Args:
            filepath: Path to load from
        """
        with open(filepath, 'rb') as f:
            self.card_info_lut = pickle.load(f)
    
    def get_bucket_key(self,
                      hole_cards: List[Card],
                      community_cards: List[Card]) -> str:
        """
        Get bucket-based key for information sets.
        
        This replaces direct card representations with abstract buckets.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
        
        Returns:
            Bucket-based key
        """
        bucket = self.get_bucket(hole_cards, community_cards)
        n_community = len(community_cards)
        return f"B{bucket}_C{n_community}"
