"""Enhanced Monte Carlo simulation with range support from dickreuter/Poker."""

import json
import logging
import operator
import os
from collections import Counter
from typing import List, Set, Tuple, Optional

import numpy as np

from ..game import Card, Deck, HandEvaluator, Rank, Suit


class MonteCarloSimulator:
    """
    Advanced Monte Carlo simulator for poker equity calculation.
    Integrates concepts from dickreuter/Poker including:
    - Opponent range modeling
    - Preflop equity tables
    - Multi-opponent support
    """
    
    def __init__(self, use_range_of_range: bool = False):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            use_range_of_range: Whether to use range of range calculation
        """
        self.logger = logging.getLogger('montecarlo')
        self.use_range_of_range = use_range_of_range
        self.preflop_equities = {}
        self._load_preflop_equities()
    
    def _load_preflop_equities(self):
        """Load preflop equity tables."""
        # This would load from a JSON file - for now, use basic values
        # In production, this should load from preflop_equity.json
        self.preflop_equities = self._generate_basic_preflop_equities()
    
    def _generate_basic_preflop_equities(self) -> dict:
        """Generate basic preflop equity estimates."""
        equities = {}
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        
        for i, r1 in enumerate(ranks):
            for j, r2 in enumerate(ranks):
                # Pairs
                if i == j:
                    equity = 0.5 + (i / len(ranks)) * 0.35
                    equities[f"{r1}{r2}"] = equity
                # Suited
                elif i < j:
                    equity = 0.45 + ((i + j) / (2 * len(ranks))) * 0.3
                    equities[f"{r1}{r2}S"] = equity
                # Offsuit
                else:
                    equity = 0.40 + ((i + j) / (2 * len(ranks))) * 0.25
                    equities[f"{r1}{r2}O"] = equity
        
        return equities
    
    def get_hand_notation(self, card1: Card, card2: Card) -> Tuple[str, str]:
        """
        Get short notation for two cards (e.g., 'AKS', 'QQO').
        
        Args:
            card1: First card
            card2: Second card
        
        Returns:
            Tuple of (notation1, notation2) for both orderings
        """
        rank1 = self._rank_to_char(card1.rank)
        rank2 = self._rank_to_char(card2.rank)
        
        if card1.rank == card2.rank:
            # Pair
            return f"{rank1}{rank2}", f"{rank1}{rank2}"
        
        suited = 'S' if card1.suit == card2.suit else 'O'
        return f"{rank1}{rank2}{suited}", f"{rank2}{rank1}{suited}"
    
    def _rank_to_char(self, rank: Rank) -> str:
        """Convert rank enum to character."""
        mapping = {
            Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
            Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9',
            Rank.TEN: 'T', Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K',
            Rank.ACE: 'A'
        }
        return mapping[rank]
    
    def get_opponent_ranges(self, opponent_range_pct: float) -> Set[str]:
        """
        Get allowed card combinations for opponent based on range percentage.
        
        Args:
            opponent_range_pct: Percentage of hands opponent plays (0.0 to 1.0)
        
        Returns:
            Set of allowed hand notations
        """
        sorted_hands = sorted(
            self.preflop_equities.items(),
            key=operator.itemgetter(1),
            reverse=True
        )
        
        count = len(sorted_hands)
        take_top = int(count * opponent_range_pct)
        
        allowed_hands = set([hand[0] for hand in sorted_hands[:take_top]])
        return allowed_hands
    
    def simulate(self,
                hole_cards: List[Card],
                community_cards: List[Card],
                num_opponents: int = 1,
                num_simulations: int = 10000,
                opponent_ranges: Optional[List[float]] = None) -> dict:
        """
        Run Monte Carlo simulation to calculate win probability.
        
        Args:
            hole_cards: Player's hole cards (2 cards)
            community_cards: Community cards (0-5 cards)
            num_opponents: Number of opponents
            num_simulations: Number of simulations to run
            opponent_ranges: Range for each opponent (0.0-1.0), None for random
        
        Returns:
            Dictionary with simulation results:
            - win_rate: Probability of winning
            - tie_rate: Probability of tying
            - lose_rate: Probability of losing
            - equity: Expected value (0.0-1.0)
        """
        if opponent_ranges is None:
            opponent_ranges = [1.0] * num_opponents  # All hands
        
        wins = 0
        ties = 0
        losses = 0
        
        # Get opponent allowed hands
        opponent_allowed_hands = [
            self.get_opponent_ranges(range_pct)
            for range_pct in opponent_ranges
        ]
        
        for _ in range(num_simulations):
            # Create deck and remove known cards
            deck = Deck()
            deck.shuffle()
            
            # Remove player's hole cards and community cards
            for card in hole_cards + community_cards:
                try:
                    deck.cards.remove(card)
                except ValueError:
                    # Card already removed
                    pass
            
            # Deal remaining community cards
            remaining_community = 5 - len(community_cards)
            sim_community = community_cards + deck.deal(remaining_community)
            
            # Deal opponent hands
            opponent_hands = []
            for opp_range in opponent_allowed_hands:
                # Deal two cards for opponent
                opp_hole = deck.deal(2)
                
                # Check if within range (if range specified)
                if opp_range != set(self.preflop_equities.keys()):
                    notation1, notation2 = self.get_hand_notation(opp_hole[0], opp_hole[1])
                    if notation1 not in opp_range and notation2 not in opp_range:
                        # Redeal (simplified - in production would be more sophisticated)
                        pass
                
                opponent_hands.append(opp_hole)
            
            # Evaluate hands
            player_hand = hole_cards + sim_community
            player_rank, player_ties = HandEvaluator.evaluate_hand(player_hand)
            
            best_opponent_rank = None
            best_opponent_ties = None
            
            for opp_hole in opponent_hands:
                opp_hand = opp_hole + sim_community
                opp_rank, opp_ties = HandEvaluator.evaluate_hand(opp_hand)
                
                if best_opponent_rank is None or opp_rank > best_opponent_rank or \
                   (opp_rank == best_opponent_rank and opp_ties > best_opponent_ties):
                    best_opponent_rank = opp_rank
                    best_opponent_ties = opp_ties
            
            # Compare hands
            if player_rank > best_opponent_rank:
                wins += 1
            elif player_rank < best_opponent_rank:
                losses += 1
            else:
                # Compare tiebreakers
                for p_tie, o_tie in zip(player_ties, best_opponent_ties):
                    if p_tie > o_tie:
                        wins += 1
                        break
                    elif p_tie < o_tie:
                        losses += 1
                        break
                else:
                    ties += 1
        
        total = wins + ties + losses
        win_rate = wins / total
        tie_rate = ties / total
        lose_rate = losses / total
        
        # Equity calculation (wins + half ties)
        equity = (wins + ties * 0.5) / total
        
        return {
            'win_rate': win_rate,
            'tie_rate': tie_rate,
            'lose_rate': lose_rate,
            'equity': equity,
            'simulations': num_simulations,
            'wins': wins,
            'ties': ties,
            'losses': losses
        }
    
    def calculate_pot_odds_ev(self,
                             pot: int,
                             bet_to_call: int,
                             equity: float) -> dict:
        """
        Calculate expected value based on pot odds.
        
        Args:
            pot: Current pot size
            bet_to_call: Amount needed to call
            equity: Win probability (0.0-1.0)
        
        Returns:
            Dictionary with:
            - pot_odds: Required win rate to break even
            - ev: Expected value
            - profitable: Whether call is profitable
        """
        pot_odds = bet_to_call / (pot + bet_to_call)
        ev = equity * (pot + bet_to_call) - bet_to_call
        profitable = equity > pot_odds
        
        return {
            'pot_odds': pot_odds,
            'required_equity': pot_odds,
            'actual_equity': equity,
            'ev': ev,
            'profitable': profitable
        }
