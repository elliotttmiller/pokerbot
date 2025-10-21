#!/usr/bin/env python3
"""
ACPC Hand History Parser for Official DeepStack Data

Parses hand histories in ACPC (Annual Computer Poker Competition) format from
official DeepStack championship matches to extract training data.

Format: STATE:<Hand#>:<Betting>:<Cards>:<Payoff>:<Player Positions> # <Timestamp>

Example:
STATE:19:r200c/cr400c/cr850f:9hQc|5dQd/4h4c6c/4s:-400|400:DeepStack|player # 1479132064.35509
"""

import os
import re
from typing import List, Dict, Tuple, Optional, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class Card:
    """Represents a poker card."""
    rank: str  # '2'-'9', 'T', 'J', 'Q', 'K', 'A'
    suit: str  # 'h', 'd', 'c', 's'
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    @classmethod
    def from_string(cls, card_str: str):
        """Parse card from string like '9h' or 'Kc'."""
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")
        return cls(rank=card_str[0], suit=card_str[1])


@dataclass
class Action:
    """Represents a poker action."""
    action_type: str  # 'f' (fold), 'c' (check/call), 'r' (raise)
    amount: Optional[int] = None  # Raise amount (total in pot)
    
    def __str__(self):
        if self.action_type == 'r' and self.amount is not None:
            return f"r{self.amount}"
        return self.action_type


@dataclass
class HandHistory:
    """Represents a complete hand history from ACPC format."""
    hand_number: int
    betting_sequence: List[List[Action]]  # Per street
    hole_cards: List[List[Card]]  # Per player
    board_cards: List[List[Card]]  # Per street (flop, turn, river)
    payoff: List[int]  # Per player
    player_names: List[str]
    timestamp: float
    
    @property
    def num_streets(self) -> int:
        """Number of betting streets."""
        return len(self.betting_sequence)
    
    @property
    def winner(self) -> int:
        """Index of winning player (0 or 1)."""
        return 0 if self.payoff[0] > 0 else 1
    
    @property
    def final_pot(self) -> int:
        """Final pot size."""
        return abs(self.payoff[0]) + abs(self.payoff[1])


class ACPCParser:
    """Parser for ACPC format hand histories."""
    
    # ACPC format regex
    STATE_PATTERN = re.compile(
        r'STATE:(\d+):([^:]*):([^:]*):([^:]*):([^#]*)#\s*([\d.]+)'
    )
    
    # LBR format regex (from DeepStack vs LBR matches)
    # Format: <hand#>:<players>:<betting>:<cards>:<payoff>(<ev>)
    LBR_PATTERN = re.compile(
        r'(\d+):([^:]+):([^:]*):([^:]*):([^(]+)\([^)]+\)'
    )
    
    def __init__(self):
        self.hands: List[HandHistory] = []
        self._stats = defaultdict(int)
    
    def parse_file(self, filepath: str) -> List[HandHistory]:
        """Parse ACPC or LBR log file and return list of hand histories."""
        hands = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('In config'):
                    continue
                
                try:
                    hand = self.parse_line(line)
                    if hand:
                        hands.append(hand)
                        self._stats['parsed'] += 1
                except Exception as e:
                    self._stats['errors'] += 1
                    # print(f"Error parsing line: {e}")
                    continue
        
        self.hands.extend(hands)
        return hands
    
    def parse_line(self, line: str) -> Optional[HandHistory]:
        """Parse single ACPC or LBR format line."""
        # Try ACPC format first
        match = self.STATE_PATTERN.match(line)
        if match:
            hand_num, betting, cards, payoff, players, timestamp = match.groups()
            timestamp = float(timestamp)
        else:
            # Try LBR format
            match = self.LBR_PATTERN.match(line)
            if not match:
                return None
            
            hand_num, players, betting, cards, payoff = match.groups()
            timestamp = 0.0  # LBR format doesn't include timestamp
        
        # Parse betting sequence
        betting_seq = self._parse_betting(betting)
        
        # Parse cards
        hole_cards, board_cards = self._parse_cards(cards)
        
        # Parse payoff
        payoffs = [int(float(p.strip())) for p in payoff.split(',')]
        
        # Parse player names
        player_names = [p.strip() for p in players.replace('|', ',').split(',')]
        
        return HandHistory(
            hand_number=int(hand_num),
            betting_sequence=betting_seq,
            hole_cards=hole_cards,
            board_cards=board_cards,
            payoff=payoffs,
            player_names=player_names,
            timestamp=timestamp
        )
    
    def _parse_betting(self, betting_str: str) -> List[List[Action]]:
        """Parse betting sequence string like 'r200c/cr400c/cr850f'."""
        if not betting_str:
            return [[]]
        
        streets = betting_str.split('/')
        betting_seq = []
        
        for street in streets:
            actions = []
            i = 0
            while i < len(street):
                action_char = street[i]
                
                if action_char in ['f', 'c']:
                    actions.append(Action(action_type=action_char))
                    i += 1
                elif action_char == 'r':
                    # Extract raise amount
                    j = i + 1
                    while j < len(street) and street[j].isdigit():
                        j += 1
                    
                    if j > i + 1:
                        amount = int(street[i+1:j])
                        actions.append(Action(action_type='r', amount=amount))
                    else:
                        actions.append(Action(action_type='r'))
                    i = j
                else:
                    i += 1
            
            betting_seq.append(actions)
        
        return betting_seq
    
    def _parse_cards(self, cards_str: str) -> Tuple[List[List[Card]], List[List[Card]]]:
        """Parse cards string like '9hQc|5dQd/4h4c6c/4s'."""
        if not cards_str:
            return [[], []], [[]]
        
        # Split by '/' for streets
        parts = cards_str.split('/')
        
        # First part is hole cards (separated by '|')
        hole_cards_str = parts[0]
        players_cards = hole_cards_str.split('|')
        hole_cards = []
        
        for player_cards in players_cards:
            cards = []
            # Parse 2-character cards
            for i in range(0, len(player_cards), 2):
                if i + 1 < len(player_cards):
                    card_str = player_cards[i:i+2]
                    try:
                        cards.append(Card.from_string(card_str))
                    except ValueError:
                        continue
            hole_cards.append(cards)
        
        # Remaining parts are board cards per street
        board_cards = []
        for street_cards_str in parts[1:]:
            street_cards = []
            for i in range(0, len(street_cards_str), 2):
                if i + 1 < len(street_cards_str):
                    card_str = street_cards_str[i:i+2]
                    try:
                        street_cards.append(Card.from_string(card_str))
                    except ValueError:
                        continue
            board_cards.append(street_cards)
        
        return hole_cards, board_cards
    
    def get_stats(self) -> Dict:
        """Get parsing statistics."""
        return dict(self._stats)
    
    def extract_training_features(self, hand: HandHistory) -> Dict:
        """Extract features suitable for training from a hand history."""
        features = {
            'num_streets': hand.num_streets,
            'final_pot': hand.final_pot,
            'winner': hand.winner,
            'num_actions_per_street': [len(actions) for actions in hand.betting_sequence],
            'has_aggression': any(
                any(a.action_type == 'r' for a in street)
                for street in hand.betting_sequence
            ),
            'preflop_actions': len(hand.betting_sequence[0]) if hand.betting_sequence else 0,
            'went_to_showdown': hand.num_streets >= 4,  # All streets played
        }
        
        # Extract bet sizing info
        bet_sizes = []
        for street in hand.betting_sequence:
            for action in street:
                if action.action_type == 'r' and action.amount:
                    bet_sizes.append(action.amount)
        
        if bet_sizes:
            features['avg_bet_size'] = np.mean(bet_sizes)
            features['max_bet_size'] = max(bet_sizes)
            features['num_raises'] = len(bet_sizes)
        else:
            features['avg_bet_size'] = 0
            features['max_bet_size'] = 0
            features['num_raises'] = 0
        
        return features


class DeepStackDataExtractor:
    """Extract training insights from official DeepStack hand histories."""
    
    def __init__(self, data_dir: str):
        """
        Initialize extractor.
        
        Args:
            data_dir: Path to official_deepstack_handhistory directory
        """
        self.data_dir = Path(data_dir)
        self.parser = ACPCParser()
        self.hands: List[HandHistory] = []
        
    def load_all_hands(self) -> int:
        """Load all available hand histories."""
        count = 0
        
        # Find all .out files (LBR format)
        for out_file in self.data_dir.rglob("*.out"):
            try:
                hands = self.parser.parse_file(str(out_file))
                count += len(hands)
            except Exception as e:
                print(f"Error loading {out_file}: {e}")
        
        self.hands = self.parser.hands
        return count
    
    def analyze_betting_patterns(self) -> Dict:
        """Analyze betting patterns across all hands."""
        analysis = {
            'total_hands': len(self.hands),
            'street_distribution': defaultdict(int),
            'bet_sizes': [],
            'pot_sizes': [],
            'aggression_frequency': 0,
            'showdown_frequency': 0,
        }
        
        for hand in self.hands:
            # Street distribution
            analysis['street_distribution'][hand.num_streets] += 1
            
            # Pot sizes
            analysis['pot_sizes'].append(hand.final_pot)
            
            # Aggression
            has_aggression = any(
                any(a.action_type == 'r' for a in street)
                for street in hand.betting_sequence
            )
            if has_aggression:
                analysis['aggression_frequency'] += 1
            
            # Showdown
            if hand.num_streets >= 4:
                analysis['showdown_frequency'] += 1
            
            # Bet sizes
            for street in hand.betting_sequence:
                for action in street:
                    if action.action_type == 'r' and action.amount:
                        analysis['bet_sizes'].append(action.amount)
        
        # Convert to percentages
        total = len(self.hands)
        if total > 0:
            analysis['aggression_frequency'] /= total
            analysis['showdown_frequency'] /= total
            
            # Normalize street distribution
            street_dist = {}
            for streets, count in analysis['street_distribution'].items():
                street_dist[f"{streets}_streets"] = count / total
            analysis['street_distribution'] = street_dist
        
        # Bet size statistics
        if analysis['bet_sizes']:
            analysis['bet_size_stats'] = {
                'mean': float(np.mean(analysis['bet_sizes'])),
                'median': float(np.median(analysis['bet_sizes'])),
                'std': float(np.std(analysis['bet_sizes'])),
                'percentiles': {
                    '25': float(np.percentile(analysis['bet_sizes'], 25)),
                    '50': float(np.percentile(analysis['bet_sizes'], 50)),
                    '75': float(np.percentile(analysis['bet_sizes'], 75)),
                    '90': float(np.percentile(analysis['bet_sizes'], 90)),
                    '95': float(np.percentile(analysis['bet_sizes'], 95)),
                }
            }
        
        return analysis
    
    def export_training_dataset(self, output_path: str) -> None:
        """Export extracted features as training dataset."""
        features_list = []
        
        for hand in self.hands:
            features = self.parser.extract_training_features(hand)
            features_list.append(features)
        
        # Save as numpy arrays
        np.savez(
            output_path,
            features=features_list,
            metadata={
                'total_hands': len(self.hands),
                'source': 'official_deepstack_handhistory'
            }
        )
        
        print(f"Exported {len(features_list)} hand features to {output_path}")


if __name__ == '__main__':
    """Example usage and testing."""
    import sys
    
    # Test with official data if available
    data_dir = 'data/official_deepstack_handhistory'
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found")
        print("Please ensure official DeepStack data is present")
        sys.exit(1)
    
    print("Loading official DeepStack hand histories...")
    extractor = DeepStackDataExtractor(data_dir)
    
    num_hands = extractor.load_all_hands()
    print(f"✓ Loaded {num_hands:,} hands")
    
    print("\nAnalyzing betting patterns...")
    analysis = extractor.analyze_betting_patterns()
    
    print(f"\n📊 Analysis Results:")
    print(f"  Total hands: {analysis['total_hands']:,}")
    print(f"  Aggression frequency: {analysis['aggression_frequency']:.1%}")
    print(f"  Showdown frequency: {analysis['showdown_frequency']:.1%}")
    
    if 'bet_size_stats' in analysis:
        print(f"\n  Bet Size Statistics:")
        stats = analysis['bet_size_stats']
        print(f"    Mean: {stats['mean']:.0f}")
        print(f"    Median: {stats['median']:.0f}")
        print(f"    Percentiles: 25%={stats['percentiles']['25']:.0f}, "
              f"75%={stats['percentiles']['75']:.0f}, "
              f"95%={stats['percentiles']['95']:.0f}")
    
    print(f"\n  Street Distribution:")
    for street_key, freq in sorted(analysis['street_distribution'].items()):
        print(f"    {street_key}: {freq:.1%}")
    
    # Export dataset
    output_file = 'data/deepstack_championship_features.npz'
    print(f"\n💾 Exporting training dataset to {output_file}...")
    extractor.export_training_dataset(output_file)
    
    print("\n✓ Complete!")
