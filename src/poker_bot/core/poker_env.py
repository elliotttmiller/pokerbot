"""
Poker Environment - The "Dojo" for Training and Decision Making.

A perfect, lightning-fast, GUI-agnostic simulator of No-Limit Texas Hold'em.
This is the foundation upon which the Strategist is trained and operates.

Key features:
- Abstract game state management (stacks, cards, pot, history)
- Methods: step(action), get_legal_actions(), is_terminal(), get_state_vector()
- Compatible with CFR solver and value network training

References:
- DeepStack: Game state representation and terminal equity
- OpenSpiel: Clean environment interface
"""

import random
from typing import List, Tuple, Optional, Dict, Any
from enum import IntEnum
from dataclasses import dataclass, field
import numpy as np

from .data_models import Card, ActionType, BettingRound


class HandRank(IntEnum):
    """Hand ranking categories."""
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_KIND = 7
    STRAIGHT_FLUSH = 8


@dataclass
class PokerPlayer:
    """Player state in the environment."""
    stack: int
    current_bet: int = 0
    folded: bool = False
    all_in: bool = False
    hole_cards: List[Card] = field(default_factory=list)


class PokerEnvironment:
    """
    Perfect poker simulator for training and real-time decision making.
    
    This is the "Dojo" - a fast, GUI-agnostic simulator that provides
    the foundation for CFR training and value network inference.
    
    Usage:
        env = PokerEnvironment(num_players=2, starting_stack=1000)
        env.reset()
        
        while not env.is_terminal():
            legal_actions = env.get_legal_actions()
            action, amount = agent.choose_action(env.get_state())
            env.step(action, amount)
        
        winner = env.get_winner()
    """
    
    def __init__(self,
                 num_players: int = 2,
                 starting_stack: int = 1000,
                 small_blind: int = 10,
                 big_blind: int = 20):
        """
        Initialize poker environment.
        
        Args:
            num_players: Number of players (2 for heads-up)
            starting_stack: Starting chip stack per player
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Game state
        self.players: List[PokerPlayer] = []
        self.deck: List[Card] = []
        self.community_cards: List[Card] = []
        self.pot: int = 0
        self.current_bet: int = 0
        self.street: BettingRound = BettingRound.PREFLOP
        self.current_player: int = 0
        self.dealer_button: int = 0
        self.action_history: List[Tuple[int, ActionType, int]] = []
        
        # Initialize
        self._init_deck()
        self._init_players()
    
    def _init_deck(self):
        """Initialize a fresh 52-card deck."""
        self.deck = [Card.from_index(i) for i in range(52)]
    
    def _init_players(self):
        """Initialize players with starting stacks."""
        self.players = [
            PokerPlayer(stack=self.starting_stack)
            for _ in range(self.num_players)
        ]
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset environment for a new hand.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Initial state dictionary
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset deck and shuffle
        self._init_deck()
        random.shuffle(self.deck)
        
        # Reset players
        for player in self.players:
            player.stack = self.starting_stack
            player.current_bet = 0
            player.folded = False
            player.all_in = False
            player.hole_cards = []
        
        # Reset game state
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.street = BettingRound.PREFLOP
        self.action_history = []
        
        # Deal hole cards
        for _ in range(2):
            for player in self.players:
                player.hole_cards.append(self.deck.pop())
        
        # Post blinds
        self._post_blinds()
        
        # Set current player (after big blind)
        self.current_player = (self.dealer_button + 3) % self.num_players
        if self.num_players == 2:
            self.current_player = self.dealer_button  # Heads-up: dealer acts first preflop
        
        return self.get_state()
    
    def _post_blinds(self):
        """Post small and big blinds."""
        sb_pos = (self.dealer_button + 1) % self.num_players
        bb_pos = (self.dealer_button + 2) % self.num_players
        
        if self.num_players == 2:
            sb_pos = self.dealer_button
            bb_pos = (self.dealer_button + 1) % self.num_players
        
        # Small blind
        sb_amount = min(self.small_blind, self.players[sb_pos].stack)
        self.players[sb_pos].stack -= sb_amount
        self.players[sb_pos].current_bet = sb_amount
        self.pot += sb_amount
        
        # Big blind
        bb_amount = min(self.big_blind, self.players[bb_pos].stack)
        self.players[bb_pos].stack -= bb_amount
        self.players[bb_pos].current_bet = bb_amount
        self.pot += bb_amount
        
        self.current_bet = bb_amount
    
    def step(self, action: ActionType, amount: int = 0) -> Tuple[Dict, float, bool]:
        """
        Execute action in environment.
        
        Args:
            action: Action type
            amount: Bet/raise amount (if applicable)
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        player = self.players[self.current_player]
        player_idx = self.current_player
        
        # Record action
        self.action_history.append((player_idx, action, amount))
        
        # Execute action
        if action == ActionType.FOLD:
            player.folded = True
        
        elif action == ActionType.CHECK:
            pass  # No action needed
        
        elif action == ActionType.CALL:
            call_amount = self.current_bet - player.current_bet
            actual_call = min(call_amount, player.stack)
            player.stack -= actual_call
            player.current_bet += actual_call
            self.pot += actual_call
            if player.stack == 0:
                player.all_in = True
        
        elif action in [ActionType.BET, ActionType.RAISE]:
            total_bet = amount
            to_add = total_bet - player.current_bet
            actual_add = min(to_add, player.stack)
            player.stack -= actual_add
            player.current_bet += actual_add
            self.pot += actual_add
            self.current_bet = player.current_bet
            if player.stack == 0:
                player.all_in = True
        
        elif action == ActionType.ALL_IN:
            all_in_amount = player.stack
            player.current_bet += all_in_amount
            self.pot += all_in_amount
            player.stack = 0
            player.all_in = True
            if player.current_bet > self.current_bet:
                self.current_bet = player.current_bet
        
        # Move to next player or street
        self._advance_game()
        
        # Check if terminal
        done = self.is_terminal()
        reward = 0.0
        
        if done:
            # Calculate rewards
            winners = self._determine_winners()
            if winners:
                reward_per_winner = self.pot / len(winners)
                for w in winners:
                    if w == player_idx:
                        reward = reward_per_winner - self.starting_stack
        
        return self.get_state(), reward, done
    
    def _advance_game(self):
        """Advance to next player or street."""
        # Find next active player
        start_pos = self.current_player
        
        for _ in range(self.num_players):
            self.current_player = (self.current_player + 1) % self.num_players
            player = self.players[self.current_player]
            
            if not player.folded and not player.all_in:
                break
        
        # Check if betting round is complete
        if self._is_betting_complete():
            self._advance_street()
    
    def _is_betting_complete(self) -> bool:
        """Check if current betting round is complete."""
        active_players = [p for p in self.players if not p.folded and not p.all_in]
        
        if len(active_players) <= 1:
            return True
        
        # All active players must have equal bets
        bets = set(p.current_bet for p in active_players)
        if len(bets) > 1:
            return False
        
        # Each player must have acted at least once this street
        if not self.action_history:
            return False
        
        # Count actions this street
        street_actions = self._get_street_actions()
        acting_players = set(a[0] for a in street_actions if not self.players[a[0]].folded)
        
        return len(acting_players) >= len(active_players)
    
    def _get_street_actions(self) -> List[Tuple[int, ActionType, int]]:
        """Get actions for current street."""
        # Find where current street started
        street_start = 0
        for i, (_, action, _) in enumerate(self.action_history):
            if action == ActionType.CHECK or action == ActionType.CALL:
                if i > 0:
                    prev_action = self.action_history[i-1][1]
                    if prev_action in [ActionType.CALL, ActionType.CHECK]:
                        # Possible street boundary
                        pass
        return self.action_history  # Simplified
    
    def _advance_street(self):
        """Advance to next street."""
        # Reset current bets
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0
        
        # Deal community cards
        if self.street == BettingRound.PREFLOP:
            # Deal flop
            for _ in range(3):
                self.community_cards.append(self.deck.pop())
            self.street = BettingRound.FLOP
        
        elif self.street == BettingRound.FLOP:
            # Deal turn
            self.community_cards.append(self.deck.pop())
            self.street = BettingRound.TURN
        
        elif self.street == BettingRound.TURN:
            # Deal river
            self.community_cards.append(self.deck.pop())
            self.street = BettingRound.RIVER
        
        # Set current player (first active player after dealer)
        for i in range(self.num_players):
            pos = (self.dealer_button + 1 + i) % self.num_players
            if not self.players[pos].folded and not self.players[pos].all_in:
                self.current_player = pos
                break
    
    def is_terminal(self) -> bool:
        """Check if hand is complete."""
        active_players = [p for p in self.players if not p.folded]
        
        # Only one player remains
        if len(active_players) <= 1:
            return True
        
        # All remaining players all-in
        not_all_in = [p for p in active_players if not p.all_in]
        if len(not_all_in) == 0:
            return True
        
        # River betting complete
        if self.street == BettingRound.RIVER and self._is_betting_complete():
            return True
        
        return False
    
    def get_legal_actions(self) -> List[Tuple[ActionType, int]]:
        """
        Get list of legal actions for current player.
        
        Returns:
            List of (action_type, amount) tuples
        """
        player = self.players[self.current_player]
        legal = []
        
        if player.folded or player.all_in:
            return legal
        
        # Fold is always legal (unless all-in)
        legal.append((ActionType.FOLD, 0))
        
        # Check if can check
        if player.current_bet >= self.current_bet:
            legal.append((ActionType.CHECK, 0))
        else:
            # Must call or raise
            call_amount = self.current_bet - player.current_bet
            if call_amount <= player.stack:
                legal.append((ActionType.CALL, 0))
        
        # Can raise if has chips
        if player.stack > 0:
            min_raise = self.current_bet + self.big_blind
            if player.stack >= min_raise - player.current_bet:
                # Add various raise sizes
                for multiplier in [0.5, 0.75, 1.0, 1.5, 2.0]:
                    raise_amount = int(self.pot * multiplier)
                    total_bet = self.current_bet + raise_amount
                    if total_bet > player.current_bet and total_bet - player.current_bet <= player.stack:
                        legal.append((ActionType.RAISE, total_bet))
            
            # All-in is always an option
            legal.append((ActionType.ALL_IN, player.stack + player.current_bet))
        
        return legal
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current game state as dictionary.
        
        Returns:
            State dictionary
        """
        return {
            'street': self.street.value,
            'pot': self.pot,
            'current_bet': self.current_bet,
            'current_player': self.current_player,
            'community_cards': [c.to_string() for c in self.community_cards],
            'players': [
                {
                    'stack': p.stack,
                    'current_bet': p.current_bet,
                    'folded': p.folded,
                    'all_in': p.all_in,
                    'hole_cards': [c.to_string() for c in p.hole_cards]
                }
                for p in self.players
            ],
            'action_history': [
                (idx, action.value, amount)
                for idx, action, amount in self.action_history
            ]
        }
    
    def get_state_vector(self, player_idx: int = 0) -> np.ndarray:
        """
        Get state as numeric vector for neural network.
        
        Args:
            player_idx: Player perspective
            
        Returns:
            Normalized state vector
        """
        vector = []
        
        # Player's hole cards (one-hot encoding)
        player = self.players[player_idx]
        for i in range(2):
            card_vec = [0.0] * 52
            if i < len(player.hole_cards):
                card_vec[player.hole_cards[i].to_index()] = 1.0
            vector.extend(card_vec)
        
        # Community cards (one-hot encoding)
        for i in range(5):
            card_vec = [0.0] * 52
            if i < len(self.community_cards):
                card_vec[self.community_cards[i].to_index()] = 1.0
            vector.extend(card_vec)
        
        # Numeric features (normalized)
        max_stack = self.starting_stack * 2
        vector.append(self.pot / max_stack)
        vector.append(self.current_bet / max_stack)
        
        for p in self.players:
            vector.append(p.stack / max_stack)
            vector.append(p.current_bet / max_stack)
            vector.append(float(p.folded))
            vector.append(float(p.all_in))
        
        # Street (one-hot)
        street_vec = [0.0] * 4
        street_idx = list(BettingRound).index(self.street)
        street_vec[street_idx] = 1.0
        vector.extend(street_vec)
        
        return np.array(vector, dtype=np.float32)
    
    def _determine_winners(self) -> List[int]:
        """Determine winner(s) at showdown."""
        active_players = [
            (i, p) for i, p in enumerate(self.players) if not p.folded
        ]
        
        if len(active_players) == 1:
            return [active_players[0][0]]
        
        # Evaluate hands
        best_rank = -1
        best_hand = None
        winners = []
        
        for idx, player in active_players:
            all_cards = player.hole_cards + self.community_cards
            rank, hand = self._evaluate_hand(all_cards)
            
            if rank > best_rank or (rank == best_rank and hand > best_hand):
                best_rank = rank
                best_hand = hand
                winners = [idx]
            elif rank == best_rank and hand == best_hand:
                winners.append(idx)
        
        return winners
    
    def _evaluate_hand(self, cards: List[Card]) -> Tuple[int, Tuple]:
        """
        Evaluate poker hand strength.
        
        Args:
            cards: List of 5-7 cards
            
        Returns:
            Tuple of (hand_rank, tiebreaker_tuple)
        """
        if len(cards) < 5:
            return (0, (0,))
        
        # Get all 5-card combinations
        from itertools import combinations
        
        best_rank = -1
        best_hand = (0,)
        
        for combo in combinations(cards, 5):
            rank, hand = self._evaluate_5_cards(list(combo))
            if rank > best_rank or (rank == best_rank and hand > best_hand):
                best_rank = rank
                best_hand = hand
        
        return best_rank, best_hand
    
    def _evaluate_5_cards(self, cards: List[Card]) -> Tuple[int, Tuple]:
        """Evaluate exactly 5 cards."""
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        is_straight = False
        straight_high = 0
        
        unique_ranks = sorted(set(ranks), reverse=True)
        if len(unique_ranks) >= 5:
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i] - unique_ranks[i+4] == 4:
                    is_straight = True
                    straight_high = unique_ranks[i]
                    break
        
        # Check for A-2-3-4-5 straight (wheel)
        if set(unique_ranks) >= {14, 2, 3, 4, 5}:
            is_straight = True
            straight_high = 5
        
        # Count rank frequencies
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        
        # Determine hand rank
        if is_straight and is_flush:
            return (HandRank.STRAIGHT_FLUSH, (straight_high,))
        
        if counts[0] == 4:
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r in ranks if r != quad_rank][0]
            return (HandRank.FOUR_OF_KIND, (quad_rank, kicker))
        
        if counts[0] == 3 and counts[1] == 2:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return (HandRank.FULL_HOUSE, (trip_rank, pair_rank))
        
        if is_flush:
            return (HandRank.FLUSH, tuple(ranks))
        
        if is_straight:
            return (HandRank.STRAIGHT, (straight_high,))
        
        if counts[0] == 3:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)[:2]
            return (HandRank.THREE_OF_KIND, (trip_rank,) + tuple(kickers))
        
        if counts[0] == 2 and counts[1] == 2:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r in ranks if rank_counts[r] == 1][0]
            return (HandRank.TWO_PAIR, tuple(pairs) + (kicker,))
        
        if counts[0] == 2:
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)[:3]
            return (HandRank.ONE_PAIR, (pair_rank,) + tuple(kickers))
        
        return (HandRank.HIGH_CARD, tuple(ranks))
    
    def get_winner(self) -> Optional[int]:
        """Get winner of current hand (if terminal)."""
        if not self.is_terminal():
            return None
        
        winners = self._determine_winners()
        return winners[0] if winners else None
    
    def clone(self) -> 'PokerEnvironment':
        """Create a deep copy of the environment."""
        import copy
        return copy.deepcopy(self)


__all__ = ['PokerEnvironment', 'PokerPlayer', 'HandRank']
