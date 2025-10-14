"""Game state management for poker."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .card import Card, Deck


class Action(Enum):
    """Possible player actions."""
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    ALL_IN = 4


class BettingRound(Enum):
    """Betting rounds in Texas Hold'em."""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass
class Player:
    """Represents a player in the game."""
    name: str
    stack: int
    hand: List[Card]
    current_bet: int = 0
    folded: bool = False
    all_in: bool = False
    
    def __init__(self, name: str, stack: int):
        """Initialize a player."""
        self.name = name
        self.stack = stack
        self.hand = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False


class GameState:
    """Manages the state of a poker game."""
    
    def __init__(self, num_players: int = 2, starting_stack: int = 1000,
                 small_blind: int = 10, big_blind: int = 20):
        """
        Initialize game state.
        
        Args:
            num_players: Number of players
            starting_stack: Starting chip stack for each player
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        self.players: List[Player] = []
        self.deck = Deck()
        self.community_cards: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.betting_round = BettingRound.PREFLOP
        self.dealer_button = 0
        
        self._initialize_players()
    
    def _initialize_players(self):
        """Initialize players with starting stacks."""
        self.players = [Player(f"Player{i+1}", self.starting_stack) 
                       for i in range(self.num_players)]
    
    def reset(self):
        """Reset game for a new hand."""
        self.deck.reset()
        self.deck.shuffle()
        
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.betting_round = BettingRound.PREFLOP
        
        # Reset players
        for player in self.players:
            player.hand = []
            player.current_bet = 0
            player.folded = False
            player.all_in = False
        
        # Deal hole cards
        for _ in range(2):
            for player in self.players:
                if player.stack > 0:
                    player.hand.extend(self.deck.deal(1))
        
        # Post blinds
        self._post_blinds()
    
    def _post_blinds(self):
        """Post small and big blinds."""
        # Determine small blind and big blind positions
        sb_position = (self.dealer_button + 1) % self.num_players
        bb_position = (self.dealer_button + 2) % self.num_players
        
        # Small blind
        sb_player = self.players[sb_position]
        sb_amount = min(self.small_blind, sb_player.stack)
        sb_player.stack -= sb_amount
        sb_player.current_bet = sb_amount
        self.pot += sb_amount
        
        # Big blind
        bb_player = self.players[bb_position]
        bb_amount = min(self.big_blind, bb_player.stack)
        bb_player.stack -= bb_amount
        bb_player.current_bet = bb_amount
        self.pot += bb_amount
        
        self.current_bet = bb_amount
    
    def apply_action(self, player_idx: int, action: Action, raise_amount: int = 0) -> bool:
        """
        Apply a player action to the game state.
        
        Args:
            player_idx: Index of the player
            action: Action to take
            raise_amount: Amount to raise (if applicable)
        
        Returns:
            True if action was valid and applied
        """
        player = self.players[player_idx]
        
        if player.folded or player.all_in:
            return False
        
        if action == Action.FOLD:
            player.folded = True
            return True
        
        elif action == Action.CHECK:
            if player.current_bet < self.current_bet:
                return False  # Cannot check if there's a bet to call
            return True
        
        elif action == Action.CALL:
            call_amount = self.current_bet - player.current_bet
            actual_call = min(call_amount, player.stack)
            player.stack -= actual_call
            player.current_bet += actual_call
            self.pot += actual_call
            
            if player.stack == 0:
                player.all_in = True
            return True
        
        elif action == Action.RAISE:
            total_bet = self.current_bet + raise_amount
            to_call = self.current_bet - player.current_bet
            total_needed = to_call + raise_amount
            
            actual_raise = min(total_needed, player.stack)
            player.stack -= actual_raise
            player.current_bet += actual_raise
            self.pot += actual_raise
            
            self.current_bet = player.current_bet
            
            if player.stack == 0:
                player.all_in = True
            return True
        
        return False
    
    def advance_betting_round(self):
        """Advance to the next betting round."""
        # Reset current bets
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0
        
        if self.betting_round == BettingRound.PREFLOP:
            # Deal flop (3 cards)
            self.community_cards.extend(self.deck.deal(3))
            self.betting_round = BettingRound.FLOP
        
        elif self.betting_round == BettingRound.FLOP:
            # Deal turn (1 card)
            self.community_cards.extend(self.deck.deal(1))
            self.betting_round = BettingRound.TURN
        
        elif self.betting_round == BettingRound.TURN:
            # Deal river (1 card)
            self.community_cards.extend(self.deck.deal(1))
            self.betting_round = BettingRound.RIVER
    
    def is_hand_complete(self) -> bool:
        """Check if the hand is complete."""
        active_players = [p for p in self.players if not p.folded]
        
        # Hand complete if only one player left
        if len(active_players) <= 1:
            return True
        
        # Hand complete if we're at river and betting is done
        if self.betting_round == BettingRound.RIVER:
            # Check if all active players have equal bets or are all-in
            active_bets = [p.current_bet for p in active_players if not p.all_in]
            if len(set(active_bets)) <= 1:
                return True
        
        return False
    
    def get_winners(self) -> List[int]:
        """
        Determine the winner(s) of the hand.
        
        Returns:
            List of player indices who won
        """
        from .hand_evaluator import HandEvaluator
        
        active_players = [(i, p) for i, p in enumerate(self.players) if not p.folded]
        
        if len(active_players) == 1:
            return [active_players[0][0]]
        
        # Evaluate all hands
        best_players = []
        best_rank = None
        best_tiebreakers = None
        
        for idx, player in active_players:
            all_cards = player.hand + self.community_cards
            rank, tiebreakers = HandEvaluator.evaluate_hand(all_cards)
            
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_tiebreakers = tiebreakers
                best_players = [idx]
            elif rank == best_rank and tiebreakers == best_tiebreakers:
                best_players.append(idx)
        
        return best_players
    
    def get_state_representation(self) -> dict:
        """
        Get a dictionary representation of the game state.
        
        Returns:
            Dictionary with game state information
        """
        return {
            'pot': self.pot,
            'current_bet': self.current_bet,
            'betting_round': self.betting_round.name,
            'community_cards': [str(card) for card in self.community_cards],
            'players': [
                {
                    'name': p.name,
                    'stack': p.stack,
                    'current_bet': p.current_bet,
                    'folded': p.folded,
                    'all_in': p.all_in,
                    'hand': [str(card) for card in p.hand]
                }
                for p in self.players
            ]
        }
