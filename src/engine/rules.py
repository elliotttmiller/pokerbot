"""
Poker Rules Engine - handles game flow and action validation
"""
import random
from typing import List, Optional
from .game_state import GameState, PlayerState, Action, ActionType, Street, Card
from .hand_evaluator import HandEvaluator


class PokerRules:
    """Enforces No-Limit Texas Hold'em rules"""
    
    @staticmethod
    def deal_hole_cards(state: GameState) -> None:
        """Deal 2 cards to each active player"""
        random.shuffle(state.deck)
        
        for player in state.get_active_players():
            player.hole_cards = [state.deck.pop(), state.deck.pop()]
    
    @staticmethod
    def deal_community_cards(state: GameState, num_cards: int) -> None:
        """Deal community cards (3 for flop, 1 for turn/river)"""
        for _ in range(num_cards):
            if state.deck:
                state.community_cards.append(state.deck.pop())
    
    @staticmethod
    def post_blinds(state: GameState) -> None:
        """Post small and big blinds"""
        num_active = len(state.get_active_players())
        if num_active < 2:
            return
        
        # Small blind position (button + 1 in heads-up, button + 1 in multi-way)
        sb_pos = (state.button_position + 1) % state.num_players
        bb_pos = (state.button_position + 2) % state.num_players
        
        # In heads-up, button is small blind
        if num_active == 2:
            sb_pos = state.button_position
            bb_pos = (state.button_position + 1) % state.num_players
        
        # Post small blind
        sb_player = state.players[sb_pos]
        sb_amount = min(state.small_blind, sb_player.stack)
        sb_player.current_bet = sb_amount
        sb_player.stack -= sb_amount
        state.pot += sb_amount
        
        # Post big blind
        bb_player = state.players[bb_pos]
        bb_amount = min(state.big_blind, bb_player.stack)
        bb_player.current_bet = bb_amount
        bb_player.stack -= bb_amount
        state.pot += bb_amount
        
        state.current_bet = bb_amount
        state.min_raise = state.big_blind
    
    @staticmethod
    def apply_action(state: GameState, action: Action) -> bool:
        """
        Apply an action to the game state
        Returns True if action was valid and applied
        """
        player = state.get_current_player_state()
        
        if action.action_type == ActionType.FOLD:
            player.is_active = False
            state.betting_history.append((state.current_player, action))
            return True
        
        elif action.action_type == ActionType.CHECK:
            if state.current_bet != player.current_bet:
                return False  # Can't check when there's a bet
            state.betting_history.append((state.current_player, action))
            return True
        
        elif action.action_type == ActionType.CALL:
            call_amount = min(state.current_bet - player.current_bet, player.stack)
            if call_amount <= 0:
                return False
            
            player.stack -= call_amount
            player.current_bet += call_amount
            player.total_invested += call_amount
            state.pot += call_amount
            
            if player.stack == 0:
                player.is_all_in = True
            
            state.betting_history.append((state.current_player, action))
            return True
        
        elif action.action_type == ActionType.BET:
            if state.current_bet != 0:
                return False  # Can't bet when there's already a bet
            
            bet_amount = min(action.amount, player.stack)
            if bet_amount < state.big_blind:
                return False  # Bet too small
            
            player.stack -= bet_amount
            player.current_bet = bet_amount
            player.total_invested += bet_amount
            state.pot += bet_amount
            state.current_bet = bet_amount
            state.min_raise = bet_amount
            
            if player.stack == 0:
                player.is_all_in = True
            
            state.betting_history.append((state.current_player, action))
            return True
        
        elif action.action_type == ActionType.RAISE:
            if state.current_bet == 0:
                return False  # Can't raise when there's no bet
            
            raise_amount = min(action.amount, player.stack)
            new_bet = player.current_bet + raise_amount
            
            if new_bet < state.current_bet + state.min_raise:
                return False  # Raise too small
            
            player.stack -= raise_amount
            player.current_bet = new_bet
            player.total_invested += raise_amount
            state.pot += raise_amount
            
            # Update min raise for next player
            raise_size = new_bet - state.current_bet
            state.min_raise = raise_size
            state.current_bet = new_bet
            
            if player.stack == 0:
                player.is_all_in = True
            
            state.betting_history.append((state.current_player, action))
            return True
        
        elif action.action_type == ActionType.ALL_IN:
            all_in_amount = player.stack
            if all_in_amount <= 0:
                return False
            
            player.stack = 0
            player.current_bet += all_in_amount
            player.total_invested += all_in_amount
            state.pot += all_in_amount
            player.is_all_in = True
            
            # Update current bet if all-in is a raise
            if player.current_bet > state.current_bet:
                state.current_bet = player.current_bet
            
            state.betting_history.append((state.current_player, action))
            return True
        
        return False
    
    @staticmethod
    def advance_to_next_player(state: GameState) -> None:
        """Move to the next active player"""
        start_pos = state.current_player
        
        while True:
            state.current_player = (state.current_player + 1) % state.num_players
            player = state.players[state.current_player]
            
            if player.is_active and not player.is_all_in:
                break
            
            # If we've cycled through all players
            if state.current_player == start_pos:
                break
    
    @staticmethod
    def is_betting_round_complete(state: GameState) -> bool:
        """Check if current betting round is complete"""
        active_players = [p for p in state.players if p.is_active and not p.is_all_in]
        
        if len(active_players) <= 1:
            return True
        
        # All active players have matched the current bet
        return all(p.current_bet == state.current_bet for p in active_players)
    
    @staticmethod
    def advance_street(state: GameState) -> None:
        """Move to the next betting street"""
        # Reset bets for new street
        for player in state.players:
            player.current_bet = 0.0
        
        state.current_bet = 0.0
        state.min_raise = state.big_blind
        
        # Deal community cards
        if state.street == Street.PREFLOP:
            PokerRules.deal_community_cards(state, 3)  # Flop
            state.street = Street.FLOP
        elif state.street == Street.FLOP:
            PokerRules.deal_community_cards(state, 1)  # Turn
            state.street = Street.TURN
        elif state.street == Street.TURN:
            PokerRules.deal_community_cards(state, 1)  # River
            state.street = Street.RIVER
        
        # First to act is left of button
        state.current_player = (state.button_position + 1) % state.num_players
        
        # Advance to first active player
        while not state.players[state.current_player].is_active or \
              state.players[state.current_player].is_all_in:
            state.current_player = (state.current_player + 1) % state.num_players
    
    @staticmethod
    def determine_winners(state: GameState) -> List[int]:
        """
        Determine winners and award pot
        Returns list of winning player IDs
        """
        active_players = [p for p in state.players if p.is_active]
        
        if len(active_players) == 1:
            return [active_players[0].player_id]
        
        # Evaluate all hands
        best_hand = None
        winners = []
        
        for player in active_players:
            if not player.hole_cards or len(state.community_cards) < 5:
                continue
            
            hand_rank, tiebreakers, _ = HandEvaluator.best_hand(
                player.hole_cards, state.community_cards
            )
            
            if best_hand is None:
                best_hand = (hand_rank, tiebreakers)
                winners = [player.player_id]
            else:
                comparison = HandEvaluator._compare_hands(
                    (hand_rank, tiebreakers), best_hand
                )
                if comparison > 0:
                    best_hand = (hand_rank, tiebreakers)
                    winners = [player.player_id]
                elif comparison == 0:
                    winners.append(player.player_id)
        
        return winners
    
    @staticmethod
    def is_hand_complete(state: GameState) -> bool:
        """Check if the hand is over"""
        active_count = len(state.get_active_players())
        
        # Only one player left
        if active_count <= 1:
            return True
        
        # Reached river and betting is complete
        if state.street == Street.RIVER and PokerRules.is_betting_round_complete(state):
            return True
        
        return False
