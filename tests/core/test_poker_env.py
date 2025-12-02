"""
Tests for Core Data Models and Poker Environment.

Tests the fundamental data structures and game simulation.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from poker_bot.core.data_models import (
    Card, GameState, Action, ActionType, BettingRound, PlayerState
)
from poker_bot.core.poker_env import PokerEnvironment, HandRank


class TestCard:
    """Test Card class."""
    
    def test_from_string_basic(self):
        """Test parsing basic card strings."""
        card = Card.from_string('As')
        assert card.rank == 14
        assert card.suit == 3  # spades
        
        card = Card.from_string('2c')
        assert card.rank == 2
        assert card.suit == 0  # clubs
    
    def test_from_string_ten(self):
        """Test parsing ten notation."""
        card = Card.from_string('Th')
        assert card.rank == 10
        
        card = Card.from_string('10d')
        assert card.rank == 10
    
    def test_to_string(self):
        """Test card to string conversion."""
        card = Card(rank=14, suit=3)
        assert card.to_string() == 'As'
        
        card = Card(rank=10, suit=2)
        assert card.to_string() == 'Th'
    
    def test_to_index(self):
        """Test card index conversion."""
        # Ace of spades should be last card
        card = Card.from_string('As')
        assert card.to_index() == 51  # (14-2)*4 + 3 = 51
        
        # Two of clubs should be first
        card = Card.from_string('2c')
        assert card.to_index() == 0
    
    def test_from_index(self):
        """Test creating card from index."""
        card = Card.from_index(0)
        assert card.to_string() == '2c'
        
        card = Card.from_index(51)
        assert card.to_string() == 'As'
    
    def test_invalid_rank(self):
        """Test invalid rank raises error."""
        with pytest.raises(ValueError):
            Card.from_string('Xs')
    
    def test_invalid_suit(self):
        """Test invalid suit raises error."""
        with pytest.raises(ValueError):
            Card.from_string('Ax')
    
    def test_equality(self):
        """Test card equality."""
        card1 = Card.from_string('As')
        card2 = Card.from_string('As')
        assert card1 == card2
        
        card3 = Card.from_string('Ks')
        assert card1 != card3


class TestGameState:
    """Test GameState class."""
    
    def test_basic_creation(self):
        """Test creating basic game state."""
        state = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[],
            pot_size=100,
            current_bet=20,
            street=BettingRound.PREFLOP
        )
        
        assert len(state.hole_cards) == 2
        assert state.pot_size == 100
        assert state.street == BettingRound.PREFLOP
    
    def test_community_cards_validation(self):
        """Test community cards validation (0, 3, 4, or 5 cards)."""
        # Valid: 0 cards (preflop)
        state = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[]
        )
        assert len(state.community_cards) == 0
        
        # Valid: 3 cards (flop)
        state = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[
                Card.from_string('Qh'),
                Card.from_string('Jd'),
                Card.from_string('Tc')
            ]
        )
        assert len(state.community_cards) == 3
    
    def test_get_state_vector(self):
        """Test state vector generation."""
        state = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[],
            pot_size=100,
            current_bet=20,
            street=BettingRound.PREFLOP
        )
        
        vector = state.get_state_vector()
        
        # Should have: 2*52 (hole) + 5*52 (community) + 6 (numeric) + 4 (street)
        expected_length = 104 + 260 + 6 + 4
        assert len(vector) == expected_length


class TestAction:
    """Test Action class."""
    
    def test_fold_action(self):
        """Test fold action."""
        action = Action(action_type=ActionType.FOLD, amount=0)
        assert action.to_command() == "fold"
        assert str(action) == "FOLD"
    
    def test_raise_action(self):
        """Test raise action."""
        action = Action(action_type=ActionType.RAISE, amount=100)
        assert action.to_command() == "raise 100"
        assert str(action) == "RAISE 100"
    
    def test_call_action(self):
        """Test call action."""
        action = Action(action_type=ActionType.CALL, amount=50)
        # Amount should be 0 for call
        assert action.amount == 0
        assert action.to_command() == "call"


class TestPokerEnvironment:
    """Test PokerEnvironment class."""
    
    def test_initialization(self):
        """Test environment initialization."""
        env = PokerEnvironment(
            num_players=2,
            starting_stack=1000,
            small_blind=10,
            big_blind=20
        )
        
        assert env.num_players == 2
        assert env.starting_stack == 1000
        assert len(env.players) == 2
    
    def test_reset(self):
        """Test environment reset."""
        env = PokerEnvironment()
        state = env.reset(seed=42)
        
        # Players should have hole cards
        for player in env.players:
            assert len(player.hole_cards) == 2
        
        # Pot should have blinds
        assert env.pot > 0
        
        # Should be preflop
        assert env.street == BettingRound.PREFLOP
    
    def test_get_legal_actions(self):
        """Test legal action generation."""
        env = PokerEnvironment()
        env.reset()
        
        actions = env.get_legal_actions()
        
        # Should have fold and some other actions
        action_types = [a[0] for a in actions]
        assert ActionType.FOLD in action_types
    
    def test_step_fold(self):
        """Test fold action."""
        env = PokerEnvironment()
        env.reset()
        
        _, _, done = env.step(ActionType.FOLD, 0)
        
        # Game should be over after fold in heads-up
        assert done or env.players[env.current_player].folded
    
    def test_is_terminal(self):
        """Test terminal state detection."""
        env = PokerEnvironment()
        env.reset()
        
        # Initially not terminal
        assert not env.is_terminal()
        
        # After fold, should be terminal
        env.step(ActionType.FOLD, 0)
        assert env.is_terminal()
    
    def test_hand_evaluation(self):
        """Test hand evaluation."""
        env = PokerEnvironment()
        
        # Create a flush
        cards = [
            Card.from_string('As'),
            Card.from_string('Ks'),
            Card.from_string('Qs'),
            Card.from_string('Js'),
            Card.from_string('9s')
        ]
        
        rank, _ = env._evaluate_hand(cards)
        assert rank == HandRank.FLUSH
    
    def test_get_state_vector(self):
        """Test state vector generation."""
        env = PokerEnvironment()
        env.reset()
        
        vector = env.get_state_vector(player_idx=0)
        
        # Should be a numpy array
        assert hasattr(vector, 'shape')
        assert len(vector) > 0


class TestPlayerState:
    """Test PlayerState class."""
    
    def test_default_values(self):
        """Test default values."""
        player = PlayerState()
        
        assert player.stack == 1000
        assert player.current_bet == 0
        assert player.is_active == True
        assert player.is_all_in == False
    
    def test_custom_values(self):
        """Test custom values."""
        player = PlayerState(
            stack=500,
            current_bet=100,
            is_all_in=True,
            position='button'
        )
        
        assert player.stack == 500
        assert player.current_bet == 100
        assert player.is_all_in == True
        assert player.position == 'button'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
