"""
Tests for Multi-Modal Vision Integration.

Tests the QWEN 2.5-7B VL detector and workflow orchestrator components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import os


class TestDetectedCard:
    """Test DetectedCard class."""
    
    def test_from_string_valid(self):
        """Test parsing valid card strings."""
        from src.vision.multimodal_detector import DetectedCard
        
        # Test all ranks
        for rank in '23456789TJQKA':
            card = DetectedCard.from_string(f'{rank}s')
            assert card.rank == rank
            assert card.suit == 's'
        
        # Test all suits
        for suit in 'shdc':
            card = DetectedCard.from_string(f'A{suit}')
            assert card.rank == 'A'
            assert card.suit == suit
    
    def test_from_string_ten(self):
        """Test parsing 10 notation."""
        from src.vision.multimodal_detector import DetectedCard
        
        card = DetectedCard.from_string('10s')
        assert card.rank == 'T'
        assert card.suit == 's'
    
    def test_from_string_invalid_rank(self):
        """Test invalid rank raises error."""
        from src.vision.multimodal_detector import DetectedCard
        
        with pytest.raises(ValueError):
            DetectedCard.from_string('Xs')
    
    def test_from_string_invalid_suit(self):
        """Test invalid suit raises error."""
        from src.vision.multimodal_detector import DetectedCard
        
        with pytest.raises(ValueError):
            DetectedCard.from_string('Ax')
    
    def test_to_string(self):
        """Test card to string conversion."""
        from src.vision.multimodal_detector import DetectedCard
        
        card = DetectedCard(rank='A', suit='s')
        assert card.to_string() == 'As'
        assert str(card) == 'As'


class TestDetectedGameState:
    """Test DetectedGameState class."""
    
    def test_from_dict(self):
        """Test creating state from dictionary."""
        from src.vision.multimodal_detector import DetectedGameState
        
        data = {
            'hole_cards': ['As', 'Kh'],
            'community_cards': ['Qd', 'Jc', 'Ts'],
            'pot_size': 150,
            'current_bet': 50,
            'player_stack': 1000,
            'opponent_stack': 1200,
            'street': 'flop',
            'action_required': True,
            'available_actions': ['fold', 'call', 'raise']
        }
        
        state = DetectedGameState.from_dict(data)
        
        assert len(state.hole_cards) == 2
        assert state.hole_cards[0].to_string() == 'As'
        assert state.hole_cards[1].to_string() == 'Kh'
        assert len(state.community_cards) == 3
        assert state.pot_size == 150
        assert state.current_bet == 50
        assert state.street == 'flop'
        assert state.action_required == True
    
    def test_to_dict(self):
        """Test serializing state to dictionary."""
        from src.vision.multimodal_detector import DetectedGameState, DetectedCard
        
        state = DetectedGameState(
            hole_cards=[DetectedCard('A', 's'), DetectedCard('K', 'h')],
            community_cards=[],
            pot_size=100,
            current_bet=20,
            player_stack=1000,
            opponent_stack=1000,
            street='preflop',
            action_required=True,
            available_actions=['fold', 'call', 'raise']
        )
        
        data = state.to_dict()
        
        assert data['hole_cards'] == ['As', 'Kh']
        assert data['community_cards'] == []
        assert data['pot_size'] == 100
        assert data['street'] == 'preflop'


class TestMultiModalVisionDetector:
    """Test MultiModalVisionDetector class."""
    
    def test_initialization(self):
        """Test detector initialization without model."""
        from src.vision.multimodal_detector import MultiModalVisionDetector
        
        detector = MultiModalVisionDetector(model_path=None, api_fallback=False)
        assert detector.model is None
        assert detector._model_loaded == False
    
    def test_mock_state_returned(self):
        """Test mock state is returned when no detection available."""
        from src.vision.multimodal_detector import MultiModalVisionDetector
        
        detector = MultiModalVisionDetector(model_path=None, api_fallback=False)
        state = detector.detect("nonexistent.png")
        
        # Should return mock state
        assert state.confidence == 0.0
        assert len(state.hole_cards) == 2
        assert state.hole_cards[0].to_string() == 'As'
    
    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        from src.vision.multimodal_detector import MultiModalVisionDetector
        
        detector = MultiModalVisionDetector(model_path=None)
        
        response = json.dumps({
            'hole_cards': ['Ah', 'Kd'],
            'community_cards': ['Qc', 'Js', '9h'],
            'pot_size': 200,
            'current_bet': 100,
            'player_stack': 800,
            'opponent_stack': 900,
            'street': 'flop',
            'action_required': True,
            'available_actions': ['fold', 'call', 'raise']
        })
        
        state = detector._parse_response(response)
        
        assert state.hole_cards[0].to_string() == 'Ah'
        assert state.pot_size == 200
        assert state.street == 'flop'
    
    def test_validate_state_duplicate_cards(self):
        """Test validation catches duplicate cards."""
        from src.vision.multimodal_detector import MultiModalVisionDetector, DetectedGameState, DetectedCard
        
        detector = MultiModalVisionDetector(model_path=None)
        
        # Create state with duplicate card
        state = DetectedGameState(
            hole_cards=[DetectedCard('A', 's'), DetectedCard('A', 's')],  # Duplicate!
            community_cards=[],
            pot_size=100,
            current_bet=20,
            player_stack=1000,
            opponent_stack=1000,
            street='preflop',
            action_required=True,
            available_actions=['fold', 'call', 'raise']
        )
        
        validated = detector._validate_state(state)
        
        # Confidence should be reduced
        assert validated.confidence < 1.0


class TestGameContext:
    """Test GameContext class."""
    
    def test_initialization(self):
        """Test context initialization."""
        from src.workflow.orchestrator import GameContext
        
        context = GameContext()
        
        assert context.hand_id == ""
        assert context.current_street == "preflop"
        assert context.player_actions == []
        assert context.opponent_actions == []
    
    def test_to_dict(self):
        """Test serialization."""
        from src.workflow.orchestrator import GameContext
        
        context = GameContext(
            hand_id="test_123",
            hole_cards=['As', 'Kh'],
            position='button'
        )
        
        data = context.to_dict()
        
        assert data['hand_id'] == "test_123"
        assert data['hole_cards'] == ['As', 'Kh']
        assert data['position'] == 'button'


class TestPokerWorkflowOrchestrator:
    """Test PokerWorkflowOrchestrator class."""
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        from src.workflow.orchestrator import PokerWorkflowOrchestrator
        
        orchestrator = PokerWorkflowOrchestrator(
            vision_model_path=None,
            strategy_model_path=None,
            enable_logging=False
        )
        
        assert orchestrator.context is None
        assert orchestrator.session_stats['hands_played'] == 0
    
    def test_start_new_hand(self):
        """Test starting new hand."""
        from src.workflow.orchestrator import PokerWorkflowOrchestrator
        
        orchestrator = PokerWorkflowOrchestrator(enable_logging=False)
        context = orchestrator.start_new_hand(position='button')
        
        assert context is not None
        assert context.position == 'button'
        assert context.current_street == 'preflop'
        assert orchestrator.session_stats['hands_played'] == 1
    
    def test_record_action(self):
        """Test recording actions."""
        from src.workflow.orchestrator import PokerWorkflowOrchestrator
        
        orchestrator = PokerWorkflowOrchestrator(enable_logging=False)
        orchestrator.start_new_hand()
        
        # Record player action
        orchestrator.record_action('raise', 50, is_opponent=False)
        assert len(orchestrator.context.player_actions) == 1
        assert orchestrator.context.player_actions[0]['action'] == 'raise'
        
        # Record opponent action
        orchestrator.record_action('call', 50, is_opponent=True)
        assert len(orchestrator.context.opponent_actions) == 1
    
    def test_get_session_stats(self):
        """Test session statistics."""
        from src.workflow.orchestrator import PokerWorkflowOrchestrator
        
        orchestrator = PokerWorkflowOrchestrator(enable_logging=False)
        orchestrator.start_new_hand()
        orchestrator.start_new_hand()
        
        stats = orchestrator.get_session_stats()
        
        assert stats['hands_played'] == 2
        assert 'session_duration' in stats


class TestDecisionResult:
    """Test DecisionResult class."""
    
    def test_creation(self):
        """Test creating decision result."""
        from src.workflow.orchestrator import DecisionResult
        
        result = DecisionResult(
            action='raise',
            amount=100,
            confidence=0.9,
            reasoning="Strong hand, building pot"
        )
        
        assert result.action == 'raise'
        assert result.amount == 100
        assert result.confidence == 0.9


# Integration tests
class TestIntegration:
    """Integration tests for the workflow."""
    
    def test_full_workflow_mock(self):
        """Test complete workflow with mocked detection."""
        from src.workflow.orchestrator import PokerWorkflowOrchestrator
        
        orchestrator = PokerWorkflowOrchestrator(
            vision_model_path=None,
            strategy_model_path=None,
            enable_logging=False
        )
        
        # Start hand
        orchestrator.start_new_hand(position='button')
        
        # Process would normally use a screenshot
        # For testing, verify the system is ready
        assert orchestrator.context is not None
        assert orchestrator.context.position == 'button'
        
        # Record some actions
        orchestrator.record_action('raise', 30, is_opponent=False)
        orchestrator.record_action('call', 30, is_opponent=True)
        
        # End hand
        orchestrator.end_hand('won', 60)
        
        assert orchestrator.context is None
        assert orchestrator.session_stats['total_winnings'] == 60


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
