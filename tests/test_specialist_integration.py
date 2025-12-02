"""
Integration Tests for Poker Bot Specialist System.

Tests the flow between modules: Perception → Decision → Action
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from poker_bot.core.data_models import (
    GameState, Card, Action, ActionType, PlayerState, BettingRound
)
from poker_bot.perception.vlm_service import VLMService
from poker_bot.decision.cfr_solver import CFRSolver
from poker_bot.action.executor import ActionExecutor, ScreenCoordinates


class TestPerceptionToDecision:
    """Test perception output feeding into decision engine."""
    
    def test_vlm_mock_state_to_cfr(self):
        """Test VLM mock state can be processed by CFR solver."""
        # Get mock game state from VLM
        vlm = VLMService(model_path=None, api_fallback=False)
        game_state = vlm._get_mock_state()
        
        # Feed to CFR solver
        solver = CFRSolver(num_buckets=10, lookahead_depth=2)
        action = solver.solve(game_state, iterations=50)
        
        # Should return valid action
        assert action is not None
        assert isinstance(action, Action)
        assert action.action_type in ActionType
    
    def test_game_state_format(self):
        """Test game state format compatibility."""
        # VLM produces game state
        game_state = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[],
            pot_size=100,
            current_bet=20,
            player=PlayerState(stack=980, current_bet=20),
            opponent=PlayerState(stack=980),
            street=BettingRound.PREFLOP,
            action_required=True,
            available_actions=[ActionType.FOLD, ActionType.CALL, ActionType.RAISE],
            confidence=0.95
        )
        
        # CFR solver can process it
        solver = CFRSolver(num_buckets=10)
        action = solver.solve(game_state, iterations=10)
        
        # Action should be one of the available actions
        assert action.action_type in [ActionType.FOLD, ActionType.CALL, 
                                       ActionType.RAISE, ActionType.CHECK]


class TestDecisionToAction:
    """Test decision output feeding into action executor."""
    
    def test_fold_action_execution(self):
        """Test fold action execution."""
        executor = ActionExecutor(simulation_mode=True)
        
        action = Action(action_type=ActionType.FOLD, amount=0)
        success = executor.execute(action)
        
        assert success
    
    def test_raise_action_execution(self):
        """Test raise action execution."""
        executor = ActionExecutor(simulation_mode=True)
        
        action = Action(action_type=ActionType.RAISE, amount=100)
        success = executor.execute(action)
        
        assert success
    
    def test_action_to_command_conversion(self):
        """Test action to command string conversion."""
        actions_and_commands = [
            (Action(action_type=ActionType.FOLD, amount=0), "fold"),
            (Action(action_type=ActionType.CHECK, amount=0), "check"),
            (Action(action_type=ActionType.CALL, amount=0), "call"),
            (Action(action_type=ActionType.BET, amount=50), "bet 50"),
            (Action(action_type=ActionType.RAISE, amount=100), "raise 100"),
            (Action(action_type=ActionType.ALL_IN, amount=0), "all_in"),
        ]
        
        for action, expected_cmd in actions_and_commands:
            assert action.to_command() == expected_cmd


class TestFullPipeline:
    """Test complete perception → decision → action pipeline."""
    
    def test_mock_pipeline(self):
        """Test complete pipeline with mocked components."""
        # 1. Perception: Get game state
        vlm = VLMService(model_path=None, api_fallback=False)
        game_state = vlm._get_mock_state()
        
        # 2. Decision: Compute action
        solver = CFRSolver(num_buckets=10, lookahead_depth=2)
        action = solver.solve(game_state, iterations=50)
        
        # 3. Action: Execute
        executor = ActionExecutor(simulation_mode=True)
        success = executor.execute(action)
        
        # Pipeline should complete successfully
        assert success
        assert action is not None
    
    def test_pipeline_with_different_streets(self):
        """Test pipeline works for different betting rounds."""
        vlm = VLMService(model_path=None, api_fallback=False)
        solver = CFRSolver(num_buckets=10, lookahead_depth=2)
        executor = ActionExecutor(simulation_mode=True)
        
        # Test preflop
        state_preflop = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[],
            pot_size=30,
            current_bet=20,
            street=BettingRound.PREFLOP
        )
        action = solver.solve(state_preflop, iterations=10)
        assert executor.execute(action)
        
        # Test flop
        state_flop = GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[
                Card.from_string('Qh'),
                Card.from_string('Jd'),
                Card.from_string('Tc')
            ],
            pot_size=100,
            current_bet=0,
            street=BettingRound.FLOP
        )
        solver.reset()
        action = solver.solve(state_flop, iterations=10)
        assert executor.execute(action)
    
    def test_json_parsing_to_action(self):
        """Test JSON response parsing to action execution."""
        vlm = VLMService(model_path=None, api_fallback=False)
        
        # Simulate JSON response from VLM
        json_response = """{
            "hole_cards": ["As", "Kh"],
            "community_cards": [],
            "pot_size": 150,
            "current_bet": 50,
            "player_stack": 950,
            "opponent_stack": 950,
            "street": "preflop",
            "action_required": true,
            "available_actions": ["fold", "call", "raise"]
        }"""
        
        game_state = vlm._parse_response(json_response)
        
        # Verify parsed correctly
        assert len(game_state.hole_cards) == 2
        assert game_state.pot_size == 150
        assert game_state.street == BettingRound.PREFLOP
        
        # Feed to solver
        solver = CFRSolver(num_buckets=10)
        action = solver.solve(game_state, iterations=10)
        
        # Execute
        executor = ActionExecutor(simulation_mode=True)
        assert executor.execute(action)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_json_handling(self):
        """Test VLM handles invalid JSON gracefully."""
        vlm = VLMService(model_path=None, api_fallback=False)
        
        # Invalid JSON should return state with reduced confidence
        state = vlm._parse_response("not valid json {}")
        
        # Should return state with reduced confidence (not full 1.0)
        assert state.confidence < 1.0
    
    def test_missing_fields_handling(self):
        """Test handling of missing fields in JSON."""
        vlm = VLMService(model_path=None, api_fallback=False)
        
        # JSON with missing fields
        json_response = """{
            "hole_cards": ["As", "Kh"],
            "pot_size": 100
        }"""
        
        state = vlm._parse_response(json_response)
        
        # Should use defaults for missing fields
        assert len(state.hole_cards) == 2
        assert state.pot_size == 100
        # community_cards should default to empty
        assert len(state.community_cards) == 0
    
    def test_solver_with_empty_state(self):
        """Test solver handles minimal state."""
        solver = CFRSolver(num_buckets=5, lookahead_depth=1)
        
        # Minimal state
        state = GameState(
            hole_cards=[],
            community_cards=[],
            pot_size=0,
            current_bet=0
        )
        
        # Should still return valid action
        action = solver.solve(state, iterations=10)
        assert action is not None
        assert action.action_type in ActionType


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
