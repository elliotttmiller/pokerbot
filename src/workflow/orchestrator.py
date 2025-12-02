"""
Poker Workflow Orchestrator - Unified Pipeline for Multi-Modal Pokerbot.

This module provides the complete orchestration layer that:
1. Captures and processes screenshots via QWEN 2.5-7B VL
2. Validates detected game states
3. Computes optimal actions using DeepStack/CFR
4. Tracks game context across hands
5. Manages opponent modeling

Architecture References:
- DeepStack.pdf: Continual re-solving pattern
- DeepStack-Leduc: Stateful resolver API
- self-operating-computer: Vision pipeline patterns
- g5-poker-bot: GPU-accelerated computation
- gto-poker-bot: GTO strategy integration
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class GameContext:
    """
    Persistent context tracking across a poker hand.
    
    Maintains state that persists across individual decisions:
    - Action history for both players
    - Range updates based on actions
    - Position information
    - Street transitions
    """
    hand_id: str = ""
    start_time: float = field(default_factory=time.time)
    
    # Hand state
    hole_cards: List[str] = field(default_factory=list)
    position: str = "unknown"  # 'button', 'big_blind', 'small_blind', 'early', 'middle', 'late'
    
    # Action tracking
    player_actions: List[Dict] = field(default_factory=list)
    opponent_actions: List[Dict] = field(default_factory=list)
    
    # Street tracking
    current_street: str = "preflop"
    pot_by_street: Dict[str, int] = field(default_factory=dict)
    
    # Range estimation
    player_range: Optional[np.ndarray] = None
    opponent_range: Optional[np.ndarray] = None
    opponent_range_updates: int = 0
    
    # Statistics
    pot_commitment: int = 0
    total_invested: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize context to dictionary."""
        return {
            'hand_id': self.hand_id,
            'start_time': self.start_time,
            'hole_cards': self.hole_cards,
            'position': self.position,
            'player_actions': self.player_actions,
            'opponent_actions': self.opponent_actions,
            'current_street': self.current_street,
            'pot_by_street': self.pot_by_street,
            'pot_commitment': self.pot_commitment,
            'total_invested': self.total_invested
        }


@dataclass
class DecisionResult:
    """Result of a strategy decision."""
    action: str  # 'fold', 'check', 'call', 'raise', 'all_in'
    amount: int  # Raise/bet amount (0 for fold/check/call)
    confidence: float  # Decision confidence (0-1)
    reasoning: str  # Human-readable reasoning
    
    # Strategy breakdown
    cfr_recommendation: Optional[str] = None
    deepstack_recommendation: Optional[str] = None
    opponent_adjustment: Optional[str] = None
    
    # Values
    expected_value: float = 0.0
    equity: float = 0.0
    pot_odds: float = 0.0


class PokerWorkflowOrchestrator:
    """
    Unified orchestrator for the multi-modal pokerbot workflow.
    
    This class connects all components into a cohesive pipeline:
    
    Screenshot → Vision Detection → Game State Validation → 
    Strategy Computation → Action Decision → Execution
    
    Usage:
        orchestrator = PokerWorkflowOrchestrator(
            vision_model_path="models/qwen_poker_vl",
            strategy_model_path="models/deepstack_champion.pt"
        )
        
        # Start new hand
        orchestrator.start_new_hand()
        
        # Process each screenshot
        result = orchestrator.process_screenshot("screenshot.png")
        print(f"Action: {result.action}, Amount: {result.amount}")
        
        # Record actual action taken
        orchestrator.record_action(result.action, result.amount)
    """
    
    def __init__(self,
                 vision_model_path: Optional[str] = None,
                 strategy_model_path: Optional[str] = None,
                 device: str = 'auto',
                 enable_opponent_modeling: bool = True,
                 enable_logging: bool = True,
                 log_dir: str = "logs/workflow"):
        """
        Initialize the workflow orchestrator.
        
        Args:
            vision_model_path: Path to fine-tuned QWEN vision model
            strategy_model_path: Path to DeepStack value network
            device: Computation device ('auto', 'cuda', 'cpu')
            enable_opponent_modeling: Track and model opponent behavior
            enable_logging: Log decisions and game states
            log_dir: Directory for log files
        """
        self.vision_model_path = vision_model_path
        self.strategy_model_path = strategy_model_path
        self.device = device
        self.enable_opponent_modeling = enable_opponent_modeling
        self.enable_logging = enable_logging
        self.log_dir = log_dir
        
        # Components (loaded lazily)
        self._vision_detector = None
        self._strategy_agent = None
        
        # Current hand context
        self.context: Optional[GameContext] = None
        
        # Session statistics
        self.session_stats = {
            'hands_played': 0,
            'decisions_made': 0,
            'total_winnings': 0,
            'session_start': datetime.now().isoformat()
        }
        
        # Create log directory
        if enable_logging:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        print("[Orchestrator] Initialized multi-modal poker workflow")
    
    @property
    def vision_detector(self):
        """Lazy load vision detector."""
        if self._vision_detector is None:
            from src.vision.multimodal_detector import MultiModalVisionDetector
            self._vision_detector = MultiModalVisionDetector(
                model_path=self.vision_model_path,
                device=self.device
            )
        return self._vision_detector
    
    @property
    def strategy_agent(self):
        """Lazy load strategy agent."""
        if self._strategy_agent is None:
            from src.agents.pokerbot_agent import PokerBotAgent
            self._strategy_agent = PokerBotAgent(
                name="MultiModalBot",
                use_cfr=True,
                use_cfr_plus=True,
                use_dqn=False,  # Disable DQN to reduce memory
                use_deepstack=True,
                use_opponent_modeling=self.enable_opponent_modeling,
                use_pretrained=True
            )
            
            # Load trained model if available
            if self.strategy_model_path and os.path.exists(self.strategy_model_path):
                try:
                    self._strategy_agent.load(self.strategy_model_path)
                    print(f"[Orchestrator] Loaded strategy model from {self.strategy_model_path}")
                except Exception as e:
                    print(f"[Orchestrator] Failed to load strategy model: {e}")
        
        return self._strategy_agent
    
    def start_new_hand(self, position: str = "unknown") -> GameContext:
        """
        Initialize context for a new hand.
        
        Args:
            position: Player's position at the table
            
        Returns:
            New GameContext instance
        """
        # Generate unique hand ID
        hand_id = f"hand_{int(time.time() * 1000)}"
        
        # Create new context
        self.context = GameContext(
            hand_id=hand_id,
            position=position,
            pot_by_street={'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0}
        )
        
        # Reset agent state
        self.strategy_agent.start_new_hand()
        
        # Update statistics
        self.session_stats['hands_played'] += 1
        
        print(f"[Orchestrator] Started new hand {hand_id}")
        return self.context
    
    def process_screenshot(self, screenshot_path: str) -> DecisionResult:
        """
        Process a screenshot and compute optimal action.
        
        Main entry point for the decision pipeline.
        
        Args:
            screenshot_path: Path to screenshot image
            
        Returns:
            DecisionResult with computed action and details
        """
        if self.context is None:
            self.start_new_hand()
        
        start_time = time.time()
        
        # Step 1: Detect game state from screenshot
        detected_state = self.vision_detector.detect(screenshot_path)
        
        # Step 2: Validate and update context
        validated_state = self._validate_and_update(detected_state)
        
        # Step 3: Check if action is required
        if not validated_state.action_required:
            return DecisionResult(
                action="wait",
                amount=0,
                confidence=1.0,
                reasoning="No action required at this time"
            )
        
        # Step 4: Compute optimal action
        result = self._compute_action(validated_state)
        
        # Step 5: Log decision
        if self.enable_logging:
            self._log_decision(screenshot_path, detected_state, result)
        
        # Update statistics
        self.session_stats['decisions_made'] += 1
        
        elapsed = time.time() - start_time
        print(f"[Orchestrator] Decision computed in {elapsed:.2f}s: {result.action} {result.amount}")
        
        return result
    
    def _validate_and_update(self, detected_state) -> Any:
        """Validate detected state and update context."""
        # Update hole cards if first detection
        if not self.context.hole_cards and detected_state.hole_cards:
            self.context.hole_cards = [c.to_string() for c in detected_state.hole_cards]
        
        # Check for street transition
        old_street = self.context.current_street
        new_street = detected_state.street
        
        if new_street != old_street:
            print(f"[Orchestrator] Street transition: {old_street} → {new_street}")
            self.context.current_street = new_street
            self.context.pot_by_street[new_street] = detected_state.pot_size
            
            # Reset agent for new street (re-solving)
            self.strategy_agent.start_new_hand()  # Triggers lookahead rebuild
        
        # Validate detection consistency
        if detected_state.confidence < 0.5:
            print(f"[Orchestrator] Warning: Low detection confidence ({detected_state.confidence:.2f})")
        
        return detected_state
    
    def _compute_action(self, state) -> DecisionResult:
        """Compute optimal action using strategy agent."""
        # Convert detected cards to Card objects
        from src.deepstack.game import Card
        
        try:
            hole_cards = [Card.from_string(c.to_string()) for c in state.hole_cards]
            community_cards = [Card.from_string(c.to_string()) for c in state.community_cards]
        except Exception as e:
            print(f"[Orchestrator] Card conversion error: {e}")
            # Use empty cards as fallback
            hole_cards = []
            community_cards = []
        
        # Get action from strategy agent
        try:
            action, raise_amount = self.strategy_agent.choose_action(
                hole_cards=hole_cards,
                community_cards=community_cards,
                pot=state.pot_size,
                current_bet=state.current_bet,
                player_stack=state.player_stack,
                opponent_bet=state.current_bet
            )
            
            # Convert action enum to string
            action_str = str(action).split('.')[-1].lower()
            
        except Exception as e:
            print(f"[Orchestrator] Strategy computation error: {e}")
            action_str = "call"
            raise_amount = 0
        
        # Compute supporting metrics
        equity = self._estimate_equity(hole_cards, community_cards)
        pot_odds = self._compute_pot_odds(state.pot_size, state.current_bet)
        ev = self._estimate_ev(action_str, equity, state.pot_size, state.current_bet, raise_amount)
        
        # Build reasoning
        reasoning = self._build_reasoning(
            action_str, raise_amount, equity, pot_odds, ev, state
        )
        
        return DecisionResult(
            action=action_str,
            amount=raise_amount,
            confidence=state.confidence,
            reasoning=reasoning,
            expected_value=ev,
            equity=equity,
            pot_odds=pot_odds
        )
    
    def _estimate_equity(self, hole_cards: List, community_cards: List) -> float:
        """Estimate hand equity."""
        if len(hole_cards) < 2:
            return 0.5  # Default equity
        
        # Simplified equity estimation based on hand strength
        try:
            # Convert rank to numeric value
            def rank_to_value(rank) -> int:
                if hasattr(rank, 'value'):
                    return rank.value
                rank_str = str(rank).upper()
                rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                           '9': 9, 'T': 10, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
                return rank_map.get(rank_str, 7)  # Default to 7 if unknown
            
            high_rank = max(rank_to_value(c.rank) for c in hole_cards)
            low_rank = min(rank_to_value(c.rank) for c in hole_cards)
            
            # Pair bonus
            is_pair = hole_cards[0].rank == hole_cards[1].rank
            
            # Suited bonus
            is_suited = hole_cards[0].suit == hole_cards[1].suit
            
            # Base equity from high card (normalized 0-1)
            base_equity = 0.3 + ((high_rank - 2) / 12.0) * 0.3
            
            if is_pair:
                base_equity += 0.2
            if is_suited:
                base_equity += 0.05
            
            return min(base_equity, 1.0)
            
        except Exception:
            return 0.5
    
    def _compute_pot_odds(self, pot: int, to_call: int) -> float:
        """Compute pot odds."""
        if to_call <= 0:
            return float('inf')
        return pot / to_call
    
    def _estimate_ev(self, action: str, equity: float, pot: int, 
                     to_call: int, raise_amount: int) -> float:
        """Estimate expected value of action."""
        if action == 'fold':
            return 0.0
        elif action in ['check', 'call']:
            # EV = (equity * pot) - ((1-equity) * to_call)
            return (equity * pot) - ((1 - equity) * to_call)
        elif action == 'raise':
            # Simplified raise EV
            total_bet = to_call + raise_amount
            return (equity * (pot + raise_amount)) - ((1 - equity) * total_bet)
        return 0.0
    
    def _build_reasoning(self, action: str, amount: int, equity: float,
                        pot_odds: float, ev: float, state) -> str:
        """Build human-readable reasoning for decision."""
        parts = []
        
        # Equity assessment
        if equity >= 0.65:
            parts.append(f"Strong hand (equity={equity:.1%})")
        elif equity >= 0.45:
            parts.append(f"Marginal hand (equity={equity:.1%})")
        else:
            parts.append(f"Weak hand (equity={equity:.1%})")
        
        # Pot odds assessment
        if pot_odds != float('inf'):
            required_equity = 1 / (1 + pot_odds)
            if equity > required_equity:
                parts.append(f"Pot odds favorable ({pot_odds:.1f}:1)")
            else:
                parts.append(f"Pot odds unfavorable ({pot_odds:.1f}:1)")
        
        # EV assessment
        if ev > 0:
            parts.append(f"+EV decision ({ev:.0f} chips)")
        elif ev < 0:
            parts.append(f"-EV but strategic ({ev:.0f} chips)")
        
        # Action context
        if action == 'raise':
            parts.append(f"Raising {amount} to build pot/semi-bluff")
        elif action == 'call':
            parts.append("Calling to see next card")
        elif action == 'fold':
            parts.append("Folding to avoid losses")
        
        return " | ".join(parts)
    
    def record_action(self, action: str, amount: int, is_opponent: bool = False):
        """
        Record an action taken in the hand.
        
        Args:
            action: Action taken
            amount: Bet/raise amount
            is_opponent: Whether this is opponent's action
        """
        if self.context is None:
            return
        
        action_record = {
            'action': action,
            'amount': amount,
            'street': self.context.current_street,
            'timestamp': time.time()
        }
        
        if is_opponent:
            self.context.opponent_actions.append(action_record)
            
            # Update opponent model
            if self.enable_opponent_modeling and self.strategy_agent.opponent_model:
                self.strategy_agent.opponent_model.observe(
                    "opponent", action
                )
        else:
            self.context.player_actions.append(action_record)
            
            # Update pot commitment
            if action in ['call', 'raise', 'bet']:
                self.context.total_invested += amount
    
    def end_hand(self, result: str, winnings: int = 0):
        """
        End current hand and record result.
        
        Args:
            result: 'won', 'lost', 'tied'
            winnings: Chip change (positive for wins, negative for losses)
        """
        if self.context is None:
            return
        
        # Record result
        self.strategy_agent.observe_result(
            won=(result == 'won'),
            amount=winnings
        )
        
        # Update session stats
        self.session_stats['total_winnings'] += winnings
        
        # Log hand summary
        if self.enable_logging:
            self._log_hand_summary(result, winnings)
        
        print(f"[Orchestrator] Hand ended: {result}, {winnings:+d} chips")
        
        # Clear context
        self.context = None
    
    def _log_decision(self, screenshot_path: str, state, result: DecisionResult):
        """Log decision for analysis."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'hand_id': self.context.hand_id if self.context else 'unknown',
            'screenshot': screenshot_path,
            'detected_state': state.to_dict() if hasattr(state, 'to_dict') else str(state),
            'decision': {
                'action': result.action,
                'amount': result.amount,
                'confidence': result.confidence,
                'ev': result.expected_value,
                'equity': result.equity,
                'pot_odds': result.pot_odds,
                'reasoning': result.reasoning
            }
        }
        
        log_file = Path(self.log_dir) / f"decisions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_hand_summary(self, result: str, winnings: int):
        """Log hand summary."""
        if self.context is None:
            return
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'hand_id': self.context.hand_id,
            'hole_cards': self.context.hole_cards,
            'result': result,
            'winnings': winnings,
            'total_invested': self.context.total_invested,
            'player_actions': self.context.player_actions,
            'opponent_actions': self.context.opponent_actions,
            'duration': time.time() - self.context.start_time
        }
        
        log_file = Path(self.log_dir) / f"hands_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        stats = self.session_stats.copy()
        stats['session_duration'] = (
            datetime.now() - datetime.fromisoformat(stats['session_start'])
        ).total_seconds()
        
        if stats['hands_played'] > 0:
            stats['avg_winnings_per_hand'] = stats['total_winnings'] / stats['hands_played']
        else:
            stats['avg_winnings_per_hand'] = 0
        
        return stats


# Convenience function for quick workflow execution
def run_single_decision(screenshot_path: str,
                       vision_model: Optional[str] = None,
                       strategy_model: Optional[str] = None) -> DecisionResult:
    """
    Run a single decision on a screenshot.
    
    Args:
        screenshot_path: Path to screenshot
        vision_model: Optional path to vision model
        strategy_model: Optional path to strategy model
        
    Returns:
        DecisionResult with recommended action
    """
    orchestrator = PokerWorkflowOrchestrator(
        vision_model_path=vision_model,
        strategy_model_path=strategy_model,
        enable_logging=False
    )
    orchestrator.start_new_hand()
    return orchestrator.process_screenshot(screenshot_path)


__all__ = [
    'PokerWorkflowOrchestrator',
    'GameContext',
    'DecisionResult',
    'run_single_decision'
]
