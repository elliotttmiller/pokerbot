"""
Poker Workflow Orchestrator

Unified pipeline for championship-grade poker AI:
Screenshot → Perception → Decision → Action

Optimized for Google Colab T4 GPU with automatic resource management.
"""

import time
import gc
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class WorkflowResult:
    """Result from workflow processing."""
    action: str
    amount: int
    confidence: float
    latency_ms: float
    strategy: Optional[Dict[str, float]] = None
    game_state: Optional[Dict] = None
    reasoning: str = ""


class PokerWorkflowOrchestrator:
    """
    Unified orchestrator for end-to-end poker AI.
    
    Connects perception (QWEN VL) with decision (DeepStack CFR)
    to produce optimal actions from raw screenshots.
    
    Usage:
        # Initialize with model paths
        orchestrator = PokerWorkflowOrchestrator(
            perception_model="drive/models/qwen_poker_vl",
            decision_model="drive/models/deepstack_champion.pt",
            simulation_mode=True
        )
        
        # Process screenshot
        result = orchestrator.process_screenshot("screenshot.png")
        print(f"Action: {result.action}, Amount: {result.amount}")
        
        # Or process dict directly (skip perception)
        result = orchestrator.solve_state({
            'hole_cards': ['As', 'Kh'],
            'pot_size': 100,
            'current_bet': 20,
            'street': 'preflop'
        })
    """
    
    def __init__(
        self,
        perception_model: Optional[str] = None,
        decision_model: Optional[str] = None,
        simulation_mode: bool = True,
        cfr_iterations: int = 2000,
        device: str = 'auto'
    ):
        """
        Initialize workflow orchestrator.
        
        Args:
            perception_model: Path to QWEN VL model (local or Drive)
            decision_model: Path to DeepStack value network
            simulation_mode: If True, don't execute GUI actions
            cfr_iterations: Default CFR iterations per solve
            device: Computation device ('auto', 'cuda', 'cpu')
        """
        self.perception_model_path = perception_model
        self.decision_model_path = decision_model
        self.simulation_mode = simulation_mode
        self.cfr_iterations = cfr_iterations
        self.device = device
        
        # Components (loaded lazily)
        self._detector = None
        self._solver = None
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'total_latency_ms': 0.0,
            'perception_calls': 0,
            'solver_calls': 0
        }
        
        print(f"[Orchestrator] Initialized (simulation_mode={simulation_mode})")
    
    @property
    def detector(self):
        """Lazy load perception detector."""
        if self._detector is None:
            from ..perception.detector import QWENPokerDetector
            self._detector = QWENPokerDetector(
                model_path=self.perception_model_path,
                device=self.device
            )
        return self._detector
    
    @property
    def solver(self):
        """Lazy load DeepStack solver."""
        if self._solver is None:
            from ..decision.solver import DeepStackSolver
            self._solver = DeepStackSolver(
                value_network_path=self.decision_model_path,
                cfr_iterations=self.cfr_iterations
            )
        return self._solver
    
    def process_screenshot(self, screenshot_path: str) -> WorkflowResult:
        """
        Full pipeline: Screenshot → Game State → Action.
        
        Args:
            screenshot_path: Path to screenshot image
            
        Returns:
            WorkflowResult with action and details
        """
        start_time = time.time()
        
        # Step 1: Perception
        game_state = self.detector.detect(screenshot_path)
        self.stats['perception_calls'] += 1
        
        # Step 2: Decision - Convert GameState to dict using Pydantic v2 or v1 compatibility
        try:
            game_state_dict = game_state.model_dump()
        except AttributeError:
            game_state_dict = game_state.dict()
        
        action = self.solver.solve(game_state_dict, iterations=self.cfr_iterations)
        self.stats['solver_calls'] += 1
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self.stats['total_decisions'] += 1
        self.stats['total_latency_ms'] += latency_ms
        
        # Build result - Get game state dict with Pydantic compatibility
        try:
            gs_dict = game_state.model_dump()
        except AttributeError:
            gs_dict = game_state.dict()
        
        result = WorkflowResult(
            action=action.action_type.value,
            amount=action.amount,
            confidence=action.confidence,
            latency_ms=latency_ms,
            strategy=action.strategy,
            game_state=gs_dict,
            reasoning=self._build_reasoning(game_state, action)
        )
        
        print(f"[Orchestrator] {result.action.upper()} {result.amount} "
              f"(conf={result.confidence:.2f}, latency={latency_ms:.0f}ms)")
        
        return result
    
    def solve_state(
        self,
        game_state: Dict[str, Any],
        iterations: Optional[int] = None
    ) -> WorkflowResult:
        """
        Solve a game state directly (skip perception).
        
        Useful for testing or when game state is already known.
        
        Args:
            game_state: Dictionary with game state
            iterations: CFR iterations (uses default if not provided)
            
        Returns:
            WorkflowResult with action and details
        """
        start_time = time.time()
        iterations = iterations or self.cfr_iterations
        
        # Decision
        action = self.solver.solve(game_state, iterations=iterations)
        self.stats['solver_calls'] += 1
        
        latency_ms = (time.time() - start_time) * 1000
        self.stats['total_decisions'] += 1
        self.stats['total_latency_ms'] += latency_ms
        
        return WorkflowResult(
            action=action.action_type.value,
            amount=action.amount,
            confidence=action.confidence,
            latency_ms=latency_ms,
            strategy=action.strategy,
            game_state=game_state,
            reasoning=f"Solved in {iterations} CFR iterations"
        )
    
    def _build_reasoning(self, game_state, action) -> str:
        """Build human-readable reasoning for decision."""
        parts = []
        
        # Hand strength
        if hasattr(game_state, 'hole_cards') and game_state.hole_cards:
            cards = [str(c) for c in game_state.hole_cards]
            parts.append(f"Hand: {' '.join(cards)}")
        
        # Action explanation
        if action.action_type.value == 'fold':
            parts.append("Folding due to weak hand or unfavorable odds")
        elif action.action_type.value in ['check', 'call']:
            parts.append("Calling to see next card")
        elif action.action_type.value == 'raise':
            parts.append(f"Raising {action.amount} for value/bluff")
        
        # Confidence
        if action.confidence >= 0.8:
            parts.append("High confidence decision")
        elif action.confidence >= 0.5:
            parts.append("Moderate confidence")
        else:
            parts.append("Low confidence (mixed strategy)")
        
        return " | ".join(parts)
    
    def cleanup(self):
        """Release resources."""
        if self._detector is not None:
            self._detector.cleanup()
            self._detector = None
        
        self._solver = None
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        print("[Orchestrator] Resources released")
    
    def get_stats(self) -> Dict:
        """Get workflow statistics."""
        stats = self.stats.copy()
        if stats['total_decisions'] > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['total_decisions']
        else:
            stats['avg_latency_ms'] = 0.0
        return stats


def process_screenshot(
    screenshot_path: str,
    perception_model: Optional[str] = None,
    decision_model: Optional[str] = None
) -> WorkflowResult:
    """
    One-call function to process a screenshot.
    
    Args:
        screenshot_path: Path to screenshot
        perception_model: Optional perception model path
        decision_model: Optional decision model path
        
    Returns:
        WorkflowResult with action and details
    """
    orchestrator = PokerWorkflowOrchestrator(
        perception_model=perception_model,
        decision_model=decision_model,
        simulation_mode=True
    )
    return orchestrator.process_screenshot(screenshot_path)


__all__ = ['PokerWorkflowOrchestrator', 'WorkflowResult', 'process_screenshot']
