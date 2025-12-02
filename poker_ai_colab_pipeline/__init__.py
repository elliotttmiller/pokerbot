"""
Poker AI Championship Pipeline

End-to-end DeepStack implementation optimized for Google Colab T4 GPU.

Components:
- Perception: QWEN 2.5-7B VL with Unsloth 4-bit quantization
- Decision: DeepStack CFR continual re-solving
- Orchestration: Unified workflow
- Validation: Compliance testing

Usage:
    from pipeline import PokerWorkflowOrchestrator
    
    orchestrator = PokerWorkflowOrchestrator(
        perception_model="drive/models/qwen_vl",
        decision_model="drive/models/deepstack.pt"
    )
    
    result = orchestrator.process_screenshot("screenshot.png")
    print(f"Action: {result.action}, Amount: {result.amount}")
"""

from .src.orchestration import PokerWorkflowOrchestrator, process_screenshot
from .src.perception import QWENPokerDetector, GameState, Card
from .src.decision import DeepStackSolver, Action, ActionType
from .src.validation import run_full_validation

__version__ = "1.0.0"

__all__ = [
    'PokerWorkflowOrchestrator',
    'process_screenshot',
    'QWENPokerDetector',
    'GameState',
    'Card',
    'DeepStackSolver',
    'Action',
    'ActionType',
    'run_full_validation'
]
