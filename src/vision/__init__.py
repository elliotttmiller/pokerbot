"""Vision package initialization.

This package provides vision-based game state detection:
- VisionDetector: GPT-4 Vision based detection (API)
- MultiModalVisionDetector: QWEN 2.5-7B VL based detection (local)
- ScreenController: Screen capture and control
- ActionMapper: Map actions to screen coordinates
"""

from .screen_controller import ActionMapper, ScreenController
from .vision_detector import VisionDetector
from .multimodal_detector import MultiModalVisionDetector, DetectedGameState, DetectedCard

__all__ = [
    'VisionDetector',
    'MultiModalVisionDetector',
    'DetectedGameState',
    'DetectedCard',
    'ScreenController',
    'ActionMapper',
]
