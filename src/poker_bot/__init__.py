"""
Poker Bot - Professional AI Poker System

Architecture based on the "Specialist" System:
- Perception Engine (Croupier): VLM-based game state detection
- Decision Engine (Strategist): CFR solver with value network
- Action Executor (Hands): GUI automation for action execution

References:
- DeepStack: https://www.deepstack.ai/
- DeepStack-Leduc: https://github.com/lifrordi/DeepStack-Leduc
- g5-poker-bot: https://github.com/Nemandza82/g5-poker-bot
"""

__version__ = "2.0.0"

from .core.data_models import (
    Card, GameState, Action, ActionType,
    BettingRound, PlayerState
)

__all__ = [
    'Card',
    'GameState', 
    'Action',
    'ActionType',
    'BettingRound',
    'PlayerState'
]
