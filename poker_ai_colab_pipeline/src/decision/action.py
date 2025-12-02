"""
Action Data Models for Decision Engine

Defines action types and structures compatible with DeepStack output format.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict


class ActionType(Enum):
    """Poker action types."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    BET = "bet"
    ALL_IN = "all_in"


@dataclass
class Action:
    """
    Represents a poker action with confidence and strategy breakdown.
    
    Attributes:
        action_type: Type of action (fold, call, raise, etc.)
        amount: Bet/raise amount (0 for fold/check/call)
        confidence: Solver confidence (0-1)
        strategy: Probability distribution over all actions
        cfr_iterations: Number of CFR iterations used
        solve_time_ms: Time taken to solve in milliseconds
    """
    action_type: ActionType
    amount: int = 0
    confidence: float = 1.0
    strategy: Optional[Dict[str, float]] = None
    cfr_iterations: int = 0
    solve_time_ms: float = 0.0
    
    @classmethod
    def fold(cls) -> 'Action':
        """Create fold action."""
        return cls(action_type=ActionType.FOLD)
    
    @classmethod
    def check(cls) -> 'Action':
        """Create check action."""
        return cls(action_type=ActionType.CHECK)
    
    @classmethod
    def call(cls) -> 'Action':
        """Create call action."""
        return cls(action_type=ActionType.CALL)
    
    @classmethod
    def raise_to(cls, amount: int, confidence: float = 1.0) -> 'Action':
        """Create raise action."""
        return cls(action_type=ActionType.RAISE, amount=amount, confidence=confidence)
    
    @classmethod
    def bet(cls, amount: int, confidence: float = 1.0) -> 'Action':
        """Create bet action."""
        return cls(action_type=ActionType.BET, amount=amount, confidence=confidence)
    
    @classmethod
    def all_in(cls, amount: int) -> 'Action':
        """Create all-in action."""
        return cls(action_type=ActionType.ALL_IN, amount=amount)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'action_type': self.action_type.value,
            'amount': self.amount,
            'confidence': self.confidence,
            'strategy': self.strategy,
            'cfr_iterations': self.cfr_iterations,
            'solve_time_ms': self.solve_time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        """Create from dictionary."""
        return cls(
            action_type=ActionType(data['action_type']),
            amount=data.get('amount', 0),
            confidence=data.get('confidence', 1.0),
            strategy=data.get('strategy'),
            cfr_iterations=data.get('cfr_iterations', 0),
            solve_time_ms=data.get('solve_time_ms', 0.0)
        )
    
    def __str__(self) -> str:
        if self.action_type in [ActionType.RAISE, ActionType.BET, ActionType.ALL_IN]:
            return f"{self.action_type.value.upper()} {self.amount}"
        return self.action_type.value.upper()


__all__ = ['Action', 'ActionType']
