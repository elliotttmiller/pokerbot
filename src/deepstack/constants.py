"""Core constants shared across DeepStack modules."""
from __future__ import annotations

from dataclasses import dataclass


PLAYERS_COUNT = 2
STREETS_COUNT = 2


@dataclass(frozen=True)
class PlayerIds:
    """Identifiers for standard poker players and the chance player."""

    CHANCE: int = 0
    P1: int = 1
    P2: int = 2


@dataclass(frozen=True)
class NodeTypes:
    """Identifiers for common node categories in the game tree."""

    TERMINAL_FOLD: int = -2
    TERMINAL_CALL: int = -1
    CHECK: int = -1
    CHANCE_NODE: int = 0
    INNER_NODE: int = 1


@dataclass(frozen=True)
class Actions:
    """Identifiers for canonical player actions in internal trees."""

    FOLD: int = -2
    CCALL: int = -1


@dataclass(frozen=True)
class AcpcActions:
    """String representation of actions for ACPC protocol output."""

    FOLD: str = "fold"
    CCALL: str = "ccall"
    RAISE: str = "raise"


PLAYERS = PlayerIds()
NODE_TYPES = NodeTypes()
ACTIONS = Actions()
ACPC_ACTIONS = AcpcActions()
