"""Opponent modeling utilities.

Tracks opponent action frequencies and simple conditional tendencies.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class OpponentProfile:
    name: str
    action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, action: str):
        self.action_counts[action] += 1

    def aggression_index(self) -> float:
        raises = self.action_counts.get("RAISE", 0)
        calls = self.action_counts.get("CALL", 0)
        folds = self.action_counts.get("FOLD", 0)
        denom = (calls + folds) or 1
        return raises / denom


class OpponentModel:
    def __init__(self):
        self.profiles: Dict[str, OpponentProfile] = {}

    def observe(self, opponent_name: str, action: str):
        prof = self.profiles.setdefault(opponent_name, OpponentProfile(opponent_name))
        prof.record(action.upper())

    def get_aggression(self, opponent_name: str) -> float:
        prof = self.profiles.get(opponent_name)
        return prof.aggression_index() if prof else 0.5
