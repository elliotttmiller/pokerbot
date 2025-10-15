"""Meta-learning wrapper that adapts strategy using an opponent model."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..game import Action, Card
from .base_agent import BaseAgent
from .opponent_model import OpponentModel


class MetaAgent(BaseAgent):
    def __init__(self, base: BaseAgent, name: str = "MetaAgent"):
        super().__init__(name)
        self.base = base
        self.opp = OpponentModel()

    def choose_action(self, 
                      hole_cards: List[Card],
                      community_cards: List[Card],
                      pot: int,
                      current_bet: int,
                      player_stack: int,
                      opponent_bet: int) -> tuple[Action, int]:
        action, amt = self.base.choose_action(hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet)
        # Simple adaptive tweak: if opponent very aggressive, tighten raises
        # (This requires integration points to feed observed opponent actions)
        return action, amt

    def integrate_opponent_action(self, opponent_name: str, action: Action):
        self.opp.observe(opponent_name, action.name)
