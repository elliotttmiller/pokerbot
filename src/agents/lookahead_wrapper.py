"""Compatibility wrapper exposing DeepStack lookahead to agent modules.

This module bridges the legacy agent interface with the ported core
implementation in ``src.deepstack.lookahead``.  It keeps the agent surface
area small while delegating the heavy lifting to the shared DeepStack logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from deepstack.game.card import Card
from deepstack.game.card_tools import get_card_tools
from deepstack.core.lookahead import Lookahead
from deepstack.core.tree_builder import PokerTreeBuilder


def _encode_board(cards: Sequence[Any]) -> Sequence[Any]:
    """Convert a list of cards to the representation expected by the core."""
    if not cards:
        return []

    encoded = []
    for card in cards:
        if isinstance(card, Card):
            # Use deck index so the tree builder can distinguish suits/ranks.
            encoded.append(card.to_index())
        else:
            encoded.append(card)
    return encoded


@dataclass
class DeepStackLookahead:
    """Thin wrapper maintaining the historical agent interface."""

    game_variant: str = "leduc"
    stack_size: int = 1200
    ante: int = 100
    bet_sizing: Optional[Sequence[float]] = None
    _core: Lookahead = field(init=False, repr=False)
    _tree_builder: PokerTreeBuilder = field(init=False, repr=False)
    _last_results: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        card_tools = get_card_tools(self.game_variant)
        num_hands = card_tools.settings.card_count
        self._core = Lookahead(game_variant=self.game_variant, num_hands=num_hands)
        self._tree_builder = PokerTreeBuilder(game_variant=self.game_variant,
                                              stack_size=self.stack_size)
        if self.bet_sizing is None:
            self.bet_sizing = [1.0]

    # ------------------------------------------------------------------
    # Legacy interface methods used by ChampionAgent and tests
    # ------------------------------------------------------------------
    def build_lookahead(self, street: int, current_player: int,
                        community_cards: Sequence[Any]) -> None:
        """Construct a fresh lookahead tree for the provided public state."""
        board = _encode_board(community_cards)
        params = {
            "street": street,
            "bets": [self.ante, self.ante],
            "current_player": current_player,
            "board": board,
            "limit_to_street": False,
            "bet_sizing": list(self.bet_sizing or [1.0]),
        }
        tree = self._tree_builder.build_tree(params)
        self._core.build_lookahead(tree)
        self._last_results = {}

    def resolve_first_node(self, player_range: np.ndarray,
                           opponent_range: np.ndarray,
                           iterations: Optional[int] = None) -> None:
        """Solve the lookahead from the root with fixed ranges."""
        if iterations is not None:
            self._core.cfr_iterations = iterations
        self._core.resolve_first_node(player_range, opponent_range)
        self._cache_results()

    def resolve(self, player_range: np.ndarray, opponent_cfvs: np.ndarray,
                iterations: Optional[int] = None) -> None:
        """Solve the lookahead using opponent CFVs (continual resolving)."""
        if iterations is not None:
            self._core.cfr_iterations = iterations
        self._core.resolve(player_range, opponent_cfvs)
        self._cache_results()

    def get_results(self) -> Dict[str, Any]:
        """Return the most recent solving artefacts."""
        if not self._last_results:
            self._cache_results()
        return self._last_results

    def get_strategy(self) -> Dict[str, np.ndarray]:
        """Expose the root strategy for downstream consumers."""
        return self._core.get_strategy()

    def get_root_cfv(self) -> np.ndarray:
        return self._core.get_root_cfv()

    def set_value_network(self, value_network) -> None:
        self._core.set_value_network(value_network)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _cache_results(self) -> None:
        results = self._core.get_results()
        if "strategy" not in results or not results["strategy"]:
            results = dict(results)
            results["strategy"] = self._core.get_strategy()
        self._last_results = results


__all__ = ["DeepStackLookahead"]
