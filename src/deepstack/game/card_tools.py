"""Utilities for manipulating cards, boards, and probability ranges.

This module ports the behaviour of the original DeepStack `card_tools.lua`.
The implementation focuses on Leduc Hold'em semantics while remaining
general enough to extend to small variants in the future.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ..utils.game_settings import GameSettings, get_game_settings


@dataclass
class CardTools:
    """Helper utilities for card-set validation and range manipulation."""

    settings: GameSettings

    def __post_init__(self) -> None:
        self._board_index_table = self._init_board_index_table()

    # ------------------------------------------------------------------
    # Basic card validation helpers
    # ------------------------------------------------------------------
    def hand_is_possible(self, hand: Sequence[int]) -> bool:
        """Return True if cards in `hand` are legal and non-repeating."""
        hand_array = _ensure_1d_array(hand)
        if hand_array.size == 0:
            return True

        if hand_array.min() < 0 or hand_array.max() >= self.settings.card_count:
            raise ValueError("Illegal cards in hand: indices out of range")

        used = np.zeros(self.settings.card_count, dtype=np.int8)
        for card in hand_array:
            used[card] += 1
            if used[card] > 1:
                return False
        return True

    # ------------------------------------------------------------------
    # Range manipulation helpers
    # ------------------------------------------------------------------
    def get_possible_hand_indexes(self, board: Sequence[int]) -> np.ndarray:
        """Return mask of hands compatible with `board` (1 = valid, 0 = invalid)."""
        if _is_empty_board(board):
            return np.ones(self.settings.card_count, dtype=np.float32)

        board_array = _ensure_1d_array(board)
        mask = np.zeros(self.settings.card_count, dtype=np.float32)
        candidate = np.empty(board_array.size + 1, dtype=int)
        candidate[:-1] = board_array

        for card in range(self.settings.card_count):
            candidate[-1] = card
            if self.hand_is_possible(candidate):
                mask[card] = 1.0
        return mask

    def get_impossible_hand_indexes(self, board: Sequence[int]) -> np.ndarray:
        """Return mask of hands incompatible with `board` (1 = invalid)."""
        return 1.0 - self.get_possible_hand_indexes(board)

    def get_uniform_range(self, board: Sequence[int]) -> np.ndarray:
        """Return uniform probability range over hands valid for `board`."""
        mask = self.get_possible_hand_indexes(board)
        total = mask.sum()
        if total == 0:
            # No valid hands remain; return zeros to avoid division by zero.
            return mask
        return mask / total

    def get_random_range(self, board: Sequence[int], seed: Optional[int] = None) -> np.ndarray:
        """Return random probability range over hands valid for `board`."""
        rng = np.random.default_rng(seed)
        raw = rng.random(self.settings.card_count, dtype=np.float32)
        raw *= self.get_possible_hand_indexes(board)
        total = raw.sum()
        if total == 0:
            return raw
        return raw / total

    def is_valid_range(self, range_vector: Sequence[float], board: Sequence[int]) -> bool:
        """Check whether `range_vector` only assigns mass to valid hands."""
        vector = np.asarray(range_vector, dtype=np.float32)
        if not math.isclose(float(vector.sum()), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            return False
        impossible_mass = np.dot(vector, self.get_impossible_hand_indexes(board))
        return math.isclose(float(impossible_mass), 0.0, abs_tol=1e-5)

    def normalize_range(self, board: Sequence[int], range_vector: Sequence[float]) -> np.ndarray:
        """Zero-out invalid hands for `board` and renormalize the range."""
        mask = self.get_possible_hand_indexes(board)
        vector = np.asarray(range_vector, dtype=np.float32)
        adjusted = vector * mask
        total = adjusted.sum()
        if total == 0:
            return np.zeros_like(adjusted)
        return adjusted / total

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------
    def board_to_street(self, board: Sequence[int]) -> int:
        """Map board length to betting round (1-indexed)."""
        board_array = _ensure_1d_array(board)
        return 1 if board_array.size == 0 else 2

    def get_second_round_boards(self) -> np.ndarray:
        """Enumerate all possible boards for the second betting round."""
        board_count = self.get_boards_count()
        if self.settings.board_card_count == 1:
            boards = np.zeros((board_count, 1), dtype=int)
            boards[:, 0] = np.arange(self.settings.card_count, dtype=int)
            return boards

        if self.settings.board_card_count == 2:
            boards = np.zeros((board_count, 2), dtype=int)
            idx = 0
            for first in range(self.settings.card_count):
                for second in range(first + 1, self.settings.card_count):
                    boards[idx, 0] = first
                    boards[idx, 1] = second
                    idx += 1
            assert idx == board_count, "Generated board count mismatch"
            return boards

        raise ValueError("Unsupported board size for enumeration")

    def get_boards_count(self) -> int:
        """Return the total number of unique boards."""
        if self.settings.board_card_count == 1:
            return self.settings.card_count
        if self.settings.board_card_count == 2:
            count = self.settings.card_count
            return (count * (count - 1)) // 2
        raise ValueError("Unsupported board size for count computation")

    def get_board_index(self, board: Sequence[int]) -> int:
        """Return deterministic index for the provided board cards."""
        board_array = _ensure_1d_array(board)
        if board_array.size == 0:
            raise ValueError("Board index requested for empty board")

        index_table = self._board_index_table
        for card in board_array:
            index_table = index_table[card]
        index_value = int(index_table)
        if index_value < 0:
            raise ValueError("Board combination is invalid or duplicates cards")
        return index_value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_board_index_table(self) -> np.ndarray:
        if self.settings.board_card_count == 1:
            return np.arange(self.settings.card_count, dtype=int)

        if self.settings.board_card_count == 2:
            table = -np.ones((self.settings.card_count, self.settings.card_count), dtype=int)
            idx = 0
            for first in range(self.settings.card_count):
                for second in range(first + 1, self.settings.card_count):
                    table[first, second] = idx
                    table[second, first] = idx
                    idx += 1
            return table

        raise ValueError("Unsupported board size for index table")


# Convenience instance mirroring Lua module behaviour
_default_card_tools = CardTools(get_game_settings("leduc"))


def get_card_tools(game_variant: str = "leduc") -> CardTools:
    """Factory returning cached `CardTools` for the requested variant."""
    if game_variant.lower() == "leduc":
        return _default_card_tools
    return CardTools(get_game_settings(game_variant))


# Utility functions ---------------------------------------------------------

def _ensure_1d_array(values: Sequence[int]) -> np.ndarray:
    array = np.asarray(values, dtype=int)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim > 1:
        return array.flatten()
    return array


def _is_empty_board(board: Optional[Sequence[int]]) -> bool:
    if board is None:
        return True
    if isinstance(board, np.ndarray):
        return board.size == 0
    try:
        return len(board) == 0  # type: ignore[arg-type]
    except TypeError:
        return True
