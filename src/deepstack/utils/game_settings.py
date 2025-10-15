"""Game configuration values for DeepStack variants.

Currently supports Leduc Hold'em as the default configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class GameSettings:
    """Encapsulates basic card game configuration."""

    suit_count: int = 2
    rank_count: int = 3
    board_card_count: int = 1
    player_count: int = 2

    @property
    def card_count(self) -> int:
        """Total number of unique cards in the deck."""
        return self.suit_count * self.rank_count


# Predefined settings for supported variants.
_VARIANT_SETTINGS: Dict[str, GameSettings] = {
    "leduc": GameSettings(),
    # Placeholders for future variants. Parameters follow standard Hold'em sizing.
    "holdem": GameSettings(suit_count=4, rank_count=13, board_card_count=5, player_count=2),
}


def get_game_settings(variant: str = "leduc") -> GameSettings:
    """Retrieve immutable game settings for the requested variant.

    Args:
        variant: Name of the poker variant (e.g., "leduc", "holdem").

    Returns:
        GameSettings instance describing the variant.
    """
    key = variant.lower()
    if key not in _VARIANT_SETTINGS:
        raise ValueError(f"Unsupported game variant '{variant}'.")
    return _VARIANT_SETTINGS[key]
