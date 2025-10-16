"""
Hand bucket utilities for Texas Hold'em 169-class abstraction and board-aware masking.

This module defines a canonical 13x13 grid ordering for hand classes:
- Rows/cols are ranks: A,K,Q,J,T,9,8,7,6,5,4,3,2 (index 0..12)
- Diagonal: pairs (AA..22)
- Upper triangle: suited (AKs..)
- Lower triangle: offsuit (AKo..)

Provides:
- rank_index/rank_from_index helpers
- enumerate_hand_classes() -> list of (r1,r2,suited) in fixed order length 169
- combos_remaining_fraction(hand_class, board_cards) -> fraction in [0,1]
  accounting for cards present on the board
"""
from __future__ import annotations
from typing import List, Tuple

RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']

def rank_index(ch: str) -> int:
    return RANKS.index(ch)

def enumerate_hand_classes() -> List[Tuple[int,int,bool]]:
    """Return the 169 classes as (r1,r2,is_suited) indices in a fixed grid order.
    r1,r2 are rank indices 0..12 (A..2). is_suited True for upper triangle, False for lower.
    Diagonal entries (pairs) use is_suited=False by convention.
    """
    classes: List[Tuple[int,int,bool]] = []
    for i in range(13):
        for j in range(13):
            if i == j:
                classes.append((i, j, False))  # pair
            elif i < j:
                classes.append((i, j, True))   # suited
            else:
                classes.append((i, j, False))  # offsuit
    return classes

_CLASSES = enumerate_hand_classes()

def _rank_count_on_board(rank_idx: int, board_ranks: List[int]) -> int:
    return sum(1 for r in board_ranks if r == rank_idx)

def combos_remaining_fraction(hand_class: Tuple[int,int,bool], board_ranks: List[int]) -> float:
    """Compute remaining combo fraction for a hand class given board ranks.

    Inputs:
      hand_class: (r1, r2, is_suited)
      board_ranks: list of rank indices (length 0..5), suits ignored for speed

    Assumptions:
      - Full deck has 4 cards per rank.
      - Board reduces available cards by the number of that rank present.
      - For simplicity/efficiency, we ignore exact suit blocking and approximate suited/offsuit combos
        by rank availability only (good approximation for masking).

    Returns: remaining combos / total combos for the class in [0,1].
    """
    r1, r2, suited = hand_class
    # Available cards for each rank after removing board cards of that rank
    avail1 = max(0, 4 - _rank_count_on_board(r1, board_ranks))
    avail2 = max(0, 4 - _rank_count_on_board(r2, board_ranks))

    if r1 == r2:
        # Pairs: total combos C(4,2)=6; remaining C(avail,2)
        total = 6
        rem = 0
        if avail1 >= 2:
            rem = avail1 * (avail1 - 1) // 2
        return 0.0 if total == 0 else rem / total

    # Non-pairs: total combos = 16 (suited: 4; offsuit: 12)
    if suited:
        total = 4
        # Roughly, remaining suited combos bounded by min(avail1, avail2)
        rem = min(avail1, avail2)
        if rem > 4:
            rem = 4
    else:
        total = 12
        # Offsuit approximated as avail1*avail2 - suited (capped to [0,12])
        est = avail1 * avail2
        suited_est = min(avail1, avail2)
        rem = max(0, min(12, est - suited_est))
    return 0.0 if total == 0 else rem / total

def board_to_rank_indices(board_cards: List[int]) -> List[int]:
    """Convert 0..51 card indices to rank indices (0..12), assuming deck order rank-major/suit-minor:
    rank = idx // 4 if deck is [A♠,A♥,A♦,A♣, K♠,...]; adjust if deck differs.
    """
    return [int(c) // 4 for c in board_cards]

def fractional_mask_169(board_cards: List[int]) -> List[float]:
    """Return a 169-length list of fractional availability masks for a given board.
    """
    ranks = board_to_rank_indices(board_cards)
    return [combos_remaining_fraction(cls, ranks) for cls in _CLASSES]
