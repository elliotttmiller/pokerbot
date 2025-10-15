"""
Card-to-string conversion utilities for DeepStack poker AI.
Ported from DeepStack Lua card_to_string_conversion.lua.
"""

class CardToString:
    suit_table = ['h', 's', 'c', 'd']
    rank_table = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

    @classmethod
    def card_to_string(cls, rank_idx, suit_idx):
        """Convert numeric rank/suit indices to string."""
        return f"{cls.rank_table[rank_idx]}{cls.suit_table[suit_idx]}"

    @classmethod
    def string_to_card(cls, card_str):
        """Convert string to (rank_idx, suit_idx)."""
        rank = card_str[0]
        suit = card_str[1]
        rank_idx = cls.rank_table.index(rank)
        suit_idx = cls.suit_table.index(suit)
        return rank_idx, suit_idx

    @classmethod
    def cards_to_string(cls, cards):
        """Convert list of (rank_idx, suit_idx) to concatenated string."""
        return ''.join([cls.card_to_string(r, s) for r, s in cards])

# Example usage:
# s = CardToString.card_to_string(1, 0)  # 'K' of hearts

def card_to_string(rank_idx, suit_idx):
    """Convert numeric rank/suit indices to string."""
    suits = ['h', 's', 'c', 'd']
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    return f"{ranks[rank_idx]}{suits[suit_idx]}"
# idxs = CardToString.string_to_card('Kh')
# s_all = CardToString.cards_to_string([(0,0), (1,1)])
