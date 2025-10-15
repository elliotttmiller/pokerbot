"""
Bet sizing utilities for DeepStack poker AI.
Ported from DeepStack Lua bet_sizing.lua.
"""
import numpy as np

class BetSizing:
    """
    Computes allowed bets during a game, restricted to predefined fractions of the pot.
    """
    def __init__(self, pot_fractions=None, stack=1000, ante=1):
        self.pot_fractions = np.array(pot_fractions) if pot_fractions is not None else np.array([1.0])
        self.stack = stack
        self.ante = ante

    def get_possible_bets(self, node):
        """
        Returns legal bets for the acting player at a game state.
        Args:
            node: dict with 'bets' (list), 'current_player' (int)
        Returns:
            np.ndarray of shape (N, 2): new commitment levels for each player
        """
        current_player = node['current_player']
        assert current_player in [1, 2], 'Wrong player for bet size computation'
        opponent = 3 - current_player
        opponent_bet = node['bets'][opponent-1]
        player_bet = node['bets'][current_player-1]

        assert player_bet <= opponent_bet

        max_raise_size = self.stack - opponent_bet
        min_raise_size = opponent_bet - player_bet
        min_raise_size = max(min_raise_size, self.ante)
        min_raise_size = min(max_raise_size, min_raise_size)

        if min_raise_size == 0:
            return np.zeros((0, 2))
        elif min_raise_size == max_raise_size:
            out = np.full((1, 2), opponent_bet)
            out[0, current_player-1] = opponent_bet + min_raise_size
            return out
        else:
            max_possible_bets_count = len(self.pot_fractions) + 1
            out = np.full((max_possible_bets_count, 2), opponent_bet)
            pot = opponent_bet * 2
            used_bets_count = 0
            for frac in self.pot_fractions:
                raise_size = pot * frac
                if raise_size >= min_raise_size and raise_size < max_raise_size:
                    out[used_bets_count, current_player-1] = opponent_bet + raise_size
                    used_bets_count += 1
            # Add all-in
            out[used_bets_count, current_player-1] = opponent_bet + max_raise_size
            used_bets_count += 1
            return out[:used_bets_count]

# Example usage:
# node = {'bets': [20, 20], 'current_player': 1}
# bet_sizer = BetSizing([0.5, 1.0], stack=100, ante=2)
# bets = bet_sizer.get_possible_bets(node)

def get_pot_sized_bet(pot_size, stack, ante=1):
    """Return a pot-sized bet value."""
    return min(pot_size, stack - ante)
