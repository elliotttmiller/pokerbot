"""
Next round value estimation for DeepStack neural network pipeline.
Ported from DeepStack Lua next_round_value.lua.
"""
import numpy as np

class NextRoundValue:
    def __init__(self, nn, bucketer, card_count, bucket_count, board_count):
        self.nn = nn
        self.bucketer = bucketer
        self.bucket_count = bucket_count
        self.card_count = card_count
        self.board_count = board_count
        self._range_matrix = np.zeros((card_count, board_count * bucket_count))
        self._reverse_value_matrix = np.zeros((bucket_count * board_count, card_count))
        # Fill matrices with dummy values for demonstration
        for c in range(card_count):
            for b in range(bucket_count * board_count):
                self._range_matrix[c, b] = 1.0 / (bucket_count * board_count)
                self._reverse_value_matrix[b, c] = 1.0 / card_count

    def card_range_to_bucket_range(self, card_range):
        return np.dot(card_range, self._range_matrix)

    def bucket_value_to_card_value(self, bucket_value):
        return np.dot(bucket_value, self._reverse_value_matrix)

    def estimate(self, card_range):
        bucket_range = self.card_range_to_bucket_range(card_range)
        return self.nn.get_value(bucket_range)


def estimate_next_round_value(card_range, nn, bucketer):
    """Estimate the next round value using the NN and bucketer."""
    next_value = NextRoundValue(nn, bucketer, len(card_range), bucketer.bucket_count, 0)
    return next_value.estimate(card_range)
