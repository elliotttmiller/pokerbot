"""
Bucket conversion utilities for DeepStack neural network pipeline.
Ported from DeepStack Lua bucket_conversion.lua.
"""
import numpy as np

class BucketConversion:
    def __init__(self, bucketer, card_count, bucket_count):
        self.bucketer = bucketer
        self.bucket_count = bucket_count
        self.card_count = card_count
        self._range_matrix = np.zeros((card_count, bucket_count))
        self._reverse_value_matrix = np.zeros((bucket_count, card_count))

    def set_board(self, board):
        buckets = self.bucketer.compute_buckets(board)
        for c in range(self.card_count):
            b = buckets[c]
            self._range_matrix[c, b] = 1
            self._reverse_value_matrix[b, c] = 1

    def card_range_to_bucket_range(self, card_range):
        return np.dot(card_range, self._range_matrix)

    def bucket_value_to_card_value(self, bucket_value):
        return np.dot(bucket_value, self._reverse_value_matrix)

    def get_possible_bucket_mask(self):
        card_indicator = np.ones(self.card_count)
        mask = np.dot(card_indicator, self._range_matrix)
        return mask


def convert_to_bucket(card_range, bucketer):
    """Convert a card range to bucket range using the provided bucketer."""
    return bucketer.card_range_to_bucket_range(card_range)
