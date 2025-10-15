"""
Unit test for next_round_value.py.
Ported from DeepStack Lua next_round_value_test.lua.
"""
import unittest
import numpy as np
from deepstack.nn.next_round_value import NextRoundValue
from deepstack.nn.mock_nn_terminal import MockNnTerminal

class TestNextRoundValue(unittest.TestCase):
    def test_estimate(self):
        bucket_count = 6
        card_count = 6
        board_count = 2
        nn = MockNnTerminal(bucket_count)
        nrv = NextRoundValue(nn, None, card_count, bucket_count, board_count)
        card_range = np.ones(card_count) / card_count
        value = nrv.estimate(card_range)
        self.assertEqual(value.shape[0], bucket_count * 2)

if __name__ == "__main__":
    unittest.main()
