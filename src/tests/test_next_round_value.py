import unittest
from deepstack.nn.next_round_value import estimate_next_round_value

class TestNextRoundValue(unittest.TestCase):
    def test_estimate_next_round_value(self):
        self.assertIsNone(estimate_next_round_value())

if __name__ == "__main__":
    unittest.main()
