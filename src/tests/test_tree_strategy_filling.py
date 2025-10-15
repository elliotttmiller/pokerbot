import unittest
from deepstack.tree.tree_strategy_filling import fill_strategy

class TestTreeStrategyFilling(unittest.TestCase):
    def test_fill_strategy(self):
        self.assertIsNone(fill_strategy('tree'))

if __name__ == "__main__":
    unittest.main()
