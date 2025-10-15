import unittest
from deepstack.tree.tree_values import compute_tree_values

class TestTreeValues(unittest.TestCase):
    def test_compute_tree_values(self):
        self.assertIsNone(compute_tree_values('tree'))

if __name__ == "__main__":
    unittest.main()
