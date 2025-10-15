import unittest
from deepstack.tree.visualization import visualize_tree

class TestVisualization(unittest.TestCase):
    def test_visualize_tree(self):
        self.assertIsNone(visualize_tree('tree'))

if __name__ == "__main__":
    unittest.main()
