import unittest
from deepstack.data.range_generator import generate_range

class TestRangeGenerator(unittest.TestCase):
    def test_generate_range(self):
        self.assertIsNone(generate_range())

if __name__ == "__main__":
    unittest.main()
