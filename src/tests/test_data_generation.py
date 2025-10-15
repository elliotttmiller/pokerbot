import unittest
from deepstack.data.data_generation import generate_training_data

class TestDataGeneration(unittest.TestCase):
    def test_generate_training_data(self):
        self.assertIsNone(generate_training_data())

if __name__ == "__main__":
    unittest.main()
