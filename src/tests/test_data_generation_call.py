import unittest
from deepstack.data.data_generation_call import call_data_generation

class TestDataGenerationCall(unittest.TestCase):
    def test_call_data_generation(self):
        self.assertIsNone(call_data_generation())

if __name__ == "__main__":
    unittest.main()
