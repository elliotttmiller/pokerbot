import unittest
from deepstack.data.main_data_generation import main

class TestMainDataGeneration(unittest.TestCase):
    def test_main(self):
        self.assertIsNone(main())

if __name__ == "__main__":
    unittest.main()
