import unittest
from deepstack.nn.mock_nn_terminal import mock_terminal_nn

class TestMockNNTerminal(unittest.TestCase):
    def test_mock_terminal_nn(self):
        self.assertIsNone(mock_terminal_nn())

if __name__ == "__main__":
    unittest.main()
