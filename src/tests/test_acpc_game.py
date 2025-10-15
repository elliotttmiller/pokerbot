import unittest
from deepstack.protocol.acpc_game import ACPCGame

class TestACPCGame(unittest.TestCase):
    def test_parse_message(self):
        game = ACPCGame()
        self.assertIsNone(game.parse_message('MATCHSTATE:...'))

if __name__ == "__main__":
    unittest.main()
