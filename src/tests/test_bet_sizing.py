import unittest
from deepstack.game.bet_sizing import get_pot_sized_bet

class TestBetSizing(unittest.TestCase):
    def test_pot_sized_bet(self):
        self.assertEqual(get_pot_sized_bet(100), 100)

if __name__ == "__main__":
    unittest.main()
