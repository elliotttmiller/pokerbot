import unittest
from deepstack.game.card_to_string_conversion import card_to_string

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

class TestCardToString(unittest.TestCase):
    def test_conversion(self):
        card = Card('K', 'h')
        self.assertEqual(card_to_string(card), 'Kh')

if __name__ == "__main__":
    unittest.main()
