import unittest
from deepstack.data.random_card_generator import generate_random_card

class TestRandomCardGenerator(unittest.TestCase):
    def test_generate_random_card(self):
        card = generate_random_card()
        self.assertIn(card, ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'])

if __name__ == "__main__":
    unittest.main()
