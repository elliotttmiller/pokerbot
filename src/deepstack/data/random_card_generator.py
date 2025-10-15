"""
Random card generator for DeepStack data generation.
Ported from DeepStack Lua random_card_generator.lua.
"""
import random

def generate_random_cards(count, deck_size=52):
    """
    Samples a random set of cards (no duplicates).
    Args:
        count: number of cards to sample
        deck_size: total number of cards in deck
    Returns:
        List of unique card indices
    """
    deck = list(range(1, deck_size+1))
    return random.sample(deck, count)

# Example usage:
# cards = generate_random_cards(5)

def generate_random_card(deck_size=52):
    """Generate a single random card from the deck."""
    return random.randint(1, deck_size)
