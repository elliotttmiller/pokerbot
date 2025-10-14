#!/usr/bin/env python3
"""Simple demo of the poker bot system."""

from src.game import Card, Rank, Suit, HandEvaluator, GameState, Action
from src.agents import RandomAgent, FixedStrategyAgent


def print_separator():
    """Print a visual separator."""
    print("\n" + "="*60 + "\n")


def demo_cards():
    """Demonstrate card system."""
    print("DEMO 1: Card System")
    print_separator()
    
    # Create cards
    card1 = Card(Rank.ACE, Suit.SPADES)
    card2 = Card(Rank.KING, Suit.HEARTS)
    card3 = Card.from_string("Q-D")
    
    print(f"Card 1: {card1} (Ace of Spades)")
    print(f"Card 2: {card2} (King of Hearts)")
    print(f"Card 3: {card3} (Queen of Diamonds from string)")
    
    print_separator()


def demo_hand_evaluation():
    """Demonstrate hand evaluation."""
    print("DEMO 2: Hand Evaluation")
    print_separator()
    
    # Royal Flush
    royal_flush = [
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.KING, Suit.SPADES),
        Card(Rank.QUEEN, Suit.SPADES),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.TEN, Suit.SPADES)
    ]
    
    print("Hand: " + ", ".join(str(c) for c in royal_flush))
    rank, _ = HandEvaluator.evaluate_hand(royal_flush)
    print(f"Evaluation: {rank.name}\n")
    
    # Full House
    full_house = [
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.KING, Suit.SPADES)
    ]
    
    print("Hand: " + ", ".join(str(c) for c in full_house))
    rank, _ = HandEvaluator.evaluate_hand(full_house)
    print(f"Evaluation: {rank.name}\n")
    
    # Pair
    pair = [
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.DIAMONDS),
        Card(Rank.QUEEN, Suit.CLUBS),
        Card(Rank.JACK, Suit.SPADES)
    ]
    
    print("Hand: " + ", ".join(str(c) for c in pair))
    rank, _ = HandEvaluator.evaluate_hand(pair)
    print(f"Evaluation: {rank.name}")
    
    print_separator()


def demo_game():
    """Demonstrate a poker game."""
    print("DEMO 3: Poker Game Simulation")
    print_separator()
    
    # Create game
    game = GameState(num_players=2, starting_stack=1000, small_blind=10, big_blind=20)
    game.reset()
    
    print(f"Starting new hand...")
    print(f"Pot: ${game.pot}")
    print(f"Current bet: ${game.current_bet}\n")
    
    # Show player hands
    for i, player in enumerate(game.players):
        print(f"{player.name}:")
        print(f"  Hand: {', '.join(str(c) for c in player.hand)}")
        print(f"  Stack: ${player.stack}")
        print(f"  Current bet: ${player.current_bet}\n")
    
    print_separator()


def demo_agents():
    """Demonstrate AI agents."""
    print("DEMO 4: AI Agents")
    print_separator()
    
    # Create game
    game = GameState(num_players=2)
    game.reset()
    
    # Create agents
    random_agent = RandomAgent()
    fixed_agent = FixedStrategyAgent()
    
    player = game.players[0]
    
    # Show game state
    print("Game State:")
    print(f"  Hole Cards: {', '.join(str(c) for c in player.hand)}")
    print(f"  Community Cards: {', '.join(str(c) for c in game.community_cards) or 'None (Pre-flop)'}")
    print(f"  Pot: ${game.pot}")
    print(f"  Current Bet: ${game.current_bet}\n")
    
    # Get agent decisions
    print("Agent Decisions:")
    
    action, raise_amt = random_agent.choose_action(
        hole_cards=player.hand,
        community_cards=game.community_cards,
        pot=game.pot,
        current_bet=game.current_bet,
        player_stack=player.stack,
        opponent_bet=game.current_bet
    )
    print(f"  Random Agent: {action.name}" + (f" (${raise_amt})" if raise_amt > 0 else ""))
    
    action, raise_amt = fixed_agent.choose_action(
        hole_cards=player.hand,
        community_cards=game.community_cards,
        pot=game.pot,
        current_bet=game.current_bet,
        player_stack=player.stack,
        opponent_bet=game.current_bet
    )
    print(f"  Fixed Strategy Agent: {action.name}" + (f" (${raise_amt})" if raise_amt > 0 else ""))
    
    print_separator()


def demo_full_hand():
    """Demonstrate a complete hand."""
    print("DEMO 5: Complete Hand Simulation")
    print_separator()
    
    # Create game with agents
    game = GameState(num_players=2, starting_stack=1000)
    game.reset()
    
    fixed_agent = FixedStrategyAgent("Player 1")
    random_agent = RandomAgent("Player 2")
    agents = [fixed_agent, random_agent]
    
    print("Starting hand between Fixed Strategy and Random Agent\n")
    
    # Show initial state
    print("Initial Deal:")
    for i, player in enumerate(game.players):
        print(f"  {agents[i].name}: {', '.join(str(c) for c in player.hand)} (${player.stack})")
    print(f"  Pot: ${game.pot}\n")
    
    # Pre-flop action
    print("Pre-flop:")
    for i in range(2):
        player = game.players[i]
        if not player.folded:
            action, raise_amt = agents[i].choose_action(
                hole_cards=player.hand,
                community_cards=game.community_cards,
                pot=game.pot,
                current_bet=game.current_bet - player.current_bet,
                player_stack=player.stack,
                opponent_bet=game.current_bet
            )
            game.apply_action(i, action, raise_amt)
            print(f"  {agents[i].name}: {action.name}" + (f" (${raise_amt})" if raise_amt > 0 else ""))
    
    # Check if anyone folded
    active_players = [p for p in game.players if not p.folded]
    if len(active_players) == 1:
        winner_idx = [i for i, p in enumerate(game.players) if not p.folded][0]
        print(f"\n{agents[winner_idx].name} wins ${game.pot}!")
    else:
        # Advance to flop
        game.advance_betting_round()
        print(f"\nFlop: {', '.join(str(c) for c in game.community_cards)}")
        print(f"Pot: ${game.pot}")
    
    print_separator()


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("         POKER BOT SYSTEM DEMONSTRATION")
    print("="*60)
    
    demo_cards()
    demo_hand_evaluation()
    demo_game()
    demo_agents()
    demo_full_hand()
    
    print("\n" + "="*60)
    print("         DEMO COMPLETE")
    print("="*60)
    print("\nFor more examples, see:")
    print("  - example.py: Full training and evaluation example")
    print("  - test.py: Comprehensive test suite")
    print("  - QUICKSTART.md: Quick start guide\n")


if __name__ == '__main__':
    main()
