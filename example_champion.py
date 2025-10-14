#!/usr/bin/env python3
"""Example of using Champion Agent in actual gameplay."""

from src.agents import ChampionAgent, RandomAgent, FixedStrategyAgent
from src.game import Action


def format_cards(cards):
    """Format a list of cards for display."""
    return ' '.join(str(c) for c in cards)


def print_action_result(agent_name, action, raise_amt):
    """Print an agent's action with optional raise amount."""
    print(f"{agent_name}: {action.name}", end="")
    if raise_amt > 0:
        print(f" ${raise_amt}")
    else:
        print()


def play_sample_hand():
    """Play a sample hand with Champion Agent."""
    print("\n" + "="*70)
    print("CHAMPION AGENT IN ACTION - SAMPLE HAND")
    print("="*70 + "\n")
    
    # Initialize agents
    champion = ChampionAgent(name="Champion", use_pretrained=True)
    opponent = FixedStrategyAgent(name="Opponent")
    
    print(f"Players: {champion.name} vs {opponent.name}\n")
    
    # Create sample game state
    from src.game import Card, Rank, Suit, Deck
    
    deck = Deck()
    deck.shuffle()
    
    # Deal hole cards
    champion_cards = deck.deal(2)
    opponent_cards = deck.deal(2)
    
    print(f"{champion.name}'s hole cards: {champion_cards[0]} {champion_cards[1]}")
    print(f"{opponent.name}'s hole cards: {opponent_cards[0]} {opponent_cards[1]}\n")
    
    # Preflop betting
    print("--- PREFLOP ---")
    pot = 30  # Blinds already posted
    print(f"Pot: ${pot}")
    
    # Champion acts first
    action, raise_amt = champion.choose_action(
        hole_cards=champion_cards,
        community_cards=[],
        pot=pot,
        current_bet=20,
        player_stack=1000,
        opponent_bet=20
    )
    print_action_result(champion.name, action, raise_amt)
    if raise_amt > 0:
        pot += raise_amt
    
    # Opponent responds
    opp_action, opp_raise = opponent.choose_action(
        hole_cards=opponent_cards,
        community_cards=[],
        pot=pot,
        current_bet=raise_amt if action == Action.RAISE else 20,
        player_stack=1000,
        opponent_bet=raise_amt if action == Action.RAISE else 20
    )
    print_action_result(opponent.name, opp_action, opp_raise)
    if opp_raise > 0:
        pot += opp_raise
    
    print(f"\nPot after preflop: ${pot}")
    
    # Deal flop
    community_cards = deck.deal(3)
    print(f"\n--- FLOP ---")
    print(f"Community: {format_cards(community_cards)}")
    print(f"Pot: ${pot}")
    
    # More betting...
    action, raise_amt = champion.choose_action(
        hole_cards=champion_cards,
        community_cards=community_cards,
        pot=pot,
        current_bet=0,
        player_stack=900,
        opponent_bet=0
    )
    print_action_result(champion.name, action, raise_amt)
    
    # Deal turn
    community_cards.append(deck.deal(1)[0])
    print(f"\n--- TURN ---")
    print(f"Community: {format_cards(community_cards)}")
    
    action, raise_amt = champion.choose_action(
        hole_cards=champion_cards,
        community_cards=community_cards,
        pot=pot,
        current_bet=0,
        player_stack=850,
        opponent_bet=0
    )
    print_action_result(champion.name, action, raise_amt)
    
    print("\n" + "="*70)
    print("END OF SAMPLE HAND")
    print("="*70 + "\n")
    
    print("This demonstrates how the Champion Agent:")
    print("  • Loads pre-trained models automatically")
    print("  • Makes intelligent decisions based on game state")
    print("  • Combines CFR, DQN, and equity strategies")
    print("  • Adapts to different situations (preflop vs postflop)")
    print()


def compare_agents_head_to_head():
    """Compare Champion Agent vs other agents."""
    print("\n" + "="*70)
    print("HEAD-TO-HEAD COMPARISON")
    print("="*70 + "\n")
    
    # Create agents
    champion = ChampionAgent(name="Champion", use_pretrained=True)
    random_agent = RandomAgent(name="Random")
    fixed_agent = FixedStrategyAgent(name="Fixed")
    
    # Test scenario: Premium hand preflop
    from src.game import Card, Rank, Suit
    hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)]
    community_cards = []
    pot = 100
    current_bet = 20
    player_stack = 1000
    opponent_bet = 20
    
    print(f"Test Scenario: Pocket Aces preflop")
    print(f"Pot: ${pot}, Bet: ${current_bet}, Stack: ${player_stack}\n")
    
    # Get decisions from each agent
    c_action, c_raise = champion.choose_action(
        hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
    )
    
    r_action, r_raise = random_agent.choose_action(
        hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
    )
    
    f_action, f_raise = fixed_agent.choose_action(
        hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
    )
    
    print_action_result("Champion Agent", c_action, c_raise)
    print("  Strategy: CFR + DQN + Equity (40%/40%/20%)")
    print()
    
    print_action_result("Random Agent", r_action, r_raise)
    print("  Strategy: Random selection")
    print()
    
    print_action_result("Fixed Agent", f_action, f_raise)
    print("  Strategy: GTO-inspired with pot odds")
    print()
    
    print("="*70)
    print("The Champion Agent typically makes the most strategic decision")
    print("by combining multiple proven approaches!")
    print("="*70 + "\n")


def demonstrate_training():
    """Demonstrate training capabilities."""
    print("\n" + "="*70)
    print("TRAINING THE CHAMPION AGENT")
    print("="*70 + "\n")
    
    print("The Champion Agent can be trained in two ways:\n")
    
    print("1. CFR Training (Self-Play)")
    print("   " + "-"*60)
    print("   Trains the game-theoretic component through self-play")
    print("   Converges to Nash equilibrium strategies")
    print()
    print("   champion = ChampionAgent(name='Champion', use_pretrained=True)")
    print("   champion.train_cfr(num_iterations=10000)")
    print()
    
    print("2. DQN Training (Reinforcement Learning)")
    print("   " + "-"*60)
    print("   Trains the neural network against opponents")
    print("   Learns patterns and exploitation strategies")
    print()
    print("   from src.evaluation import Trainer")
    print("   from src.agents import RandomAgent")
    print()
    print("   opponent = RandomAgent()")
    print("   trainer = Trainer(champion, opponent)")
    print("   trainer.train(num_episodes=1000, batch_size=32)")
    print()
    
    print("3. Combined Training (Best Results)")
    print("   " + "-"*60)
    print("   Train both components for a truly superior agent")
    print()
    print("   # First: CFR training")
    print("   champion.train_cfr(num_iterations=5000)")
    print()
    print("   # Second: DQN training")
    print("   trainer = Trainer(champion, RandomAgent())")
    print("   trainer.train(num_episodes=500)")
    print()
    print("   # Save the trained agent")
    print("   champion.save_strategy('models/my_champion')")
    print()
    
    print("="*70)
    print("With pre-trained models, you START with 50K+ epochs of knowledge!")
    print("="*70 + "\n")


def main():
    """Run all examples."""
    play_sample_hand()
    compare_agents_head_to_head()
    demonstrate_training()
    
    print("\n" + "="*70)
    print("🏆 CHAMPION AGENT EXAMPLES COMPLETE 🏆")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Run: python demo_champion.py")
    print("  2. Read: CHAMPION_AGENT.md")
    print("  3. Test: python test_champion.py")
    print("  4. Train: Create your own champion variant")
    print()
    print("The Champion Agent is ready to dominate the poker table!")
    print()


if __name__ == '__main__':
    main()
