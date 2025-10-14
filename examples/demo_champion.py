#!/usr/bin/env python3
"""Demonstrate the Champion Agent - unified CFR + DQN with pre-trained models."""

from src.agents import ChampionAgent, RandomAgent
from src.game import Card, Rank, Suit


def demo_champion_agent():
    """Demonstrate the Champion Agent's capabilities."""
    print("\n" + "="*70)
    print(" "*15 + "🏆 CHAMPION AGENT DEMONSTRATION 🏆")
    print("="*70 + "\n")
    
    print("Initializing Champion Agent...")
    print("-" * 70)
    
    # Create champion agent with pre-trained models
    champion = ChampionAgent(
        name="Champion",
        use_pretrained=True,
        cfr_weight=0.4,  # 40% CFR strategy
        dqn_weight=0.4,  # 40% DQN neural network
        equity_weight=0.2  # 20% equity-based decisions
    )
    
    print("\n" + "="*70)
    print("TESTING CHAMPION AGENT DECISION-MAKING")
    print("="*70 + "\n")
    
    # Test scenario 1: Premium hand preflop
    print("Scenario 1: Premium Hand (Pocket Aces) Preflop")
    print("-" * 70)
    hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)]
    community_cards = []
    pot = 100
    current_bet = 20
    player_stack = 1000
    opponent_bet = 20
    
    print(f"Hole Cards: {hole_cards[0]} {hole_cards[1]}")
    print(f"Community: {community_cards}")
    print(f"Pot: ${pot}, Current Bet: ${current_bet}, Stack: ${player_stack}")
    
    action, raise_amount = champion.choose_action(
        hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
    )
    print(f"Champion's Decision: {action.name}", end="")
    if raise_amount > 0:
        print(f" ${raise_amount}")
    else:
        print()
    print()
    
    # Test scenario 2: Weak hand preflop
    print("Scenario 2: Weak Hand (7-2 offsuit) Preflop")
    print("-" * 70)
    hole_cards = [Card(Rank.SEVEN, Suit.CLUBS), Card(Rank.TWO, Suit.DIAMONDS)]
    community_cards = []
    pot = 100
    current_bet = 50
    player_stack = 1000
    opponent_bet = 50
    
    print(f"Hole Cards: {hole_cards[0]} {hole_cards[1]}")
    print(f"Community: {community_cards}")
    print(f"Pot: ${pot}, Current Bet: ${current_bet}, Stack: ${player_stack}")
    
    action, raise_amount = champion.choose_action(
        hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
    )
    print(f"Champion's Decision: {action.name}", end="")
    if raise_amount > 0:
        print(f" ${raise_amount}")
    else:
        print()
    print()
    
    # Test scenario 3: Good hand on flop
    print("Scenario 3: Top Pair on Flop (AK with Ace on board)")
    print("-" * 70)
    hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
    community_cards = [
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.NINE, Suit.CLUBS),
        Card(Rank.FOUR, Suit.SPADES)
    ]
    pot = 300
    current_bet = 50
    player_stack = 900
    opponent_bet = 50
    
    print(f"Hole Cards: {hole_cards[0]} {hole_cards[1]}")
    print(f"Community: {' '.join(str(c) for c in community_cards)}")
    print(f"Pot: ${pot}, Current Bet: ${current_bet}, Stack: ${player_stack}")
    
    action, raise_amount = champion.choose_action(
        hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
    )
    print(f"Champion's Decision: {action.name}", end="")
    if raise_amount > 0:
        print(f" ${raise_amount}")
    else:
        print()
    print()
    
    # Test scenario 4: Drawing hand
    print("Scenario 4: Flush Draw on Turn")
    print("-" * 70)
    hole_cards = [Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS)]
    community_cards = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.FOUR, Suit.CLUBS),
        Card(Rank.TWO, Suit.SPADES)
    ]
    pot = 500
    current_bet = 100
    player_stack = 800
    opponent_bet = 100
    
    print(f"Hole Cards: {hole_cards[0]} {hole_cards[1]}")
    print(f"Community: {' '.join(str(c) for c in community_cards)}")
    print(f"Pot: ${pot}, Current Bet: ${current_bet}, Stack: ${player_stack}")
    
    action, raise_amount = champion.choose_action(
        hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
    )
    print(f"Champion's Decision: {action.name}", end="")
    if raise_amount > 0:
        print(f" ${raise_amount}")
    else:
        print()
    print()
    
    print("="*70)
    print("CHAMPION AGENT FEATURES")
    print("="*70)
    print()
    print("✓ Unified CFR + DQN Strategy")
    print("  - CFR provides game-theoretic optimal play")
    print("  - DQN adds learned patterns and adaptability")
    print("  - Weighted ensemble combines both approaches")
    print()
    print("✓ Pre-trained Champion Knowledge")
    print("  - DeepStack neural network models (50K+ epochs)")
    print("  - Preflop equity tables (169 starting hands)")
    print("  - Proven champion-level strategies")
    print()
    print("✓ Intelligent Decision Making")
    print("  - Multi-component voting system")
    print("  - Equity-based preflop decisions")
    print("  - Adaptive to game situations")
    print()
    print("✓ Continuous Learning")
    print("  - DQN replay memory and training")
    print("  - CFR regret minimization")
    print("  - Combines exploration with exploitation")
    print()
    
    print("="*70)
    print("SAVING AND LOADING STRATEGIES")
    print("="*70)
    print()
    
    # Demonstrate saving
    print("Saving champion strategy...")
    champion.save_strategy("models/champion_strategy")
    print()
    
    # Demonstrate loading (create new agent)
    print("Loading strategy into new agent...")
    champion2 = ChampionAgent(name="ChampionClone", use_pretrained=True)
    champion2.load_strategy("models/champion_strategy")
    print()
    
    print("="*70)
    print("TRAINING CAPABILITIES")
    print("="*70)
    print()
    print("The Champion Agent can be trained in two ways:")
    print()
    print("1. CFR Training (self-play, game-theoretic optimal)")
    print("   champion.train_cfr(num_iterations=10000)")
    print()
    print("2. DQN Training (reinforcement learning against opponents)")
    print("   Use the standard training framework with champion.replay()")
    print()
    print("Training combines both approaches to create a superior agent!")
    print()
    
    print("="*70)
    print("🏆 CHAMPION AGENT DEMO COMPLETE 🏆")
    print("="*70)
    print()
    print("The Champion Agent represents the pinnacle of our poker AI:")
    print("- Starts with advanced pre-trained knowledge")
    print("- Combines multiple proven strategies")
    print("- Continuously learns and adapts")
    print("- Championship-level decision making from day one!")
    print()


def compare_agents():
    """Compare Champion Agent against basic agents."""
    print("\n" + "="*70)
    print("AGENT COMPARISON")
    print("="*70 + "\n")
    
    # Create agents
    champion = ChampionAgent(name="Champion", use_pretrained=True)
    random_agent = RandomAgent(name="Random")
    
    # Test scenario
    hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
    community_cards = []
    pot = 100
    current_bet = 20
    player_stack = 1000
    opponent_bet = 20
    
    print(f"Test Scenario: {hole_cards[0]} {hole_cards[1]} preflop")
    print(f"Pot: ${pot}, Bet: ${current_bet}\n")
    
    # Champion decision
    action, raise_amt = champion.choose_action(
        hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
    )
    print(f"Champion Agent: {action.name}" + (f" ${raise_amt}" if raise_amt > 0 else ""))
    print("  (Using CFR + DQN + equity tables)")
    
    # Random decision
    action, raise_amt = random_agent.choose_action(
        hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet
    )
    print(f"\nRandom Agent: {action.name}" + (f" ${raise_amt}" if raise_amt > 0 else ""))
    print("  (Random choice)")
    
    print("\n" + "="*70)
    print("The Champion Agent makes informed, strategic decisions based on")
    print("champion-level pre-trained knowledge and game theory!")
    print("="*70 + "\n")


def main():
    """Run all demonstrations."""
    demo_champion_agent()
    compare_agents()


if __name__ == '__main__':
    main()
