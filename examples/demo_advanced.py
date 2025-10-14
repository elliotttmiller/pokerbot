#!/usr/bin/env python3
"""Demonstrate advanced features from world-class poker projects."""

from src.game import Card, GameState, MonteCarloSimulator, Rank, Suit
from src.agents import CFRAgent, FixedStrategyAgent


def demo_monte_carlo():
    """Demonstrate Monte Carlo simulation with opponent ranges."""
    print("="*70)
    print("MONTE CARLO SIMULATION WITH OPPONENT RANGES")
    print("="*70 + "\n")
    
    # Create Monte Carlo simulator
    mc = MonteCarloSimulator()
    
    # Example hand: Player has A♠ K♠
    hole_cards = [
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.KING, Suit.SPADES)
    ]
    
    # Flop: Q♠ J♥ 2♦
    community_cards = [
        Card(Rank.QUEEN, Suit.SPADES),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.TWO, Suit.DIAMONDS)
    ]
    
    print(f"Hole Cards: {', '.join(str(c) for c in hole_cards)}")
    print(f"Community Cards: {', '.join(str(c) for c in community_cards)}")
    print()
    
    # Simulate against random opponent
    print("1. Against Random Opponent (100% range):")
    result = mc.simulate(
        hole_cards=hole_cards,
        community_cards=community_cards,
        num_opponents=1,
        num_simulations=1000,
        opponent_ranges=[1.0]
    )
    
    print(f"   Win Rate: {result['win_rate']:.1%}")
    print(f"   Tie Rate: {result['tie_rate']:.1%}")
    print(f"   Equity: {result['equity']:.1%}")
    print()
    
    # Simulate against tight opponent (top 20%)
    print("2. Against Tight Opponent (20% range):")
    result = mc.simulate(
        hole_cards=hole_cards,
        community_cards=community_cards,
        num_opponents=1,
        num_simulations=1000,
        opponent_ranges=[0.2]
    )
    
    print(f"   Win Rate: {result['win_rate']:.1%}")
    print(f"   Tie Rate: {result['tie_rate']:.1%}")
    print(f"   Equity: {result['equity']:.1%}")
    print()
    
    # Pot odds calculation
    print("3. Pot Odds Analysis:")
    pot = 100
    bet_to_call = 50
    equity = result['equity']
    
    pot_odds = mc.calculate_pot_odds_ev(pot, bet_to_call, equity)
    print(f"   Pot: ${pot}")
    print(f"   Bet to Call: ${bet_to_call}")
    print(f"   Required Equity: {pot_odds['required_equity']:.1%}")
    print(f"   Actual Equity: {pot_odds['actual_equity']:.1%}")
    print(f"   Expected Value: ${pot_odds['ev']:.2f}")
    print(f"   Call Profitable: {pot_odds['profitable']}")
    print()


def demo_cfr_agent():
    """Demonstrate CFR agent training."""
    print("="*70)
    print("COUNTERFACTUAL REGRET MINIMIZATION (CFR) AGENT")
    print("="*70 + "\n")
    
    print("Training CFR agent for 100 iterations...")
    print("(In production, would train for 10,000+ iterations)\n")
    
    # Create CFR agent
    cfr_agent = CFRAgent("CFR_Agent")
    
    # Train (using small number for demo)
    game_state = GameState(num_players=2, starting_stack=1000)
    cfr_agent.train(num_iterations=100, game_state=game_state)
    
    print(f"Training complete!")
    print(f"Total iterations: {cfr_agent.iterations}")
    print(f"Information sets learned: {len(cfr_agent.infosets)}")
    print()
    
    # Demonstrate learned strategy
    print("Example Learned Strategy:")
    if cfr_agent.infosets:
        # Get first infoset
        infoset_key = list(cfr_agent.infosets.keys())[0]
        infoset = cfr_agent.infosets[infoset_key]
        strategy = infoset.get_average_strategy()
        
        print(f"  Infoset: {infoset_key[:50]}...")
        print(f"  Actions: {[a.name for a in infoset.actions]}")
        print(f"  Strategy: {[f'{p:.2%}' for p in strategy]}")
    print()


def demo_comparison():
    """Compare different agent types."""
    print("="*70)
    print("AGENT COMPARISON")
    print("="*70 + "\n")
    
    # Create agents
    fixed_agent = FixedStrategyAgent("Fixed_GTO")
    cfr_agent = CFRAgent("CFR_Trained")
    
    # Quick CFR training
    game_state = GameState(num_players=2)
    cfr_agent.train(num_iterations=50, game_state=game_state)
    
    # Test scenario
    hole_cards = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS)
    ]
    community_cards = [
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.TEN, Suit.DIAMONDS)
    ]
    
    print("Test Scenario:")
    print(f"  Hole Cards: {', '.join(str(c) for c in hole_cards)}")
    print(f"  Community Cards: {', '.join(str(c) for c in community_cards)}")
    print(f"  Pot: $100, Bet to Call: $50")
    print()
    
    # Get decisions from each agent
    print("Agent Decisions:")
    
    action, raise_amt = fixed_agent.choose_action(
        hole_cards, community_cards, 100, 50, 500, 50
    )
    print(f"  1. Fixed Strategy Agent: {action.name}" + 
          (f" (${raise_amt})" if raise_amt > 0 else ""))
    
    action, raise_amt = cfr_agent.choose_action(
        hole_cards, community_cards, 100, 50, 500, 50
    )
    print(f"  2. CFR Agent: {action.name}" +
          (f" (${raise_amt})" if raise_amt > 0 else ""))
    print()


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" "*10 + "ADVANCED POKER BOT FEATURES DEMONSTRATION")
    print(" "*15 + "(From World-Class Projects)")
    print("="*70 + "\n")
    
    demo_monte_carlo()
    demo_cfr_agent()
    demo_comparison()
    
    print("="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nNew Features Integrated:")
    print("  ✓ Monte Carlo simulation with opponent range modeling")
    print("  ✓ Pot odds and EV calculation")
    print("  ✓ Counterfactual Regret Minimization (CFR)")
    print("  ✓ Information set abstraction")
    print("  ✓ Nash equilibrium approximation")
    print("\nNext Steps:")
    print("  • Train CFR agent for 10,000+ iterations")
    print("  • Implement External Sampling MCCFR")
    print("  • Add DeepStack neural network integration")
    print("  • Implement strategy persistence")
    print()


if __name__ == '__main__':
    main()
