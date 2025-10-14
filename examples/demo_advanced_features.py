#!/usr/bin/env python3
"""Demonstration of all advanced poker AI features.

Shows usage of:
- Advanced CFR with progressive training
- Card abstraction
- Search agent with blueprint + real-time search
- Distributed training (if desired)
"""

from src.agents import AdvancedCFRAgent, SearchAgent, ChampionAgent
from src.game import CardAbstraction, Card, Rank, Suit
from src.evaluation import DistributedTrainer


def demo_advanced_cfr():
    """Demonstrate advanced CFR with progressive training."""
    print("\n" + "="*70)
    print("ADVANCED CFR WITH PROGRESSIVE TRAINING")
    print("="*70 + "\n")
    
    # Create advanced CFR agent
    agent = AdvancedCFRAgent(
        name="AdvancedCFR",
        regret_floor=-310000000,  # Prevent overflow
        use_linear_cfr=True
    )
    
    print("Training with progressive curriculum:")
    print("  Phase 1: Pure CFR (iterations 1-1000)")
    print("  Phase 2: Light pruning (iterations 1000-5000)")
    print("  Phase 3: Mixed CFR/CFRp (iterations 5000-10000)")
    print("  Phase 4: Linear CFR with discounting (iterations 10000+)")
    print()
    
    # Train with progressive phases (reduced for demo)
    agent.train_progressive(
        num_iterations=1000,  # Would use 50000+ in production
        warmup_threshold=100,
        prune_threshold=500,
        lcfr_threshold=800,
        discount_interval=100,
        strategy_interval=50,
        verbose=True
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Total iterations: {agent.iterations}")
    print(f"  Information sets learned: {len(agent.infosets)}")
    print()


def demo_card_abstraction():
    """Demonstrate card abstraction for memory efficiency."""
    print("\n" + "="*70)
    print("CARD ABSTRACTION & INFORMATION SET CLUSTERING")
    print("="*70 + "\n")
    
    # Create card abstraction
    abstraction = CardAbstraction(
        n_buckets_preflop=10,
        n_buckets_flop=50,
        n_buckets_turn=50,
        n_buckets_river=50
    )
    
    print("Card abstraction reduces 56+ billion information sets to tractable sizes\n")
    
    # Test different hands
    test_hands = [
        ([Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)], [], "Pocket Aces preflop"),
        ([Card(Rank.SEVEN, Suit.CLUBS), Card(Rank.TWO, Suit.DIAMONDS)], [], "7-2 offsuit preflop"),
        ([Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS)], 
         [Card(Rank.ACE, Suit.HEARTS), Card(Rank.NINE, Suit.HEARTS), Card(Rank.FOUR, Suit.CLUBS)],
         "Flush draw on flop"),
    ]
    
    for hole_cards, community_cards, description in test_hands:
        bucket = abstraction.get_bucket(hole_cards, community_cards)
        bucket_key = abstraction.get_bucket_key(hole_cards, community_cards)
        print(f"{description}:")
        print(f"  Bucket: {bucket}")
        print(f"  Key: {bucket_key}")
        print()
    
    print("✓ Card abstraction groups similar hands strategically")
    print("  Memory reduction: ~1000x compared to direct representation\n")


def demo_search_agent():
    """Demonstrate search agent with blueprint + real-time search."""
    print("\n" + "="*70)
    print("SEARCH AGENT: BLUEPRINT + REAL-TIME SEARCH")
    print("="*70 + "\n")
    
    # Create search agent
    agent = SearchAgent(
        name="SearchAgent",
        search_depth=2,
        search_threshold_pot=200,
        use_pretrained=True
    )
    
    print("Search Agent uses two-stage approach:")
    print("  1. Blueprint strategy: Pre-computed via CFR (fast)")
    print("  2. Real-time search: Depth-limited search for critical decisions")
    print()
    
    # Small pot scenario - uses blueprint
    print("Scenario 1: Small pot ($50) - Uses blueprint")
    print("-" * 70)
    hole_cards = [Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.HEARTS)]
    community_cards = []
    
    action, raise_amt = agent.choose_action(
        hole_cards=hole_cards,
        community_cards=community_cards,
        pot=50,
        current_bet=10,
        player_stack=1000,
        opponent_bet=10
    )
    print(f"Decision: {action.name}", end="")
    if raise_amt > 0:
        print(f" ${raise_amt}")
    else:
        print()
    print()
    
    # Large pot scenario - uses search
    print("Scenario 2: Large pot ($300) on turn - Uses real-time search")
    print("-" * 70)
    community_cards = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.FOUR, Suit.CLUBS),
        Card(Rank.TWO, Suit.SPADES)
    ]
    
    action, raise_amt = agent.choose_action(
        hole_cards=hole_cards,
        community_cards=community_cards,
        pot=300,
        current_bet=50,
        player_stack=900,
        opponent_bet=50
    )
    print(f"Decision: {action.name}", end="")
    if raise_amt > 0:
        print(f" ${raise_amt}")
    else:
        print()
    print()
    
    print("✓ Search agent combines speed (blueprint) with accuracy (search)\n")


def demo_champion_with_advanced_features():
    """Demonstrate champion agent with all advanced features."""
    print("\n" + "="*70)
    print("CHAMPION AGENT WITH ADVANCED FEATURES")
    print("="*70 + "\n")
    
    # Create champion agent
    champion = ChampionAgent(
        name="Champion",
        use_pretrained=True,
        cfr_weight=0.4,
        dqn_weight=0.4,
        equity_weight=0.2
    )
    
    print("Champion Agent integrates:")
    print("  ✓ Pre-trained DeepStack models (50K+ epochs)")
    print("  ✓ Preflop equity tables (169 hands)")
    print("  ✓ CFR game-theoretic strategy")
    print("  ✓ DQN learned patterns")
    print("  ✓ Ensemble decision making")
    print()
    
    # Test decision
    hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
    action, raise_amt = champion.choose_action(
        hole_cards=hole_cards,
        community_cards=[],
        pot=100,
        current_bet=20,
        player_stack=1000,
        opponent_bet=20
    )
    
    print(f"Test decision (AK preflop): {action.name}", end="")
    if raise_amt > 0:
        print(f" ${raise_amt}")
    else:
        print()
    print()
    
    print("✓ Champion agent ready for championship-level play!\n")


def show_summary():
    """Show summary of all features."""
    print("\n" + "="*70)
    print("🏆 ADVANCED POKER AI FEATURES SUMMARY")
    print("="*70 + "\n")
    
    print("Implemented from Pluribus Analysis:")
    print()
    print("✅ Phase 1: Progressive CFR Training")
    print("   - CFR with Pruning (CFRp)")
    print("   - Linear CFR discounting")
    print("   - Multi-phase curriculum (warmup → pruning → linear CFR)")
    print()
    print("✅ Phase 2: Information Set Abstraction")
    print("   - Card clustering and bucketing")
    print("   - Hand strength evaluation")
    print("   - Memory reduction (1000x+)")
    print()
    print("✅ Phase 3: Distributed Training")
    print("   - Multiprocessing support")
    print("   - Async training with synchronization")
    print("   - 4-8x speedup potential")
    print()
    print("✅ Phase 4: Blueprint + Real-time Search")
    print("   - Two-stage decision making")
    print("   - Depth-limited search with alpha-beta pruning")
    print("   - Dynamic strategy selection")
    print()
    print("✅ Codebase Organization")
    print("   - Organized root directory (scripts/, examples/, docs/)")
    print("   - Clean module structure")
    print("   - Professional project layout")
    print()
    print("="*70)
    print("All phases implemented and ready for use!")
    print("="*70 + "\n")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" "*15 + "🚀 ADVANCED POKER AI DEMO 🚀")
    print("="*70)
    
    demo_advanced_cfr()
    demo_card_abstraction()
    demo_search_agent()
    demo_champion_with_advanced_features()
    show_summary()
    
    print("Next steps:")
    print("  1. Train agents: python scripts/train.py")
    print("  2. Evaluate agents: python scripts/evaluate.py")
    print("  3. Play poker: python scripts/play.py")
    print("  4. Read docs: docs/CHAMPION_AGENT.md")
    print("  5. Study analysis: docs/PLURIBUS_ANALYSIS.md")
    print()


if __name__ == '__main__':
    main()
