#!/usr/bin/env python3
"""
Demo: Enhanced Champion Agent with CFR+ and DeepStack

This example demonstrates the enhanced training capabilities integrated
into the ChampionAgent from the poker-ai repository:
1. CFR+ algorithm with pruning (integrated into CFRAgent)
2. DeepStack value network (integrated into ChampionAgent)
3. Consolidated, unified architecture

Usage:
    python examples/demo_champion_enhanced.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import ChampionAgent, CFRAgent
from src.game import Card, Rank, Suit


def demo_cfr_plus():
    """Demonstrate CFR+ enhancements in CFRAgent."""
    print("\n" + "="*70)
    print("CFR+ ENHANCEMENTS DEMO")
    print("="*70)
    print()
    
    print("CFR+ enhancements are now integrated into CFRAgent:")
    print("  • Regret matching+ (negative regrets reset to 0)")
    print("  • Action pruning (skip actions with very negative regret)")
    print("  • Linear CFR (discount old iterations)")
    print()
    
    # Create CFR agent and enable CFR+ features
    print("Creating CFR agent with CFR+ enhancements...")
    cfr = CFRAgent(name="CFRDemo")
    cfr.enable_cfr_plus(
        use_regret_matching_plus=True,
        use_pruning=True,
        use_linear_cfr=True
    )
    print()
    
    # Train with CFR+
    print("Training with CFR+ for 100 iterations...")
    cfr.train_with_cfr_plus(num_iterations=100)
    print()
    
    # Get statistics
    stats = cfr.get_training_stats()
    print("Training Statistics:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Information sets: {stats['infosets']}")
    print(f"  Average regret: {stats['average_regret']:.6f}")
    if 'prune_percentage' in stats:
        print(f"  Pruned actions: {stats['prune_percentage']:.1f}%")
    print()


def demo_enhanced_champion():
    """Demonstrate enhanced ChampionAgent with CFR+ and DeepStack."""
    print("\n" + "="*70)
    print("ENHANCED CHAMPION AGENT DEMO")
    print("="*70)
    print()
    
    print("The enhanced ChampionAgent now supports:")
    print("  • CFR+ enhancements (enable with use_cfr_plus=True)")
    print("  • DeepStack value network (enable with use_deepstack=True)")
    print("  • All features integrated into single agent class")
    print()
    
    # Create enhanced champion with CFR+ only (no PyTorch required)
    print("Creating Enhanced Champion with CFR+ (no PyTorch needed)...")
    champion = ChampionAgent(
        name="EnhancedDemo",
        use_cfr_plus=True,
        use_deepstack=False,  # Disabled to avoid PyTorch requirement
        use_pretrained=False
    )
    print()
    
    # Train CFR+ component
    print("Training CFR+ component (50 iterations)...")
    champion.train_cfr_plus(num_iterations=50)
    print()
    
    # Test decision making
    print("Testing decision making with strong hand (Ace-King)...")
    hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
    community_cards = []
    pot = 100
    current_bet = 20
    player_stack = 1000
    
    action, raise_amt = champion.choose_action(
        hole_cards, community_cards, pot,
        current_bet, player_stack, current_bet
    )
    
    print(f"  Decision: {action}")
    print(f"  Raise amount: {raise_amt}")
    print()
    
    # Show statistics
    stats = champion.get_enhanced_stats()
    print("Agent Statistics:")
    print(f"  CFR+ enabled: {stats['use_cfr_plus']}")
    print(f"  DeepStack enabled: {stats['use_deepstack']}")
    print(f"  CFR iterations: {stats['cfr']['iterations']}")
    print(f"  Information sets: {stats['cfr']['infosets']}")
    print()


def demo_with_deepstack():
    """Demonstrate with DeepStack if PyTorch is available."""
    print("\n" + "="*70)
    print("DEEPSTACK VALUE NETWORK DEMO")
    print("="*70)
    print()
    
    try:
        import torch
        
        print("Creating Champion with DeepStack value network...")
        champion = ChampionAgent(
            name="DeepStackDemo",
            use_cfr_plus=True,
            use_deepstack=True,
            use_pretrained=False
        )
        print()
        
        if champion.use_deepstack and champion.value_network:
            print("Testing DeepStack value estimation...")
            import numpy as np
            
            # Create dummy ranges
            my_range = np.ones(169) / 169
            opp_range = np.ones(169) / 169
            pot_size = 100.0
            
            my_values, opp_values = champion.estimate_hand_values(
                my_range, opp_range, pot_size
            )
            
            print(f"  Value estimate for uniform ranges:")
            print(f"    My expected value: {my_values.mean():.2f}")
            print(f"    Opponent expected value: {opp_values.mean():.2f}")
            print()
            
            print("✓ DeepStack value network is functional")
        else:
            print("⚠ DeepStack initialization failed")
        
    except ImportError:
        print("⚠ PyTorch not available. Install with: pip install torch")
        print("  DeepStack value network requires PyTorch")
    print()


def main():
    """Run all demonstrations."""
    print("="*70)
    print("CONSOLIDATED CHAMPION AGENT - ENHANCED FEATURES DEMO")
    print("="*70)
    print()
    print("This demo showcases enhancements integrated from poker-ai:")
    print("  • CFR+ algorithm (integrated into CFRAgent)")
    print("  • DeepStack value network (integrated into ChampionAgent)")
    print("  • Unified, consolidated architecture")
    print()
    print("All features are now part of the existing agent classes,")
    print("not separate files. Simply enable them with parameters:")
    print("  ChampionAgent(use_cfr_plus=True, use_deepstack=True)")
    print()
    
    try:
        # Demo 1: CFR+ in CFRAgent
        demo_cfr_plus()
        
        # Demo 2: Enhanced Champion with CFR+
        demo_enhanced_champion()
        
        # Demo 3: DeepStack (if PyTorch available)
        demo_with_deepstack()
        
        # Summary
        print("="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print()
        print("Key Takeaways:")
        print("  ✓ CFR+ enhancements integrated into CFRAgent")
        print("  ✓ DeepStack value network integrated into ChampionAgent")
        print("  ✓ Enable features with simple parameters")
        print("  ✓ No separate files needed - clean, unified codebase")
        print()
        print("Usage:")
        print("  # Standard champion")
        print("  agent = ChampionAgent()")
        print()
        print("  # Enhanced champion with CFR+")
        print("  agent = ChampionAgent(use_cfr_plus=True)")
        print()
        print("  # Fully enhanced (requires PyTorch)")
        print("  agent = ChampionAgent(use_cfr_plus=True, use_deepstack=True)")
        print()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
