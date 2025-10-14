#!/usr/bin/env python3
"""
Demo: Enhanced Champion Agent with DeepStack and CFR+

This example demonstrates the enhanced training capabilities integrated
from the poker-ai repository:
1. CFR+ algorithm with pruning
2. DeepStack value network
3. Enhanced training statistics

Usage:
    python examples/demo_enhanced_champion.py
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import EnhancedChampionAgent, CFRPlusAgent
from src.game import Card, Rank, Suit


def demo_cfr_plus():
    """Demonstrate CFR+ algorithm improvements."""
    print("\n" + "="*70)
    print("CFR+ ALGORITHM DEMO")
    print("="*70)
    print()
    
    print("CFR+ is an enhanced version of CFR that includes:")
    print("  1. Regret matching+ (negative regrets reset to 0)")
    print("  2. Action pruning (skip actions with very negative regret)")
    print("  3. Linear CFR (discount old iterations)")
    print("  4. Faster convergence to Nash equilibrium")
    print()
    
    # Create CFR+ agent
    cfr_plus = CFRPlusAgent(name="CFR+Demo")
    
    print("Training CFR+ agent for 200 iterations...")
    cfr_plus.train(num_iterations=200)
    print()
    
    # Get statistics
    stats = cfr_plus.get_training_stats()
    print("Training Statistics:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Information sets learned: {stats['infosets']}")
    print(f"  Average regret: {stats['average_regret']:.6f}")
    print(f"  Pruned actions: {stats['pruned_actions']}/{stats['total_actions']} ({stats['prune_percentage']:.1f}%)")
    print()


def demo_deepstack_value_network():
    """Demonstrate DeepStack value network."""
    print("\n" + "="*70)
    print("DEEPSTACK VALUE NETWORK DEMO")
    print("="*70)
    print()
    
    print("DeepStack uses a neural network to estimate counterfactual values.")
    print("This enables real-time decision making via continual re-solving.")
    print()
    
    try:
        from src.agents.deepstack_value_network import build_deepstack_network
        import torch
        
        # Build network
        print("Building DeepStack value network (medium architecture)...")
        value_net = build_deepstack_network(
            bucket_count=169,  # Texas Hold'em hand buckets
            architecture='medium'
        )
        print()
        
        # Generate sample input
        print("Testing value prediction...")
        p1_range = torch.rand(169)  # Random range for player 1
        p2_range = torch.rand(169)  # Random range for player 2
        p1_range = p1_range / p1_range.sum()  # Normalize
        p2_range = p2_range / p2_range.sum()  # Normalize
        pot_size = 100.0
        
        # Predict values
        p1_values, p2_values = value_net.predict_values(p1_range, p2_range, pot_size)
        
        print(f"  Player 1 value estimate: {p1_values.mean().item():.2f} (mean over all hands)")
        print(f"  Player 2 value estimate: {p2_values.mean().item():.2f} (mean over all hands)")
        print(f"  Value network output shape: {p1_values.shape}")
        print()
        
        print("✓ DeepStack value network is functional")
        print()
        
    except ImportError:
        print("⚠ PyTorch not available. DeepStack value network requires PyTorch.")
        print("  Install with: pip install torch")
        print()


def demo_enhanced_champion():
    """Demonstrate enhanced champion agent."""
    print("\n" + "="*70)
    print("ENHANCED CHAMPION AGENT DEMO")
    print("="*70)
    print()
    
    print("The Enhanced Champion Agent integrates:")
    print("  • CFR+ algorithm (faster convergence)")
    print("  • DeepStack value network (better value estimation)")
    print("  • Range-based decision making")
    print("  • Comprehensive training statistics")
    print()
    
    # Create enhanced champion
    print("Creating Enhanced Champion Agent...")
    champion = EnhancedChampionAgent(
        name="EnhancedDemo",
        use_deepstack=True,
        use_cfr_plus=True,
        deepstack_architecture='small',  # Use small for demo
        use_pretrained=False
    )
    print()
    
    # Show agent configuration
    stats = champion.get_enhanced_stats()
    print("Agent Configuration:")
    print(f"  Using CFR+: {stats['use_cfr_plus']}")
    print(f"  Using DeepStack: {stats['use_deepstack']}")
    print(f"  Epsilon: {stats['epsilon']}")
    print()
    
    # Train CFR+ component
    print("Training CFR+ component (100 iterations)...")
    champion.train_cfr_plus(num_iterations=100)
    print()
    
    # Test decision making
    print("Testing decision making with strong hand (Ace-King suited)...")
    hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)]
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
    
    # Show final statistics
    final_stats = champion.get_enhanced_stats()
    print("Final Agent Statistics:")
    print(f"  CFR+ iterations: {final_stats['cfr']['iterations']}")
    print(f"  Information sets: {final_stats['cfr']['infosets']}")
    if 'average_regret' in final_stats['cfr']:
        print(f"  Average regret: {final_stats['cfr']['average_regret']:.6f}")
    print()


def demo_value_network_training():
    """Demonstrate value network training (with synthetic data)."""
    print("\n" + "="*70)
    print("VALUE NETWORK TRAINING DEMO")
    print("="*70)
    print()
    
    print("Training a value network requires counterfactual value data.")
    print("In production, this data comes from CFR solutions.")
    print()
    
    try:
        import torch
        
        # Create enhanced champion
        champion = EnhancedChampionAgent(
            name="TrainingDemo",
            use_deepstack=True,
            use_cfr_plus=False,
            deepstack_architecture='small'
        )
        
        if not champion.use_deepstack:
            print("⚠ DeepStack not available (PyTorch required)")
            return
        
        print("Generating synthetic training data...")
        # Generate synthetic data for demonstration
        n_samples = 1000
        bucket_count = 169
        input_size = 2 * bucket_count + 1
        output_size = 2 * bucket_count
        
        # Random range vectors + pot sizes -> random CFVs
        X_train = np.random.randn(n_samples, input_size).astype(np.float32)
        y_train = np.random.randn(n_samples, output_size).astype(np.float32)
        
        print(f"  Training samples: {n_samples}")
        print(f"  Input dimensions: {input_size}")
        print(f"  Output dimensions: {output_size}")
        print()
        
        # Train for a few epochs
        print("Training value network (20 epochs, demo only)...")
        history = champion.train_value_network(
            training_data=(X_train, y_train),
            epochs=20,
            batch_size=64,
            learning_rate=0.001
        )
        
        if history:
            print(f"  Initial loss: {history[0]:.6f}")
            print(f"  Final loss: {history[-1]:.6f}")
            print(f"  Improvement: {((history[0] - history[-1]) / history[0] * 100):.1f}%")
        print()
        
        print("✓ Value network training complete")
        print("  (In production, train on real CFR-derived counterfactual values)")
        print()
        
    except ImportError:
        print("⚠ PyTorch not available. Value network training requires PyTorch.")
        print()


def main():
    """Run all demonstrations."""
    print("="*70)
    print("ENHANCED CHAMPION AGENT - COMPREHENSIVE DEMO")
    print("="*70)
    print()
    print("This demo showcases advanced components integrated from poker-ai:")
    print("  • CFR+ algorithm with pruning and LCFR")
    print("  • DeepStack value network architecture")
    print("  • Enhanced training capabilities")
    print("  • Comprehensive statistics and monitoring")
    print()
    
    try:
        # Demo 1: CFR+
        demo_cfr_plus()
        
        # Demo 2: DeepStack Value Network
        demo_deepstack_value_network()
        
        # Demo 3: Enhanced Champion Agent
        demo_enhanced_champion()
        
        # Demo 4: Value Network Training
        demo_value_network_training()
        
        # Summary
        print("="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print()
        print("Key Takeaways:")
        print("  ✓ CFR+ converges faster than vanilla CFR")
        print("  ✓ DeepStack value network enables range-based reasoning")
        print("  ✓ Enhanced Champion integrates both seamlessly")
        print("  ✓ Training statistics provide detailed insights")
        print()
        print("Next Steps:")
        print("  1. Train Enhanced Champion with full iterations")
        print("  2. Generate real counterfactual value data from CFR")
        print("  3. Train DeepStack network on that data")
        print("  4. Use trained agent in production")
        print()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
