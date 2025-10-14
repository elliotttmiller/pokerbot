#!/usr/bin/env python3
"""Demonstrate pre-trained champion models and training data."""

from src.utils import initialize_champion_models, TrainingDataManager


def demo_pretrained_models():
    """Show pre-trained models from champion projects."""
    print("\n" + "="*70)
    print(" "*15 + "CHAMPION PRE-TRAINED MODELS DEMO")
    print("="*70 + "\n")
    
    # Initialize champion models and data
    model_loader, data_manager = initialize_champion_models()
    
    print("\n" + "="*70)
    print("DEMONSTRATING TRAINING DATA")
    print("="*70 + "\n")
    
    # Demo 1: Show equity for specific hands
    print("1. Preflop Equity for Specific Hands:")
    test_hands = ['AA', 'AKS', 'AKO', 'QQ', 'JJ', 'AQS', '72O']
    
    for hand in test_hands:
        equity = data_manager.get_hand_equity(hand)
        if equity is not None:
            print(f"   {hand:4s} - {equity:.3f} equity")
        else:
            print(f"   {hand:4s} - Not found")
    
    print("\n2. Top 20% of Starting Hands (Tight Range):")
    top_20_percent = data_manager.get_top_hands(percentile=0.2)
    print(f"   Total hands in top 20%: {len(top_20_percent)}")
    print(f"   Best 15 hands:")
    for i, (hand, equity) in enumerate(top_20_percent[:15], 1):
        print(f"     {i:2d}. {hand:4s} - {equity:.3f}")
    
    print("\n3. Dataset Statistics:")
    stats = data_manager.get_dataset_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*70)
    print("EXPORTING UNIFIED DATASET")
    print("="*70 + "\n")
    
    # Export unified dataset
    output_file = "data/unified_training_data.json"
    data_manager.export_unified_dataset(output_file)
    
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70 + "\n")
    
    # Show available models
    models = model_loader.list_available_models()
    print(f"Available Pre-trained Models: {len(models)}")
    for model in models:
        print(f"\n  Model: {model['name']}")
        print(f"  Path: {model['path']}")
        print(f"  Size: {model['size_mb']:.2f} MB")
    
    # Try to get DeepStack model info
    deepstack_info = model_loader.get_model_info('deepstack')
    if deepstack_info:
        print(f"\nDeepStack Model Details:")
        print(f"  Type: {deepstack_info.get('model_type', 'N/A')}")
        print(f"  Size: {deepstack_info.get('size_mb', 0):.2f} MB")
        print(f"  Path: {deepstack_info.get('model_path', 'N/A')}")
        print(f"\n  This is a champion neural network trained on 50,000+ epochs")
        print(f"  from the DeepStack-Leduc project for value prediction.")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nChampion Models Integrated:")
    print("  ✓ DeepStack pre-trained neural networks (2 models)")
    print("  ✓ Preflop equity tables from dickreuter/Poker (169 hands)")
    print("  ✓ Training data samples from champion projects")
    print("\nReady for:")
    print("  • Training new agents with champion baselines")
    print("  • Fine-tuning on top of pre-trained models")
    print("  • Leveraging proven equity calculations")
    print()


def main():
    """Run the demonstration."""
    demo_pretrained_models()


if __name__ == '__main__':
    main()
