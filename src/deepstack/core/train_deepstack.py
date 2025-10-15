"""
Main entry point for DeepStack Poker neural net training (Python)
Equivalent to main_train.lua
"""

import json
from deepstack.core.deepstack_trainer import DeepStackTrainer

def main():
    # Load training parameters from config file
    with open('scripts/config/training.json', 'r') as f:
        config = json.load(f)

    # Example: Pass config parameters to tree builder and bucketer
    from deepstack.core.tree_builder import PokerTreeBuilder
    from deepstack.utils.bucketer import Bucketer

    tree_builder = PokerTreeBuilder(game_variant='holdem', stack_size=20000)
    tree_params = {
        'street': 0,
        'bets': [config.get('big_blind', 20), config.get('big_blind', 20)],
        'current_player': 1,
        'board': [],
        'bet_sizing': config.get('bet_sizing', [1, 2])
    }
    root = tree_builder.build_tree(tree_params)

    bucketer = Bucketer(bucket_count=config.get('bucket_count', 10))

    trainer = DeepStackTrainer(
        num_buckets=config.get('num_buckets', 10),
        data_path=config.get('data_path', '../data/deepstacked_training/samples/train_samples'),
        batch_size=config.get('batch_size', 32),
        hidden_sizes=config.get('hidden_sizes', [128, 128]),
        activation=config.get('activation', 'relu'),
        bet_sizing=config.get('bet_sizing', [1, 2]),
        bucket_count=config.get('bucket_count', 10),
        use_gpu=config.get('use_gpu', False),
        lr=config.get('lr', 0.001),
        epochs=config.get('epochs', 10)
    )
    trainer.train()

    # Optimization guidance:
    # - To tune parameters like DeepStack Lua, adjust 'num_buckets', 'hidden_sizes', 'lr', and 'epochs' in training.json.
    # - For larger datasets, increase 'epochs' and 'hidden_sizes'.
    # - For faster training, use_gpu: true if you have a compatible GPU.
    # - Match Lua settings for bet sizing, abstraction, and model architecture as needed.

if __name__ == '__main__':
    main()
