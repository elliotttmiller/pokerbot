"""
Main entry point for DeepStack Poker neural net training (Python)
Equivalent to main_train.lua
"""
import argparse
from .deepstack_trainer import DeepStackTrainer

def main():
    parser = argparse.ArgumentParser(description='Train DeepStack Poker neural net')
    parser.add_argument('--num-buckets', type=int, default=10)
    parser.add_argument('--data-path', type=str, default='data/deepstacked_training')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[128,128])
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    trainer = DeepStackTrainer(
        num_buckets=args.num_buckets,
        data_path=args.data_path,
        batch_size=args.batch_size,
        hidden_sizes=args.hidden_sizes,
        use_gpu=args.use_gpu,
        lr=args.lr,
        epochs=args.epochs
    )
    trainer.train()

if __name__ == '__main__':
    main()
