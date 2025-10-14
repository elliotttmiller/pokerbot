#!/usr/bin/env python3
"""Train a DQN poker agent."""

import argparse
import sys

from src.agents import DQNAgent, RandomAgent
from src.evaluation import Trainer
from src.utils import Config, Logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train a poker bot agent')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save model every N episodes')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--verbose', action='store_true',
                       help='Print training progress')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger(verbose=args.verbose)
    logger.info("Starting poker bot training")
    
    # Create config
    config = Config.from_env()
    config.num_episodes = args.episodes
    config.batch_size = args.batch_size
    config.model_dir = args.model_dir
    
    logger.info(f"Training configuration: {config.to_dict()}")
    
    # Create agents
    logger.info("Initializing DQN agent")
    agent = DQNAgent(
        state_size=60,
        action_size=3,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay
    )
    
    logger.info("Initializing opponent (Random agent)")
    opponent = RandomAgent()
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = Trainer(agent, opponent)
    
    # Start training
    logger.info(f"Training for {args.episodes} episodes")
    try:
        trainer.train(
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            save_dir=args.model_dir,
            verbose=args.verbose
        )
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
