#!/usr/bin/env python3
"""Example: Train and evaluate a DQN poker agent."""

from src.agents import DQNAgent, FixedStrategyAgent, RandomAgent
from src.evaluation import Evaluator, Trainer
from src.utils import Config, Logger


def main():
    """Main example function."""
    # Initialize logger
    logger = Logger(verbose=True)
    logger.info("=== Poker Bot Training Example ===\n")
    
    # Create configuration
    config = Config()
    config.num_episodes = 100  # Short training for demo
    config.batch_size = 32
    
    # Step 1: Create and train DQN agent
    logger.info("Step 1: Creating and training DQN agent")
    dqn_agent = DQNAgent(
        state_size=60,
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Create opponent for training
    opponent = RandomAgent("Training Opponent")
    
    # Create trainer
    trainer = Trainer(dqn_agent, opponent)
    
    # Train
    logger.info(f"Training for {config.num_episodes} episodes...")
    trainer.train(
        num_episodes=config.num_episodes,
        batch_size=config.batch_size,
        save_interval=50,
        verbose=True
    )
    
    # Step 2: Evaluate all agents
    logger.info("\n\nStep 2: Evaluating all agents")
    
    # Set DQN to evaluation mode
    dqn_agent.set_training_mode(False)
    
    # Create other agents
    fixed_agent = FixedStrategyAgent()
    random_agent = RandomAgent()
    
    # Create evaluator
    agents = [dqn_agent, fixed_agent, random_agent]
    evaluator = Evaluator(agents)
    
    # Run evaluation
    logger.info("Running evaluation over 100 hands...")
    results = evaluator.evaluate_agents(num_hands=100, verbose=True)
    
    # Step 3: Head-to-head comparison
    logger.info("\n\nStep 3: Head-to-head - DQN vs Fixed Strategy")
    evaluator.head_to_head(0, 1, num_hands=100)
    
    logger.info("\n\n=== Example Complete ===")


if __name__ == '__main__':
    main()
