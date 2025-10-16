#!/usr/bin/env python3
"""Evaluate poker bot agents."""

import argparse

from src.agents import DQNAgent, FixedStrategyAgent, RandomAgent
from src.deepstack.game import Evaluator
from src.utils import Config, Logger


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate poker bot agents')
    parser.add_argument('--num-hands', type=int, default=1000,
                       help='Number of hands to play')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained DQN model')
    parser.add_argument('--agents', nargs='+', 
                       choices=['dqn', 'fixed', 'random'],
                       default=['dqn', 'fixed', 'random'],
                       help='Agents to evaluate')
    parser.add_argument('--verbose', action='store_true',
                       help='Print evaluation progress')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger(verbose=args.verbose)
    logger.info("Starting poker bot evaluation")
    
    # Create config
    config = Config.from_env()
    if args.model_path:
        config.model_path = args.model_path
    
    # Create agents
    agents = []
    
    if 'dqn' in args.agents:
        logger.info("Creating DQN agent")
        dqn_agent = DQNAgent(state_size=60, action_size=3)
        dqn_agent.set_training_mode(False)
        
        if config.model_path:
            logger.info(f"Loading DQN model from {config.model_path}")
            try:
                dqn_agent.load_model(config.model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        else:
            logger.warning("No model path provided, using untrained DQN agent")
        
        agents.append(dqn_agent)
    
    if 'fixed' in args.agents:
        logger.info("Creating Fixed Strategy agent")
        agents.append(FixedStrategyAgent())
    
    if 'random' in args.agents:
        logger.info("Creating Random agent")
        agents.append(RandomAgent())
    
    if len(agents) < 2:
        logger.error("Need at least 2 agents to evaluate")
        return
    
    # Create evaluator
    logger.info(f"Evaluating {len(agents)} agents over {args.num_hands} hands")
    evaluator = Evaluator(agents)
    
    # Run evaluation
    try:
        results = evaluator.evaluate_agents(
            num_hands=args.num_hands,
            verbose=args.verbose
        )
        
        logger.info("\nEvaluation complete!")
        logger.info(f"Results: {results}")
        
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == '__main__':
    main()
