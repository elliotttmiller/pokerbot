
#!/usr/bin/env python3
"""Play poker with automated agent."""

import argparse
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath:
    for p in pythonpath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)
# Fallback: always add src path directly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.agents import create_agent
from src.utils import Config, Logger
from src.vision import ActionMapper, ScreenController, VisionDetector


def main():
    """Main playing function."""
    parser = argparse.ArgumentParser(description='Play poker with automated agent')
    parser.add_argument('--agent', type=str, choices=['dqn', 'fixed'],
                       default='fixed',
                       help='Agent type to use')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained DQN model (for DQN agent)')
    parser.add_argument('--max-hands', type=int, default=20,
                       help='Maximum number of hands to play')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between actions in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger(verbose=args.verbose)
    logger.info("Starting automated poker bot")
    
    # Create config
    config = Config.from_env()
    
    # Check for OpenAI API key
    if not config.openai_api_key:
        logger.warning("No OpenAI API key found. Vision detection will use mock data.")
        logger.warning("Set OPENAI_API_KEY environment variable for actual vision detection.")
    
    # Create screenshot directory
    os.makedirs(config.screenshot_dir, exist_ok=True)
    
    # Create agent
    logger.info(f"Creating agent via factory: {args.agent}")
    if args.agent == 'dqn':
        agent = create_agent('dqn', state_size=60, action_size=3)
        try:
            agent.set_training_mode(False)
        except Exception:
            pass
        if args.model_path:
            logger.info(f"Loading model from {args.model_path}")
            try:
                agent.load_model(args.model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Could not load model: {e}")
                return
        else:
            logger.warning("No model path provided, using untrained DQN agent")
    else:
        agent = create_agent('fixed')
    
    # Create vision detector
    logger.info("Initializing vision detector")
    detector = VisionDetector(api_key=config.openai_api_key)
    
    # Create action mapper
    logger.info("Initializing action mapper")
    mapper = ActionMapper()
    
    # Create screen controller
    controller = ScreenController()
    
    logger.info("\n" + "="*60)
    logger.info("POKER BOT READY")
    logger.info("="*60)
    logger.info(f"Agent: {agent.name}")
    logger.info(f"Max hands: {args.max_hands}")
    logger.info(f"Delay: {args.delay}s")
    logger.info("="*60 + "\n")
    
    # Play loop
    hand_count = 0
    
    try:
        while hand_count < args.max_hands:
            logger.info(f"\n--- Hand {hand_count + 1}/{args.max_hands} ---")
            
            # Wait before taking screenshot
            time.sleep(args.delay)
            
            # Capture screenshot
            screenshot_path = os.path.join(config.screenshot_dir, "current_game.png")
            logger.info("Capturing screenshot...")
            controller.capture_screenshot(screenshot_path)
            
            # Detect game state
            logger.info("Detecting game state...")
            game_state = detector.detect_game_state(screenshot_path)
            
            # Check if game is over
            if game_state.get('is_game_over', False):
                logger.info("Game over detected, waiting for next hand...")
                time.sleep(args.delay * 2)
                continue
            
            # Parse cards
            hole_cards = detector.parse_cards(game_state.get('hole_cards', []))
            community_cards = detector.parse_cards(game_state.get('community_cards', []))
            
            if not hole_cards:
                logger.info("No hole cards detected, waiting...")
                time.sleep(args.delay)
                continue
            
            # Get game state info
            pot = game_state.get('pot_value', 30)
            raised_amounts = game_state.get('raised_amounts', [])
            current_bet = max(raised_amounts) if raised_amounts else 0
            
            logger.info(f"Hole cards: {[str(c) for c in hole_cards]}")
            logger.info(f"Community cards: {[str(c) for c in community_cards]}")
            logger.info(f"Pot: {pot}, Current bet: {current_bet}")
            
            # Choose action
            action, raise_amount = agent.choose_action(
                hole_cards=hole_cards,
                community_cards=community_cards,
                pot=pot,
                current_bet=current_bet,
                player_stack=1000,  # Default stack
                opponent_bet=current_bet
            )
            
            logger.info(f"Action: {action.name}" + 
                       (f" (raise: {raise_amount})" if raise_amount > 0 else ""))
            
            # Execute action
            action_name = action.name.lower()
            mapper.execute_action(action_name, raise_amount)
            
            hand_count += 1
            
            # Wait before next action
            time.sleep(args.delay)
        
        logger.info(f"\nCompleted {hand_count} hands")
        
    except KeyboardInterrupt:
        logger.info("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Error during play: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
