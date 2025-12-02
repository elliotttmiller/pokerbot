#!/usr/bin/env python3
"""
Poker AI System - Main Entry Point

The single entry point for the poker bot. Initializes and runs the main loop
that coordinates the Perception Engine, Decision Engine, and Action Executor.

Architecture: "Specialist" System
- Croupier (Perception): VLM-based game state detection
- Strategist (Decision): CFR solver with value network
- Hands (Action): GUI automation for action execution

Usage:
    python main.py                    # Run with default config
    python main.py --config my.yaml   # Run with custom config
    python main.py --simulate         # Run in simulation mode (no GUI)
    python main.py --calibrate        # Run calibration mode

References:
- DeepStack: https://www.deepstack.ai/
- DeepStack-Leduc: https://github.com/lifrordi/DeepStack-Leduc
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from poker_bot.core import GameState, Action, ActionType
from poker_bot.perception import VLMService
from poker_bot.decision import CFRSolver, ValueNetworkWrapper
from poker_bot.action import ActionExecutor
from poker_bot.utils import (
    load_config, get_default_config, merge_configs,
    setup_logger, capture_screen
)


class PokerBot:
    """
    Main orchestrator for the Poker AI system.
    
    Implements the Loop of Dominance:
    1. Sense: Capture screenshot of poker table
    2. Perceive: VLM extracts GameState from pixels
    3. Decide: CFR solver computes optimal Action
    4. Act: Action Executor interacts with GUI
    5. Wait: Pause before next iteration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the poker bot system.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        self.config = get_default_config()
        if config_path:
            user_config = load_config(config_path)
            self.config = merge_configs(self.config, user_config)
        
        # Setup logging
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            name="poker_bot",
            level=log_config.get('level', 'INFO'),
            log_dir=log_config.get('log_dir', 'logs')
        )
        
        self.logger.info("=" * 60)
        self.logger.info("Poker AI System Initializing")
        self.logger.info("=" * 60)
        
        # Initialize components
        self._init_perception()
        self._init_decision()
        self._init_action()
        
        # State tracking
        self.running = False
        self.hands_played = 0
        self.session_start = None
        
        self.logger.info("Poker AI System Ready")
    
    def _init_perception(self):
        """Initialize the Perception Engine (Croupier)."""
        self.logger.info("Initializing Perception Engine (Croupier)...")
        
        perception_config = self.config.get('perception', {})
        model_path = self.config.get('models', {}).get('perception')
        
        self.perception = VLMService(
            model_path=model_path,
            device=perception_config.get('device', 'auto'),
            use_quantization=perception_config.get('use_quantization', True),
            api_fallback=perception_config.get('api_fallback', True)
        )
        
        self.logger.info("Perception Engine ready")
    
    def _init_decision(self):
        """Initialize the Decision Engine (Strategist)."""
        self.logger.info("Initializing Decision Engine (Strategist)...")
        
        decision_config = self.config.get('decision', {})
        model_path = self.config.get('models', {}).get('decision')
        
        # Try to load value network
        self.value_network = None
        if model_path and Path(model_path).exists():
            try:
                self.value_network = ValueNetworkWrapper(
                    model_path=model_path,
                    num_buckets=decision_config.get('num_buckets', 169)
                )
                self.logger.info(f"Loaded value network from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load value network: {e}")
        
        # Initialize CFR solver
        self.decision = CFRSolver(
            num_buckets=decision_config.get('num_buckets', 169),
            value_network=self.value_network,
            lookahead_depth=decision_config.get('lookahead_depth', 3),
            use_cfr_plus=decision_config.get('use_cfr_plus', True)
        )
        
        self.logger.info("Decision Engine ready")
    
    def _init_action(self):
        """Initialize the Action Executor (Hands)."""
        self.logger.info("Initializing Action Executor (Hands)...")
        
        action_config = self.config.get('action', {})
        
        self.action = ActionExecutor(
            action_delay=action_config.get('action_delay', 0.5),
            typing_delay=action_config.get('typing_delay', 0.05),
            click_delay=action_config.get('click_delay', 0.1),
            simulation_mode=action_config.get('simulation_mode', True)
        )
        
        # Load screen coordinates
        coords = self.config.get('screen_coordinates', {})
        self.action.set_coordinates(coords)
        
        self.logger.info("Action Executor ready")
    
    def run(self, max_hands: Optional[int] = None):
        """
        Run the main bot loop.
        
        Args:
            max_hands: Maximum number of hands to play (None for unlimited)
        """
        max_hands = max_hands or self.config.get('main_loop', {}).get('max_hands', 100)
        loop_delay = self.config.get('main_loop', {}).get('loop_delay', 0.5)
        
        self.running = True
        self.session_start = time.time()
        self.hands_played = 0
        
        self.logger.info(f"Starting main loop (max_hands={max_hands})")
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running and (max_hands is None or self.hands_played < max_hands):
                self._run_iteration()
                time.sleep(loop_delay)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self._shutdown()
    
    def _run_iteration(self):
        """Run single iteration of the main loop."""
        try:
            # 1. SENSE: Capture screenshot
            screenshot = self._capture_screenshot()
            if screenshot is None:
                return
            
            # 2. PERCEIVE: Extract game state
            game_state = self._perceive(screenshot)
            if game_state is None:
                return
            
            # Check if action required
            if not game_state.action_required:
                self.logger.debug("No action required, waiting...")
                return
            
            # 3. DECIDE: Compute optimal action
            action = self._decide(game_state)
            
            # 4. ACT: Execute action on GUI
            self._act(action)
            
            # Track statistics
            self.hands_played += 1
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
    
    def _capture_screenshot(self):
        """Capture screenshot of poker table."""
        screenshot = capture_screen()
        if screenshot is None:
            self.logger.warning("Failed to capture screenshot")
        return screenshot
    
    def _perceive(self, screenshot) -> Optional[GameState]:
        """Run perception to extract game state."""
        try:
            game_state = self.perception.get_game_state(screenshot)
            
            self.logger.debug(
                f"Perceived: street={game_state.street.value}, "
                f"pot={game_state.pot_size}, confidence={game_state.confidence:.2f}"
            )
            
            return game_state
            
        except Exception as e:
            self.logger.error(f"Perception error: {e}")
            return None
    
    def _decide(self, game_state: GameState) -> Action:
        """Compute optimal action for game state."""
        try:
            action = self.decision.solve(
                game_state,
                iterations=self.config.get('decision', {}).get('cfr_iterations', 1000)
            )
            
            self.logger.info(
                f"Decision: {action.action_type.value} {action.amount} "
                f"(confidence={action.confidence:.2f})"
            )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Decision error: {e}")
            # Default to call
            return Action(action_type=ActionType.CALL, amount=0)
    
    def _act(self, action: Action):
        """Execute action on poker client."""
        try:
            success = self.action.execute(action)
            if success:
                self.logger.info(f"Executed: {action}")
            else:
                self.logger.warning(f"Failed to execute: {action}")
                
        except Exception as e:
            self.logger.error(f"Action error: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Received shutdown signal")
        self.running = False
    
    def _shutdown(self):
        """Clean shutdown of the system."""
        elapsed = time.time() - self.session_start if self.session_start else 0
        
        self.logger.info("=" * 60)
        self.logger.info("Session Summary")
        self.logger.info(f"  Hands played: {self.hands_played}")
        self.logger.info(f"  Duration: {elapsed:.1f} seconds")
        self.logger.info("=" * 60)
        self.logger.info("Poker AI System Shutdown Complete")
    
    def calibrate(self):
        """Run calibration mode to set screen coordinates."""
        self.logger.info("Starting calibration mode...")
        
        elements = ['fold', 'check', 'call', 'raise', 'bet_input', 'all_in']
        
        for element in elements:
            print(f"\nCalibrating '{element}' button...")
            self.action.calibrate(element)
        
        # Save coordinates to config
        coords = self.action.get_coordinates_dict()
        self.config['screen_coordinates'] = coords
        
        # Optionally save to file
        print("\nCalibration complete!")
        print(f"Coordinates: {coords}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Poker AI System - Professional poker bot using VLM and CFR"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--simulate', '-s',
        action='store_true',
        help='Run in simulation mode (no GUI interaction)'
    )
    
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Run calibration mode to set screen coordinates'
    )
    
    parser.add_argument(
        '--max-hands', '-m',
        type=int,
        default=None,
        help='Maximum number of hands to play'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = PokerBot(config_path=args.config)
    
    # Set simulation mode if requested
    if args.simulate:
        bot.action.set_simulation_mode(True)
    
    # Set verbose logging
    if args.verbose:
        bot.logger.setLevel('DEBUG')
    
    # Run appropriate mode
    if args.calibrate:
        bot.calibrate()
    else:
        bot.run(max_hands=args.max_hands)


if __name__ == '__main__':
    main()
