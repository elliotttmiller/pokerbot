"""Configuration management."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for poker bot."""
    
    # Game settings
    starting_stack: int = 1000
    small_blind: int = 10
    big_blind: int = 20
    
    # Training settings
    num_episodes: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    
    # Vision settings
    openai_api_key: Optional[str] = None
    screenshot_dir: str = "screenshots"
    
    # Model settings
    model_dir: str = "models"
    model_path: Optional[str] = None
    
    # Logging
    verbose: bool = True
    log_dir: str = "logs"
    
    # DeepStack settings
    use_lookahead: bool = True  # Enable DeepStack lookahead by default
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        return cls(
            starting_stack=int(os.getenv('STARTING_STACK', '1000')),
            small_blind=int(os.getenv('SMALL_BLIND', '10')),
            big_blind=int(os.getenv('BIG_BLIND', '20')),
            num_episodes=int(os.getenv('NUM_EPISODES', '1000')),
            batch_size=int(os.getenv('BATCH_SIZE', '32')),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            screenshot_dir=os.getenv('SCREENSHOT_DIR', 'screenshots'),
            model_dir=os.getenv('MODEL_DIR', 'models'),
            model_path=os.getenv('MODEL_PATH'),
            verbose=os.getenv('VERBOSE', 'true').lower() == 'true',
            log_dir=os.getenv('LOG_DIR', 'logs'),
            use_lookahead=os.getenv('USE_LOOKAHEAD', 'true').lower() == 'true'
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'starting_stack': self.starting_stack,
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'num_episodes': self.num_episodes,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'screenshot_dir': self.screenshot_dir,
            'model_dir': self.model_dir,
            'model_path': self.model_path,
            'verbose': self.verbose,
            'log_dir': self.log_dir,
            'use_lookahead': self.use_lookahead
        }


@dataclass
class EnhancedConfig(Config):
    """
    Enhanced configuration for modular DeepStack upgrades and feature flags.
    Ensures compatibility, performance, and safe rollout.
    """
    use_deepstack: bool = True
    use_cfr_plus: bool = True
    lookahead_enabled: bool = True
    enable_analysis_report: bool = True
    enable_strategy_visualization: bool = True
    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            'use_deepstack': self.use_deepstack,
            'use_cfr_plus': self.use_cfr_plus,
            'lookahead_enabled': self.lookahead_enabled,
            'enable_analysis_report': self.enable_analysis_report,
            'enable_strategy_visualization': self.enable_strategy_visualization
        })
        return d
