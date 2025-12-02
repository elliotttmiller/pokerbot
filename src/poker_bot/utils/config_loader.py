"""
Configuration Loader Utility.

Helper functions for loading and parsing YAML configuration files.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        print(f"[ConfigLoader] Config not found: {config_path}")
        return {}
    
    if not YAML_AVAILABLE:
        print("[ConfigLoader] PyYAML not available, returning empty config")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"[ConfigLoader] Loaded config from {config_path}")
        return config or {}
        
    except Exception as e:
        print(f"[ConfigLoader] Error loading config: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    if not YAML_AVAILABLE:
        print("[ConfigLoader] PyYAML not available, cannot save config")
        return
    
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"[ConfigLoader] Saved config to {config_path}")
        
    except Exception as e:
        print(f"[ConfigLoader] Error saving config: {e}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration dictionary.
    
    Returns:
        Default configuration
    """
    return {
        # Model paths
        'models': {
            'perception': 'models/perception_v1',
            'decision': 'models/decision_v1/value_network.pt'
        },
        
        # Perception settings
        'perception': {
            'device': 'auto',
            'use_quantization': True,
            'api_fallback': True
        },
        
        # Decision settings
        'decision': {
            'cfr_iterations': 1000,
            'lookahead_depth': 3,
            'use_cfr_plus': True,
            'num_buckets': 169
        },
        
        # Action settings
        'action': {
            'simulation_mode': True,
            'action_delay': 0.5,
            'typing_delay': 0.05,
            'click_delay': 0.1
        },
        
        # Screen coordinates (to be calibrated)
        'screen_coordinates': {
            'fold': [0, 0],
            'check': [0, 0],
            'call': [0, 0],
            'raise': [0, 0],
            'bet_input': [0, 0],
            'all_in': [0, 0]
        },
        
        # Main loop settings
        'main_loop': {
            'loop_delay': 0.5,
            'screenshot_interval': 0.1,
            'max_hands': 100
        },
        
        # Logging
        'logging': {
            'level': 'INFO',
            'log_dir': 'logs',
            'log_decisions': True,
            'log_screenshots': False
        }
    }


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


__all__ = ['load_config', 'save_config', 'get_default_config', 'merge_configs']
