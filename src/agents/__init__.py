"""Agents package initialization."""

# Only import lightweight core components directly
from .base_agent import BaseAgent
from .agent import create_agent, get_available_agents, register


# Unified PokerBot agent (recommended for all new training)
from .pokerbot_agent import PokerBotAgent



# Use create_agent('champion') or create_agent('elite') for legacy agent instantiation.

__all__ = [
    'BaseAgent',
    'create_agent',
    'get_available_agents',
    'register',
    'PokerBotAgent',
]
