"""Agents package initialization."""

# Only import lightweight core components directly
from .base_agent import BaseAgent
from .agent import create_agent, get_available_agents, register

# Heavy imports are lazy-loaded through create_agent()
# This avoids import errors and improves startup time

__all__ = [
    'BaseAgent',
    'create_agent',
    'get_available_agents',
    'register',
]
