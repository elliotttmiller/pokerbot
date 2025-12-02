"""
Action module - The "Hands" of the System.

Translates abstract decisions into physical GUI interactions.
"""

from .executor import ActionExecutor, ScreenCoordinates

__all__ = ['ActionExecutor', 'ScreenCoordinates']
