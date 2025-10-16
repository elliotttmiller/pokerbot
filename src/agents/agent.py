"""
Agent facade: registry, factory, and convenience re-exports.

This module intentionally stays lightweight. Concrete agents live in
their own modules (cfr_agent.py, advanced_cfr.py, dqn_agent.py, champion_agent.py, etc.).
Use create_agent(kind, **kwargs) to instantiate agents by name.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

import importlib

from .base_agent import BaseAgent
from .meta_agent import MetaAgent  # lightweight


# ---------------------------------------------------------------------------
# Agent registry and factory
# ---------------------------------------------------------------------------

AgentFactory = Dict[str, Callable[..., BaseAgent]]
_REGISTRY: AgentFactory = {}


def register(name: str):
    """Decorator to register an agent class under a given name (case-insensitive)."""

    def _decorator(cls: Type[BaseAgent]):
        _REGISTRY[name.lower()] = cls
        return cls

    return _decorator


def _lazy(module: str, cls: str) -> Callable[..., BaseAgent]:
    def _factory(**kwargs: Any) -> BaseAgent:
        mod = importlib.import_module(module, package=__package__)
        klass = getattr(mod, cls)
        return klass(**kwargs)
    return _factory


# Built-in registrations (lazy import heavy modules)
_REGISTRY.update({
    'cfr': _lazy('.cfr_agent', 'CFRAgent'),
    'advanced_cfr': _lazy('.cfr_agent', 'AdvancedCFRAgent'),
    'dqn': _lazy('.dqn_agent', 'DQNAgent'),
    'champion': _lazy('.champion_agent', 'ChampionAgent'),
    'fixed': _lazy('.fixed_strategy_agent', 'FixedStrategyAgent'),
    'random': _lazy('.random_agent', 'RandomAgent'),
    'search': _lazy('.search_agent', 'SearchAgent'),
    'elite': _lazy('.elite_unified_agent', 'EliteUnifiedAgent'),
    'pokerbot': _lazy('.pokerbot_agent', 'PokerBotAgent'),
})


def get_available_agents() -> List[str]:
    """List available agent kinds registered with the factory."""
    return sorted(_REGISTRY.keys())


def create_agent(kind: str, /, **kwargs: Any) -> BaseAgent:
    """Instantiate an agent by registry name.

    Example:
        agent = create_agent('champion', use_pretrained=True)
    """
    cls = _REGISTRY.get(kind.lower())
    if not cls:
        raise ValueError(f"Unknown agent kind '{kind}'. Available: {', '.join(get_available_agents())}")

    # Special handling: meta agent expects a 'base' agent; allow base_kind shortcut
    if cls is MetaAgent and 'base' not in kwargs:
        base_kind = kwargs.pop('base_kind', 'fixed')
        base_kwargs = kwargs.pop('base_kwargs', {})
        base_agent = create_agent(base_kind, **base_kwargs)
        return MetaAgent(base=base_agent, **kwargs)

    return cls(**kwargs)


__all__ = [
    'BaseAgent',
    'CFRAgent',
    'AdvancedCFRAgent',
    'DQNAgent',
    'ChampionAgent',
    'FixedStrategyAgent',
    'RandomAgent',
    'SearchAgent',
    'MetaAgent',
    'EliteUnifiedAgent',
    'PokerBotAgent',
    'get_available_agents',
    'create_agent',
    'register',
]

