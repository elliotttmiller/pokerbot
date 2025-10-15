"""Compatibility shim for AdvancedCFRAgent.

AdvancedCFRAgent has been merged into cfr_agent.AdvancedCFRAgent.
Importing from this module is deprecated and will be removed in a future release.
"""

import warnings

from .cfr_agent import AdvancedCFRAgent  # re-export

warnings.warn(
    "src.agents.advanced_cfr.AdvancedCFRAgent is deprecated; "
    "use src.agents.cfr_agent.AdvancedCFRAgent instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AdvancedCFRAgent"]
