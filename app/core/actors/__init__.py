"""
Actor system for power management decision making.

This package provides the base Actor class and concrete implementations
like Heuristic for making optimal power management decisions.
"""

from .base import Actor
from .config import (
    AgentConfig,
    HeuristicConfig,
    create_agent_from_config,
)
from .heuristic import Heuristic

__all__ = [
    "Actor",
    "Heuristic",
    "AgentConfig",
    "HeuristicConfig",
    "create_agent_from_config",
]
