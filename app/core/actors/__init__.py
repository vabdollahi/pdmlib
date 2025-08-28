"""
Actor system for power management decision making.

This package provides the base Actor class and concrete implementations
like BasicHeuristic for making optimal power management decisions.
"""

from .base import Actor
from .config import (
    AgentConfig,
    BasicHeuristicConfig,
    create_agent_from_config,
    create_default_agent,
)
from .heuristic import BasicHeuristic

__all__ = [
    "Actor",
    "BasicHeuristic",
    "AgentConfig",
    "BasicHeuristicConfig",
    "create_agent_from_config",
    "create_default_agent",
]
