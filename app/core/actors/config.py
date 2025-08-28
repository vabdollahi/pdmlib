"""
Agent configuration models and factory for automatic agent creation.

This module provides Pydantic models and factory functions to support
automatic agent instantiation from JSON configuration specifications.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.utils.logging import get_logger

from .base import Actor
from .heuristic import BasicHeuristic

logger = get_logger("agent_config")


class BasicHeuristicConfig(BaseModel):
    """Configuration parameters for BasicHeuristic agent."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    max_lookahead_steps: int = Field(
        default=12,
        description="Maximum number of forecast steps to consider",
        ge=1,
        le=168,  # 1 week at hourly intervals
    )
    charge_threshold_ratio: float = Field(
        default=0.3,
        description="Price percentile below which to charge battery",
        ge=0.0,
        le=1.0,
    )
    discharge_threshold_ratio: float = Field(
        default=0.7,
        description="Price percentile above which to discharge battery",
        ge=0.0,
        le=1.0,
    )
    soc_buffer: float = Field(
        default=0.1,
        description="SOC buffer from min/max limits for battery safety",
        ge=0.0,
        le=0.5,
    )


class AgentConfig(BaseModel):
    """Configuration for automatic agent creation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    type: Literal["BasicHeuristic"] = Field(
        default="BasicHeuristic", description="Type of agent to create"
    )
    parameters: BasicHeuristicConfig = Field(
        default_factory=BasicHeuristicConfig,
        description="Agent-specific configuration parameters",
    )
    enabled: bool = Field(
        default=True, description="Whether the agent is enabled for execution"
    )
    name: Optional[str] = Field(
        default=None, description="Optional name for the agent instance"
    )


def create_agent_from_config(agent_config: AgentConfig) -> Optional[Actor]:
    """
    Create an agent instance from configuration.

    Args:
        agent_config: Agent configuration specification

    Returns:
        Actor: Configured agent instance, or None if disabled

    Raises:
        ValueError: If agent type is not supported
        TypeError: If parameters are invalid for the agent type
    """
    if not agent_config.enabled:
        logger.info(f"Agent {agent_config.type} is disabled, skipping creation")
        return None

    agent_type = agent_config.type
    logger.info(f"Creating agent of type: {agent_type}")

    try:
        if agent_type == "BasicHeuristic":
            # Parameters are always BasicHeuristicConfig due to Pydantic conversion
            params = agent_config.parameters.model_dump()

            agent = BasicHeuristic(**params)
            logger.info(
                f"Successfully created BasicHeuristic with parameters: {params}"
            )
            return agent

        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    except Exception as e:
        logger.error(f"Failed to create agent {agent_type}: {e}")
        raise


def create_default_agent() -> Actor:
    """
    Create a default agent with standard configuration.

    Returns:
        Actor: Default BasicHeuristic agent
    """
    logger.info("Creating default BasicHeuristic agent")
    default_config = AgentConfig(
        type="BasicHeuristic",
        parameters=BasicHeuristicConfig(),
        enabled=True,  # Ensure enabled for default agent
    )
    agent = create_agent_from_config(default_config)
    if agent is None:
        # This should never happen since we ensure enabled=True
        raise RuntimeError("Failed to create default agent")
    return agent


def validate_agent_config(config_dict: Dict[str, Any]) -> AgentConfig:
    """
    Validate and create AgentConfig from dictionary.

    Args:
        config_dict: Dictionary containing agent configuration

    Returns:
        AgentConfig: Validated configuration

    Raises:
        ValidationError: If configuration is invalid
    """
    return AgentConfig(**config_dict)
