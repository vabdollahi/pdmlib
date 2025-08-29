"""
Agent configuration models and factory for automatic agent creation.

This module provides Pydantic models and factory functions to support
automatic agent instantiation from JSON configuration specifications.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.utils.logging import get_logger

from .base import Actor
from .heuristic import Heuristic

logger = get_logger("agent_config")


class HeuristicConfig(BaseModel):
    """Configuration parameters for Heuristic agent."""

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
    max_discharge_intensity: float = Field(
        default=0.8,
        description="Maximum battery discharge intensity during high prices",
        ge=0.1,
        le=1.0,
    )
    max_charge_intensity: float = Field(
        default=0.3,
        description="Maximum battery charge intensity during low prices",
        ge=0.1,
        le=1.0,
    )
    price_trend_threshold: float = Field(
        default=0.1,
        description="Price trend threshold for battery decisions (10%)",
        ge=0.01,
        le=0.5,
    )
    strong_trend_threshold: float = Field(
        default=0.15,
        description="Strong price trend threshold for aggressive actions (15%)",
        ge=0.05,
        le=0.5,
    )
    min_solar_generation: float = Field(
        default=0.3,
        description="Minimum solar generation factor during low prices",
        ge=0.1,
        le=1.0,
    )
    default_solar_generation: float = Field(
        default=0.6,
        description="Default solar generation factor for neutral conditions",
        ge=0.1,
        le=1.0,
    )


class AgentConfig(BaseModel):
    """Configuration for automatic agent creation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    type: Literal["Heuristic"] = Field(
        default="Heuristic", description="Type of agent to create"
    )
    parameters: HeuristicConfig = Field(
        default_factory=HeuristicConfig,
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
        if agent_type == "Heuristic":
            # For heuristic, we store config-less initialization
            # The config will be injected during simulation setup
            params = agent_config.parameters.model_dump()

            # Create a placeholder agent that will be properly configured later
            agent = Heuristic(config=None, **params)
            logger.info(f"Successfully created Heuristic with parameters: {params}")
            return agent

        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    except Exception as e:
        logger.error(f"Failed to create agent {agent_type}: {e}")
        raise


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
