"""
Tests for agent configuration system.

Tests automatic agent creation from JSON specifications and
validation of agent configuration models.
"""

import pytest
from pydantic import ValidationError

from app.core.actors import (
    AgentConfig,
    Heuristic,
    HeuristicConfig,
    create_agent_from_config,
)


class TestAgentConfiguration:
    """Test agent configuration models and factory functions."""

    def test_heuristic_config_defaults(self):
        """Test HeuristicConfig with default values."""
        config = HeuristicConfig()

        assert config.max_lookahead_steps == 12
        assert config.charge_threshold_ratio == 0.3
        assert config.discharge_threshold_ratio == 0.7
        assert config.soc_buffer == 0.1

    def test_heuristic_config_custom_values(self):
        """Test HeuristicConfig with custom values."""
        config = HeuristicConfig(
            max_lookahead_steps=24,
            charge_threshold_ratio=0.2,
            discharge_threshold_ratio=0.8,
            soc_buffer=0.05,
        )

        assert config.max_lookahead_steps == 24
        assert config.charge_threshold_ratio == 0.2
        assert config.discharge_threshold_ratio == 0.8
        assert config.soc_buffer == 0.05

    def test_heuristic_config_validation(self):
        """Test HeuristicConfig validation rules."""
        # Test invalid max_lookahead_steps
        with pytest.raises(ValidationError):
            HeuristicConfig(max_lookahead_steps=0)

        with pytest.raises(ValidationError):
            HeuristicConfig(max_lookahead_steps=200)

        # Test invalid charge_threshold_ratio
        with pytest.raises(ValidationError):
            HeuristicConfig(charge_threshold_ratio=-0.1)

        with pytest.raises(ValidationError):
            HeuristicConfig(charge_threshold_ratio=1.1)

        # Test invalid soc_buffer
        with pytest.raises(ValidationError):
            HeuristicConfig(soc_buffer=0.6)

    def test_agent_config_defaults(self):
        """Test AgentConfig with default values."""
        config = AgentConfig()

        assert config.type == "Heuristic"
        assert isinstance(config.parameters, HeuristicConfig)
        assert config.enabled is True
        assert config.name is None

    def test_agent_config_with_dict_parameters(self):
        """Test AgentConfig with dict parameters (automatically converted)."""
        config = AgentConfig(
            type="Heuristic",
            parameters={"max_lookahead_steps": 6, "charge_threshold_ratio": 0.25},
        )

        assert config.type == "Heuristic"
        # Pydantic automatically converts dict to HeuristicConfig
        assert isinstance(config.parameters, HeuristicConfig)
        assert config.parameters.max_lookahead_steps == 6
        assert config.parameters.charge_threshold_ratio == 0.25
        # Check defaults for unspecified parameters
        assert config.parameters.discharge_threshold_ratio == 0.7
        assert config.parameters.soc_buffer == 0.1

    def test_agent_config_with_structured_parameters(self):
        """Test AgentConfig with structured HeuristicConfig."""
        heuristic_config = HeuristicConfig(max_lookahead_steps=18, soc_buffer=0.15)

        config = AgentConfig(
            type="Heuristic",
            parameters=heuristic_config,
            enabled=True,
            name="test_agent",
        )

        assert config.type == "Heuristic"
        assert isinstance(config.parameters, HeuristicConfig)
        assert config.parameters.max_lookahead_steps == 18
        assert config.parameters.soc_buffer == 0.15
        assert config.enabled is True
        assert config.name == "test_agent"

    def test_create_agent_from_config_structured_params(self):
        """Test creating agent with structured parameters."""
        heuristic_config = HeuristicConfig(
            max_lookahead_steps=8,
            charge_threshold_ratio=0.4,
            discharge_threshold_ratio=0.6,
        )

        agent_config = AgentConfig(type="Heuristic", parameters=heuristic_config)

        agent = create_agent_from_config(agent_config)

        assert isinstance(agent, Heuristic)
        assert agent.max_lookahead_steps == 8
        assert agent.charge_threshold_ratio == 0.4
        assert agent.discharge_threshold_ratio == 0.6

    def test_create_agent_from_config_dict_params(self):
        """Test creating agent with dict parameters."""
        agent_config = AgentConfig(
            type="Heuristic",
            parameters={"max_lookahead_steps": 10, "soc_buffer": 0.2},
        )

        agent = create_agent_from_config(agent_config)

        assert isinstance(agent, Heuristic)
        assert agent.max_lookahead_steps == 10
        assert agent.soc_buffer == 0.2
        # Check defaults for unspecified parameters
        assert agent.charge_threshold_ratio == 0.3  # default
        assert agent.discharge_threshold_ratio == 0.7  # default

    def test_create_agent_from_config_disabled(self):
        """Test creating agent when disabled."""
        agent_config = AgentConfig(type="Heuristic", enabled=False)

        agent = create_agent_from_config(agent_config)

        assert agent is None

    def test_create_agent_invalid_parameters(self):
        """Test creating agent with invalid parameters."""
        # Pydantic validates parameters at AgentConfig creation
        with pytest.raises(ValidationError):
            # This will fail validation due to negative max_lookahead_steps
            HeuristicConfig(max_lookahead_steps=-5)

        # Also test validation through AgentConfig
        with pytest.raises(ValidationError):
            # This bypasses type checking by using model_validate
            AgentConfig.model_validate(
                {
                    "type": "Heuristic",
                    "parameters": {
                        "max_lookahead_steps": -5,  # Invalid
                        "charge_threshold_ratio": 0.3,
                    },
                }
            )

    def test_agent_config_from_json_dict(self):
        """Test creating AgentConfig from JSON-like dictionary."""
        json_dict = {
            "type": "Heuristic",
            "parameters": {
                "max_lookahead_steps": 16,
                "charge_threshold_ratio": 0.25,
                "discharge_threshold_ratio": 0.75,
                "soc_buffer": 0.08,
            },
            "enabled": True,
            "name": "production_agent",
        }

        config = AgentConfig(**json_dict)
        agent = create_agent_from_config(config)

        assert isinstance(agent, Heuristic)
        assert agent.max_lookahead_steps == 16
        assert agent.charge_threshold_ratio == 0.25
        assert agent.discharge_threshold_ratio == 0.75
        assert agent.soc_buffer == 0.08
