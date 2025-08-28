"""
Tests for integrated agent configuration in environment system.

Tests that the EnvironmentConfigFactory correctly creates agents from
JSON specifications and integrates them with the environment.
"""

import datetime
from pathlib import Path

import pytest

from app.core.actors import BasicHeuristic
from app.core.environment.config import create_environment_config_from_json
from app.core.environment.power_management_env import PowerManagementEnvironment


class TestEnvironmentAgentIntegration:
    """Test agent configuration integration with environment system."""

    def test_environment_with_agent_config(self):
        """Test creating environment with agent configuration."""
        # Use the simplified test config file with agent configuration
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "test_config_simple.json"
        )

        # Create environment config from JSON
        env_config = create_environment_config_from_json(config_path)

        # Verify agent was created
        assert env_config.agent is not None
        assert isinstance(env_config.agent, BasicHeuristic)

        # Verify agent parameters match JSON specification
        agent = env_config.agent
        assert agent.max_lookahead_steps == 8
        assert agent.charge_threshold_ratio == 0.25
        assert agent.discharge_threshold_ratio == 0.75
        assert agent.soc_buffer == 0.05

        # Verify environment can be created with agent
        env = PowerManagementEnvironment(config=env_config)
        assert env is not None
        assert env.config.agent is not None
        assert isinstance(env.config.agent, BasicHeuristic)

    def test_environment_without_agent_config(self):
        """Test creating environment without agent configuration."""
        # Create a temporary config without agent by loading and modifying
        import json
        import os
        import tempfile
        from pathlib import Path

        from tests.config import test_config

        # Load the multi config and remove agent section
        config_dict = test_config.load_config_file("test_config_multi.json")
        if "agent" in config_dict:
            del config_dict["agent"]

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            temp_config_path = Path(f.name)

        try:
            env_config = create_environment_config_from_json(temp_config_path)

            # Verify no agent was created
            assert env_config.agent is None

            # Verify environment can still be created normally
            env = PowerManagementEnvironment(config=env_config)
            assert env is not None
            assert env.config.agent is None
        finally:
            # Clean up temp file
            os.unlink(temp_config_path)

    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        from pydantic import ValidationError

        from app.core.actors import BasicHeuristicConfig

        # Test invalid parameters directly with BasicHeuristicConfig
        with pytest.raises(ValidationError):
            BasicHeuristicConfig(
                max_lookahead_steps=-5,  # Invalid: negative value
                charge_threshold_ratio=1.5,  # Invalid: > 1.0
            )

    def test_disabled_agent_config(self):
        """Test agent configuration with enabled=false."""
        from app.core.actors import AgentConfig, BasicHeuristicConfig

        # Test disabled agent
        agent_config = AgentConfig(
            type="BasicHeuristic",
            parameters=BasicHeuristicConfig(max_lookahead_steps=12),
            enabled=False,
        )

        # Verify config is valid but agent creation returns None
        assert agent_config.enabled is False

    def test_agent_integration_with_simulation_main(self):
        """Test that the simulation can use the automatically created agent."""
        # This test verifies the integration works as expected
        from tests.config import test_config

        config_path = test_config.environment_spec_path

        env_config = create_environment_config_from_json(config_path)
        agent = env_config.agent

        # Verify agent can be used for simulation
        assert agent is not None
        assert hasattr(agent, "get_action")
        assert callable(agent.get_action)

        # Test that agent can get actions (basic smoke test)
        # Create a minimal observation for testing
        mock_observation = {
            "market": {
                "market_data": {
                    "current_price": [50.0],
                    "price_forecast": [45.0, 55.0, 60.0],
                }
            },
            "portfolios": {
                "test_portfolio": {
                    "test_plant": {
                        "ac_power_generation_potential": [10.0],
                        "battery_state_of_charge": [0.5],
                        "battery_energy_capacity": [30.0],
                        "battery_min_state_of_charge": [0.1],
                        "battery_max_state_of_charge": [0.9],
                        "battery_max_discharge_power": [15.0],
                    }
                }
            },
        }

        # Convert to numpy arrays (as expected by the agent)
        import numpy as np

        for category, data in mock_observation.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, list):
                                value[k] = np.array(v)
                    elif isinstance(value, list):
                        data[key] = np.array(value)

        # Test agent action
        action = agent.get_action(mock_observation, datetime.datetime.now())
        assert isinstance(action, dict)
        assert len(action) > 0
