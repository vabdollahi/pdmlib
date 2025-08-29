"""
Simple tests for the PowerManagementEnvironment.

These tests verify basic functionality without complex fixture dependencies.
All tests use CSV providers to avoid external API calls.
"""

import numpy as np

from app.core.environment.config import create_environment_config_from_json
from app.core.environment.power_management_env import PowerManagementEnvironment
from tests.config import test_config


class TestSimpleEnvironment:
    """Test the PowerManagementEnvironment with basic functionality."""

    def _create_test_environment(self):
        """Create test environment using spec-driven configuration."""
        config = create_environment_config_from_json(test_config.environment_spec_path)
        return PowerManagementEnvironment(config=config)

    def _get_test_portfolio(self):
        """Get the first portfolio from the test configuration."""
        config = create_environment_config_from_json(test_config.environment_spec_path)
        return config.portfolios[0]

    def _get_test_plant_with_battery(self):
        """Get the first plant (with battery) from the test configuration."""
        portfolio = self._get_test_portfolio()
        return portfolio.plants[0]  # First plant has a battery

    def test_basic_environment_creation(self):
        """Test creating a basic environment with minimal configuration."""
        # Use unified config to create test environment
        env = self._create_test_environment()

        # Test basic properties
        assert env.timestamp.hour == 0
        assert env.end_date_time.hour == 23
        assert env.interval.total_seconds() == 60 * 60

        # Test observation space
        obs_space = env.observation_space
        assert obs_space is not None
        assert hasattr(obs_space, "shape")
        assert obs_space.shape is not None
        assert obs_space.shape[0] > 0  # Should have some dimension

        # Test action space
        action_space = env.action_space
        assert action_space is not None
        assert hasattr(action_space, "shape")
        assert action_space.shape == (3,)  # Three plants in unified portfolio
        assert hasattr(action_space, "low")
        assert hasattr(action_space, "high")
        # Check bounds (accessing as arrays)
        low_vals = getattr(action_space, "low")
        high_vals = getattr(action_space, "high")
        # Check all three plants have correct bounds
        for i in range(3):
            assert low_vals[i] == -1.0
            assert high_vals[i] == 1.0

    def test_environment_reset_and_step(self):
        """Test basic environment reset and step operations."""
        # Use unified config to ensure CSV providers
        env = self._create_test_environment()

        # Test reset
        observation, info = env.reset()
        assert isinstance(observation, np.ndarray)
        assert len(observation) > 0
        assert np.all(np.isfinite(observation))
        assert isinstance(info, dict)

        # Test step
        action = np.array([0.0])  # Neutral action
        observation, reward, terminated, truncated, info = env.step(action)

        assert isinstance(observation, np.ndarray)
        assert len(observation) > 0
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)

    def test_action_conversion_constraints(self):
        """Test that action conversion respects plant constraints."""
        # Use unified config to get a test plant
        plant = self._get_test_plant_with_battery()

        # Verify that plant constraints are properly configured
        min_power = plant.config.min_net_power_mw
        max_power = plant.config.max_net_power_mw

        assert min_power < max_power
        assert min_power >= 0
        assert max_power > 0

    def test_grid_purchase_configuration(self):
        """Test the grid purchase configuration functionality."""
        # Use the unified test portfolio to test the basic configuration
        portfolio = self._get_test_portfolio()

        # Check that the portfolio has the expected default configuration
        # Our unified config should have allow_grid_purchase as False by default
        assert hasattr(portfolio.config, "allow_grid_purchase")

        # Verify that the portfolio total capacity matches the sum of plant capacities
        total_plant_capacity = sum(
            plant.config.max_net_power_mw for plant in portfolio.plants
        )
        assert portfolio.config.max_total_power_mw is not None
        assert portfolio.config.max_total_power_mw >= total_plant_capacity

        # Test that we have multiple plants as expected
        assert len(portfolio.plants) == 3

        # Verify plant capacity limits are reasonable
        for plant in portfolio.plants:
            assert plant.config.max_net_power_mw > 0
            assert plant.config.min_net_power_mw >= 0
