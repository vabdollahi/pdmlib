"""
Tests for the heuristic actor system.

These tests verify that the BasicHeuristic agent can work with
the observation system.
"""

import datetime

import numpy as np
import pytest

from app.core.actors import BasicHeuristic
from app.core.environment.actions import ActionName
from app.core.environment.observations import ObservationName
from tests.config import test_config


class TestHeuristicActor:
    """Test the BasicHeuristic actor with the observation system."""

    def test_basic_heuristic_creation(self):
        """Test creating a BasicHeuristic agent."""
        heuristic = BasicHeuristic(max_lookahead_steps=12)

        assert heuristic.max_lookahead_steps == 12
        assert heuristic.charge_threshold_ratio == 0.3
        assert heuristic.discharge_threshold_ratio == 0.7
        assert heuristic.soc_buffer == 0.1

    @pytest.mark.asyncio
    async def test_heuristic_with_observations(self):
        """Test heuristic agent with observations."""
        # Get observation factory from environment using spec-driven config
        from app.core.environment.config import create_environment_config_from_json
        from app.core.environment.observations import ObservationFactory

        config = create_environment_config_from_json(test_config.environment_spec_path)
        portfolios = config.portfolios

        obs_factory = ObservationFactory(
            portfolios=portfolios,
            historic_data_intervals=6,
            forecast_data_intervals=12,
        )

        # Create heuristic agent
        heuristic = BasicHeuristic(max_lookahead_steps=6)

        # Get current timestamp
        timestamp = datetime.datetime.now(datetime.timezone.utc)

        # Get observations
        observation = await obs_factory.create_observation(timestamp)

        # Verify observation structure
        assert "market" in observation
        assert "portfolios" in observation
        assert "market_data" in observation["market"]

        # Check that market observations contain required fields
        market_data = observation["market"]["market_data"]
        assert ObservationName.CURRENT_PRICE in market_data
        assert ObservationName.PRICE_FORECAST in market_data

        # Check portfolio structure
        assert len(observation["portfolios"]) > 0
        portfolio_name = list(observation["portfolios"].keys())[0]
        portfolio_obs = observation["portfolios"][portfolio_name]

        # Check plant observations
        assert len(portfolio_obs) > 0
        plant_name = list(portfolio_obs.keys())[0]
        plant_obs = portfolio_obs[plant_name]

        # Verify plant observations contain fields
        assert ObservationName.AC_POWER_GENERATION_POTENTIAL in plant_obs
        assert ObservationName.BATTERY_STATE_OF_CHARGE in plant_obs

        # Test heuristic action
        action = heuristic.get_action(observation, timestamp)

        # Verify action structure
        assert isinstance(action, dict)
        assert portfolio_name in action
        assert plant_name in action[portfolio_name]

        # Verify action contains expected keys
        plant_action = action[portfolio_name][plant_name]
        assert ActionName.AC_POWER_GENERATION_TARGET.value in plant_action
        assert ActionName.BATTERY_POWER_TARGET.value in plant_action

        # Verify action values are reasonable
        ac_target = plant_action[ActionName.AC_POWER_GENERATION_TARGET.value]
        battery_target = plant_action[ActionName.BATTERY_POWER_TARGET.value]
        assert isinstance(ac_target, (int, float))
        assert isinstance(battery_target, (int, float))
        assert ac_target >= 0

    def test_heuristic_battery_logic(self):
        """Test heuristic battery charging/discharging logic."""
        heuristic = BasicHeuristic(
            max_lookahead_steps=4,
            charge_threshold_ratio=0.3,
            discharge_threshold_ratio=0.7,
        )

        # Test battery action with low price (should charge)
        battery_action_low = heuristic._determine_battery_action(
            current_price=20.0,
            price_forecast=np.array([30.0, 40.0, 50.0, 60.0]),
            current_soc=0.5,
            min_soc=0.2,
            max_soc=0.8,
            max_battery_power=5.0,
        )
        assert battery_action_low < 0  # Negative = charging

        # Test battery action with high price (should discharge)
        battery_action_high = heuristic._determine_battery_action(
            current_price=80.0,
            price_forecast=np.array([30.0, 40.0, 50.0, 60.0]),
            current_soc=0.5,
            min_soc=0.2,
            max_soc=0.8,
            max_battery_power=5.0,
        )
        assert battery_action_high > 0  # Positive = discharging

        # Test SOC constraints (low SOC should force charging)
        battery_action_low_soc = heuristic._determine_battery_action(
            current_price=80.0,  # High price
            price_forecast=np.array([30.0, 40.0, 50.0, 60.0]),
            current_soc=0.15,  # Very low SOC
            min_soc=0.2,
            max_soc=0.8,
            max_battery_power=5.0,
        )
        assert battery_action_low_soc < 0  # Should charge despite high price

        # Test SOC constraints (high SOC should force discharging)
        battery_action_high_soc = heuristic._determine_battery_action(
            current_price=20.0,  # Low price
            price_forecast=np.array([30.0, 40.0, 50.0, 60.0]),
            current_soc=0.85,  # Very high SOC
            min_soc=0.2,
            max_soc=0.8,
            max_battery_power=5.0,
        )
        assert battery_action_high_soc > 0  # Should discharge despite low price

    def test_heuristic_no_battery_case(self):
        """Test heuristic behavior when plant has no battery."""
        # Create a mock observation for a plant without battery
        mock_observation = {
            "market": {
                "market_data": {
                    ObservationName.CURRENT_PRICE: np.array([50.0]),
                    ObservationName.PRICE_FORECAST: np.array([]),
                }
            },
            "portfolios": {
                "test_portfolio": {
                    "test_plant": {
                        ObservationName.AC_POWER_GENERATION_POTENTIAL: np.array([8.5]),
                        ObservationName.BATTERY_ENERGY_CAPACITY: np.array([0.0]),
                        ObservationName.BATTERY_STATE_OF_CHARGE: np.array([0.0]),
                        ObservationName.BATTERY_MIN_STATE_OF_CHARGE: np.array([0.0]),
                        ObservationName.BATTERY_MAX_STATE_OF_CHARGE: np.array([0.0]),
                        ObservationName.BATTERY_MAX_DISCHARGE_POWER: np.array([0.0]),
                    }
                }
            },
        }

        heuristic = BasicHeuristic()
        action = heuristic.get_action(mock_observation)

        # Should generate at full potential with no battery action
        plant_action = action["test_portfolio"]["test_plant"]
        ac_target = plant_action[ActionName.AC_POWER_GENERATION_TARGET.value]
        battery_target = plant_action[ActionName.BATTERY_POWER_TARGET.value]
        assert ac_target == 8.5
        assert battery_target == 0.0

    def test_heuristic_error_handling(self):
        """Test heuristic error handling with malformed observations."""
        # Test with empty observation
        heuristic = BasicHeuristic()

        empty_obs = {"market": {}, "portfolios": {}}
        action = heuristic.get_action(empty_obs)
        assert isinstance(action, dict)
        assert len(action) == 0
