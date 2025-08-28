"""
Tests for the heuristic actor system.

These tests verify that the BasicHeuristic agent can work with
the enhanced observation system.
"""

import numpy as np

from app.core.actors import BasicHeuristic
from app.core.environment.observations import ObservationName
from tests.config import test_config


class TestHeuristicActor:
    """Test the BasicHeuristic actor with the enhanced observation system."""

    def test_basic_heuristic_creation(self):
        """Test creating a BasicHeuristic agent."""
        heuristic = BasicHeuristic(max_lookahead_steps=12)

        assert heuristic.max_lookahead_steps == 12
        assert heuristic.charge_threshold_ratio == 0.3
        assert heuristic.discharge_threshold_ratio == 0.7
        assert heuristic.soc_buffer == 0.1

    def test_heuristic_with_enhanced_observations(self):
        """Test heuristic agent with enhanced observations."""
        # Create test environment with observation factory
        env = test_config.create_test_environment()
        obs_factory = env.observation_factory

        # Create heuristic agent
        heuristic = BasicHeuristic(max_lookahead_steps=6)

        # Get enhanced observations
        enhanced_obs = obs_factory.create_enhanced_observation(env.timestamp)

        # Verify enhanced observation structure
        assert "market" in enhanced_obs
        assert "portfolios" in enhanced_obs
        assert "market_data" in enhanced_obs["market"]

        # Check that market observations contain required fields
        market_data = enhanced_obs["market"]["market_data"]
        assert ObservationName.CURRENT_PRICE in market_data
        assert ObservationName.PRICE_FORECAST in market_data

        # Check portfolio structure
        assert len(enhanced_obs["portfolios"]) > 0
        portfolio_name = list(enhanced_obs["portfolios"].keys())[0]
        portfolio_obs = enhanced_obs["portfolios"][portfolio_name]

        # Check plant observations
        assert len(portfolio_obs) > 0
        plant_name = list(portfolio_obs.keys())[0]
        plant_obs = portfolio_obs[plant_name]

        # Verify plant observations contain enhanced fields
        assert ObservationName.AC_POWER_GENERATION_POTENTIAL in plant_obs
        assert ObservationName.BATTERY_STATE_OF_CHARGE in plant_obs

        # Test heuristic action
        action = heuristic.get_action(enhanced_obs, env.timestamp, obs_factory)

        # Verify action structure
        assert isinstance(action, dict)
        assert portfolio_name in action
        assert plant_name in action[portfolio_name]

        # Verify action contains expected keys
        plant_action = action[portfolio_name][plant_name]
        assert "ac_power_generation_target_mw" in plant_action
        assert "battery_power_target_mw" in plant_action

        # Verify action values are reasonable
        assert isinstance(plant_action["ac_power_generation_target_mw"], (int, float))
        assert isinstance(plant_action["battery_power_target_mw"], (int, float))
        assert plant_action["ac_power_generation_target_mw"] >= 0

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
        assert plant_action["ac_power_generation_target_mw"] == 8.5
        assert plant_action["battery_power_target_mw"] == 0.0

    def test_heuristic_error_handling(self):
        """Test heuristic error handling with malformed observations."""
        # Test with empty observation
        heuristic = BasicHeuristic()

        empty_obs = {"market": {}, "portfolios": {}}
        action = heuristic.get_action(empty_obs)
        assert isinstance(action, dict)
        assert len(action) == 0

        # Test with malformed plant data
        malformed_obs = {
            "market": {
                "market_data": {
                    ObservationName.CURRENT_PRICE: np.array([50.0]),
                }
            },
            "portfolios": {
                "test_portfolio": {
                    "test_plant": "invalid_data"  # Should be dict
                }
            },
        }

        # Should not crash and return default action
        action = heuristic.get_action(malformed_obs)
        assert isinstance(action, dict)
