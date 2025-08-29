"""
Tests for the heuristic actor system.

These tests verify that the BasicHeuristic agent can work with
the observation system.
"""

import datetime

import numpy as np
import pytest

from app.core.actors import Heuristic
from app.core.simulation.price_provider import CSVPriceProvider
from tests.config import test_config


class TestHeuristicActor:
    """Test the Heuristic actor with the observation system."""

    def test_basic_heuristic_creation(self):
        """Test creating a Heuristic agent."""
        heuristic = Heuristic(max_lookahead_steps=12)

        assert heuristic.max_lookahead_steps == 12
        assert heuristic.charge_threshold_ratio == 0.3
        assert heuristic.discharge_threshold_ratio == 0.7
        assert heuristic.soc_buffer == 0.1

    @pytest.mark.asyncio
    async def test_heuristic_with_observations(self, test_data_dir):
        """Test heuristic agent with observations."""
        from app.core.environment.observations import ObservationFactory
        from app.core.environment.power_management_env import PowerManagementEnvironment
        from app.core.simulation.simulation_manager import MarketData

        env = PowerManagementEnvironment.from_json(test_config.environment_spec_path)
        config = env.config
        portfolios = config.portfolios

        # Create a price provider for the observation factory
        price_provider = CSVPriceProvider(
            csv_file_path=test_data_dir / "sample_price_data.csv"
        )
        market_data = MarketData(price_providers={"default": price_provider})
        await market_data.load_market_data(config.start_date_time, config.end_date_time)

        obs_factory = ObservationFactory(
            portfolios=portfolios,
            historic_data_intervals=6,
            forecast_data_intervals=12,
            market_data=market_data,
        )
        heuristic = Heuristic(max_lookahead_steps=6)
        heuristic.configure(config)  # Configure agent with env spec

        timestamp = config.start_date_time + datetime.timedelta(hours=4)
        observation = await obs_factory.create_observation(timestamp)

        assert "market" in observation
        assert "portfolios" in observation
        assert "market_data" in observation["market"]

        market_data = observation["market"]["market_data"]
        assert "current_price_dollar_mwh" in market_data
        assert "price_forecast_dollar_mwh" in market_data

        assert len(observation["portfolios"]) > 0
        portfolio_name = list(observation["portfolios"].keys())[0]
        portfolio_obs = observation["portfolios"][portfolio_name]

        plant_name = list(portfolio_obs.keys())[0]
        plant_obs = portfolio_obs[plant_name]

        assert "ac_power_generation_potential_mw" in plant_obs
        assert "battery_state_of_charge" in plant_obs

        action = heuristic.get_action(observation)

        assert isinstance(action, np.ndarray)
        assert action.shape == (heuristic.num_plants,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_heuristic_optimal_action_logic(self):
        """Test the core logic of the _get_optimal_action method."""
        heuristic = Heuristic()
        # Normalized values for testing
        ac_potential = 0.8
        soc = 0.5
        prices = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        history = prices[:2]
        forecast = prices[3:]

        # Scenario 1: Low current price, rising forecast -> Charge
        # current_price is at 20th percentile, forecast is high
        action_charge = heuristic._get_optimal_action(
            0.18, forecast, history, ac_potential, soc
        )
        assert action_charge < ac_potential  # Should charge

        # Scenario 2: High current price -> Discharge
        # current_price is at 95th percentile
        action_discharge = heuristic._get_optimal_action(
            0.88, forecast, history, ac_potential, soc
        )
        assert action_discharge > 0  # Should discharge

        # Scenario 3: Medium price, neutral forecast -> Generate from solar
        action_hold = heuristic._get_optimal_action(
            0.48, np.array([0.5, 0.51]), history, ac_potential, soc
        )
        assert action_hold > 0
        assert abs(action_hold - ac_potential) < 0.1

        # Scenario 4: High price but low SOC -> Limit discharge
        action_low_soc = heuristic._get_optimal_action(
            0.9, forecast, history, ac_potential, battery_soc=0.1
        )
        assert action_low_soc >= 0  # Should not charge, but discharge is limited

        # Scenario 5: Low price but high SOC -> Limit charge
        action_high_soc = heuristic._get_optimal_action(
            0.1, forecast, history, ac_potential, battery_soc=0.95
        )
        assert action_high_soc > 0  # Should not discharge, but charge is limited

    def test_heuristic_no_battery_case(self):
        """Test heuristic behavior when plant has no battery."""
        mock_observation = {
            "market": {
                "market_data": {
                    "current_price_dollar_mwh": 0.5,  # Normalized
                    "price_forecast_dollar_mwh": np.array([0.5, 0.6]),
                    "price_history_dollar_mwh": np.array([0.4, 0.45]),
                }
            },
            "portfolios": {
                "test_portfolio": {
                    "test_plant": {
                        "ac_power_generation_potential_mw": 0.8,
                        "battery_state_of_charge": 0.0,  # Empty battery
                    },
                }
            },
        }

        heuristic = Heuristic()
        heuristic.configure(
            {"portfolios": [{"plants": [{"name": "test_plant"}]}]}
        )  # Basic config for num_plants=1
        action = heuristic.get_action(mock_observation)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        # With neutral price and no battery, should generate near default solar
        assert abs(action[0] - 0.5) < 0.1

    def test_heuristic_error_handling(self):
        """Test heuristic error handling with malformed observations."""
        heuristic = Heuristic()
        heuristic.configure({"portfolios": []})

        # Test with completely empty observation
        empty_obs = {}
        with pytest.raises(ValueError):
            heuristic.get_action(empty_obs)

        # Test with missing portfolio data
        missing_portfolio_obs = {"market": {"market_data": {}}}
        with pytest.raises(ValueError):
            heuristic.get_action(missing_portfolio_obs)
