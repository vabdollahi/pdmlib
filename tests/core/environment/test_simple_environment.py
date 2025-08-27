"""
Simple tests for the PowerManagementEnvironment.

These tests verify basic functionality without complex fixture dependencies.
"""

import datetime
import json
from pathlib import Path

import numpy as np

from app.core.environment import EnvironmentConfig
from app.core.environment.power_management_env import PowerManagementEnvironment
from app.core.simulation.battery_simulator import (
    BatteryConfiguration,
    LinearBatterySimulator,
)
from app.core.simulation.plant import PlantConfiguration, SolarBatteryPlant
from app.core.simulation.portfolio import PortfolioConfiguration, PowerPlantPortfolio
from app.core.simulation.price_provider import CSVPriceProvider
from app.core.simulation.pv_model import PVModel
from app.core.simulation.pvlib_models import PVLibModel
from app.core.simulation.solar_revenue import SolarRevenueCalculator
from app.core.simulation.weather_provider import CSVWeatherProvider
from app.core.utils.location import GeospatialLocation


class TestSimpleEnvironment:
    """Test the PowerManagementEnvironment with basic functionality."""

    def test_basic_environment_creation(self):
        """Test creating a basic environment with minimal configuration."""
        # Create simple plant configuration from JSON
        config_path = (
            Path(__file__).parent.parent.parent
            / "config"
            / "test_portfolio_config.json"
        )
        with open(config_path, "r") as f:
            data = json.load(f)

        # Get the first plant data
        plant_data = data["plants"][0]

        # Create location
        location = GeospatialLocation(
            latitude=plant_data["location"]["latitude"],
            longitude=plant_data["location"]["longitude"],
        )

        # Create weather provider
        weather_csv = (
            Path(__file__).parent.parent.parent / "data" / "sample_weather_data.csv"
        )
        weather_provider = CSVWeatherProvider(
            location=location, file_path=str(weather_csv)
        )
        weather_provider.set_range(
            start_time=datetime.datetime(2025, 7, 15, 0, 0),
            end_time=datetime.datetime(2025, 7, 15, 23, 0),
        )

        # Create price provider
        price_csv = (
            Path(__file__).parent.parent.parent / "data" / "sample_price_data.csv"
        )
        price_provider = CSVPriceProvider(csv_file_path=str(price_csv))
        price_provider.set_range(
            start_time=datetime.datetime(2025, 7, 15, 0, 0),
            end_time=datetime.datetime(2025, 7, 15, 23, 0),
        )

        # Create PV model
        pv_config = PVLibModel.model_validate(plant_data)
        pv_model = PVModel(pv_config=pv_config, weather_provider=weather_provider)

        # Create battery
        bat_config = plant_data["batteries"][0]
        battery_config = BatteryConfiguration(
            energy_capacity_mwh=bat_config["energy_capacity_mwh"],
            max_power_mw=bat_config["max_power_mw"],
            round_trip_efficiency=bat_config["round_trip_efficiency"],
            initial_soc=bat_config.get("initial_soc", 0.5),
            min_soc=bat_config.get("min_soc", 0.1),
            max_soc=bat_config.get("max_soc", 0.9),
        )
        battery = LinearBatterySimulator(config=battery_config)

        # Create revenue calculator
        revenue_calc = SolarRevenueCalculator(
            price_provider=price_provider, pv_model=pv_model
        )

        # Create plant configuration
        plant_cfg_data = plant_data["plant_config"]
        plant_config = PlantConfiguration(
            name=plant_cfg_data["name"],
            plant_id=plant_cfg_data["plant_id"],
            max_net_power_mw=plant_cfg_data["max_net_power_mw"],
            min_net_power_mw=plant_cfg_data["min_net_power_mw"],
            enable_market_participation=plant_cfg_data["enable_market_participation"],
        )

        # Create the plant
        plant = SolarBatteryPlant(
            config=plant_config,
            pv_model=pv_model,
            batteries=[battery],
            revenue_calculator=revenue_calc,
        )

        # Create portfolio
        portfolio_config = PortfolioConfiguration(
            name="Test Environment Portfolio",
            max_total_power_mw=50.0,
            allow_grid_purchase=False,
        )

        portfolio = PowerPlantPortfolio(config=portfolio_config, plants=[plant])

        # Create environment
        env_config = EnvironmentConfig(
            portfolios=[portfolio],
            start_date_time=datetime.datetime(2025, 7, 15, 8, 0, 0),
            end_date_time=datetime.datetime(2025, 7, 15, 18, 0, 0),
            interval_min=15.0,
        )

        env = PowerManagementEnvironment(env_config)

        # Test basic properties
        assert env.timestamp.hour == 8
        assert env.end_date_time.hour == 18
        assert env.interval.total_seconds() == 15 * 60

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
        assert action_space.shape == (1,)  # One plant
        assert hasattr(action_space, "low")
        assert hasattr(action_space, "high")
        # Check bounds (accessing as arrays)
        low_vals = getattr(action_space, "low")
        high_vals = getattr(action_space, "high")
        assert low_vals[0] == -1.0
        assert high_vals[0] == 1.0

    def test_environment_reset_and_step(self):
        """Test basic environment reset and step operations."""
        # Reuse the environment creation logic but simpler
        config_path = (
            Path(__file__).parent.parent.parent
            / "config"
            / "test_portfolio_config.json"
        )
        with open(config_path, "r") as f:
            data = json.load(f)

        plant_data = data["plants"][0]
        location = GeospatialLocation(
            latitude=plant_data["location"]["latitude"],
            longitude=plant_data["location"]["longitude"],
        )

        weather_csv = (
            Path(__file__).parent.parent.parent / "data" / "sample_weather_data.csv"
        )
        weather_provider = CSVWeatherProvider(
            location=location, file_path=str(weather_csv)
        )
        weather_provider.set_range(
            start_time=datetime.datetime(2025, 7, 15, 0, 0),
            end_time=datetime.datetime(2025, 7, 15, 23, 0),
        )

        price_csv = (
            Path(__file__).parent.parent.parent / "data" / "sample_price_data.csv"
        )
        price_provider = CSVPriceProvider(csv_file_path=str(price_csv))
        price_provider.set_range(
            start_time=datetime.datetime(2025, 7, 15, 0, 0),
            end_time=datetime.datetime(2025, 7, 15, 23, 0),
        )

        pv_config = PVLibModel.model_validate(plant_data)
        pv_model = PVModel(pv_config=pv_config, weather_provider=weather_provider)

        bat_config = plant_data["batteries"][0]
        battery_config = BatteryConfiguration(
            energy_capacity_mwh=bat_config["energy_capacity_mwh"],
            max_power_mw=bat_config["max_power_mw"],
            round_trip_efficiency=bat_config["round_trip_efficiency"],
            initial_soc=bat_config.get("initial_soc", 0.5),
        )
        battery = LinearBatterySimulator(config=battery_config)

        revenue_calc = SolarRevenueCalculator(
            price_provider=price_provider, pv_model=pv_model
        )

        plant_cfg_data = plant_data["plant_config"]
        plant_config = PlantConfiguration(
            name=plant_cfg_data["name"],
            plant_id=plant_cfg_data["plant_id"],
            max_net_power_mw=plant_cfg_data["max_net_power_mw"],
            min_net_power_mw=plant_cfg_data["min_net_power_mw"],
            enable_market_participation=plant_cfg_data["enable_market_participation"],
        )

        plant = SolarBatteryPlant(
            config=plant_config,
            pv_model=pv_model,
            batteries=[battery],
            revenue_calculator=revenue_calc,
        )

        portfolio_config = PortfolioConfiguration(
            name="Test Portfolio", max_total_power_mw=50.0, allow_grid_purchase=False
        )

        portfolio = PowerPlantPortfolio(config=portfolio_config, plants=[plant])

        env_config = EnvironmentConfig(
            portfolios=[portfolio],
            start_date_time=datetime.datetime(2025, 7, 15, 8, 0, 0),
            end_date_time=datetime.datetime(2025, 7, 15, 10, 0, 0),  # Short episode
            interval_min=15.0,
        )

        env = PowerManagementEnvironment(env_config)

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
        # Test configuration part since action converter is internal
        config_path = (
            Path(__file__).parent.parent.parent
            / "config"
            / "test_portfolio_config.json"
        )
        with open(config_path, "r") as f:
            data = json.load(f)

        plant_data = data["plants"][0]
        plant_cfg_data = plant_data["plant_config"]

        # Verify that plant constraints are properly configured
        min_power = plant_cfg_data["min_net_power_mw"]
        max_power = plant_cfg_data["max_net_power_mw"]

        assert min_power < max_power
        assert min_power >= 0
        assert max_power > 0

        # Verify we can create a plant configuration with these constraints
        plant_config = PlantConfiguration(
            name=plant_cfg_data["name"],
            plant_id=plant_cfg_data["plant_id"],
            max_net_power_mw=max_power,
            min_net_power_mw=min_power,
            enable_market_participation=plant_cfg_data["enable_market_participation"],
        )

        assert plant_config.min_net_power_mw == min_power
        assert plant_config.max_net_power_mw == max_power

    def test_grid_purchase_configuration(self):
        """Test the grid purchase configuration functionality."""
        # Test traditional portfolio (no grid purchase)
        traditional_config = PortfolioConfiguration(
            name="Traditional Plant", allow_grid_purchase=False
        )
        assert not traditional_config.allow_grid_purchase

        # Test small producer portfolio (with grid purchase)
        small_producer_config = PortfolioConfiguration(
            name="Small Producer", allow_grid_purchase=True
        )
        assert small_producer_config.allow_grid_purchase

        # Test default behavior (should default to False for traditional plants)
        default_config = PortfolioConfiguration(name="Default Plant")
        assert not default_config.allow_grid_purchase
