"""
Unified test configuration management.

Provides centralized configuration loading and object creation for tests,
similar to the app config system. Creates a unified test portfolio with:
- Three plants total
- Two plants with batteries (California and Arizona)
- One plant without battery (Nevada)
"""

import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

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


class TestConfigManager:
    """Centralized configuration manager for tests."""

    def __init__(self):
        """Initialize the config manager."""
        self.config_dir = Path(__file__).parent
        self.data_dir = Path(__file__).parent.parent / "data"

    def load_config_file(self, filename: str) -> Dict:
        """Load a JSON configuration file."""
        config_path = self.config_dir / filename
        with open(config_path, "r") as f:
            return json.load(f)

    def create_test_location(
        self, location_data: Optional[Dict] = None
    ) -> GeospatialLocation:
        """Create a test location from configuration data."""
        if location_data is None:
            # Default test location data
            location_data = {
                "latitude": 40.7589,
                "longitude": -111.8883,
            }

        return GeospatialLocation(
            latitude=location_data["latitude"],
            longitude=location_data["longitude"],
        )

    def create_test_weather_provider(
        self, location: Optional[GeospatialLocation] = None
    ) -> CSVWeatherProvider:
        """Create a test weather provider."""
        if location is None:
            location = self.create_test_location()

        weather_csv = self.data_dir / "sample_weather_data.csv"
        weather_provider = CSVWeatherProvider(
            location=location, file_path=str(weather_csv)
        )
        weather_provider.set_range(
            start_time=datetime.datetime(2025, 7, 15, 0, 0),
            end_time=datetime.datetime(2025, 7, 15, 23, 0),
        )
        return weather_provider

    def create_test_price_provider(self) -> CSVPriceProvider:
        """Create a test price provider."""
        price_csv = self.data_dir / "sample_price_data.csv"
        price_provider = CSVPriceProvider(csv_file_path=str(price_csv))
        price_provider.set_range(
            start_time=datetime.datetime(2025, 7, 15, 0, 0),
            end_time=datetime.datetime(2025, 7, 15, 23, 0),
        )
        return price_provider

    def create_test_battery(
        self, battery_data: Optional[Dict] = None
    ) -> LinearBatterySimulator:
        """Create a test battery from configuration data."""
        if battery_data is None:
            # Get default from config
            config_data = self.load_config_file("test_portfolio_config.json")
            battery_data = config_data["plants"][0]["batteries"][0]

        # Type assertion to help type checker
        assert battery_data is not None

        battery_config = BatteryConfiguration(
            energy_capacity_mwh=battery_data["energy_capacity_mwh"],
            max_power_mw=battery_data["max_power_mw"],
            round_trip_efficiency=battery_data["round_trip_efficiency"],
            initial_soc=battery_data.get("initial_soc", 0.5),
            min_soc=battery_data.get("min_soc", 0.1),
            max_soc=battery_data.get("max_soc", 0.9),
        )
        return LinearBatterySimulator(config=battery_config)

    def create_test_plant_config(
        self, config_data: Optional[Dict] = None
    ) -> PlantConfiguration:
        """Create a plant configuration from test data."""
        if config_data is None:
            # Default test plant
            return PlantConfiguration(
                name="Test Plant",
                plant_id="TST-001",
                max_net_power_mw=25.0,
                min_net_power_mw=0.0,
                enable_market_participation=True,
            )

        plant_config_data = config_data["plant_config"]
        return PlantConfiguration(
            name=plant_config_data["name"],
            plant_id=plant_config_data["plant_id"],
            max_net_power_mw=plant_config_data["max_net_power_mw"],
            min_net_power_mw=plant_config_data["min_net_power_mw"],
            enable_market_participation=plant_config_data[
                "enable_market_participation"
            ],
        )

    def create_test_portfolio(
        self, portfolio_data: Optional[Dict] = None
    ) -> PowerPlantPortfolio:
        """Create a test portfolio from configuration data."""
        if portfolio_data is None:
            # Use the unified test portfolio config
            config_data = self.load_config_file("test_portfolio_config.json")
            portfolio_data = config_data["portfolio_config"]

        # Type assertion to help type checker
        assert portfolio_data is not None

        # Create all three plants
        plants = self.create_all_test_plants()

        # Create portfolio configuration
        portfolio_config = PortfolioConfiguration(
            name=portfolio_data["name"],
            max_total_power_mw=portfolio_data["max_total_power_mw"],
            allow_grid_purchase=portfolio_data.get("allow_grid_purchase", False),
        )

        return PowerPlantPortfolio(config=portfolio_config, plants=plants)

    def create_test_plant(self, plant_data: Optional[Dict] = None) -> SolarBatteryPlant:
        """Create a test plant from configuration data."""
        if plant_data is None:
            # Get default from config - use first plant (with battery)
            config_data = self.load_config_file("test_portfolio_config.json")
            plant_data = config_data["plants"][0]

        # Type assertion to help type checker
        assert plant_data is not None

        # Create location and providers
        location = self.create_test_location(plant_data["location"])
        weather_provider = self.create_test_weather_provider(location)
        price_provider = self.create_test_price_provider()

        # Create PV model
        pv_config = PVLibModel.model_validate(plant_data)
        pv_model = PVModel(pv_config=pv_config, weather_provider=weather_provider)

        # Create batteries (if any)
        batteries = []
        if plant_data.get("batteries") and len(plant_data["batteries"]) > 0:
            for battery_data in plant_data["batteries"]:
                battery = self.create_test_battery(battery_data)
                batteries.append(battery)

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
        return SolarBatteryPlant(
            config=plant_config,
            pv_model=pv_model,
            batteries=batteries,
            revenue_calculator=revenue_calc,
        )

    def create_all_test_plants(self) -> List[SolarBatteryPlant]:
        """Create all three test plants: two with batteries, one without."""
        config_data = self.load_config_file("test_portfolio_config.json")
        plants = []

        for plant_data in config_data["plants"]:
            plant = self.create_test_plant(plant_data)
            plants.append(plant)

        return plants

    def create_test_plant_with_battery_1(self) -> SolarBatteryPlant:
        """Create the first test plant (California) with battery."""
        config_data = self.load_config_file("test_portfolio_config.json")
        return self.create_test_plant(config_data["plants"][0])

    def create_test_plant_with_battery_2(self) -> SolarBatteryPlant:
        """Create the second test plant (Arizona) with battery."""
        config_data = self.load_config_file("test_portfolio_config.json")
        return self.create_test_plant(config_data["plants"][1])

    def create_test_plant_without_battery(self) -> SolarBatteryPlant:
        """Create the third test plant (Nevada) without battery."""
        config_data = self.load_config_file("test_portfolio_config.json")
        return self.create_test_plant(config_data["plants"][2])

    def create_test_environment_config(self):
        """Create a standard test environment configuration."""
        try:
            from app.core.environment import EnvironmentConfig

            portfolio = self.create_test_portfolio()
            return EnvironmentConfig(
                portfolios=[portfolio],
                start_date_time=datetime.datetime(2025, 7, 15, 8, 0, 0),
                end_date_time=datetime.datetime(2025, 7, 15, 18, 0, 0),
                interval_min=15.0,
            )
        except ImportError:
            # Skip environment tests if gymnasium is not available
            pytest.skip("Environment testing requires gymnasium")

    def create_test_environment(self):
        """Create a complete test environment ready for use."""
        try:
            from app.core.environment.power_management_env import (
                PowerManagementEnvironment,
            )

            env_config = self.create_test_environment_config()
            if env_config is not None:
                return PowerManagementEnvironment(env_config)
        except ImportError:
            # Skip environment tests if gymnasium is not available
            pytest.skip("Environment testing requires gymnasium")


# Global instance for easy access
test_config = TestConfigManager()
