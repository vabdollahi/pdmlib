"""
Production configuration management.

Provides centralized configuration loading and object creation for the main application,
using the same unified configuration system as tests but for production use.
"""

import datetime
import json
from pathlib import Path
from typing import Dict, Optional

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
from app.core.simulation.weather_provider import (
    CSVWeatherProvider,
    WeatherProvider,
)
from app.core.utils.location import GeospatialLocation


class ConfigManager:
    """Centralized configuration manager for production."""

    def __init__(self):
        """Initialize the config manager."""
        self.config_dir = Path(__file__).parent.parent.parent / "tests" / "config"
        self.data_dir = Path(__file__).parent.parent.parent / "tests" / "data"

    def load_config_file(self, filename: str) -> Dict:
        """Load a JSON configuration file."""
        config_path = self.config_dir / filename
        with open(config_path, "r") as f:
            return json.load(f)

    def create_location(
        self, location_data: Optional[Dict] = None
    ) -> GeospatialLocation:
        """Create a location from configuration data."""
        if location_data is None:
            # Default San Francisco Bay Area location
            location_data = {
                "latitude": 37.7749,
                "longitude": -122.4194,
            }

        return GeospatialLocation(
            latitude=location_data["latitude"],
            longitude=location_data["longitude"],
        )

    def create_weather_provider(
        self,
        location: Optional[GeospatialLocation] = None,
        use_csv: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Create a weather provider (CSV or API-based)."""
        if location is None:
            location = self.create_location()

        if use_csv:
            # Use CSV weather provider for testing/demo (no caching)
            weather_csv = self.data_dir / "sample_weather_data.csv"
            provider = CSVWeatherProvider(location=location, file_path=str(weather_csv))
            provider.set_range(
                start_time=datetime.datetime(2025, 7, 15, 0, 0),
                end_time=datetime.datetime(2025, 7, 15, 23, 0),
            )
            return provider
        else:
            # Use API weather provider for production (with caching)
            from app.core.utils.date_handling import TimeInterval
            from app.core.utils.storage import DataStorage

            return WeatherProvider(
                location=location,
                start_date=start_date or "2025-07-15",
                end_date=end_date or "2025-07-16",
                organization="SolarRevenue",
                asset="Weather",
                interval=TimeInterval.HOURLY,
                storage=DataStorage(base_path="data"),
            )

    def create_csv_weather_provider(
        self, location: Optional[GeospatialLocation] = None
    ):
        """Create a CSV weather provider for testing (no caching)."""
        return self.create_weather_provider(location=location, use_csv=True)

    def create_api_weather_provider(
        self,
        location: Optional[GeospatialLocation] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Create an API weather provider for production (with caching)."""
        return self.create_weather_provider(
            location=location, use_csv=False, start_date=start_date, end_date=end_date
        )

    def create_pv_model_from_config(
        self, plant_config: Dict, weather_provider, enable_caching: bool = True
    ) -> PVModel:
        """Create a PV model from plant configuration."""
        # Create PV model
        pv_config = PVLibModel.model_validate(plant_config)
        pv_model = PVModel(pv_config=pv_config, weather_provider=weather_provider)

        # Enable caching manually if needed (e.g., when using CSV weather provider)
        if enable_caching and not pv_model.is_caching_enabled:
            from app.core.utils.storage import DataStorage

            # Get organization and asset from weather provider or use defaults
            organization = getattr(weather_provider, "organization", "SolarRevenue")
            asset = getattr(weather_provider, "asset", "PV_Generation")

            # Ensure asset is unique for PV data
            if asset == "Weather":
                asset = "PV_Generation"
            elif not asset.startswith("PV_"):
                asset = f"PV_{asset}"

            storage = DataStorage(base_path="data")
            pv_model.enable_caching(
                storage=storage, organization=organization, asset=asset
            )

        return pv_model

    def create_simple_solar_farm(
        self,
        capacity_mw: float = 10.0,
        location: Optional[GeospatialLocation] = None,
        weather_provider=None,
        use_api_weather: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> PVModel:
        """Create a simple solar farm for demo purposes."""
        if location is None:
            location = self.create_location()

        if weather_provider is None:
            if use_api_weather:
                # Use API weather provider with caching
                weather_provider = self.create_api_weather_provider(
                    location=location, start_date=start_date, end_date=end_date
                )
            else:
                # Use CSV weather provider for testing
                weather_provider = self.create_csv_weather_provider(location=location)

        # Create a simple 10 MW solar farm configuration
        simple_config = {
            "location": {
                "name": "Demo_Solar_Farm",
                "latitude": location.latitude,
                "longitude": location.longitude,
                "tz": "America/Los_Angeles",
                "altitude": 100,
            },
            "pv_systems": [
                {
                    "inverters": {
                        "count": int(capacity_mw / 5),  # 5MW per inverter
                        "database": "CECInverter",
                        "record": "SMA_America__SB5000US__240V_",
                    },
                    "pv_arrays": [
                        {
                            "pv_modules": {
                                "count": 20,
                                "name": f"Solar Array {capacity_mw}MW",
                                "database": "CECMod",
                                "record": "Canadian_Solar_Inc__CS5P_220M",
                            },
                            "array_setup": {
                                "name": f"Fixed Mount Array ({capacity_mw} MW)",
                                "mount": {
                                    "type": "fixed_mount",
                                    "tilt_degrees": 30.0,
                                    "azimuth_degrees": 180.0,
                                },
                                "number_of_strings": int(
                                    capacity_mw * 100
                                ),  # Scale strings with capacity
                                "temperature_model": {
                                    "database": "sapm",
                                    "record": "open_rack_glass_glass",
                                },
                            },
                        }
                    ],
                }
            ],
            "physical_simulation": {
                "aoi_model": "physical",
                "spectral_model": "no_loss",
            },
        }

        return self.create_pv_model_from_config(
            simple_config,
            weather_provider,
            enable_caching=use_api_weather,  # Enable caching only when using API
        )

    def create_portfolio_from_unified_config(self) -> PowerPlantPortfolio:
        """Create a portfolio using the unified test configuration."""
        config_data = self.load_config_file("test_portfolio_config.json")

        # Create portfolio configuration
        portfolio_config_data = config_data["portfolio_config"]
        portfolio_config = PortfolioConfiguration(
            name=portfolio_config_data["name"],
            portfolio_id=portfolio_config_data["portfolio_id"],
            strategy=portfolio_config_data["strategy"],
            max_portfolio_risk=portfolio_config_data["max_portfolio_risk"],
            diversification_weight=portfolio_config_data["diversification_weight"],
            enable_market_arbitrage=portfolio_config_data["enable_market_arbitrage"],
            enable_ancillary_services=portfolio_config_data[
                "enable_ancillary_services"
            ],
            max_total_power_mw=portfolio_config_data["max_total_power_mw"],
            min_operating_plants=portfolio_config_data["min_operating_plants"],
        )

        # Create plants
        plants = []
        for plant_data in config_data["plants"]:
            # Create location and providers
            location = self.create_location(plant_data["location"])
            weather_provider = self.create_weather_provider(location, use_csv=True)

            # Create PV model
            pv_model = self.create_pv_model_from_config(plant_data, weather_provider)

            # Create batteries (if any)
            batteries = []
            if plant_data.get("batteries") and len(plant_data["batteries"]) > 0:
                for battery_data in plant_data["batteries"]:
                    battery_config = BatteryConfiguration(
                        energy_capacity_mwh=battery_data["energy_capacity_mwh"],
                        max_power_mw=battery_data["max_power_mw"],
                        round_trip_efficiency=battery_data["round_trip_efficiency"],
                        initial_soc=battery_data.get("initial_soc", 0.5),
                        min_soc=battery_data.get("min_soc", 0.1),
                        max_soc=battery_data.get("max_soc", 0.9),
                    )
                    battery = LinearBatterySimulator(config=battery_config)
                    batteries.append(battery)

            # Create plant configuration
            plant_config_data = plant_data["plant_config"]
            plant_config = PlantConfiguration(
                name=plant_config_data["name"],
                plant_id=plant_config_data["plant_id"],
                max_net_power_mw=plant_config_data["max_net_power_mw"],
                min_net_power_mw=plant_config_data["min_net_power_mw"],
                enable_market_participation=plant_config_data[
                    "enable_market_participation"
                ],
            )

            # Create revenue calculator
            price_csv_path = self.data_dir / "sample_price_data.csv"
            price_provider = CSVPriceProvider(csv_file_path=str(price_csv_path))
            revenue_calculator = SolarRevenueCalculator(
                price_provider=price_provider, pv_model=pv_model
            )

            # Create plant
            plant = SolarBatteryPlant(
                config=plant_config,
                pv_model=pv_model,
                batteries=batteries,
                revenue_calculator=revenue_calculator,
            )
            plants.append(plant)

        # Create portfolio
        portfolio = PowerPlantPortfolio(config=portfolio_config, plants=plants)
        return portfolio


# Global instance for easy access
config_manager = ConfigManager()
