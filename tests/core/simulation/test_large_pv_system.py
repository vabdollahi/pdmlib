"""
Test using the 10MW solar farm configuration.

This module tests using the JSON configuration file for large-scale PV systems.
"""

import json
from pathlib import Path

import pytest

from app.core.simulation.pv_model import PVModel
from app.core.simulation.pvlib_models import PVLibModel
from app.core.simulation.weather import WeatherProvider
from app.core.utils.date_handling import TimeInterval
from app.core.utils.location import GeospatialLocation


class TestLargePVSystem:
    """Test large PV system using JSON configuration."""

    @pytest.fixture
    def config_10mw(self):
        """Load the 10MW solar farm configuration."""
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "10mw_solar_farm.json"
        )
        with open(config_path, "r") as f:
            return json.load(f)

    @pytest.fixture
    def sample_weather_provider_10mw(self, config_10mw):
        """Create a sample weather provider for the 10MW farm."""
        location_config = config_10mw["location"]
        location = GeospatialLocation(
            latitude=location_config["latitude"], longitude=location_config["longitude"]
        )
        return WeatherProvider(
            location=location,
            start_date="2023-06-15",
            end_date="2023-06-16",
            organization="test_org",
            asset=location_config["name"],
            interval=TimeInterval.HOURLY,
        )

    def test_10mw_pv_model_from_json(self, config_10mw, sample_weather_provider_10mw):
        """Test creating 10MW PV model from JSON configuration."""
        # Create PV model in one shot from JSON
        pvlib_config = PVLibModel(**config_10mw)

        # Create final PV model
        pv_model = PVModel(
            pv_config=pvlib_config, weather_provider=sample_weather_provider_10mw
        )

        # Verify the configuration
        assert pv_model.pv_config is not None
        assert pv_model.weather_provider is not None
        assert pv_model.pv_config.location.name == "LA_10MW_Solar_Farm"
        assert len(pv_model.pv_config.pv_systems) == 1
        assert len(pv_model.pv_config.pv_systems[0].pv_arrays) == 4

        # Verify array names
        array_names = [
            array.array_setup.name
            for array in pv_model.pv_config.pv_systems[0].pv_arrays
        ]
        expected_names = [
            "Block 1 - Fixed Mount (2.75 MW)",
            "Block 2 - Fixed Mount (2.75 MW)",
            "Block 3 - Single Axis Tracking (2.75 MW)",
            "Block 4 - Single Axis Tracking (2.0 MW)",
        ]
        assert array_names == expected_names

        # Verify physical simulation
        assert pv_model.pv_config.physical_simulation is not None
        assert pv_model.pv_config.physical_simulation.aoi_model == "physical"
        assert pv_model.pv_config.physical_simulation.spectral_model == "no_loss"

        # Verify inverter configuration
        inverter = pv_model.pv_config.pv_systems[0].inverters
        assert inverter.count == 5

    def test_10mw_system_capacity_calculation(
        self, config_10mw, sample_weather_provider_10mw
    ):
        """Test capacity calculation for 10MW system."""
        # Create PV model
        pvlib_config = PVLibModel(**config_10mw)
        pv_model = PVModel(
            pv_config=pvlib_config, weather_provider=sample_weather_provider_10mw
        )

        # Calculate capacity
        capacity = pv_model.get_system_capacity()

        # For database-driven modules using Canadian Solar 220W modules,
        # verify reasonable capacity ranges around 10MW
        assert capacity["dc_capacity_w"] > 9_000_000  # At least 9MW
        assert capacity["dc_capacity_w"] < 12_000_000  # Less than 12MW

        # Verify the system has reasonable capacity structure
        assert capacity["dc_capacity_w"] > 0
        assert "ac_capacity_w" in capacity
        assert "dc_ac_ratio" in capacity
