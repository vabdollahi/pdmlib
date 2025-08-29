"""
Minimal and efficient tests for a solar plant with a battery using CSV data.

This keeps things simple (no conftest), mirrors main-style factory usage,
and only validates key behaviors with real CSV weather input.
All tests use the unified configuration system for consistency.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.core.environment.power_management_env import PowerManagementEnvironment
from app.core.simulation.weather_provider import CSVWeatherProvider
from app.core.utils.location import GeospatialLocation
from tests.config import test_config


@pytest.fixture
def test_plant():
    """Create a test plant using spec-driven configuration."""
    env = PowerManagementEnvironment.from_json(test_config.environment_spec_path)
    # Get the first plant from the first portfolio (has battery)
    return env.config.portfolios[0].plants[0]


@pytest.fixture
def location():
    """Create test location from spec configuration."""
    # California location from the JSON spec
    return GeospatialLocation(latitude=34.0522, longitude=-118.2437)


@pytest.fixture
def weather_provider(location):
    # Use CSVWeatherProvider directly; it now implements async get_data
    csv_path = Path(__file__).parent.parent.parent / "data" / "sample_weather_data.csv"
    provider = CSVWeatherProvider(location=location, file_path=str(csv_path))
    provider.set_range(
        datetime(2025, 7, 15, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 7, 15, 23, 59, 59, tzinfo=timezone.utc),
    )
    return provider


@pytest.mark.asyncio
async def test_weather_csv_loaded(weather_provider):
    start = datetime(2025, 7, 15, 10, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 7, 15, 14, 0, 0, tzinfo=timezone.utc)
    weather_provider.set_range(start, end)
    df = await weather_provider.get_data()
    assert not df.empty
    assert "ghi" in df.columns or "shortwave_radiation" in df.columns


def test_battery_limits_from_config(test_plant):
    max_charge, max_discharge = test_plant.get_battery_available_power()
    assert max_charge > 0
    assert max_discharge > 0
    # Config sets 15 MW limit for California plant; allow small tolerance
    assert max_charge <= 15.1
    assert max_discharge <= 15.1


def test_basic_plant_simulation(test_plant):
    """Test basic plant simulation functionality."""
    # Test that the plant can provide its configuration
    # Updated to match our validated config
    assert test_plant.config.name == "Plant With Battery 1"
    # Updated from 25.0 to match our config
    assert test_plant.config.max_net_power_mw == 7.0

    # Test that the plant has batteries
    assert len(test_plant.batteries) == 1

    # Test battery properties
    battery = test_plant.batteries[0]
    assert battery.config.energy_capacity_mwh == 30.0
    assert battery.config.max_power_mw == 15.0


@pytest.mark.asyncio
async def test_plant_pv_generation(test_plant):
    """Test that PV generation simulation can run without errors."""
    # Get PV generation data using the correct method
    pv_data = await test_plant.pv_model.run_simulation()

    # Check that the simulation completed successfully and returned a DataFrame
    assert isinstance(pv_data, pd.DataFrame)

    # Check that we have the expected columns structure
    expected_columns = ["date_time", "Total AC power (W)"]
    for col in expected_columns:
        assert col in pv_data.columns, f"Missing expected column: {col}"
