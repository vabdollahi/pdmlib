"""
Minimal and efficient tests for a solar plant with a battery using CSV data.

This keeps things simple (no conftest), mirrors main-style factory usage,
and only validates key behaviors with real CSV weather input.
All tests use the unified configuration system for consistency.
"""

from datetime import datetime
from pathlib import Path

import pytest

from app.core.simulation.weather_provider import CSVWeatherProvider
from tests.config import test_config


@pytest.fixture
def test_plant():
    """Create a test plant using unified configuration."""
    return test_config.create_test_plant_with_battery_1()


@pytest.fixture
def location():
    """Create test location using unified configuration."""
    return test_config.create_test_location()


@pytest.fixture
def weather_provider(location):
    # Use CSVWeatherProvider directly; it now implements async get_data
    csv_path = Path(__file__).parent.parent.parent / "data" / "sample_weather_data.csv"
    provider = CSVWeatherProvider(location=location, file_path=str(csv_path))
    provider.set_range(
        datetime(2025, 7, 15, 0, 0, 0), datetime(2025, 7, 15, 23, 59, 59)
    )
    return provider


@pytest.mark.asyncio
async def test_weather_csv_loaded(weather_provider):
    start = datetime(2025, 7, 15, 10, 0, 0)
    end = datetime(2025, 7, 15, 14, 0, 0)
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
    assert test_plant.config.name == "California Solar Plant 1"
    assert test_plant.config.max_net_power_mw == 25.0

    # Test that the plant has batteries
    assert len(test_plant.batteries) == 1

    # Test battery properties
    battery = test_plant.batteries[0]
    assert battery.config.energy_capacity_mwh == 30.0
    assert battery.config.max_power_mw == 15.0


@pytest.mark.asyncio
async def test_plant_pv_generation(test_plant):
    """Test that the plant can generate PV power."""
    # Disable caching to avoid time range conflicts
    test_plant.pv_model._cached_provider = None

    # Get PV generation data using the correct method
    pv_data = await test_plant.pv_model.run_simulation()
    assert not pv_data.empty
    assert len(pv_data) > 0
    # Check that we have data with positive power generation during daylight hours
    assert pv_data["Total AC power (W)"].max() > 0
