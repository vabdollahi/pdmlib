"""
Minimal and efficient tests for a solar plant with a battery using CSV data.

This keeps things simple (no conftest), mirrors main-style factory usage,
and only validates key behaviors with real CSV weather input.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.core.simulation.plant import SolarBatteryPlant
from app.core.simulation.pv_model import PVModel
from app.core.simulation.pvlib_models import PVLibModel
from app.core.simulation.weather_provider import CSVWeatherProvider
from app.core.utils.location import GeospatialLocation


@pytest.fixture
def plant_config_data():
    cfg = (
        Path(__file__).parent.parent.parent / "config" / "test_solar_battery_plant.json"
    )
    with open(cfg, "r") as f:
        return json.load(f)


@pytest.fixture
def location(plant_config_data):
    loc = plant_config_data["location"]
    return GeospatialLocation(latitude=loc["latitude"], longitude=loc["longitude"])


@pytest.fixture
def weather_provider(location):
    # Use CSVWeatherProvider directly; it now implements async get_data
    csv_path = Path(__file__).parent.parent.parent / "data" / "sample_weather_data.csv"
    provider = CSVWeatherProvider(location=location, file_path=str(csv_path))
    provider.set_range(
        datetime(2025, 7, 15, 0, 0, 0), datetime(2025, 7, 15, 23, 59, 59)
    )
    return provider


@pytest.fixture
def pv_model(plant_config_data, weather_provider):
    pv_conf = PVLibModel.model_validate(plant_config_data)
    return PVModel(pv_config=pv_conf, weather_provider=weather_provider)


@pytest.fixture
def plant(plant_config_data, pv_model):
    from app.core.simulation.battery_simulator import (
        BatteryConfiguration,
        LinearBatterySimulator,
    )
    from app.core.simulation.plant import PlantConfiguration

    cfg = plant_config_data["plant_config"]
    plant_cfg = PlantConfiguration(
        name=cfg.get("name", "Test Plant"),
        plant_id=cfg.get("plant_id"),
        max_net_power_mw=cfg.get("max_net_power_mw", 10.0),
        min_net_power_mw=cfg.get("min_net_power_mw", 0.0),
        enable_market_participation=cfg.get("enable_market_participation", True),
    )

    # Only pass supported fields to BatteryConfiguration
    bat_data = plant_config_data["batteries"][0]
    bat_cfg = BatteryConfiguration(
        energy_capacity_mwh=bat_data["energy_capacity_mwh"],
        max_power_mw=bat_data["max_power_mw"],
        round_trip_efficiency=bat_data["round_trip_efficiency"],
        initial_soc=bat_data["initial_soc"],
        min_soc=bat_data["min_soc"],
        max_soc=bat_data["max_soc"],
    )
    battery = LinearBatterySimulator(config=bat_cfg)

    return SolarBatteryPlant(config=plant_cfg, pv_model=pv_model, batteries=[battery])


@pytest.mark.asyncio
async def test_weather_csv_loaded(weather_provider):
    start = datetime(2025, 7, 15, 10, 0, 0)
    end = datetime(2025, 7, 15, 14, 0, 0)
    weather_provider.set_range(start, end)
    df = await weather_provider.get_data()
    assert not df.empty
    assert "ghi" in df.columns or "shortwave_radiation" in df.columns


def test_battery_limits_from_config(plant):
    max_charge, max_discharge = plant.get_battery_available_power()
    assert max_charge > 0
    assert max_discharge > 0
    # Config sets 10 MW limit; allow small tolerance
    assert max_charge <= 10.1
    assert max_discharge <= 10.1


@pytest.mark.asyncio
async def test_dispatch_smoke(plant):
    # Simple smoke test to ensure dispatch runs with CSV-driven PV
    ts = datetime(2025, 7, 15, 12, 0, 0)
    actual, state, valid = await plant.dispatch_power(5.0, ts, 60.0)
    assert isinstance(actual, float)
    assert isinstance(state, dict)
    assert isinstance(valid, bool)
