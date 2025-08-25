"""
Tests for the weather fetching and processing logic.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.core.simulation.weather import (
    MAX_FORECAST_DAYS,
    OPEN_METEO_ARCHIVE_API_URL,
    OPEN_METEO_FORECAST_API_URL,
    OpenMeteoClient,
    WeatherProvider,
)
from app.core.utils.location import GeospatialLocation
from app.core.utils.storage import DataStorage


# --- Fixtures ---
@pytest.fixture
def mock_storage():
    """Provides a mock DataStorage instance using an in-memory filesystem."""
    return DataStorage(base_path="memory://test-storage")


@pytest.fixture
def mock_api_response():
    """Provides a mock JSON response from the Open-Meteo API."""
    return {
        "latitude": 52.52,
        "longitude": 13.41,
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "shortwave_radiation": [0.0, 0.0],
            "direct_normal_irradiance": [0.0, 0.0],
            "diffuse_radiation": [0.0, 0.0],
            "temperature_2m": [5.0, 5.4],
        },
    }


# --- WeatherProvider Tests ---
def test_weather_provider_valid_historical_range(mock_storage):
    """Tests that a valid historical date range passes validation."""
    provider = WeatherProvider(
        location=GeospatialLocation(latitude=52.52, longitude=13.41),
        start_date="2025-01-01",
        end_date="2025-01-10",
        organization="test-org",
        asset="test-asset",
        storage=mock_storage,
    )
    assert provider is not None


def test_weather_provider_valid_forecast_range(mock_storage):
    """Tests that a valid future forecast range passes validation."""
    start = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=MAX_FORECAST_DAYS - 1)).strftime("%Y-%m-%d")
    provider = WeatherProvider(
        location=GeospatialLocation(latitude=52.52, longitude=13.41),
        start_date=start,
        end_date=end,
        organization="test-org",
        asset="test-asset",
        storage=mock_storage,
    )
    assert provider is not None


def test_weather_provider_invalid_forecast_range_raises_error(mock_storage):
    """Tests that a forecast range exceeding the max limit raises a ValueError."""
    start = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=MAX_FORECAST_DAYS + 1)).strftime("%Y-%m-%d")
    with pytest.raises(
        ValueError, match=f"Forecast range cannot exceed {MAX_FORECAST_DAYS} days."
    ):
        WeatherProvider(
            location=GeospatialLocation(latitude=52.52, longitude=13.41),
            start_date=start,
            end_date=end,
            organization="test-org",
            asset="test-asset",
            storage=mock_storage,
        )


# --- OpenMeteoClient Tests ---
@pytest.mark.asyncio
@patch("aiohttp.ClientSession.get")
async def test_client_uses_archive_url_for_historical_data(mock_get, mock_api_response):
    """Tests that the client calls the correct API for historical dates."""
    mock_get.return_value.__aenter__.return_value.json = AsyncMock(
        return_value=mock_api_response
    )
    mock_get.return_value.__aenter__.return_value.raise_for_status = lambda: None

    client = OpenMeteoClient()
    await client.get_weather_data(
        latitude=52.52,
        longitude=13.41,
        start_date="2025-01-01",
        end_date="2025-01-02",
        variables=[],
    )

    called_url = mock_get.call_args[0][0]
    assert called_url == OPEN_METEO_ARCHIVE_API_URL


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.get")
async def test_client_uses_forecast_url_for_future_data(mock_get, mock_api_response):
    """Tests that the client calls the correct API for future dates."""
    mock_get.return_value.__aenter__.return_value.json = AsyncMock(
        return_value=mock_api_response
    )
    mock_get.return_value.__aenter__.return_value.raise_for_status = lambda: None

    start = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")

    client = OpenMeteoClient()
    await client.get_weather_data(
        latitude=52.52, longitude=13.41, start_date=start, end_date=end, variables=[]
    )

    called_url = mock_get.call_args[0][0]
    assert called_url == OPEN_METEO_FORECAST_API_URL


@pytest.mark.asyncio
async def test_client_processes_response_correctly(mock_api_response):
    """Tests that the client correctly processes a valid API response."""
    client = OpenMeteoClient()
    variables = [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "temperature_2m",
    ]
    df = client._process_response(mock_api_response, variables)

    assert not df.empty
    assert list(df.columns) == ["ghi", "dni", "dhi", "temp_air"]
    assert len(df) == 2
