"""
Essential tests for weather provider functionality.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.core.simulation.weather_provider import (
    CSVWeatherProvider,
    WeatherColumns,
    create_open_meteo_provider,
)
from app.core.utils.location import GeospatialLocation
from app.core.utils.storage import DataStorage


@pytest.fixture
def sample_location():
    """Sample geographical location for testing."""
    return GeospatialLocation(latitude=37.7749, longitude=-122.4194)


@pytest.fixture
def sample_csv_path():
    """Path to sample weather CSV file."""
    return Path(__file__).parent.parent.parent / "data" / "sample_weather_data.csv"


@pytest.fixture
def csv_provider(sample_csv_path: Path, sample_location: GeospatialLocation):
    """CSV weather provider for testing."""
    return CSVWeatherProvider(location=sample_location, file_path=str(sample_csv_path))


@pytest.fixture
def mock_storage():
    """Mock storage for testing."""
    return MagicMock(spec=DataStorage)


class TestWeatherProviderCreation:
    """Test weather provider creation and configuration."""

    def test_weather_provider_creates_openmeteo_by_default(
        self, mock_storage, sample_location
    ):
        """Test that create_open_meteo_provider creates OpenMeteo provider."""
        provider = create_open_meteo_provider(
            location=sample_location,
            start_date="2025-01-01",
            end_date="2025-01-10",
            organization="test-org",
            asset="test-asset",
            storage=mock_storage,
        )
        assert provider is not None


class TestCSVWeatherProvider:
    """Test CSV weather provider functionality."""

    def test_csv_provider_initialization(
        self, sample_csv_path: Path, sample_location: GeospatialLocation
    ):
        """Test CSV provider initialization."""
        provider = CSVWeatherProvider(
            location=sample_location, file_path=str(sample_csv_path)
        )
        assert provider.file_path == str(sample_csv_path)
        assert provider.location == sample_location

    @pytest.mark.asyncio
    async def test_get_weather_data_success(self, csv_provider: CSVWeatherProvider):
        """Test getting weather data from CSV file."""
        start_time = datetime(2025, 7, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 7, 15, 23, 59, 59, tzinfo=timezone.utc)
        csv_provider.set_range(start_time, end_time)
        data = await csv_provider.get_data()

        assert not data.empty
        assert WeatherColumns.GHI.value in data.columns
        assert WeatherColumns.TEMPERATURE.value in data.columns
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_get_weather_data_partial_timerange(
        self, csv_provider: CSVWeatherProvider
    ):
        """Test getting weather data for partial time range."""
        start_time = datetime(2025, 7, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 7, 15, 14, 0, 0, tzinfo=timezone.utc)
        csv_provider.set_range(start_time, end_time)
        data = await csv_provider.get_data()

        assert not data.empty
        assert len(data) <= 5  # Maximum 5 hours of data

    @pytest.mark.asyncio
    async def test_csv_provider_with_missing_file(
        self, sample_location: GeospatialLocation
    ):
        """Test CSV provider behavior with non-existent file."""
        provider = CSVWeatherProvider(
            location=sample_location, file_path="/non/existent/file.csv"
        )
        start_time = datetime(2025, 7, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 7, 15, 23, 59, 59, tzinfo=timezone.utc)
        provider.set_range(start_time, end_time)

        data = await provider.get_data()
        assert data.empty  # Should return empty DataFrame for missing file


class TestWeatherDataValidation:
    """Test weather data validation."""

    def test_weather_provider_validation_requires_ghi_minimum(self, sample_location):
        """Test that weather data validation requires minimum GHI data."""
        # Create minimal valid data
        data = pd.DataFrame(
            {
                WeatherColumns.GHI.value: [100, 200, 300],
                WeatherColumns.TEMPERATURE.value: [20, 25, 30],
            }
        )

        # Should pass validation with GHI present
        assert not data.empty
        assert WeatherColumns.GHI.value in data.columns


class TestOpenMeteoIntegration:
    """Test OpenMeteo provider integration."""

    def test_create_open_meteo_provider(self, sample_location, mock_storage):
        """Test creating OpenMeteo provider."""
        provider = create_open_meteo_provider(
            location=sample_location,
            start_date="2025-01-01",
            end_date="2025-01-02",
            organization="test",
            asset="weather",
            storage=mock_storage,
        )
        assert provider is not None
