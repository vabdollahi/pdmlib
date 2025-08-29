"""
Tests for provider configuration models and factory functions.

This module tests:
- Price provider configuration models
- Weather provider configuration models
- Provider factory functions
- Integration with existing provider system
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.core.simulation.provider_config import (
    CAISOPriceProviderConfig,
    CSVPriceProviderConfig,
    CSVWeatherProviderConfig,
    IESOPriceProviderConfig,
    OpenMeteoWeatherProviderConfig,
    create_price_provider_from_config,
    create_weather_provider_from_config,
)
from app.core.utils.location import GeospatialLocation


class TestPriceProviderConfigs:
    """Test price provider configuration models."""

    def test_csv_price_provider_config(self):
        """Test CSV price provider configuration."""
        config = CSVPriceProviderConfig(
            name="Test CSV Price", data="tests/data/sample_price_data.csv"
        )

        assert config.type == "csv_file"
        assert config.name == "Test CSV Price"
        assert config.data == "tests/data/sample_price_data.csv"

    def test_caiso_price_provider_config(self):
        """Test CAISO price provider configuration."""
        config = CAISOPriceProviderConfig(
            name="CAISO LMP", organization="TestOrg", asset="TestAsset"
        )

        assert config.type == "caiso"
        assert config.name == "CAISO LMP"
        assert config.organization == "TestOrg"
        assert config.asset == "TestAsset"
        assert config.data_type == "caiso_lmp"

    def test_ieso_price_provider_config(self):
        """Test IESO price provider configuration."""
        config = IESOPriceProviderConfig(
            name="IESO HOEP", organization="TestOrg", asset="TestAsset"
        )

        assert config.type == "ieso"
        assert config.name == "IESO HOEP"
        assert config.organization == "TestOrg"
        assert config.asset == "TestAsset"
        assert config.data_type == "ieso_hoep"


class TestWeatherProviderConfigs:
    """Test weather provider configuration models."""

    def test_csv_weather_provider_config(self):
        """Test CSV weather provider configuration."""
        config = CSVWeatherProviderConfig(data="tests/data/sample_weather_data.csv")

        assert config.type == "csv_file"
        assert config.data == "tests/data/sample_weather_data.csv"

    def test_openmeteo_weather_provider_config(self):
        """Test OpenMeteo weather provider configuration."""
        config = OpenMeteoWeatherProviderConfig(
            organization="TestOrg", asset="TestWeather", fetch_all_radiation=True
        )

        assert config.type == "openmeteo"
        assert config.organization == "TestOrg"
        assert config.asset == "TestWeather"
        assert config.data_type == "weather"
        assert config.fetch_all_radiation is True


class TestProviderFactoryFunctions:
    """Test provider factory functions."""

    @pytest.fixture
    def sample_location(self) -> GeospatialLocation:
        """Create sample location for testing."""
        return GeospatialLocation(latitude=37.7749, longitude=-122.4194)

    @pytest.fixture
    def sample_datetime_range(self) -> tuple[datetime, datetime]:
        """Create sample datetime range for testing."""
        start = datetime(2025, 7, 15, 8, 0, 0)
        end = datetime(2025, 7, 15, 18, 0, 0)
        return start, end

    def test_csv_price_provider_creation(self, sample_location, sample_datetime_range):
        """Test CSV price provider creation."""
        start_date, end_date = sample_datetime_range
        config = CSVPriceProviderConfig(data="tests/data/sample_price_data.csv")

        # Mock the CSV provider since we don't want to test actual file I/O here
        with patch(
            "app.core.simulation.price_provider.CSVPriceProvider"
        ) as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance

            provider = create_price_provider_from_config(
                config, sample_location, start_date, end_date
            )

            # Verify the provider was created with correct parameters
            mock_provider.assert_called_once_with(
                csv_file_path="tests/data/sample_price_data.csv"
            )
            mock_instance.set_range.assert_called_once_with(start_date, end_date)
            assert provider == mock_instance

    def test_csv_weather_provider_creation(
        self, sample_location, sample_datetime_range
    ):
        """Test CSV weather provider creation."""
        start_date, end_date = sample_datetime_range
        config = CSVWeatherProviderConfig(data="tests/data/sample_weather_data.csv")

        # Mock the CSV provider
        with patch(
            "app.core.simulation.weather_provider.CSVWeatherProvider"
        ) as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance

            provider = create_weather_provider_from_config(
                config, sample_location, start_date, end_date
            )

            # Verify the provider was created with correct parameters
            mock_provider.assert_called_once_with(
                location=sample_location, file_path="tests/data/sample_weather_data.csv"
            )
            mock_instance.set_range.assert_called_once_with(start_date, end_date)
            assert provider == mock_instance

    @patch("app.core.simulation.caiso_data.CAISOPriceProvider")
    def test_caiso_price_provider_creation(
        self, mock_provider, sample_location, sample_datetime_range
    ):
        """Test CAISO price provider creation."""
        start_date, end_date = sample_datetime_range
        config = CAISOPriceProviderConfig(organization="TestOrg", asset="TestAsset")

        mock_instance = MagicMock()
        mock_provider.return_value = mock_instance

        provider = create_price_provider_from_config(
            config, sample_location, start_date, end_date
        )

        # Verify the provider was created with correct parameters
        expected_kwargs = {
            "location": sample_location,
            "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
            "organization": "TestOrg",
            "asset": "TestAsset",
            "data_type": "caiso_lmp",
            "interval": config.interval,
            "storage": mock_provider.call_args[1]["storage"],  # DataStorage instance
        }

        mock_provider.assert_called_once()
        call_kwargs = mock_provider.call_args[1]

        # Check individual parameters (excluding storage which is complex)
        assert call_kwargs["location"] == expected_kwargs["location"]
        assert call_kwargs["start_date"] == expected_kwargs["start_date"]
        assert call_kwargs["end_date"] == expected_kwargs["end_date"]
        assert call_kwargs["organization"] == expected_kwargs["organization"]
        assert call_kwargs["asset"] == expected_kwargs["asset"]
        assert call_kwargs["data_type"] == expected_kwargs["data_type"]
        assert call_kwargs["interval"] == expected_kwargs["interval"]

        assert provider == mock_instance

    @patch("app.core.simulation.open_meteo_data.OpenMeteoWeatherProvider")
    def test_openmeteo_weather_provider_creation(
        self, mock_provider, sample_location, sample_datetime_range
    ):
        """Test OpenMeteo weather provider creation."""
        start_date, end_date = sample_datetime_range
        config = OpenMeteoWeatherProviderConfig(
            organization="TestOrg", asset="TestWeather", fetch_all_radiation=True
        )

        mock_instance = MagicMock()
        mock_provider.return_value = mock_instance

        provider = create_weather_provider_from_config(
            config, sample_location, start_date, end_date
        )

        # Verify the provider was created with correct parameters
        mock_provider.assert_called_once()
        call_kwargs = mock_provider.call_args[1]

        assert call_kwargs["location"] == sample_location
        assert call_kwargs["start_date"] == start_date.strftime("%Y-%m-%d %H:%M:%S")
        assert call_kwargs["end_date"] == end_date.strftime("%Y-%m-%d %H:%M:%S")
        assert call_kwargs["organization"] == "TestOrg"
        assert call_kwargs["asset"] == "TestWeather"
        assert call_kwargs["data_type"] == "weather"
        assert call_kwargs["fetch_all_radiation"] is True

        assert provider == mock_instance

    def test_unsupported_price_provider_type(
        self, sample_location, sample_datetime_range
    ):
        """Test error handling for unsupported price provider type."""
        start_date, end_date = sample_datetime_range

        # Create an invalid config object
        invalid_config = MagicMock()
        invalid_config.__class__ = type("InvalidConfig", (), {})

        with pytest.raises(ValueError, match="Unsupported price provider type"):
            create_price_provider_from_config(
                invalid_config, sample_location, start_date, end_date
            )

    def test_unsupported_weather_provider_type(
        self, sample_location, sample_datetime_range
    ):
        """Test error handling for unsupported weather provider type."""
        start_date, end_date = sample_datetime_range

        # Create an invalid config object
        invalid_config = MagicMock()
        invalid_config.__class__ = type("InvalidConfig", (), {})

        with pytest.raises(ValueError, match="Unsupported weather provider type"):
            create_weather_provider_from_config(
                invalid_config, sample_location, start_date, end_date
            )


class TestConfigurationValidation:
    """Test configuration validation and edge cases."""

    def test_price_config_validation(self):
        """Test price configuration validation."""
        # Test required fields
        with pytest.raises(Exception):  # Pydantic validation error
            CSVPriceProviderConfig()  # Missing 'data' field

    def test_weather_config_validation(self):
        """Test weather configuration validation."""
        # Test required fields
        with pytest.raises(Exception):  # Pydantic validation error
            CSVWeatherProviderConfig()  # Missing 'data' field

    def test_config_defaults(self):
        """Test configuration default values."""
        caiso_config = CAISOPriceProviderConfig()
        assert caiso_config.organization == "SolarRevenue"
        assert caiso_config.asset == "LMP_Data"
        assert caiso_config.data_type == "caiso_lmp"

        openmeteo_config = OpenMeteoWeatherProviderConfig()
        assert openmeteo_config.organization == "SolarRevenue"
        assert openmeteo_config.asset == "Weather"
        assert openmeteo_config.data_type == "weather"
        assert openmeteo_config.fetch_all_radiation is False
