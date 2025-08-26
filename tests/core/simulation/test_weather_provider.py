"""
Tests for the weather fetching and processing logic.

These tests prioritize GHI as the only required radiation type, with DNI and DHI
being optional and calculated using PVLib decomposition models when missing.
Most tests mock API calls to avoid actual OpenMeteo requests.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.core.simulation.open_meteo_data import (
    MAX_FORECAST_DAYS,
    OpenMeteoClient,
)
from app.core.simulation.weather_provider import (
    BaseWeatherProvider,
    CSVWeatherProvider,
    WeatherColumns,
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
def sample_location():
    """Provides a sample geographic location (Berlin)."""
    return GeospatialLocation(latitude=52.52, longitude=13.41)


@pytest.fixture
def ghi_only_api_response():
    """Mock API response with only GHI and temperature (optimized mode)."""
    return {
        "latitude": 52.52,
        "longitude": 13.41,
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00", "2025-01-01T02:00"],
            "shortwave_radiation": [0.0, 120.5, 340.8],  # GHI values
            "temperature_2m": [5.0, 5.4, 6.2],  # Temperature values
        },
    }


@pytest.fixture
def full_radiation_api_response():
    """Mock API response with all radiation components."""
    return {
        "latitude": 52.52,
        "longitude": 13.41,
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00", "2025-01-01T02:00"],
            "shortwave_radiation": [0.0, 120.5, 340.8],  # GHI
            "direct_normal_irradiance": [0.0, 180.2, 520.1],  # DNI
            "diffuse_radiation": [0.0, 60.3, 85.7],  # DHI
            "temperature_2m": [5.0, 5.4, 6.2],  # Temperature
        },
    }


# --- WeatherProvider Factory Tests ---
def test_weather_provider_creates_openmeteo_by_default(mock_storage, sample_location):
    """Tests that WeatherProvider creates OpenMeteo provider by default."""
    provider = WeatherProvider(
        location=sample_location,
        start_date="2025-01-01",
        end_date="2025-01-10",
        organization="test-org",
        asset="test-asset",
        storage=mock_storage,
    )
    # Should be OpenMeteoWeatherProvider instance
    assert provider.__class__.__name__ == "OpenMeteoWeatherProvider"


def test_weather_provider_optimized_by_default(mock_storage, sample_location):
    """Tests that WeatherProvider uses optimized fetching (GHI only) by default."""
    provider = WeatherProvider(
        location=sample_location,
        start_date="2025-01-01",
        end_date="2025-01-10",
        organization="test-org",
        asset="test-asset",
        storage=mock_storage,
    )
    # Cast to OpenMeteoWeatherProvider to access the attribute
    assert hasattr(provider, "fetch_all_radiation")
    assert provider.fetch_all_radiation is False  # type: ignore[attr-defined]


def test_weather_provider_can_enable_full_radiation(mock_storage, sample_location):
    """Tests that WeatherProvider can be configured to fetch all components."""
    provider = WeatherProvider(
        location=sample_location,
        start_date="2025-01-01",
        end_date="2025-01-10",
        organization="test-org",
        asset="test-asset",
        storage=mock_storage,
        fetch_all_radiation=True,
    )
    assert provider.fetch_all_radiation is True  # type: ignore[attr-defined]


# --- Radiation Decomposition Tests ---
@patch("pvlib.location")
@patch("pvlib.irradiance")
def test_base_weather_provider_decomposes_ghi_to_dni_dhi(
    mock_irradiance, mock_pvlib_location, sample_location
):
    """Tests that BaseWeatherProvider correctly decomposes GHI into DNI and DHI."""
    # Setup mock data with only GHI and temperature
    ghi_only_data = pd.DataFrame(
        {
            "ghi": [0.0, 120.5, 340.8],
            "temperature_celsius": [5.0, 5.4, 6.2],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="h"),
    )

    # Setup mocks
    mock_location_instance = MagicMock()
    mock_pvlib_location.Location.return_value = mock_location_instance
    mock_solar_position = pd.DataFrame(
        {"zenith": [90.0, 60.0, 45.0]}, index=ghi_only_data.index
    )
    mock_location_instance.get_solarposition.return_value = mock_solar_position

    mock_decomp_result = {
        "dni": pd.Series([0.0, 180.2, 520.1], index=ghi_only_data.index),
        "dhi": pd.Series([0.0, 60.3, 85.7], index=ghi_only_data.index),
    }
    mock_irradiance.erbs.return_value = mock_decomp_result

    # Create provider and test decomposition
    provider = BaseWeatherProvider(location=sample_location)
    result = provider.calculate_radiation_components(ghi_only_data)

    # Verify result has all radiation components
    assert "ghi" in result.columns
    assert "dni" in result.columns
    assert "dhi" in result.columns
    assert "temperature_celsius" in result.columns
    assert len(result) == 3

    # Verify PVLib was called correctly
    mock_irradiance.erbs.assert_called_once()


def test_base_weather_provider_skips_decomposition_when_all_present(sample_location):
    """Tests that decomposition is skipped when all radiation components are present."""
    # Setup data with all radiation components
    complete_data = pd.DataFrame(
        {
            "ghi": [0.0, 120.5, 340.8],
            "dni": [0.0, 180.2, 520.1],
            "dhi": [0.0, 60.3, 85.7],
            "temperature_celsius": [5.0, 5.4, 6.2],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="h"),
    )

    provider = BaseWeatherProvider(location=sample_location)
    result = provider.calculate_radiation_components(complete_data)

    # Should return data unchanged
    pd.testing.assert_frame_equal(result, complete_data)


# --- OpenMeteo Client Tests ---
@patch("app.core.simulation.open_meteo_data.requests.get")
def test_openmeteo_client_fetches_ghi_only_by_default(
    mock_get, ghi_only_api_response, sample_location
):
    """Tests that OpenMeteoClient fetches only GHI and temperature by default."""
    mock_response = MagicMock()
    mock_response.json.return_value = ghi_only_api_response
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    client = OpenMeteoClient()
    result = client.fetch_weather_data(
        location=sample_location,
        start_date="2025-01-01",
        end_date="2025-01-02",
        fetch_all_radiation=False,  # Default optimization mode
    )

    # Verify API was called with only GHI and temperature
    mock_get.assert_called_once()
    call_args = mock_get.call_args
    params = call_args[1]["params"]
    hourly_vars = params["hourly"].split(",")
    assert "shortwave_radiation" in hourly_vars  # GHI
    assert "temperature_2m" in hourly_vars  # Temperature
    assert "direct_normal_irradiance" not in hourly_vars  # DNI should not be requested
    assert "diffuse_radiation" not in hourly_vars  # DHI should not be requested

    # Verify result has standardized column names
    assert "ghi" in result.columns
    assert "temperature_celsius" in result.columns
    assert len(result) == 3


@patch("app.core.simulation.open_meteo_data.requests.get")
def test_openmeteo_client_fetches_all_radiation_when_requested(
    mock_get, full_radiation_api_response, sample_location
):
    """Tests that OpenMeteoClient fetches all radiation components when requested."""
    mock_response = MagicMock()
    mock_response.json.return_value = full_radiation_api_response
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    client = OpenMeteoClient()
    result = client.fetch_weather_data(
        location=sample_location,
        start_date="2025-01-01",
        end_date="2025-01-02",
        fetch_all_radiation=True,  # Fetch all components
    )

    # Verify API was called with all radiation variables
    mock_get.assert_called_once()
    call_args = mock_get.call_args
    params = call_args[1]["params"]
    hourly_vars = params["hourly"].split(",")
    assert "shortwave_radiation" in hourly_vars
    assert "direct_normal_irradiance" in hourly_vars
    assert "diffuse_radiation" in hourly_vars
    assert "temperature_2m" in hourly_vars

    # Verify result has all radiation components
    assert "ghi" in result.columns
    assert "dni" in result.columns
    assert "dhi" in result.columns
    assert "temperature_celsius" in result.columns


# --- Integration Tests ---
@patch("app.core.simulation.open_meteo_data.requests.get")
@patch("pvlib.irradiance")
@patch("pvlib.location")
@pytest.mark.asyncio
async def test_openmeteo_provider_full_workflow_with_decomposition(
    mock_pvlib_location,
    mock_irradiance,
    mock_get,
    ghi_only_api_response,
    sample_location,
    mock_storage,
):
    """Tests the full workflow: fetch GHI only → decompose → return complete data."""
    # Setup API mock
    mock_response = MagicMock()
    mock_response.json.return_value = ghi_only_api_response
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # Setup PVLib mocks
    mock_location_instance = MagicMock()
    mock_pvlib_location.Location.return_value = mock_location_instance
    mock_solar_position = pd.DataFrame(
        {"zenith": [90.0, 60.0, 45.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="h"),
    )
    mock_location_instance.get_solarposition.return_value = mock_solar_position

    mock_decomp_result = {
        "dni": pd.Series([0.0, 180.2, 520.1]),
        "dhi": pd.Series([0.0, 60.3, 85.7]),
    }
    mock_irradiance.erbs.return_value = mock_decomp_result

    # Create provider (should default to optimized mode)
    provider = WeatherProvider(
        location=sample_location,
        start_date="2025-01-01",
        end_date="2025-01-02",
        organization="test-org",
        asset="test-asset",
        storage=mock_storage,
    )

    # Get weather data using async interface
    provider.set_range(datetime(2025, 1, 1), datetime(2025, 1, 2))
    result = await provider.get_data()  # type: ignore[attr-defined]

    # Verify we got complete radiation data from GHI-only API call
    assert not result.empty
    assert "ghi" in result.columns
    assert "dni" in result.columns  # Should be calculated
    assert "dhi" in result.columns  # Should be calculated
    assert "temperature_celsius" in result.columns

    # Verify API was called in optimized mode (GHI + temp only)
    call_args = mock_get.call_args
    params = call_args[1]["params"]
    hourly_vars = params["hourly"].split(",")
    assert len(hourly_vars) == 2  # Only GHI and temperature


def test_weather_provider_validation_requires_ghi_minimum(sample_location):
    """Tests that weather data validation requires at least GHI and temperature."""
    provider = BaseWeatherProvider(location=sample_location)

    # Test with missing GHI
    missing_ghi = pd.DataFrame(
        {
            "dni": [100.0, 200.0],
            "temperature_celsius": [15.0, 16.0],
        }
    )
    assert not provider.validate_data_format(missing_ghi)

    # Test with missing temperature
    missing_temp = pd.DataFrame(
        {
            "ghi": [100.0, 200.0],
            "dni": [100.0, 200.0],
        }
    )
    assert not provider.validate_data_format(missing_temp)

    # Test with GHI and temperature (minimum requirement)
    valid_minimum = pd.DataFrame(
        {
            "ghi": [100.0, 200.0],
            "temperature_celsius": [15.0, 16.0],
        }
    )
    assert provider.validate_data_format(valid_minimum)


# --- Historical Provider Validation Tests ---
def test_weather_provider_valid_historical_range(mock_storage, sample_location):
    """Tests that a valid historical date range passes validation."""
    provider = WeatherProvider(
        location=sample_location,
        start_date="2025-01-01",
        end_date="2025-01-10",
        organization="test-org",
        asset="test-asset",
        storage=mock_storage,
    )
    assert provider is not None


def test_weather_provider_valid_forecast_range(mock_storage, sample_location):
    """Tests that a valid future forecast range passes validation."""
    start = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=MAX_FORECAST_DAYS - 1)).strftime("%Y-%m-%d")
    provider = WeatherProvider(
        location=sample_location,
        start_date=start,
        end_date=end,
        organization="test-org",
        asset="test-asset",
        storage=mock_storage,
    )
    assert provider is not None


def test_weather_provider_invalid_forecast_range_raises_error(
    mock_storage, sample_location
):
    """Tests that a forecast range exceeding the max limit raises a ValueError."""
    start = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=MAX_FORECAST_DAYS + 1)).strftime("%Y-%m-%d")
    with pytest.raises(
        ValueError, match=f"Forecast range cannot exceed {MAX_FORECAST_DAYS} days."
    ):
        WeatherProvider(
            location=sample_location,
            start_date=start,
            end_date=end,
            organization="test-org",
            asset="test-asset",
            storage=mock_storage,
        )


class TestCSVWeatherProvider:
    """Test CSV weather provider functionality."""

    @pytest.fixture
    def sample_csv_path(self) -> Path:
        """Provide path to sample CSV file with all weather data."""
        return Path(__file__).parent.parent.parent / "data" / "sample_weather_data.csv"

    @pytest.fixture
    def ghi_only_csv_path(self) -> Path:
        """Provide path to sample CSV file with only GHI and temperature."""
        return (
            Path(__file__).parent.parent.parent / "data" / "sample_weather_ghi_only.csv"
        )

    @pytest.fixture
    def sample_location(self) -> GeospatialLocation:
        """Provide a sample location for testing."""
        return GeospatialLocation(latitude=52.52, longitude=13.41)

    @pytest.fixture
    def csv_provider(
        self, sample_csv_path: Path, sample_location: GeospatialLocation
    ) -> CSVWeatherProvider:
        """Create CSV weather provider with sample data."""
        return CSVWeatherProvider(
            location=sample_location, file_path=str(sample_csv_path)
        )

    @pytest.fixture
    def ghi_only_csv_provider(
        self, ghi_only_csv_path: Path, sample_location: GeospatialLocation
    ) -> CSVWeatherProvider:
        """Create CSV weather provider with GHI-only sample data."""
        return CSVWeatherProvider(
            location=sample_location, file_path=str(ghi_only_csv_path)
        )

    def test_csv_provider_initialization(
        self, sample_csv_path: Path, sample_location: GeospatialLocation
    ):
        """Test CSV provider can be initialized with valid file path."""
        provider = CSVWeatherProvider(
            location=sample_location, file_path=str(sample_csv_path)
        )
        assert provider.file_path == str(sample_csv_path)
        assert provider.location == sample_location

    @pytest.mark.asyncio
    async def test_get_weather_data_success(self, csv_provider: CSVWeatherProvider):
        """Test getting weather data from CSV file."""
        start_time = datetime(2025, 7, 15, 0, 0, 0)
        end_time = datetime(2025, 7, 15, 23, 59, 59)
        csv_provider.set_range(start_time, end_time)
        data = await csv_provider.get_data()

        assert not data.empty
        assert WeatherColumns.GHI.value in data.columns
        assert WeatherColumns.TEMPERATURE.value in data.columns
        assert WeatherColumns.DNI.value in data.columns
        assert WeatherColumns.DHI.value in data.columns
        assert len(data) == 24  # 24 hours of data

    @pytest.mark.asyncio
    async def test_get_weather_data_partial_timerange(
        self, csv_provider: CSVWeatherProvider
    ):
        """Test getting weather data for a partial time range."""
        start_time = datetime(2025, 7, 15, 10, 0, 0)
        end_time = datetime(2025, 7, 15, 14, 0, 0)
        csv_provider.set_range(start_time, end_time)
        data = await csv_provider.get_data()

        assert not data.empty
        assert len(data) == 5  # 10:00, 11:00, 12:00, 13:00, 14:00
        # Check that we get expected GHI values for midday hours
        assert data[WeatherColumns.GHI.value].max() > 700

    @pytest.mark.asyncio
    async def test_ghi_only_csv_with_radiation_decomposition(
        self, ghi_only_csv_provider: CSVWeatherProvider
    ):
        """Test CSV provider with only GHI data - decompose radiation components."""
        start_time = datetime(2025, 7, 15, 6, 0, 0)
        end_time = datetime(2025, 7, 15, 18, 0, 0)
        ghi_only_csv_provider.set_range(start_time, end_time)
        data = await ghi_only_csv_provider.get_data()

        assert not data.empty
        assert WeatherColumns.GHI.value in data.columns
        assert WeatherColumns.TEMPERATURE.value in data.columns
        assert WeatherColumns.DNI.value in data.columns  # Should be calculated
        assert WeatherColumns.DHI.value in data.columns  # Should be calculated

        # Verify radiation decomposition worked - DNI and DHI should be non-zero
        noon_row = data[data.index == datetime(2025, 7, 15, 12, 0, 0)]
        assert not noon_row.empty
        assert noon_row[WeatherColumns.GHI.value].iloc[0] > 0
        assert noon_row[WeatherColumns.DNI.value].iloc[0] > 0
        assert noon_row[WeatherColumns.DHI.value].iloc[0] > 0

    @pytest.mark.asyncio
    async def test_csv_provider_with_missing_file(
        self, sample_location: GeospatialLocation
    ):
        """Test CSV provider behavior with non-existent file."""
        nonexistent_path = "/nonexistent/path/weather.csv"
        provider = CSVWeatherProvider(
            location=sample_location, file_path=nonexistent_path
        )

        start_time = datetime(2025, 7, 15, 0, 0, 0)
        end_time = datetime(2025, 7, 15, 23, 59, 59)
        provider.set_range(start_time, end_time)
        data = await provider.get_data()
        assert data.empty

    @pytest.mark.asyncio
    async def test_csv_provider_calls_radiation_decomposition(
        self, ghi_only_csv_provider
    ):
        """Test that CSV provider calls radiation decomposition for GHI-only data."""
        # Mock the calculate_radiation_components method
        with patch.object(
            ghi_only_csv_provider, "calculate_radiation_components"
        ) as mock_calculate:
            mock_calculate.return_value = pd.DataFrame(
                {
                    WeatherColumns.GHI.value: [500.0],
                    WeatherColumns.DNI.value: [700.0],
                    WeatherColumns.DHI.value: [200.0],
                    WeatherColumns.TEMPERATURE.value: [25.0],
                }
            )

            start_time = datetime(2025, 7, 15, 12, 0, 0)
            end_time = datetime(2025, 7, 15, 12, 0, 0)

            ghi_only_csv_provider.set_range(start_time, end_time)
            data = await ghi_only_csv_provider.get_data()

            # Should call radiation decomposition
            mock_calculate.assert_called_once()
            assert not data.empty

    @pytest.mark.asyncio
    async def test_csv_provider_validates_required_columns(
        self, tmp_path, sample_location: GeospatialLocation
    ):
        """Test CSV provider behavior with missing required columns."""
        # Create CSV with missing required columns
        invalid_csv = tmp_path / "invalid_weather.csv"
        invalid_csv.write_text("timestamp,some_other_column\n2025-07-15 12:00:00,123\n")

        provider = CSVWeatherProvider(
            location=sample_location, file_path=str(invalid_csv)
        )
        start_time = datetime(2025, 7, 15, 0, 0, 0)
        end_time = datetime(2025, 7, 15, 23, 59, 59)
        provider.set_range(start_time, end_time)
        data = await provider.get_data()
        # Should still load the data but with warning about missing radiation components
        assert not data.empty
        assert "some_other_column" in data.columns
