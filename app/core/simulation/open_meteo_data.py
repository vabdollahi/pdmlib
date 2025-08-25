"""
OpenMeteo weather data provider implementation.

This module provides OpenMeteo-specific functionality for fetching weather data,
including the optimized approach that fetches only GHI and temperature by default
to reduce API costs, then calculates other radiation components using PVLib.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.simulation.weather_provider import BaseWeatherProvider, WeatherDataColumns
from app.core.utils.caching import BaseProvider
from app.core.utils.date_handling import TimeInterval, parse_date_string
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger
from app.core.utils.storage import DataStorage

logger = get_logger("open_meteo_data")


# --- Constants ---

OPEN_METEO_FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
RETRY_COUNT = 3
MAX_FORECAST_DAYS = 7


# -----------------------------------------------------------------------------
# OpenMeteo Client Configuration
# -----------------------------------------------------------------------------


class OpenMeteoConfig(BaseModel):
    """Configuration for OpenMeteo API requests."""

    base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    hourly_variables: List[str] = Field(default_factory=list)
    timezone: str = "UTC"
    format: str = "json"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OpenMeteoClient:
    """
    Client for fetching weather data from OpenMeteo API.

    Supports optimized fetching that requests only GHI and temperature
    to reduce API costs, with radiation decomposition handled separately.
    """

    def __init__(self, config: Optional[OpenMeteoConfig] = None):
        """Initialize with configuration."""
        self.config = config or OpenMeteoConfig()

    def fetch_weather_data(
        self,
        location: GeospatialLocation,
        start_date: str,
        end_date: str,
        fetch_all_radiation: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch weather data from OpenMeteo API.

        Args:
            location: Geographic location
            start_date: Start date in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format
            end_date: End date in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format
            fetch_all_radiation: If True, fetch all radiation components.
                                If False (default), fetch only GHI for optimization.

        Returns:
            DataFrame with weather data using standardized column names
        """
        try:
            # Convert datetime strings to date-only format for OpenMeteo API
            # OpenMeteo archive API expects YYYY-MM-DD format, not full datetime
            start_date_formatted = start_date.split(" ")[0]  # Extract date part
            end_date_formatted = end_date.split(" ")[0]  # Extract date part

            # Select variables based on optimization setting
            if fetch_all_radiation:
                variables = [
                    WeatherDataColumns.OPENMETEO_GHI.value,
                    WeatherDataColumns.OPENMETEO_DNI.value,
                    WeatherDataColumns.OPENMETEO_DHI.value,
                    WeatherDataColumns.OPENMETEO_TEMP.value,
                ]
                logger.info("Fetching all radiation components from OpenMeteo API")
            else:
                variables = [
                    WeatherDataColumns.OPENMETEO_GHI.value,
                    WeatherDataColumns.OPENMETEO_TEMP.value,
                ]
                logger.info(
                    "Fetching optimized variables (GHI + temperature) "
                    "from OpenMeteo API"
                )

            # Build API request parameters
            params = {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "start_date": start_date_formatted,
                "end_date": end_date_formatted,
                "hourly": ",".join(variables),
                "timezone": self.config.timezone,
                "format": self.config.format,
            }

            # Make API request
            response = requests.get(self.config.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse response into DataFrame
            hourly_data = data.get("hourly", {})
            if not hourly_data:
                logger.warning("No hourly data in OpenMeteo response")
                return pd.DataFrame()

            # Create DataFrame from API response
            df_data = {}

            # Add timestamp
            timestamps = hourly_data.get("time", [])
            if not timestamps:
                logger.error("No timestamps in OpenMeteo response")
                return pd.DataFrame()

            df_data["timestamp"] = pd.to_datetime(timestamps)

            # Add weather variables with standardized names
            variable_mapping = {
                WeatherDataColumns.OPENMETEO_GHI.value: WeatherDataColumns.GHI.value,
                WeatherDataColumns.OPENMETEO_DNI.value: WeatherDataColumns.DNI.value,
                WeatherDataColumns.OPENMETEO_DHI.value: WeatherDataColumns.DHI.value,
                WeatherDataColumns.OPENMETEO_TEMP.value: (
                    WeatherDataColumns.TEMPERATURE.value
                ),
            }

            for api_var, std_var in variable_mapping.items():
                if api_var in hourly_data:
                    df_data[std_var] = hourly_data[api_var]

            # Create DataFrame and set timestamp as index
            df = pd.DataFrame(df_data)
            df.set_index("timestamp", inplace=True)

            logger.info(
                f"Successfully fetched {len(df)} weather records from OpenMeteo"
            )
            return df

        except requests.RequestException as e:
            logger.error(f"HTTP error fetching weather data from OpenMeteo: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching weather data from OpenMeteo: {e}")
            return pd.DataFrame()


# -----------------------------------------------------------------------------
# OpenMeteo Weather Provider
# -----------------------------------------------------------------------------


class OpenMeteoWeatherProvider(BaseProvider, BaseWeatherProvider):
    """
    OpenMeteo weather provider with caching and radiation decomposition.

    By default, fetches only GHI and temperature to optimize API costs,
    then calculates DNI and DHI using PVLib decomposition models.
    """

    data_type: str = Field(default="weather", description="Type of data.")
    fetch_all_radiation: bool = Field(
        default=False, description="If True, fetch all radiation components from API"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        location: GeospatialLocation,
        start_date: str,
        end_date: str,
        organization: str,
        asset: str,
        interval: TimeInterval = TimeInterval.HOURLY,
        storage: Optional[DataStorage] = None,
        fetch_all_radiation: bool = False,
        **kwargs,
    ):
        """
        Initialize OpenMeteo weather provider.

        Args:
            location: Geographic location for weather data
            start_date: Start date for data range
            end_date: End date for data range
            organization: Organization name for storage
            asset: Asset name for storage
            interval: Data interval (default: hourly)
            storage: Data storage configuration
            fetch_all_radiation: If True, fetch all radiation components.
                               If False (default), fetch only GHI for optimization.
        """
        # Initialize BaseProvider for caching
        BaseProvider.__init__(
            self,
            location=location,
            start_date=start_date,
            end_date=end_date,
            organization=organization,
            asset=asset,
            data_type="weather",
            interval=interval,
            storage=storage or DataStorage(),
        )

        # Initialize BaseWeatherProvider for weather-specific functionality
        BaseWeatherProvider.__init__(self, location=location, **kwargs)

        # Set instance attributes that may not be handled by Pydantic
        self.fetch_all_radiation = fetch_all_radiation
        self.client = OpenMeteoClient()

    @model_validator(mode="after")
    def validate_date_range(self) -> "OpenMeteoWeatherProvider":
        """Validate forecast requests don't exceed maximum allowed days."""
        start_date = parse_date_string(self.start_date)
        end_date = parse_date_string(self.end_date)
        today = datetime.now().date()

        if start_date >= today:
            if (end_date - start_date).days >= MAX_FORECAST_DAYS:
                raise ValueError(
                    f"Forecast range cannot exceed {MAX_FORECAST_DAYS} days."
                )
        return self

    async def _fetch_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch weather data for a specific date range.

        This method is called by BaseProvider's caching system.
        """
        try:
            # Fetch data from OpenMeteo API
            raw_df = self.client.fetch_weather_data(
                location=self.location,  # type: ignore[arg-type]
                start_date=start_date,
                end_date=end_date,
                fetch_all_radiation=self.fetch_all_radiation,
            )

            if raw_df.empty:
                logger.warning(f"No data returned for range {start_date} to {end_date}")
                return pd.DataFrame()

            # Validate data format
            if not self.validate_data_format(raw_df):
                logger.error("Weather data failed validation")
                return pd.DataFrame()

            # Apply radiation decomposition if needed
            enhanced_df = self.calculate_radiation_components(raw_df)

            # Log optimization information
            if not self.fetch_all_radiation:
                has_decomposed = (
                    WeatherDataColumns.DNI in enhanced_df.columns
                    and WeatherDataColumns.DHI in enhanced_df.columns
                )
                if has_decomposed:
                    logger.info(
                        "API optimization: fetched 2 variables, "
                        "calculated 2 additional (50% cost reduction)"
                    )

            return enhanced_df

        except Exception as e:
            logger.error(
                f"Error fetching OpenMeteo data range {start_date}-{end_date}: {e}"
            )
            return pd.DataFrame()

    def _get_cache_key_suffix(self) -> str:
        """Generate cache key suffix for this provider configuration."""
        optimization_flag = "optimized" if not self.fetch_all_radiation else "full"
        # Type: ignore is safe here since we enforce GeospatialLocation in __init__
        lat = self.location.latitude  # type: ignore[attr-defined]
        lon = self.location.longitude  # type: ignore[attr-defined]
        return f"openmeteo_{optimization_flag}_{lat}_{lon}"
