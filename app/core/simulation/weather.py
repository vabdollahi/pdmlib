"""
This module provides a weather data client for the Open-Meteo API, which offers
free historical, live, and forecast weather data. It is designed to be used
with the pvlib library for photovoltaic system modeling.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

import pandas as pd
from pydantic import ConfigDict, Field, model_validator

from app.core.utils.api_clients import BaseAPIClient
from app.core.utils.caching import BaseProvider
from app.core.utils.data_processing import (
    resample_timeseries_data,
    standardize_column_names,
)
from app.core.utils.date_handling import TimeInterval, parse_date_string
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger

logger = get_logger("weather")


# --- Constants and Enums ---

OPEN_METEO_FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
RETRY_COUNT = 3
MAX_FORECAST_DAYS = 7


class WeatherDataColumns(str, Enum):
    """Standardized column names for weather data."""

    DATE_TIME = "date_time"
    GHI = "shortwave_radiation"
    DNI = "direct_normal_irradiance"
    DHI = "diffuse_radiation"
    TEMPERATURE = "temperature_2m"


# --- Open-Meteo API Client ---


class OpenMeteoClient(BaseAPIClient):
    """
    An asynchronous client for fetching weather data from the Open-Meteo API.
    It handles API requests and retries using the base API client functionality.
    """

    def __init__(self):
        """Initialize the Open-Meteo client with appropriate retry settings."""
        # Don't set a base URL since we use different endpoints for date ranges
        super().__init__(base_url="", retry_count=RETRY_COUNT, retry_wait=1)

    async def get_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: List[str],
        interval: Optional["TimeInterval"] = None,
    ) -> pd.DataFrame:
        """
        Fetches weather data for a given location and time range.
        Dynamically selects the API endpoint based on the requested date range.

        Args:
            latitude: The latitude of the location.
            longitude: The longitude of the location.
            start_date: The start date in YYYY-MM-DD format.
            end_date: The end date in YYYY-MM-DD format.
            variables: A list of weather variables to fetch.
            interval: Time interval for data (Note: Open-Meteo API only supports
                hourly for most data).

        Returns:
            A pandas DataFrame with the requested weather data.

        Raises:
            aiohttp.ClientError: If the API request fails.
            ValueError: If interval is not supported by Open-Meteo API.
        """
        # Import here to avoid circular imports
        from app.core.utils.date_handling import TimeInterval

        if interval is None:
            interval = TimeInterval.HOURLY

        # Always fetch hourly data from Open-Meteo API
        # We'll resample to other intervals if needed
        if interval != TimeInterval.HOURLY:
            logger.warning(
                f"Open-Meteo API doesn't natively support {interval.display_name} "
                f"intervals. Fetching hourly data and will resample to "
                f"{interval.display_name}."
            )

        start_date_dt = parse_date_string(start_date)
        end_date_dt = parse_date_string(end_date)
        ninety_days_ago = datetime.now().date() - timedelta(days=90)

        api_url = (
            OPEN_METEO_ARCHIVE_API_URL
            if start_date_dt < ninety_days_ago
            else OPEN_METEO_FORECAST_API_URL
        )

        # Convert datetime strings to date-only format for API (YYYY-MM-DD)
        api_start_date = start_date_dt.strftime("%Y-%m-%d")
        api_end_date = end_date_dt.strftime("%Y-%m-%d")

        # Always use hourly data from the API
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": api_start_date,
            "end_date": api_end_date,
            "hourly": ",".join(variables),
            "timezone": "auto",
        }

        # Use the base API client functionality for the request
        # Since we need different base URLs, we'll override the URL construction
        endpoint = api_url
        data = await self.get_json(endpoint, params)
        df = self._process_response(data, variables)

        # Resample if needed (all intervals except hourly need resampling)
        if interval != TimeInterval.HOURLY:
            df = resample_timeseries_data(df, interval)

        return df

    def _process_response(
        self,
        response_data: dict,
        variables: List[str],
    ) -> pd.DataFrame:
        """Processes the JSON response from the Open-Meteo API (hourly data only)."""

        # Always use hourly data section
        data_section = response_data.get("hourly", {})

        time_range = pd.to_datetime(data_section["time"])
        data = {var: data_section[var] for var in variables}

        df = pd.DataFrame(data=data, index=time_range)
        df.index.name = WeatherDataColumns.DATE_TIME.value

        # Rename columns to pvlib standard (hourly data only)
        column_mapping = {
            "shortwave_radiation": "ghi",
            "direct_normal_irradiance": "dni",
            "diffuse_radiation": "dhi",
            "temperature_2m": "temp_air",
        }
        return standardize_column_names(df, column_mapping)


# --- Weather Data Provider ---


class WeatherProvider(BaseProvider):
    """
    Provides weather data, using a caching layer before falling back to the
    Open-Meteo API. This class is the main entry point for fetching weather data.
    """

    data_type: str = Field(default="weather", description="Type of data.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, location: GeospatialLocation, **kwargs):
        """Initialize with a GeospatialLocation."""
        super().__init__(location=location, **kwargs)

    @model_validator(mode="after")
    def validate_date_range(self) -> "WeatherProvider":
        """
        Validates that forecast requests do not exceed the maximum allowed days.
        """
        start_date = parse_date_string(self.start_date)
        end_date = parse_date_string(self.end_date)
        today = datetime.now().date()

        # Check if the request is for a future forecast, considering a small buffer
        if start_date >= today:
            # The forecast API allows up to 16 days, but we limit it
            if (end_date - start_date).days >= MAX_FORECAST_DAYS:
                raise ValueError(
                    f"Forecast range cannot exceed {MAX_FORECAST_DAYS} days."
                )
        return self

    async def _fetch_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches weather data for a specific date range from the Open-Meteo API.

        Args:
            start_date: The start date in YYYY-MM-DD format.
            end_date: The end date in YYYY-MM-DD format.

        Returns:
            A pandas DataFrame with the requested weather data.
        """
        client = OpenMeteoClient()
        variables = [
            WeatherDataColumns.GHI,
            WeatherDataColumns.DNI,
            WeatherDataColumns.DHI,
            WeatherDataColumns.TEMPERATURE,
        ]
        variables_str = [v.value for v in variables]

        # Ensure we have a GeospatialLocation
        if not isinstance(self.location, GeospatialLocation):
            raise ValueError("WeatherProvider requires a GeospatialLocation")

        return await client.get_weather_data(
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            start_date=start_date,
            end_date=end_date,
            variables=variables_str,
            interval=self.interval,  # Pass the interval to the client
        )
