"""
This module provides a weather data client for the Open-Meteo API, which offers
free historical, live, and forecast weather data. It is designed to be used
with the pvlib library for photovoltaic system modeling.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_fixed

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


class OpenMeteoClient:
    """
    An asynchronous client for fetching weather data from the Open-Meteo API.
    It handles API requests and retries.
    """

    @retry(stop=stop_after_attempt(RETRY_COUNT), wait=wait_fixed(1))
    async def get_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: List[str],
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

        Returns:
            A pandas DataFrame with the requested weather data.

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        ninety_days_ago = datetime.now().date() - timedelta(days=90)

        api_url = (
            OPEN_METEO_ARCHIVE_API_URL
            if start_date_dt < ninety_days_ago
            else OPEN_METEO_FORECAST_API_URL
        )

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(api_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return self._process_response(data, variables)
            except aiohttp.ClientError as e:
                print(f"Error fetching weather data: {e}")
                raise

    def _process_response(
        self, response_data: dict, variables: List[str]
    ) -> pd.DataFrame:
        """Processes the JSON response from the Open-Meteo API."""
        hourly_data = response_data["hourly"]

        time_range = pd.to_datetime(hourly_data["time"])

        data = {var: hourly_data[var] for var in variables}

        df = pd.DataFrame(data=data, index=time_range)
        df.index.name = WeatherDataColumns.DATE_TIME.value

        # Rename columns to pvlib standard
        return df.rename(
            columns={
                "shortwave_radiation": "ghi",
                "direct_normal_irradiance": "dni",
                "diffuse_radiation": "dhi",
                "temperature_2m": "temp_air",
            }
        )


# --- Weather Data Provider ---


class WeatherProvider(BaseModel):
    """
    Provides weather data using the Open-Meteo client. This class acts as the
    main entry point for fetching weather data.
    """

    latitude: float = Field(description="Latitude of the location.")
    longitude: float = Field(description="Longitude of the location.")
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")

    @model_validator(mode="after")
    def validate_date_range(self) -> "WeatherProvider":
        """
        Validates that forecast requests do not exceed the maximum allowed days.
        """
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        today = datetime.now().date()

        # Check if the request is for a future forecast
        if start_date > today:
            if (end_date - start_date).days > MAX_FORECAST_DAYS:
                raise ValueError(
                    f"Forecast range cannot exceed {MAX_FORECAST_DAYS} days."
                )
        return self

    async def get_weather_data(self) -> pd.DataFrame:
        """
        Retrieves weather data for the configured location and time range.

        Returns:
            A pandas DataFrame containing the weather data.
        """
        client = OpenMeteoClient()
        variables = [
            WeatherDataColumns.GHI,
            WeatherDataColumns.DNI,
            WeatherDataColumns.DHI,
            WeatherDataColumns.TEMPERATURE,
        ]
        variables_str = [v.value for v in variables]
        return await client.get_weather_data(
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=self.start_date,
            end_date=self.end_date,
            variables=variables_str,
        )
