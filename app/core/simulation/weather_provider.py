"""
Shared weather provider primitives and facade.

This module defines:
- WeatherColumns: unified column names for weather data
- BaseWeatherProvider: common provider interface with radiation decomposition
- CSVWeatherProvider: generic CSV-backed provider
- create_weather_provider: factory that routes to OpenMeteo/CSV
- WeatherProvider facade

Provider-specific implementations (OpenMeteo) remain in their modules and
inherit from BaseProvider (cache/crawl) and BaseWeatherProvider (interface/helpers).

By default, providers fetch only GHI and temperature, then calculate DNI and DHI
using PVLib radiation decomposition models for API cost optimization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core.utils.data_processing import standardize_column_names
from app.core.utils.date_handling import TimeInterval
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger
from app.core.utils.storage import DataStorage

logger = get_logger("weather_provider")


# -----------------------------------------------------------------------------
# Unified columns
# -----------------------------------------------------------------------------


class WeatherColumns(str, Enum):
    """Standardized column names for weather data."""

    DATE_TIME = "date_time"
    GHI = "ghi"
    DNI = "dni"
    DHI = "dhi"
    TEMPERATURE = "temperature_celsius"

    # OpenMeteo API variable names
    OPENMETEO_GHI = "shortwave_radiation"
    OPENMETEO_DNI = "direct_normal_irradiance"
    OPENMETEO_DHI = "diffuse_radiation"
    OPENMETEO_TEMP = "temperature_2m"


# -----------------------------------------------------------------------------
# Base provider interface and helpers
# -----------------------------------------------------------------------------


@runtime_checkable
class WeatherProviderProtocol(Protocol):
    """Protocol for weather data providers.

    Single, canonical async method get_data() must be implemented.
    """

    async def get_data(self, force_refresh: bool = False) -> pd.DataFrame: ...

    def validate_data_format(self, df: pd.DataFrame) -> bool: ...


## Removed SyncWeatherProviderProtocol: async-only providers are supported.


class BaseWeatherProvider:
    """
    Base interface and helpers for weather data providers.

    Providers fetch GHI and temperature by default, then calculate DNI and DHI
    using PVLib radiation decomposition models to optimize API costs.

    Providers typically also inherit from app.core.utils.caching.BaseProvider to
    gain async cache-first `get_data()` and range fetching.
    """

    def __init__(self, location: GeospatialLocation, **kwargs):
        """Initialize with location for solar position calculations."""
        self.location = location
        # Optional range hint for providers that support local filtering
        self._range_start: Optional[datetime] = None
        self._range_end: Optional[datetime] = None

    def set_range(self, start_time: datetime, end_time: datetime) -> None:
        """Optionally hint a time range for subsequent get_data() calls."""
        self._range_start = start_time
        self._range_end = end_time

    def calculate_radiation_components(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate missing radiation components using PVLib decomposition models.

        If only GHI is available, calculates DNI and DHI using the Erbs model.
        Returns the enhanced weather DataFrame with all radiation components.
        """
        if weather_df.empty:
            return weather_df

        # Check what radiation components we have
        has_ghi = WeatherColumns.GHI.value in weather_df.columns
        has_dni = WeatherColumns.DNI.value in weather_df.columns
        has_dhi = WeatherColumns.DHI.value in weather_df.columns

        # If we have all components, no decomposition needed
        if has_ghi and has_dni and has_dhi:
            logger.debug("All radiation components available, no decomposition needed")
            return weather_df

        # If we only have GHI, calculate DNI and DHI
        if has_ghi and not has_dni and not has_dhi:
            logger.info("Calculating DNI and DHI from GHI using PVLib Erbs model")
            return self._decompose_ghi_to_dni_dhi(weather_df)

        # For other partial cases, warn about unexpected combination
        logger.warning(
            f"Unexpected radiation component combination: "
            f"GHI={has_ghi}, DNI={has_dni}, DHI={has_dhi}"
        )
        return weather_df

    def _decompose_ghi_to_dni_dhi(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate DNI and DHI from GHI using PVLib Erbs decomposition model."""
        from pvlib import irradiance
        from pvlib import location as pvlib_location

        # Create PVLib location for solar position calculations
        # Handle timezone properly for datetime index
        tz = getattr(weather_df.index, "tz", None) or "UTC"
        pvlib_loc = pvlib_location.Location(
            latitude=self.location.latitude, longitude=self.location.longitude, tz=tz
        )

        # Get solar position for the time range
        solar_position = pvlib_loc.get_solarposition(weather_df.index)

        # Get day of year from datetime index
        day_of_year = weather_df.index.to_series().dt.dayofyear

        # Use Erbs model to decompose GHI into DNI and DHI
        decomp_result = irradiance.erbs(
            ghi=weather_df[WeatherColumns.GHI.value],
            zenith=solar_position["zenith"],
            datetime_or_doy=day_of_year,
        )

        # Add calculated DNI and DHI to the DataFrame
        enhanced_df = weather_df.copy()
        enhanced_df[WeatherColumns.DNI.value] = decomp_result["dni"]
        enhanced_df[WeatherColumns.DHI.value] = decomp_result["dhi"]

        logger.debug(
            f"Decomposed GHI into DNI and DHI for {len(weather_df)} data points"
        )
        return enhanced_df

    def validate_data_format(self, df: pd.DataFrame) -> bool:
        """Validate that the weather data has required columns and format."""
        required_columns = [
            WeatherColumns.GHI.value,
            WeatherColumns.TEMPERATURE.value,
        ]

        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False

        if df.empty:
            logger.error("Weather data is empty")
            return False

        return True


# -----------------------------------------------------------------------------
# CSV Weather Provider
# -----------------------------------------------------------------------------


class CSVWeatherProvider(BaseWeatherProvider):
    """
    CSV-based weather provider that reads weather data from files.

    Supports automatic radiation decomposition if only GHI is available.
    """

    def __init__(self, location: GeospatialLocation, file_path: str, **kwargs):
        """Initialize with location and CSV file path."""
        super().__init__(location=location, **kwargs)
        self.file_path = file_path

    # Removed sync get_weather_data; async-only interface below.

    async def get_data(
        self, force_refresh: bool = False
    ) -> pd.DataFrame:  # pragma: no cover - thin async bridge
        """Async interface delegating to get_weather_data.

        If callers previously set a date range via BaseWeatherProvider.get_weather_data,
        use that range. Otherwise, return the entire file by querying an intentionally
        wide window that captures all rows.
        """
        # Determine desired range using optional hints
        start = self._range_start or datetime(1900, 1, 1)
        end = self._range_end or datetime(2100, 1, 1)

        try:
            # Read CSV file
            df = pd.read_csv(self.file_path)

            # Ensure datetime index with UTC timezone
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)
            elif WeatherColumns.DATE_TIME.value in df.columns:
                df[WeatherColumns.DATE_TIME.value] = pd.to_datetime(
                    df[WeatherColumns.DATE_TIME.value], utc=True
                )
                df.set_index(WeatherColumns.DATE_TIME.value, inplace=True)
            else:
                # Assume first column is datetime
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True)
                df.set_index(df.columns[0], inplace=True)

            # Filter by date range (if any was set), aligning timezone awareness
            idx_tz = getattr(df.index, "tz", None)

            # Prepare comparable bounds respecting index tz-awareness
            start_cmp: datetime | pd.Timestamp
            end_cmp: datetime | pd.Timestamp

            if idx_tz is None:
                # Index is timezone-naive; compare against naive datetimes
                start_cmp = (
                    start.replace(tzinfo=None)
                    if hasattr(start, "tzinfo") and start.tzinfo is not None
                    else start
                )
                end_cmp = (
                    end.replace(tzinfo=None)
                    if hasattr(end, "tzinfo") and end.tzinfo is not None
                    else end
                )
            else:
                # Index is timezone-aware; convert bounds to the same timezone
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                start_cmp = (
                    start_ts.tz_localize(idx_tz)
                    if start_ts.tz is None
                    else start_ts.tz_convert(idx_tz)
                )
                end_cmp = (
                    end_ts.tz_localize(idx_tz)
                    if end_ts.tz is None
                    else end_ts.tz_convert(idx_tz)
                )

            mask = (df.index >= start_cmp) & (df.index <= end_cmp)
            filtered_df = df[mask].copy()

            # Standardize column names if needed
            column_mapping = {
                "ghi": WeatherColumns.GHI.value,
                "dni": WeatherColumns.DNI.value,
                "dhi": WeatherColumns.DHI.value,
                "temp_air": WeatherColumns.TEMPERATURE.value,
                "temperature": WeatherColumns.TEMPERATURE.value,
                "temperature_celsius": WeatherColumns.TEMPERATURE.value,
                "shortwave_radiation": WeatherColumns.GHI.value,
                "direct_normal_irradiance": WeatherColumns.DNI.value,
                "diffuse_radiation": WeatherColumns.DHI.value,
                "temperature_2m": WeatherColumns.TEMPERATURE.value,
            }

            standardized_df = standardize_column_names(filtered_df, column_mapping)

            # Apply radiation decomposition if needed
            enhanced_df = self.calculate_radiation_components(standardized_df)

            logger.info(f"Loaded {len(enhanced_df)} weather records from CSV")
            return enhanced_df

        except Exception as e:
            logger.error(f"Failed to read weather data from CSV: {e}")
            return pd.DataFrame()


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def create_weather_provider(
    location: GeospatialLocation, provider_type: str = "openmeteo", **kwargs
) -> BaseWeatherProvider:
    """
    Factory function to create weather providers.

    Args:
        location: Geographic location for weather data
        provider_type: Type of provider ("openmeteo", "csv")
        **kwargs: Additional arguments passed to provider constructor

    Returns:
        Configured weather provider instance

    Raises:
        ValueError: If provider_type is not supported
    """
    if provider_type.lower() == "openmeteo":
        from app.core.simulation.open_meteo_data import OpenMeteoWeatherProvider

        return OpenMeteoWeatherProvider(location=location, **kwargs)
    elif provider_type.lower() == "csv":
        if "file_path" not in kwargs:
            raise ValueError("CSV provider requires 'file_path' argument")
        return CSVWeatherProvider(location=location, **kwargs)
    else:
        raise ValueError(f"Unsupported weather provider type: {provider_type}")


# -----------------------------------------------------------------------------
# Convenience Facade and Configuration
# -----------------------------------------------------------------------------


class WeatherProviderConfig(BaseModel):
    """Configuration builder for weather providers."""

    location: GeospatialLocation = Field(description="Geographic location")
    provider_type: str = Field(default="openmeteo", description="Provider type")
    start_date: str = Field(description="Start date for data")
    end_date: str = Field(description="End date for data")
    organization: str = Field(description="Organization name for data storage")
    asset: str = Field(description="Asset name for data storage")
    interval: TimeInterval = Field(
        default=TimeInterval.HOURLY, description="Data interval"
    )
    storage: Optional[DataStorage] = Field(
        default=None, description="Data storage config"
    )
    fetch_all_radiation: bool = Field(
        default=False, description="If True, fetch all radiation components"
    )
    file_path: Optional[str] = Field(
        default=None, description="CSV file path for CSV provider"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_provider(self) -> BaseWeatherProvider:
        """Create weather provider from configuration."""
        kwargs = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "organization": self.organization,
            "asset": self.asset,
            "interval": self.interval,
            "storage": self.storage,
        }

        if self.provider_type == "openmeteo":
            kwargs["fetch_all_radiation"] = self.fetch_all_radiation
        elif self.provider_type == "csv":
            if self.file_path is None:
                raise ValueError("CSV provider requires file_path")
            kwargs["file_path"] = self.file_path

        return create_weather_provider(self.location, self.provider_type, **kwargs)


def create_open_meteo_provider(
    location: GeospatialLocation, **kwargs
) -> BaseWeatherProvider:
    """
    Convenience facade for weather data access.

    Defaults to OpenMeteo provider with optimized fetching (GHI + temperature only).
    Use create_weather_provider() for more control over provider selection.
    """
    return create_weather_provider(location, "openmeteo", **kwargs)
