"""
Provider configuration models and factory functions for automatic provider creation.

This module provides:
- Pydantic configuration models for all provider types
- Factory functions for automatic provider instantiation
- Type-safe configuration with validation

Supports:
- Price providers: CSV, CAISO, IESO
- Weather providers: CSV, OpenMeteo
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from app.core.utils.date_handling import TimeInterval
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger
from app.core.utils.storage import DataStorage

logger = get_logger("provider_config")


# -----------------------------------------------------------------------------
# Price Provider Configuration Models
# -----------------------------------------------------------------------------


class CSVPriceProviderConfig(BaseModel):
    """Configuration for CSV-based price provider."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["csv_file"] = "csv_file"
    name: Optional[str] = Field(default=None, description="Provider name")
    data: str = Field(description="Path to CSV file")


class CAISOPriceProviderConfig(BaseModel):
    """Configuration for CAISO price provider."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["caiso"] = "caiso"
    name: Optional[str] = Field(default=None, description="Provider name")
    organization: str = Field(default="SolarRevenue", description="Organization name")
    asset: str = Field(default="LMP_Data", description="Asset name")
    data_type: str = Field(default="caiso_lmp", description="Data type")
    interval: TimeInterval = Field(
        default=TimeInterval.HOURLY, description="Data interval"
    )


class IESOPriceProviderConfig(BaseModel):
    """Configuration for IESO price provider."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ieso"] = "ieso"
    name: Optional[str] = Field(default=None, description="Provider name")
    organization: str = Field(default="SolarRevenue", description="Organization name")
    asset: str = Field(default="HOEP_Data", description="Asset name")
    data_type: str = Field(default="ieso_hoep", description="Data type")
    interval: TimeInterval = Field(
        default=TimeInterval.HOURLY, description="Data interval"
    )


# Union type for all price provider configurations
PriceProviderConfig = Union[
    CSVPriceProviderConfig, CAISOPriceProviderConfig, IESOPriceProviderConfig
]


# -----------------------------------------------------------------------------
# Weather Provider Configuration Models
# -----------------------------------------------------------------------------


class CSVWeatherProviderConfig(BaseModel):
    """Configuration for CSV-based weather provider."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["csv_file"] = "csv_file"
    data: str = Field(description="Path to CSV file")


class OpenMeteoWeatherProviderConfig(BaseModel):
    """Configuration for OpenMeteo weather provider."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["openmeteo"] = "openmeteo"
    organization: str = Field(default="SolarRevenue", description="Organization name")
    asset: str = Field(default="Weather", description="Asset name")
    data_type: str = Field(default="weather", description="Data type")
    interval: TimeInterval = Field(
        default=TimeInterval.HOURLY, description="Data interval"
    )
    fetch_all_radiation: bool = Field(
        default=False, description="If True, fetch all radiation components from API"
    )


# Union type for all weather provider configurations
WeatherProviderConfig = Union[CSVWeatherProviderConfig, OpenMeteoWeatherProviderConfig]


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def create_price_provider_from_config(
    config: PriceProviderConfig,
    location: GeospatialLocation,
    start_date_time: datetime,
    end_date_time: datetime,
    storage: Optional[DataStorage] = None,
) -> Any:
    """
    Create a price provider from configuration.

    Args:
        config: Price provider configuration
        location: Geographic location for the provider
        start_date_time: Start datetime for data range
        end_date_time: End datetime for data range
        storage: Optional data storage configuration

    Returns:
        Configured price provider instance

    Raises:
        ValueError: If provider type is not supported
    """
    start_date = start_date_time.strftime("%Y-%m-%d %H:%M:%S")
    end_date = end_date_time.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(config, CSVPriceProviderConfig):
        logger.info(f"Creating CSV price provider from file: {config.data}")
        from app.core.simulation.price_provider import CSVPriceProvider

        provider = CSVPriceProvider(csv_file_path=config.data)
        provider.set_range(start_date_time, end_date_time)
        return provider

    elif isinstance(config, CAISOPriceProviderConfig):
        logger.info("Creating CAISO price provider")
        from app.core.simulation.caiso_data import CAISOPriceProvider

        return CAISOPriceProvider(
            location=location,
            start_date=start_date,
            end_date=end_date,
            organization=config.organization,
            asset=config.asset,
            data_type=config.data_type,
            interval=config.interval,
            storage=storage or DataStorage(),
        )

    elif isinstance(config, IESOPriceProviderConfig):
        logger.info("Creating IESO price provider")
        from app.core.simulation.ieso_data import IESOPriceProvider

        return IESOPriceProvider(
            location=location,
            start_date=start_date,
            end_date=end_date,
            organization=config.organization,
            asset=config.asset,
            data_type=config.data_type,
            interval=config.interval,
            storage=storage or DataStorage(),
        )

    else:
        raise ValueError(f"Unsupported price provider type: {type(config)}")


def create_weather_provider_from_config(
    config: WeatherProviderConfig,
    location: GeospatialLocation,
    start_date_time: datetime,
    end_date_time: datetime,
    storage: Optional[DataStorage] = None,
) -> Any:
    """
    Create a weather provider from configuration.

    Args:
        config: Weather provider configuration
        location: Geographic location for the provider
        start_date_time: Start datetime for data range
        end_date_time: End datetime for data range
        storage: Optional data storage configuration

    Returns:
        Configured weather provider instance

    Raises:
        ValueError: If provider type is not supported
    """
    start_date = start_date_time.strftime("%Y-%m-%d %H:%M:%S")
    end_date = end_date_time.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(config, CSVWeatherProviderConfig):
        logger.info(f"Creating CSV weather provider from file: {config.data}")
        from app.core.simulation.weather_provider import CSVWeatherProvider

        provider = CSVWeatherProvider(location=location, file_path=config.data)
        provider.set_range(start_date_time, end_date_time)
        return provider

    elif isinstance(config, OpenMeteoWeatherProviderConfig):
        logger.info("Creating OpenMeteo weather provider")
        from app.core.simulation.open_meteo_data import OpenMeteoWeatherProvider

        return OpenMeteoWeatherProvider(
            location=location,
            start_date=start_date,
            end_date=end_date,
            organization=config.organization,
            asset=config.asset,
            data_type=config.data_type,
            interval=config.interval,
            storage=storage or DataStorage(),
            fetch_all_radiation=config.fetch_all_radiation,
        )

    else:
        raise ValueError(f"Unsupported weather provider type: {type(config)}")
