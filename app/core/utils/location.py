"""
This module defines a set of Pydantic models for representing different
types of locations. This allows for a flexible and extensible way to handle
location-specific data, such as weather data (geospatial) or electricity
prices (regional/market-based).
"""

from typing import Union

from pydantic import BaseModel, Field


class GeospatialLocation(BaseModel):
    """Represents a location defined by latitude and longitude."""

    latitude: float = Field(..., description="Latitude of the location.")
    longitude: float = Field(..., description="Longitude of the location.")

    def to_path_string(self) -> str:
        """Returns a filesystem-safe string representation."""
        return f"lat{self.latitude}_lon{self.longitude}".replace(".", "_")


class RegionalLocation(BaseModel):
    """Represents a location defined by a country and a region (state/province)."""

    country: str = Field(..., description="The country of the location.")
    region: str = Field(..., description="The region, state, or province.")

    def to_path_string(self) -> str:
        """Returns a filesystem-safe string representation."""
        country_safe = self.country.lower().replace(" ", "_")
        region_safe = self.region.lower().replace(" ", "_")
        return f"country_{country_safe}_region_{region_safe}"


class MarketLocation(BaseModel):
    """Represents a location defined by an electricity market area."""

    market_area: str = Field(
        ..., description="The electricity market area (e.g., 'DE-LU')."
    )

    def to_path_string(self) -> str:
        """Returns a filesystem-safe string representation."""
        return f"market_{self.market_area.lower()}"


# A union of all possible location types for type hinting
Location = Union[GeospatialLocation, RegionalLocation, MarketLocation]
