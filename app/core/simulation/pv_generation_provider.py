"""
PV generation data provider with caching support.

This module provides a cache-aware provider for PV generation data that inherits
from BaseProvider to leverage the existing caching infrastructure for weather
and price data.
"""

import hashlib
import json

import pandas as pd
from pydantic import Field

from app.core.simulation.pv_model import PVModel
from app.core.utils.caching import BaseProvider
from app.core.utils.logging import get_logger

logger = get_logger("pv_generation_provider")


class PVGenerationProvider(BaseProvider):
    """
    Cache-aware PV generation data provider.

    This provider implements caching for PV generation data using the same
    patterns as weather and price data providers. It caches results per
    portfolio/plant/time range in monthly parquet files.
    """

    data_type: str = Field(default="pv_generation", description="Data type identifier")
    pv_model: PVModel = Field(description="PV model for simulation")

    def _get_pv_config_hash(self) -> str:
        """
        Generate hash for PV configuration to use in cache key.

        This ensures that cache is invalidated when PV system configuration
        changes (modules, inverters, mounting, etc.).

        Returns:
            8-character hash string of PV configuration
        """
        try:
            config_dict = self.pv_model.pv_config.model_dump()
            config_str = json.dumps(config_dict, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"Could not generate PV config hash: {e}")
            return "default"

    def _get_cache_key_suffix(self) -> str:
        """
        Generate cache key suffix including PV configuration.

        Returns:
            Cache key suffix in format: pv_gen_{config_hash}_{lat}_{lon}
        """
        pv_hash = self._get_pv_config_hash()

        # Handle different location types - only GeospatialLocation has lat/lon
        if hasattr(self.location, "latitude") and hasattr(self.location, "longitude"):
            lat = round(self.location.latitude, 4)  # Round to ~10m precision
            lon = round(self.location.longitude, 4)
            return f"pv_gen_{pv_hash}_{lat}_{lon}"
        else:
            # For non-geospatial locations, use path string representation
            location_str = self.location.to_path_string()
            return f"pv_gen_{pv_hash}_{location_str}"

    async def _fetch_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch PV generation data by running simulation.

        This method is called by BaseProvider when cache misses occur.
        It updates the weather provider dates and runs the PV simulation.

        Args:
            start_date: Start date in YYYY-MM-DD HH:MM:SS format
            end_date: End date in YYYY-MM-DD HH:MM:SS format

        Returns:
            DataFrame with PV generation results
        """
        logger.info(f"Running PV simulation for {start_date} to {end_date}")

        # Update weather provider dates to match requested range if possible
        if hasattr(self.pv_model.weather_provider, "start_date"):
            self.pv_model.weather_provider.start_date = start_date  # type: ignore
        if hasattr(self.pv_model.weather_provider, "end_date"):
            self.pv_model.weather_provider.end_date = end_date  # type: ignore

        # Run the actual PVLib simulation
        try:
            results = await self.pv_model._run_simulation_uncached()
            logger.info(f"PV simulation completed with {len(results)} data points")

            # Ensure the DataFrame has a DatetimeIndex for caching compatibility
            if not results.empty and "date_time" in results.columns:
                # Set date_time as index if it's currently a column
                results = results.set_index("date_time")
                # Ensure index is datetime type
                if not isinstance(results.index, pd.DatetimeIndex):
                    results.index = pd.to_datetime(results.index)
            elif not results.empty and not isinstance(results.index, pd.DatetimeIndex):
                # If no date_time column but index isn't datetime, try to convert
                results.index = pd.to_datetime(results.index)

            return results
        except Exception as e:
            logger.error(f"PV simulation failed: {e}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=["Total AC power (W)", "Total DC power (W)"])
