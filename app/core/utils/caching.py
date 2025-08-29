"""
This module provides a generic, cache-first data provider framework.
"""

from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core.utils.date_handling import (
    TimeInterval,
    find_missing_intervals,
    normalize_date_range,
    parse_datetime_input,
)
from app.core.utils.location import Location
from app.core.utils.logging import get_logger
from app.core.utils.storage import DataStorage

logger = get_logger("cache")


class BaseProvider(BaseModel, ABC):
    """
    An abstract base class for data providers that implements a cache-first
    strategy for data retrieval with configurable time intervals.
    """

    location: Location
    start_date: str = Field(
        description="Start date in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format."
    )
    end_date: str = Field(
        description="End date in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format."
    )
    organization: str = Field(description="Name of the organization.")
    asset: str = Field(description="Name of the asset.")
    data_type: str = Field(description="Type of data being handled (e.g., 'weather').")
    interval: TimeInterval = Field(
        default=TimeInterval.HOURLY,
        description="Time interval for data fetching (5min, 15min, 1h, 1d).",
    )
    storage: DataStorage = Field(
        default_factory=DataStorage,
        description="Data storage handler for caching.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context):
        """Apply interval rounding after model validation."""
        # Apply interval rounding to ensure start/end times align with boundaries
        start_rounded, end_rounded = normalize_date_range(
            self.start_date, self.end_date, interval=self.interval
        )

        # Convert back to string format for consistency with existing API
        self.start_date = start_rounded.strftime("%Y-%m-%d %H:%M:%S")
        self.end_date = end_rounded.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(
            f"Rounded to {self.interval.display_name} intervals: "
            f"{self.start_date} to {self.end_date}"
        )

    @abstractmethod
    async def _fetch_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        An abstract method that subclasses must implement to fetch data
        for a specific date range from the underlying data source.
        """
        raise NotImplementedError

    def _get_storage_key_suffix(self) -> str | None:
        """
        Optional provider-specific suffix for cache partitioning.

        Subclasses may implement `_get_cache_key_suffix()` to provide a
        stable discriminator (e.g., PV config hash). If not provided, None.
        """
        # Prefer subclass hook if available
        try:
            get_suffix = getattr(self, "_get_cache_key_suffix", None)
            if callable(get_suffix):
                val = get_suffix()
                return str(val) if val is not None else None
        except Exception:
            pass
        return None

    async def get_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieves data, utilizing a cache-first strategy.

        Args:
            force_refresh: If True, bypass the cache and fetch fresh data.

        Returns:
            A pandas DataFrame containing the requested data.
        """
        # If force_refresh is True, skip cache check and fetch all data
        if force_refresh:
            logger.info("Force refresh requested, fetching fresh data")
            fresh_data = await self._fetch_range(
                start_date=self.start_date, end_date=self.end_date
            )

            # Save the fresh data to cache only if there's actual data
            if not fresh_data.empty and isinstance(fresh_data.index, pd.DatetimeIndex):
                logger.debug("Saving fresh data to cache")
                self.storage.write_data(
                    df=fresh_data,
                    organization=self.organization,
                    asset=self.asset,
                    data_type=self.data_type,
                    location=self.location,
                    key_suffix=self._get_storage_key_suffix(),
                )
            else:
                logger.info("No fresh data to save (empty or invalid DataFrame)")

            return fresh_data

        # 1. Check the cache first
        cached_data = self.storage.read_data_for_range(
            organization=self.organization,
            asset=self.asset,
            data_type=self.data_type,
            location=self.location,
            start_date=self.start_date,
            end_date=self.end_date,
            key_suffix=self._get_storage_key_suffix(),
        )

        # 2. Identify missing date ranges using the date handling utility
        if cached_data.empty:
            # No cached data, fetch everything
            missing_ranges = [(self.start_date, self.end_date)]
        else:
            # Convert date strings to datetime objects for the utility function
            start_dt = parse_datetime_input(self.start_date)
            end_dt = parse_datetime_input(self.end_date)
            cached_data_index = pd.to_datetime(cached_data.index)

            # Use the missing intervals detection
            missing_intervals = find_missing_intervals(
                start_dt, end_dt, cached_data_index, self.interval
            )

            # Convert datetime tuples back to string format for API compatibility
            missing_ranges = [
                (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"))
                for start, end in missing_intervals
            ]

        if not missing_ranges and not cached_data.empty:
            logger.debug("All data found in cache")
            result = cached_data.loc[self.start_date : self.end_date]
            return result if isinstance(result, pd.DataFrame) else result.to_frame().T

        # 3. Fetch new data only for the missing ranges
        logger.info(f"Fetching data for missing ranges: {missing_ranges}")
        new_data_list = []
        for start, end in missing_ranges:
            api_data = await self._fetch_range(start_date=start, end_date=end)
            new_data_list.append(api_data)

        if not new_data_list and cached_data.empty:
            return pd.DataFrame()
        if not new_data_list:
            result = cached_data.loc[self.start_date : self.end_date]
            return result if isinstance(result, pd.DataFrame) else result.to_frame().T

        # 4. Combine and save the new data
        new_data_df = pd.concat(new_data_list)

        # Only save if there's actual data to save
        if not new_data_df.empty and isinstance(new_data_df.index, pd.DatetimeIndex):
            logger.debug("Saving newly fetched data to cache")
            self.storage.write_data(
                df=new_data_df,
                organization=self.organization,
                asset=self.asset,
                data_type=self.data_type,
                location=self.location,
                key_suffix=self._get_storage_key_suffix(),
            )
        else:
            logger.info("No new data to save (empty or invalid DataFrame)")

        # 5. Return the complete, combined dataset
        final_df = pd.concat([cached_data, new_data_df]).sort_index()
        final_df = final_df[~final_df.index.duplicated(keep="last")]
        result = final_df.loc[self.start_date : self.end_date]
        return result if isinstance(result, pd.DataFrame) else result.to_frame().T
