"""
This module provides a generic, cache-first data provider framework.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core.utils.location import Location
from app.core.utils.storage import DataStorage


def _find_missing_date_ranges(
    start_date: str, end_date: str, existing_dates: pd.DatetimeIndex
) -> List[Tuple[str, str]]:
    """
    Compares a required date range with existing dates to find gaps.

    Args:
        start_date: The required start date (YYYY-MM-DD).
        end_date: The required end date (YYYY-MM-DD).
        existing_dates: A DatetimeIndex of dates that are already available.

    Returns:
        A list of tuples, where each tuple contains the start and end date
        of a missing period.
    """
    required_range = pd.date_range(
        start=start_date, end=f"{end_date} 23:59:59", freq="h", tz="UTC"
    )

    # Ensure existing_dates is timezone-aware (UTC) to match required_range
    if existing_dates.tz is None:
        existing_dates = existing_dates.tz_localize("UTC")
    else:
        existing_dates = existing_dates.tz_convert("UTC")

    missing_dates = required_range.difference(existing_dates)

    if missing_dates.empty:
        return []

    # Find contiguous blocks of missing dates. A new block starts when the
    # time difference to the previous entry is > 1 hour.
    missing_series = missing_dates.to_series()
    contiguous_blocks = (missing_series.diff().dt.total_seconds().gt(3600)).cumsum()
    groups = missing_series.groupby(contiguous_blocks)

    gaps = []
    for _, group in groups:
        start_gap = group.index.min().strftime("%Y-%m-%d")
        end_gap = group.index.max().strftime("%Y-%m-%d")
        gaps.append((start_gap, end_gap))

    return gaps


class BaseProvider(BaseModel, ABC):
    """
    An abstract base class for data providers that implements a cache-first
    strategy for data retrieval.
    """

    location: Location
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    organization: str = Field(description="Name of the organization.")
    asset: str = Field(description="Name of the asset.")
    data_type: str = Field(description="Type of data being handled (e.g., 'weather').")
    storage: DataStorage = Field(
        default_factory=DataStorage,
        description="Data storage handler for caching.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def _fetch_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        An abstract method that subclasses must implement to fetch data
        for a specific date range from the underlying data source.
        """
        raise NotImplementedError

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
            print("--- Force refresh requested, fetching fresh data ---")
            fresh_data = await self._fetch_range(
                start_date=self.start_date, end_date=self.end_date
            )

            # Save the fresh data to cache
            print("\n--- Saving fresh data to Parquet store... ---")
            self.storage.write_data(
                df=fresh_data,
                organization=self.organization,
                asset=self.asset,
                data_type=self.data_type,
                location=self.location,
            )

            return fresh_data

        # 1. Check the cache first
        cached_data = self.storage.read_data_for_range(
            organization=self.organization,
            asset=self.asset,
            data_type=self.data_type,
            location=self.location,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        # 2. Identify missing date ranges
        if cached_data.empty:
            # No cached data, fetch everything
            missing_ranges = [(self.start_date, self.end_date)]
        else:
            cached_data_index = pd.to_datetime(cached_data.index)
            missing_ranges = _find_missing_date_ranges(
                self.start_date, self.end_date, cached_data_index
            )

        if not missing_ranges and not cached_data.empty:
            print("--- All data found in cache. ---")
            result = cached_data.loc[self.start_date : self.end_date]
            return result if isinstance(result, pd.DataFrame) else result.to_frame().T

        # 3. Fetch new data only for the missing ranges
        print(f"--- Fetching data for missing ranges: {missing_ranges} ---")
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
        print("\n--- Saving newly fetched data to Parquet store... ---")
        self.storage.write_data(
            df=new_data_df,
            organization=self.organization,
            asset=self.asset,
            data_type=self.data_type,
            location=self.location,
        )

        # 5. Return the complete, combined dataset
        final_df = pd.concat([cached_data, new_data_df]).sort_index()
        final_df = final_df[~final_df.index.duplicated(keep="last")]
        result = final_df.loc[self.start_date : self.end_date]
        return result if isinstance(result, pd.DataFrame) else result.to_frame().T
