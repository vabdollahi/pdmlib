"""
This module provides a data storage solution for saving and retrieving
weather and simulation data using the Parquet file format. It supports
a partitioned directory structure to organize data logically across
local and Google Cloud Storage filesystems.
"""

import os

import fsspec
import pandas as pd

from app.core.utils.location import Location
from app.core.utils.logging import get_logger
from app.core.utils.portfolio_context import (
    get_portfolio_organization,
    is_processed_data_type,
    is_raw_data_type,
    is_test_environment,
    should_use_unified_structure,
)

logger = get_logger("storage")


class DataStorage:
    """
    Data storage with separation of raw and processed data.

    Raw data (weather, price): Always cached independently at global level
    Processed data (PV generation): Cached with context-awareness
    """

    def __init__(self, base_path: str = "data"):
        """
        Initializes the DataStorage.

        Args:
            base_path: The base path for storage. Can be a local path or a
                       GCS URI (e.g., 'gcs://my-bucket/data').
        """
        self.base_path = base_path
        self.fs, _ = fsspec.core.url_to_fs(base_path)

    def _get_raw_data_path(self, data_type: str, location: Location) -> str:
        """
        Get path for raw data (always independent).

        Price data is organized by market region (caiso, ieso), not geographic location.
        Weather data is organized by geographic location (lat/lon).
        """
        # Price data: organize by market region, not geographic coordinates
        if data_type in ["caiso_lmp", "ieso_hoep"]:
            # Extract market region from data_type
            if data_type == "caiso_lmp":
                market_region = "caiso"
            elif data_type == "ieso_hoep":
                market_region = "ieso"
            else:
                market_region = "unknown"

            return "/".join([self.base_path, "raw", "price", market_region])

        # Weather and other data: use location-based organization
        location_str = location.to_path_string()
        normalized_type = "price" if data_type == "price" else data_type
        return "/".join([self.base_path, "raw", normalized_type, location_str])

    def _get_processed_data_path(
        self,
        organization: str,
        asset: str,
        data_type: str,
        location: Location,
        key_suffix: str | None = None,
    ) -> str:
        """
        Get path for processed data (context-aware, plant-specific).

        Processed data is always organization/asset specific to ensure
        proper isolation between different organizations and their assets.
        Each plant's location creates a separate cache even within the same asset.
        """
        location_str = location.to_path_string()

        if should_use_unified_structure():
            # Portfolio context: use unified structure
            # organization/asset/data/category/([suffix]/)location
            category = "power_generation" if data_type == "pv_generation" else data_type
            parts = [self.base_path, organization, asset, "data", category]
            if key_suffix:
                parts.append(key_suffix)
            parts.append(location_str)
            return "/".join(parts)
        else:
            # Standalone context: use legacy structure
            # organization/asset/data_type/location
            return "/".join(
                [self.base_path, organization, asset, data_type, location_str]
            )

    def _is_storage_test(self) -> bool:
        """
        Check if we're running storage-specific tests that should be allowed to cache.
        """
        # Check if we're in a memory filesystem (test setup)
        if self.base_path.startswith("memory://"):
            return True

        # Check if the test file is storage-related
        test_file = os.environ.get("PYTEST_CURRENT_TEST", "")
        if "test_storage" in test_file:
            return True

        return False

    def _get_partition_path(
        self,
        organization: str,
        asset: str,
        data_type: str,
        location: Location,
        key_suffix: str | None = None,
    ) -> str:
        """
        Constructs the path for a given data partition using best practices.

        Raw data (weather, price): Always cached independently by location
        - Ignores organization/asset to enable sharing across portfolios
        - Uses global location-based structure: data/raw/{type}/{location}/

        Processed data (PV generation): Organization and asset specific
        - Each plant gets separate cache using its own asset name
        - Portfolio organization provides isolation between organizations
        - Uses structure: data/{org}/{asset}/data/{category}/{location}/

        Args:
            organization: The name of the organization (for processed data isolation).
            asset: The name of the asset (plant-specific for processed data).
            data_type: The type of data (e.g., 'weather', 'price', 'pv_generation').
            location: The location object.

        Returns:
            The constructed path as a string.
        """
        # Skip caching in test environment (except for storage tests)
        if is_test_environment() and not self._is_storage_test():
            return "/tmp/test_cache_disabled"

        # Use portfolio organization if available for processed data
        if is_processed_data_type(data_type):
            portfolio_org = get_portfolio_organization()
            if portfolio_org:
                organization = portfolio_org

        # Route based on data type using best practices
        if is_raw_data_type(data_type):
            # Raw data: always independent, location-based caching
            return self._get_raw_data_path(data_type, location)
        elif is_processed_data_type(data_type):
            # Processed data: context-aware caching with plant-specific assets
            return self._get_processed_data_path(
                organization, asset, data_type, location, key_suffix
            )
        else:
            # Fallback to legacy behavior for unknown data types
            location_str = location.to_path_string()
            return "/".join(
                [self.base_path, organization, asset, data_type, location_str]
            )

    def write_data(
        self,
        df: pd.DataFrame,
        organization: str,
        asset: str,
        data_type: str,
        location: Location,
        key_suffix: str | None = None,
    ):
        """
        Writes a DataFrame to monthly Parquet files with smart caching strategy.

        Args:
            df: The DataFrame to save. It must have a DatetimeIndex.
            organization: The name of the organization.
            asset: The name of the asset.
            data_type: The type of data (e.g., 'weather', 'price', 'pv_generation').
            location: The location object.
        """
        # Skip writing in test environment only for production-level tests,
        # but allow storage-specific tests to run
        if is_test_environment() and not self._is_storage_test():
            logger.debug("Skipping data write in test environment")
            return

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        partition_path = self._get_partition_path(
            organization, asset, data_type, location, key_suffix
        )

        # Log caching strategy being used
        if is_raw_data_type(data_type):
            logger.debug(
                f"Caching raw {data_type} data independently: {partition_path}"
            )
        elif is_processed_data_type(data_type):
            context = "unified" if should_use_unified_structure() else "legacy"
            logger.debug(
                f"Caching processed {data_type} using {context} structure: "
                f"{partition_path}"
            )

        self.fs.mkdirs(partition_path, exist_ok=True)

        # Group data by year and month and save to separate files
        for (year, month), group in df.groupby([df.index.year, df.index.month]):
            file_path = f"{partition_path}/{year}_{month:02d}.parquet"

            # If a file already exists, read it and merge the new data
            if self.fs.exists(file_path):
                logger.info(f"Updating existing file: {file_path}")
                with self.fs.open(file_path, "rb") as f:
                    existing_df = pd.read_parquet(f)
                    if "date_time" in existing_df.columns:
                        existing_df = existing_df.set_index("date_time")
                        # Ensure existing index is datetime type
                        if not pd.api.types.is_datetime64_any_dtype(existing_df.index):
                            existing_df.index = pd.to_datetime(existing_df.index)

                # Ensure new data (group) also has datetime index
                if not pd.api.types.is_datetime64_any_dtype(group.index):
                    group.index = pd.to_datetime(group.index)

                # Try to combine old and new data
                try:
                    combined_df = pd.concat([existing_df, group])
                    # By keeping the 'last' duplicate, we ensure new data overwrites old
                    group_to_save = combined_df[
                        ~combined_df.index.duplicated(keep="last")
                    ].sort_index()
                except (TypeError, ValueError) as e:
                    # If there's a type or value issue, log and use only new data
                    logger.warning(
                        f"Could not combine with existing data: {e}. "
                        f"Using new data only."
                    )
                    group_to_save = group
            else:
                group_to_save = group

            # Reset index to make date_time a column, and don't save the new index
            group_to_save_final = group_to_save.reset_index()
            # Ensure the index column is named 'date_time'
            if group_to_save_final.columns[0] not in ["date_time"]:
                group_to_save_final = group_to_save_final.rename(
                    columns={group_to_save_final.columns[0]: "date_time"}
                )
            with self.fs.open(file_path, "wb") as f:
                group_to_save_final.to_parquet(f, engine="pyarrow", index=False)
            logger.info(f"Data saved to {file_path}")

    def read_data_for_range(
        self,
        organization: str,
        asset: str,
        data_type: str,
        location: Location,
        start_date: str,
        end_date: str,
        key_suffix: str | None = None,
    ) -> pd.DataFrame:
        """
        Reads and combines monthly Parquet files with smart caching strategy.

        Args:
            organization: The name of the organization.
            asset: The name of the asset.
            data_type: The type of data (e.g., 'weather', 'price', 'pv_generation').
            location: The location object.
            start_date: The start date in YYYY-MM-DD format.
            end_date: The end date in YYYY-MM-DD format.

        Returns:
            A DataFrame containing the combined data, or an empty DataFrame
            if no data is found.
        """
        # Return empty DataFrame in test environment (except for storage tests)
        if is_test_environment() and not self._is_storage_test():
            logger.debug("Returning empty DataFrame in test environment")
            return pd.DataFrame()

        partition_path = self._get_partition_path(
            organization, asset, data_type, location, key_suffix
        )

        if not self.fs.exists(partition_path):
            cache_type = "raw" if is_raw_data_type(data_type) else "processed"
            logger.debug(f"No {cache_type} cache found at {partition_path}")
            return pd.DataFrame()

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        all_data = []
        # Use fsspec's glob to find all parquet files in the directory
        file_paths = self.fs.glob(f"{partition_path}/*.parquet")
        for file_path_full in file_paths:
            try:
                # Add protocol if it's a cloud path
                file_path = (
                    f"gcs://{file_path_full}"
                    if self.fs.protocol == "gcs"
                    else file_path_full
                )

                # Extract year and month from filename
                filename = file_path.split("/")[-1]
                year, month = map(int, filename.replace(".parquet", "").split("_"))

                # Check if the file's month is within the requested date range
                file_date = pd.Timestamp(year=year, month=month, day=1)
                if not (
                    file_date.to_period("M") >= start.to_period("M")
                    and file_date.to_period("M") <= end.to_period("M")
                ):
                    continue

                with self.fs.open(file_path, "rb") as f:
                    monthly_df = pd.read_parquet(f)

                # Clean up any unwanted index columns that might remain
                if "index" in monthly_df.columns:
                    monthly_df = monthly_df.drop(columns=["index"])

                # Set date_time back to be the index
                if "date_time" in monthly_df.columns:
                    monthly_df = monthly_df.set_index("date_time")
                    # Ensure index is datetime type
                    if not pd.api.types.is_datetime64_any_dtype(monthly_df.index):
                        monthly_df.index = pd.to_datetime(monthly_df.index)

                all_data.append(monthly_df)
            except (ValueError, TypeError):
                # Ignore files that don't match the naming convention
                continue

        if not all_data:
            return pd.DataFrame()

        # Combine all dataframes and filter to the exact date range
        combined_df = pd.concat(all_data).sort_index()

        # Ensure index is datetime type and timezone-aware if needed
        if not pd.api.types.is_datetime64_any_dtype(combined_df.index):
            logger.warning(
                (
                    "Index is not datetime type: %s. Converting..."
                    % combined_df.index.dtype
                )
            )
            combined_df.index = pd.to_datetime(combined_df.index)

        index_tz = getattr(combined_df.index, "tz", None)
        if hasattr(combined_df.index, "tz") and index_tz is not None:
            # Convert start/end to match the index timezone
            start = start.tz_localize("UTC") if start.tz is None else start
            end = end.tz_localize("UTC") if end.tz is None else end

        try:
            return combined_df.loc[start:end]
        except (KeyError, TypeError) as e:
            # Fallback: use boolean indexing
            logger.warning(
                f"Direct datetime slicing failed: {e}. Using boolean filtering."
            )
            # Use simple boolean filtering
            end_inclusive = end + pd.Timedelta(days=1)  # Include end day
            return combined_df[
                (combined_df.index >= start) & (combined_df.index < end_inclusive)
            ]
