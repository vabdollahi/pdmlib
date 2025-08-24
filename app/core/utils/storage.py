"""
This module provides a data storage solution for saving and retrieving
weather and simulation data using the Parquet file format. It supports
a partitioned directory structure to organize data logically across
local and Google Cloud Storage filesystems.
"""

import fsspec
import pandas as pd

from app.core.utils.location import Location
from app.core.utils.logging import get_logger

logger = get_logger("storage")


class DataStorage:
    """
    Handles reading from and writing to a partitioned Parquet data store.
    Supports local filesystems and GCS.
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

    def _get_partition_path(
        self,
        organization: str,
        asset: str,
        data_type: str,
        location: Location,
    ) -> str:
        """
        Constructs the path for a given data partition.

        Args:
            organization: The name of the organization.
            asset: The name of the asset.
            data_type: The type of data (e.g., 'weather', 'price').
            location: The location object.

        Returns:
            The constructed path as a string.
        """
        # Create a location string that is safe for file paths
        location_str = location.to_path_string()
        # Use standard string join for paths, as fsspec handles the separator
        return "/".join([self.base_path, organization, asset, data_type, location_str])

    def write_data(
        self,
        df: pd.DataFrame,
        organization: str,
        asset: str,
        data_type: str,
        location: Location,
    ):
        """
        Writes a DataFrame to monthly Parquet files within a partitioned structure.

        Args:
            df: The DataFrame to save. It must have a DatetimeIndex.
            organization: The name of the organization.
            asset: The name of the asset.
            data_type: The type of data (e.g., 'weather', 'price').
            location: The location object.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        partition_path = self._get_partition_path(
            organization, asset, data_type, location
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
    ) -> pd.DataFrame:
        """
        Reads and combines monthly Parquet files for a given date range.

        Args:
            organization: The name of the organization.
            asset: The name of the asset.
            data_type: The type of data (e.g., 'weather', 'price').
            location: The location object.
            start_date: The start date in YYYY-MM-DD format.
            end_date: The end date in YYYY-MM-DD format.

        Returns:
            A DataFrame containing the combined data, or an empty DataFrame
            if no data is found.
        """
        partition_path = self._get_partition_path(
            organization, asset, data_type, location
        )
        if not self.fs.exists(partition_path):
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
                f"Index is not datetime type: {combined_df.index.dtype}. Converting..."
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
