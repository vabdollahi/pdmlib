"""
This module provides a data storage solution for saving and retrieving
weather and simulation data using the Parquet file format. It supports
a partitioned directory structure to organize data logically across
local and Google Cloud Storage filesystems.
"""

import fsspec
import pandas as pd


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
        self, organization: str, asset: str, latitude: float, longitude: float
    ) -> str:
        """
        Constructs the path for a given data partition.

        Args:
            organization: The name of the organization.
            asset: The name of the asset.
            latitude: The latitude of the location.
            longitude: The longitude of the location.

        Returns:
            The constructed path as a string.
        """
        # Create a location string that is safe for file paths
        location_str = f"lat{latitude}_lon{longitude}".replace(".", "_")
        # Use standard string join for paths, as fsspec handles the separator
        return "/".join([self.base_path, organization, asset, location_str])

    def write_data(
        self,
        df: pd.DataFrame,
        organization: str,
        asset: str,
        latitude: float,
        longitude: float,
    ):
        """
        Writes a DataFrame to monthly Parquet files within a partitioned structure.

        Args:
            df: The DataFrame to save. It must have a DatetimeIndex.
            organization: The name of the organization.
            asset: The name of the asset.
            latitude: The latitude of the location.
            longitude: The longitude of the location.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        partition_path = self._get_partition_path(
            organization, asset, latitude, longitude
        )
        self.fs.mkdirs(partition_path, exist_ok=True)

        # Group data by year and month and save to separate files
        for (year, month), group in df.groupby([df.index.year, df.index.month]):
            file_path = f"{partition_path}/{year}_{month:02d}.parquet"
            # Reset index to make date_time a column, and don't save the new index
            group_to_save = group.reset_index()
            with self.fs.open(file_path, "wb") as f:
                group_to_save.to_parquet(f, engine="pyarrow", index=False)
            print(f"Data saved to {file_path}")

    def read_data_for_range(
        self,
        organization: str,
        asset: str,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Reads and combines monthly Parquet files for a given date range.

        Args:
            organization: The name of the organization.
            asset: The name of the asset.
            latitude: The latitude of the location.
            longitude: The longitude of the location.
            start_date: The start date in YYYY-MM-DD format.
            end_date: The end date in YYYY-MM-DD format.

        Returns:
            A DataFrame containing the combined data, or an empty DataFrame
            if no data is found.
        """
        partition_path = self._get_partition_path(
            organization, asset, latitude, longitude
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
                if not (
                    (year == start.year and month >= start.month)
                    or (year == end.year and month <= end.month)
                    or (start.year < year < end.year)
                ):
                    continue

                with self.fs.open(file_path, "rb") as f:
                    monthly_df = pd.read_parquet(f)
                    # Set date_time back to be the index
                    if "date_time" in monthly_df.columns:
                        monthly_df = monthly_df.set_index("date_time")
                all_data.append(monthly_df)
            except (ValueError, TypeError):
                # Ignore files that don't match the naming convention
                continue

        if not all_data:
            return pd.DataFrame()

        # Combine all dataframes and filter to the exact date range
        combined_df = pd.concat(all_data).sort_index()
        return combined_df.loc[start:end]
