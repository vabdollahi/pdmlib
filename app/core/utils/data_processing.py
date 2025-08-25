"""
Data processing utilities for time series data operations.

This module provides reusable functionality for data transformation,
resampling, and processing that can be used across different data types
like weather, price, or other time series data.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.utils.date_handling import TimeInterval


def resample_timeseries_data(
    df: pd.DataFrame, target_interval: TimeInterval, aggregation_method: str = "mean"
) -> pd.DataFrame:
    """
    Resample time series data to the target interval.

    This method converts data between different time frequencies:
    - For higher frequency (e.g., hourly to 5min): uses interpolation
    - For lower frequency (e.g., hourly to daily): uses aggregation

    Args:
        df: DataFrame with datetime index
        target_interval: Target time interval
        aggregation_method: Method for downsampling ('mean', 'sum', 'max', 'min')

    Returns:
        Resampled DataFrame

    Examples:
        >>> # Downsample hourly to daily using mean
        >>> daily_df = resample_timeseries_data(hourly_df, TimeInterval.DAILY, "mean")

        >>> # Upsample hourly to 15-minute using interpolation
        >>> high_freq_df = resample_timeseries_data(
        ...     hourly_df, TimeInterval.FIFTEEN_MINUTES
        ... )
    """
    if target_interval == TimeInterval.DAILY:
        # Aggregate to daily using specified method
        if aggregation_method == "mean":
            return df.resample("D").mean()
        elif aggregation_method == "sum":
            return df.resample("D").sum()
        elif aggregation_method == "max":
            return df.resample("D").max()
        elif aggregation_method == "min":
            return df.resample("D").min()
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    elif target_interval in [
        TimeInterval.FIVE_MINUTES,
        TimeInterval.FIFTEEN_MINUTES,
    ]:
        # Interpolate to higher frequency
        target_freq = target_interval.pandas_frequency

        # Create a new index with the target frequency
        new_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=target_freq,
            tz=getattr(df.index, "tz", None),
        )

        # Reindex and interpolate
        df_resampled = df.reindex(new_index)
        df_resampled = df_resampled.interpolate(method="linear")

        return df_resampled
    else:
        # For hourly or same frequency, return as-is
        return df


def standardize_column_names(
    df: pd.DataFrame, column_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Standardize column names in a DataFrame using a mapping dictionary.

    Args:
        df: DataFrame to standardize
        column_mapping: Dictionary mapping original names to standard names

    Returns:
        DataFrame with standardized column names

    Examples:
        >>> mapping = {
        ...     "temperature_2m": "temperature_celsius",
        ...     "shortwave_radiation": "ghi"
        ... }
        >>> standardized_df = standardize_column_names(df, mapping)
    """
    return df.rename(columns=column_mapping)


def validate_timeseries_data(
    df: pd.DataFrame, required_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate that a DataFrame is properly formatted for time series processing.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names (optional)

    Returns:
        True if valid, raises ValueError if invalid

    Raises:
        ValueError: If DataFrame is not properly formatted
    """
    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for time series data")

    # Check if index is sorted
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted in ascending order")

    # Check required columns if specified
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    return True


def fill_missing_timestamps(
    df: pd.DataFrame, interval: TimeInterval, method: str = "interpolate"
) -> pd.DataFrame:
    """
    Fill missing timestamps in a time series DataFrame.

    Args:
        df: DataFrame with datetime index
        interval: Expected time interval between data points
        method: Method for filling ('interpolate', 'forward_fill', 'backward_fill',
               'zero')

    Returns:
        DataFrame with filled missing timestamps
    """
    # Create complete time range
    complete_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=interval.pandas_frequency,
        tz=getattr(df.index, "tz", None),
    )

    # Reindex to include all timestamps
    df_complete = df.reindex(complete_range)

    # Fill missing values based on method
    if method == "interpolate":
        df_complete = df_complete.interpolate(method="linear")
    elif method == "forward_fill":
        df_complete = df_complete.ffill()
    elif method == "backward_fill":
        df_complete = df_complete.bfill()
    elif method == "zero":
        df_complete = df_complete.fillna(0)
    else:
        raise ValueError(f"Unsupported fill method: {method}")

    return df_complete


def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect common data quality issues in time series data.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with detected issues and statistics
    """
    issues = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_timestamps": df.index.duplicated().sum(),
        "negative_values": {},
        "outliers": {},
        "data_gaps": [],
    }

    # Check for negative values in numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            issues["negative_values"][col] = negative_count

    # Simple outlier detection using IQR method
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            issues["outliers"][col] = outlier_count

    return issues
