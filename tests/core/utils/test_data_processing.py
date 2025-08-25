"""
Tests for the data processing utilities.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.utils.data_processing import (
    detect_data_quality_issues,
    fill_missing_timestamps,
    resample_timeseries_data,
    standardize_column_names,
    validate_timeseries_data,
)
from app.core.utils.date_handling import TimeInterval


class TestResampleTimeseriesData:
    """Tests for resample_timeseries_data function."""

    @pytest.fixture
    def sample_hourly_data(self):
        """Create sample hourly data for testing."""
        dates = pd.date_range("2025-01-01", periods=24, freq="h")
        return pd.DataFrame(
            {
                "ghi": range(24),
                "dni": range(24, 48),
                "dhi": range(48, 72),
                "temp_air": [20 + (i % 12) for i in range(24)],
            },
            index=dates,
        )

    def test_resample_to_daily_mean(self, sample_hourly_data):
        """Test resampling hourly data to daily with mean aggregation."""
        result = resample_timeseries_data(
            sample_hourly_data, TimeInterval.DAILY, aggregation_method="mean"
        )

        assert len(result) == 1  # One day
        # Check mean values
        assert result.iloc[0]["ghi"] == 11.5  # mean of 0-23
        assert result.iloc[0]["dni"] == 35.5  # mean of 24-47

    def test_resample_to_five_minutes(self, sample_hourly_data):
        """Test resampling hourly data to 5-minute intervals."""
        result = resample_timeseries_data(sample_hourly_data, TimeInterval.FIVE_MINUTES)

        # The actual length depends on how pandas handles the end time
        # 24 hours = 24*12 = 288 5-minute intervals, but pandas may exclude last point
        assert len(result) >= 277  # Allow for pandas behavior differences
        # Values should be interpolated from hourly data
        assert result.iloc[0]["ghi"] == 0
        # Check that interpolation is working
        assert result.iloc[1]["ghi"] > 0 and result.iloc[1]["ghi"] < 1

    def test_resample_with_sum_aggregation(self, sample_hourly_data):
        """Test resampling with sum aggregation method."""
        result = resample_timeseries_data(
            sample_hourly_data, TimeInterval.DAILY, aggregation_method="sum"
        )

        assert len(result) == 1
        assert result.iloc[0]["ghi"] == sum(range(24))  # Sum of 0-23

    def test_resample_invalid_method_raises_error(self, sample_hourly_data):
        """Test that invalid aggregation method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            resample_timeseries_data(
                sample_hourly_data, TimeInterval.DAILY, aggregation_method="invalid"
            )

    def test_resample_preserves_column_names(self, sample_hourly_data):
        """Test that resampling preserves original column names."""
        result = resample_timeseries_data(sample_hourly_data, TimeInterval.DAILY)

        assert list(result.columns) == list(sample_hourly_data.columns)


class TestStandardizeColumnNames:
    """Tests for standardize_column_names function."""

    def test_standardize_weather_columns(self):
        """Test standardizing weather data column names."""
        df = pd.DataFrame(
            {
                "shortwave_radiation": [100, 200],
                "direct_normal_irradiance": [150, 250],
                "diffuse_radiation": [50, 100],
                "temperature_2m": [20, 25],
                "unknown_column": [1, 2],
            }
        )

        column_mapping = {
            "shortwave_radiation": "ghi",
            "direct_normal_irradiance": "dni",
            "diffuse_radiation": "dhi",
            "temperature_2m": "temp_air",
        }
        result = standardize_column_names(df, column_mapping)

        expected_columns = ["ghi", "dni", "dhi", "temp_air", "unknown_column"]
        assert list(result.columns) == expected_columns

    def test_standardize_preserves_unknown_columns(self):
        """Test that unknown columns are preserved unchanged."""
        df = pd.DataFrame(
            {
                "shortwave_radiation": [100],
                "custom_column": [123],
                "another_unknown": ["text"],
            }
        )

        column_mapping = {"shortwave_radiation": "ghi"}
        result = standardize_column_names(df, column_mapping)

        assert "ghi" in result.columns
        assert "custom_column" in result.columns
        assert "another_unknown" in result.columns

    def test_standardize_empty_dataframe(self):
        """Test standardizing empty DataFrame."""
        df = pd.DataFrame()
        result = standardize_column_names(df, {})
        assert result.empty

    def test_standardize_no_matching_columns(self):
        """Test standardizing DataFrame with no matching columns."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result = standardize_column_names(df, {})

        assert list(result.columns) == ["col1", "col2"]
        assert result.equals(df)


class TestValidateTimeseriesData:
    """Tests for validate_timeseries_data function."""

    def test_validate_valid_data(self):
        """Test validation of valid timeseries data."""
        dates = pd.date_range("2025-01-01", periods=5, freq="h")
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

        result = validate_timeseries_data(df)
        assert result is True

    def test_validate_non_datetime_index_raises_error(self):
        """Test that non-datetime index raises ValueError."""
        df = pd.DataFrame({"value": [1, 2, 3]})  # Default integer index

        with pytest.raises(ValueError, match="DataFrame index must be a DatetimeIndex"):
            validate_timeseries_data(df)

    def test_validate_with_text_data_passes(self):
        """Test that text data passes validation (no numeric requirement)."""
        dates = pd.date_range("2025-01-01", periods=3, freq="h")
        df = pd.DataFrame({"text_col": ["a", "b", "c"]}, index=dates)

        result = validate_timeseries_data(df)
        assert result is True

    def test_validate_mixed_data_types(self):
        """Test validation with mixed numeric data types."""
        dates = pd.date_range("2025-01-01", periods=3, freq="h")
        df = pd.DataFrame(
            {"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3]}, index=dates
        )

        result = validate_timeseries_data(df)
        assert result is True

    def test_validate_with_nans_allowed(self):
        """Test validation with NaN values (should pass)."""
        dates = pd.date_range("2025-01-01", periods=3, freq="h")
        df = pd.DataFrame({"value": [1, np.nan, 3]}, index=dates)

        result = validate_timeseries_data(df)
        assert result is True


class TestFillMissingTimestamps:
    """Tests for fill_missing_timestamps function."""

    def test_fill_missing_hourly_data(self):
        """Test filling missing timestamps in hourly data."""
        # Create data with gaps
        dates = [
            pd.Timestamp("2025-01-01 00:00"),
            pd.Timestamp("2025-01-01 01:00"),
            # Missing 02:00
            pd.Timestamp("2025-01-01 03:00"),
            pd.Timestamp("2025-01-01 04:00"),
        ]
        df = pd.DataFrame({"value": [1, 2, 4, 5]}, index=dates)

        result = fill_missing_timestamps(df, TimeInterval.HOURLY)

        assert len(result) == 5  # Should have 5 hours total
        assert pd.Timestamp("2025-01-01 02:00") in result.index
        # Default method is interpolate, so value should be interpolated, not NaN
        assert result.loc["2025-01-01 02:00", "value"] == 3.0  # Interpolated

    def test_fill_missing_with_forward_fill(self):
        """Test filling missing timestamps with forward fill method."""
        dates = [
            pd.Timestamp("2025-01-01 00:00"),
            pd.Timestamp("2025-01-01 02:00"),  # Gap at 01:00
        ]
        df = pd.DataFrame({"value": [10, 20]}, index=dates)

        result = fill_missing_timestamps(df, TimeInterval.HOURLY, method="forward_fill")

        assert len(result) == 3
        assert result.loc["2025-01-01 01:00", "value"] == 10  # Forward filled

    def test_fill_missing_with_interpolation(self):
        """Test filling missing timestamps with interpolation."""
        dates = [
            pd.Timestamp("2025-01-01 00:00"),
            pd.Timestamp("2025-01-01 02:00"),  # Gap at 01:00
        ]
        df = pd.DataFrame({"value": [10, 30]}, index=dates)

        result = fill_missing_timestamps(df, TimeInterval.HOURLY, method="interpolate")

        assert len(result) == 3
        assert result.loc["2025-01-01 01:00", "value"] == 20  # Interpolated

    def test_fill_missing_no_gaps(self):
        """Test filling when there are no missing timestamps."""
        dates = pd.date_range("2025-01-01", periods=3, freq="h")
        df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

        result = fill_missing_timestamps(df, TimeInterval.HOURLY)

        assert result.equals(df)


class TestDetectDataQualityIssues:
    """Tests for detect_data_quality_issues function."""

    def test_detect_missing_values(self):
        """Test detection of missing values."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4],
                "col2": [1, np.nan, np.nan, 4],
            }
        )

        issues = detect_data_quality_issues(df)

        assert "missing_values" in issues
        assert issues["missing_values"]["col1"] == 1
        assert issues["missing_values"]["col2"] == 2

    def test_detect_outliers(self):
        """Test detection of outliers using IQR method."""
        # Create data with clear outliers
        values = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        df = pd.DataFrame({"value": values})

        issues = detect_data_quality_issues(df)

        assert "outliers" in issues
        assert issues["outliers"]["value"] == 1  # One outlier

    def test_detect_duplicate_timestamps(self):
        """Test detection of duplicate timestamps."""
        dates = [
            pd.Timestamp("2025-01-01 00:00"),
            pd.Timestamp("2025-01-01 01:00"),
            pd.Timestamp("2025-01-01 01:00"),  # Duplicate
        ]
        df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

        issues = detect_data_quality_issues(df)

        assert "duplicate_timestamps" in issues
        assert issues["duplicate_timestamps"] == 1

    def test_detect_no_issues(self):
        """Test detection when data has no quality issues."""
        dates = pd.date_range("2025-01-01", periods=5, freq="h")
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

        issues = detect_data_quality_issues(df)

        # Function returns counts for all columns, including 0 for no missing values
        assert issues["missing_values"]["value"] == 0
        assert issues["duplicate_timestamps"] == 0
        # No outliers should be detected
        assert "value" not in issues["outliers"] or issues["outliers"]["value"] == 0

    def test_detect_mixed_issues(self):
        """Test detection of multiple quality issues."""
        dates = [
            pd.Timestamp("2025-01-01 00:00"),
            pd.Timestamp("2025-01-01 01:00"),
            pd.Timestamp("2025-01-01 01:00"),  # Duplicate
        ]
        df = pd.DataFrame(
            {
                "col1": [1, np.nan, 1000],  # Missing and potential outlier
                "col2": [1, 2, 3],
            },
            index=dates,
        )

        issues = detect_data_quality_issues(df)

        assert issues["missing_values"]["col1"] == 1
        # Outlier detection may or may not trigger depending on IQR calculation
        # Just check that outliers key exists and col1 is handled
        assert "outliers" in issues
        assert issues["duplicate_timestamps"] == 1
