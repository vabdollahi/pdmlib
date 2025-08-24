"""
Tests for the DataStorage class.
"""

import pandas as pd
import pytest

from app.core.utils.location import GeospatialLocation, RegionalLocation
from app.core.utils.storage import DataStorage


@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame for testing."""
    dates = pd.to_datetime(
        pd.date_range(start="2024-01-15", end="2024-03-10", freq="D")
    )
    data = {
        "ghi": range(len(dates)),
        "dni": range(len(dates)),
        "dhi": range(len(dates)),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date_time"
    return df


@pytest.fixture
def storage():
    """Initializes DataStorage with an in-memory filesystem for testing."""
    return DataStorage(base_path="memory://test-data")


def test_write_data_creates_partitions_and_files(storage, sample_dataframe):
    """
    Tests that write_data creates the correct directory structure and monthly files.
    """
    location = GeospatialLocation(latitude=12.34, longitude=56.78)
    storage.write_data(
        df=sample_dataframe,
        organization="TestOrg",
        asset="TestAsset",
        data_type="weather",
        location=location,
    )

    # Check that files for each month have been created
    expected_files = [
        "memory://test-data/TestOrg/TestAsset/weather/lat12_34_lon56_78/2024_01.parquet",
        "memory://test-data/TestOrg/TestAsset/weather/lat12_34_lon56_78/2024_02.parquet",
        "memory://test-data/TestOrg/TestAsset/weather/lat12_34_lon56_78/2024_03.parquet",
    ]
    for f in expected_files:
        assert storage.fs.exists(f)

    # Check content of one file
    with storage.fs.open(expected_files[0], "rb") as f:
        df_read = pd.read_parquet(f)
    assert "date_time" in df_read.columns
    assert len(df_read) == 17  # 17 days in Jan from the sample data


def test_read_data_for_range_combines_and_filters(storage, sample_dataframe):
    """
    Tests that read_data_for_range correctly reads, combines, and filters data.
    """
    # First, write the data
    location = GeospatialLocation(latitude=12.34, longitude=56.78)
    storage.write_data(
        df=sample_dataframe,
        organization="TestOrg",
        asset="TestAsset",
        data_type="weather",
        location=location,
    )

    # Now, read a range that spans multiple months
    df_read = storage.read_data_for_range(
        organization="TestOrg",
        asset="TestAsset",
        data_type="weather",
        location=location,
        start_date="2024-01-20",
        end_date="2024-02-10",
    )

    assert isinstance(df_read.index, pd.DatetimeIndex)
    assert df_read.index.name == "date_time"
    assert df_read.index.min() == pd.to_datetime("2024-01-20")
    assert df_read.index.max() == pd.to_datetime("2024-02-10")
    assert len(df_read) == 22  # 12 days in Jan + 10 in Feb


def test_read_from_non_existent_path_returns_empty(storage):
    """
    Tests that reading from a path that doesn't exist returns an empty DataFrame.
    """
    location = GeospatialLocation(latitude=0, longitude=0)
    df_read = storage.read_data_for_range(
        organization="NoOrg",
        asset="NoAsset",
        data_type="weather",
        location=location,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    assert df_read.empty


def test_write_data_updates_existing_file(storage):
    """
    Tests that writing data to a month that already has a file correctly
    updates and merges the data without losing existing records.
    """
    # Create initial data for January
    location = GeospatialLocation(latitude=12.34, longitude=56.78)
    jan_dates_1 = pd.to_datetime(pd.date_range("2024-01-01", "2024-01-10", freq="D"))
    jan_df_1 = pd.DataFrame({"value": range(10)}, index=jan_dates_1)
    jan_df_1.index.name = "date_time"

    storage.write_data(
        df=jan_df_1,
        organization="TestOrg",
        asset="TestAsset",
        data_type="test_data",
        location=location,
    )

    # Create new data for January, partially overlapping
    jan_dates_2 = pd.to_datetime(pd.date_range("2024-01-05", "2024-01-15", freq="D"))
    # Use different values to check the update
    jan_df_2 = pd.DataFrame({"value": range(100, 111)}, index=jan_dates_2)
    jan_df_2.index.name = "date_time"

    storage.write_data(
        df=jan_df_2,
        organization="TestOrg",
        asset="TestAsset",
        data_type="test_data",
        location=location,
    )

    # Read the entire month back
    full_jan_data = storage.read_data_for_range(
        organization="TestOrg",
        asset="TestAsset",
        data_type="test_data",
        location=location,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    # Check that the total number of days is correct (15 days)
    assert len(full_jan_data) == 15
    # Check that the first part of the old data is still there
    assert full_jan_data.loc["2024-01-01"]["value"] == 0
    # Check that the overlapping data has been updated with the new values
    assert full_jan_data.loc["2024-01-10"]["value"] == 105
    # Check that the newest data is present
    assert full_jan_data.loc["2024-01-15"]["value"] == 110


def test_storage_with_regional_location(storage):
    """
    Tests that the storage works correctly with a different location type.
    """
    location = RegionalLocation(country="Germany", region="Bavaria")
    dates = pd.to_datetime(pd.date_range("2024-01-01", "2024-01-05", freq="D"))
    df = pd.DataFrame({"price": [100, 110, 105, 120, 115]}, index=dates)
    df.index.name = "date_time"

    storage.write_data(
        df=df,
        organization="EnergyCorp",
        asset="GridDE",
        data_type="price",
        location=location,
    )

    expected_file = "memory://test-data/EnergyCorp/GridDE/price/country_germany_region_bavaria/2024_01.parquet"
    assert storage.fs.exists(expected_file)

    df_read = storage.read_data_for_range(
        organization="EnergyCorp",
        asset="GridDE",
        data_type="price",
        location=location,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    assert len(df_read) == 5
    assert df_read.iloc[0]["price"] == 100
