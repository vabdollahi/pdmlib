"""
Tests for the DataStorage class.
"""

import pandas as pd
import pytest

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
    storage.write_data(
        df=sample_dataframe,
        organization="TestOrg",
        asset="TestAsset",
        latitude=12.34,
        longitude=56.78,
    )

    # Check that files for each month have been created
    expected_files = [
        "memory://test-data/TestOrg/TestAsset/lat12_34_lon56_78/2024_01.parquet",
        "memory://test-data/TestOrg/TestAsset/lat12_34_lon56_78/2024_02.parquet",
        "memory://test-data/TestOrg/TestAsset/lat12_34_lon56_78/2024_03.parquet",
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
    storage.write_data(
        df=sample_dataframe,
        organization="TestOrg",
        asset="TestAsset",
        latitude=12.34,
        longitude=56.78,
    )

    # Now, read a range that spans multiple months
    df_read = storage.read_data_for_range(
        organization="TestOrg",
        asset="TestAsset",
        latitude=12.34,
        longitude=56.78,
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
    df_read = storage.read_data_for_range(
        organization="NoOrg",
        asset="NoAsset",
        latitude=0,
        longitude=0,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    assert df_read.empty
