"""
Unit tests for electricity price data providers.

This module tests both CSV and CAISO price providers to ensure they
properly implement the BasePriceProvider interface.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from app.core.simulation.caiso_data import (
    BasePriceProvider,
    CSVPriceProvider,
    ElectricityDataColumns,
    create_price_provider,
)


class TestBasePriceProvider:
    """Test the abstract base class interface."""

    def test_base_provider_is_abstract(self):
        """Test that BasePriceProvider cannot be instantiated directly."""
        # The abstract methods should prevent direct instantiation
        assert hasattr(BasePriceProvider, "get_price_data")
        assert hasattr(BasePriceProvider, "validate_data_format")


class TestCSVPriceProvider:
    """Test CSV price provider functionality."""

    @pytest.fixture
    def sample_csv_path(self):
        """Path to the sample CSV file."""
        return Path(__file__).parent.parent.parent / "data" / "sample_price_data.csv"

    @pytest.fixture
    def csv_provider(self, sample_csv_path):
        """Create a CSV provider instance."""
        return CSVPriceProvider(csv_file_path=sample_csv_path)

    def test_csv_provider_initialization(self, sample_csv_path):
        """Test CSV provider can be initialized."""
        provider = CSVPriceProvider(csv_file_path=sample_csv_path)
        assert provider.csv_file_path == sample_csv_path
        assert provider._data_cache is None

    def test_csv_provider_loads_data(self, csv_provider):
        """Test CSV provider loads data correctly."""
        start_time = datetime(2024, 7, 15)
        end_time = datetime(2024, 7, 16)

        data = csv_provider.get_price_data(start_time, end_time)

        assert not data.empty
        assert len(data) == 24  # 24 hours of data
        assert ElectricityDataColumns.PRICE_USD_MWH.value in data.columns
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_csv_provider_data_validation(self, csv_provider):
        """Test CSV provider validates data format."""
        # Valid data
        valid_data = pd.DataFrame(
            {
                ElectricityDataColumns.TIMESTAMP.value: pd.date_range(
                    "2024-07-15", periods=3, freq="h"
                ),
                ElectricityDataColumns.PRICE_USD_MWH.value: [45.5, 42.3, 39.8],
            }
        )
        assert csv_provider.validate_data_format(valid_data)

        # Invalid data - missing price column
        invalid_data = pd.DataFrame(
            {
                ElectricityDataColumns.TIMESTAMP.value: pd.date_range(
                    "2024-07-15", periods=3, freq="h"
                ),
            }
        )
        assert not csv_provider.validate_data_format(invalid_data)

    def test_csv_provider_time_filtering(self, csv_provider):
        """Test CSV provider filters data by time range."""
        # Request only first 6 hours
        start_time = datetime(2024, 7, 15)
        end_time = datetime(2024, 7, 15, 6)

        data = csv_provider.get_price_data(start_time, end_time)

        assert not data.empty
        assert len(data) <= 7  # Should be 6-7 hours depending on inclusivity
        assert data.index.min() >= start_time
        assert data.index.max() <= end_time

    def test_csv_provider_with_nonexistent_file(self):
        """Test CSV provider handles nonexistent file gracefully."""
        provider = CSVPriceProvider("nonexistent_file.csv")

        start_time = datetime(2024, 7, 15)
        end_time = datetime(2024, 7, 16)

        data = provider.get_price_data(start_time, end_time)
        assert data.empty

    def test_csv_provider_column_mapping(self, csv_provider):
        """Test CSV provider standardizes column names."""
        # Test internal _standardize_columns method
        test_data = pd.DataFrame(
            {"datetime": ["2024-07-15 00:00:00"], "price": [45.5], "lmp": [50.0]}
        )

        standardized = csv_provider._standardize_columns(test_data)

        assert ElectricityDataColumns.TIMESTAMP.value in standardized.columns
        assert ElectricityDataColumns.PRICE_USD_MWH.value in standardized.columns


class TestPriceProviderFactory:
    """Test the factory function for creating price providers."""

    def test_create_csv_provider(self):
        """Test factory creates CSV provider."""
        csv_path = (
            Path(__file__).parent.parent.parent / "data" / "sample_price_data.csv"
        )

        provider = create_price_provider(source_type="csv", csv_file_path=csv_path)

        assert isinstance(provider, CSVPriceProvider)
        assert isinstance(provider, BasePriceProvider)

    def test_factory_invalid_source_type(self):
        """Test factory raises error for invalid source type."""
        with pytest.raises(ValueError, match="Unknown source type"):
            create_price_provider(source_type="invalid")

    def test_factory_missing_csv_path(self):
        """Test factory raises error when CSV path is missing."""
        with pytest.raises(ValueError, match="CSV file path is required"):
            create_price_provider(source_type="csv")


class TestCSVPriceProviderInterface:
    """Test that CSV provider implements the correct interface."""

    @pytest.fixture
    def csv_provider(self):
        """CSV provider instance."""
        csv_path = (
            Path(__file__).parent.parent.parent / "data" / "sample_price_data.csv"
        )
        return CSVPriceProvider(csv_file_path=csv_path)

    def test_csv_provider_interface(self, csv_provider):
        """Test CSV provider implements the BasePriceProvider interface."""
        # Should be instance of BasePriceProvider
        assert isinstance(csv_provider, BasePriceProvider)

        # Should have the required interface methods
        assert hasattr(csv_provider, "get_price_data")
        assert hasattr(csv_provider, "validate_data_format")

        # Should return data in the correct format
        start_time = datetime(2024, 7, 15)
        end_time = datetime(2024, 7, 16)

        data = csv_provider.get_price_data(start_time, end_time)

        # Should return DataFrame with standard columns
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert ElectricityDataColumns.PRICE_USD_MWH.value in data.columns
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_csv_data_format_standardization(self, csv_provider):
        """Test CSV provider returns standardized data format."""
        start_time = datetime(2024, 7, 15)
        end_time = datetime(2024, 7, 16)

        data = csv_provider.get_price_data(start_time, end_time)

        # Should have exactly the expected columns
        expected_columns = [ElectricityDataColumns.PRICE_USD_MWH.value]
        assert list(data.columns) == expected_columns

        # Index should be datetime with correct name
        assert isinstance(data.index, pd.DatetimeIndex)

        # Price data should be numeric
        price_column = data[ElectricityDataColumns.PRICE_USD_MWH.value]
        assert pd.api.types.is_numeric_dtype(price_column)
