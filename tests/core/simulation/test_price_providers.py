"""
Unit tests for electricity price data providers.

This module tests CSV, CAISO, and IESO price providers to ensure they
properly implement the BasePriceProvider interface and handle data correctly.
Tests are designed for efficiency using mocks and fixtures without API calls.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.core.simulation.caiso_data import CAISOPriceProvider
from app.core.simulation.ieso_data import IESOPriceProvider
from app.core.simulation.price_provider import (
    BasePriceProvider,
    CSVPriceProvider,
    PriceColumns,
    create_price_provider,
)
from app.core.utils.location import GeospatialLocation


class TestBasePriceProvider:
    """Test base class helpers only (no abstractness)."""

    def test_standardize_columns_mapping(self):
        """Test that column standardization maps various names correctly."""

        # Create a concrete implementation for testing
        class TestProvider(BasePriceProvider):
            async def get_data(self):
                return pd.DataFrame()

            def validate_data_format(self, df):
                return True

        provider = TestProvider()

        # Test various column name mappings
        test_cases = [
            (
                {"datetime": [datetime(2025, 7, 15)], "price": [45.50]},
                {"timestamp": [datetime(2025, 7, 15)], "price_dollar_mwh": [45.50]},
            ),
            (
                {"date_time": [datetime(2025, 7, 15)], "lmp": [45.50]},
                {"timestamp": [datetime(2025, 7, 15)], "price_dollar_mwh": [45.50]},
            ),
            (
                {"time": [datetime(2025, 7, 15)], "lmp_price": [45.50]},
                {"timestamp": [datetime(2025, 7, 15)], "price_dollar_mwh": [45.50]},
            ),
        ]

        for input_data, expected_cols in test_cases:
            df = pd.DataFrame(input_data)
            result = provider._standardize_columns(df)
            for col in expected_cols.keys():
                assert col in result.columns


class TestCSVPriceProvider:
    """Test CSV price provider functionality."""

    @pytest.fixture
    def sample_csv_path(self) -> Path:
        """Provide path to sample CSV file."""
        return Path(__file__).parent.parent.parent / "data" / "sample_price_data.csv"

    @pytest.fixture
    def csv_provider(self, sample_csv_path: Path) -> CSVPriceProvider:
        """Create CSV price provider with sample data."""
        return CSVPriceProvider(csv_file_path=sample_csv_path)

    def test_csv_provider_initialization(self, sample_csv_path: Path):
        """Test CSV provider can be initialized with valid file path."""
        provider = CSVPriceProvider(csv_file_path=sample_csv_path)
        assert provider.csv_file_path == sample_csv_path
        assert provider._data_cache is None

    @pytest.mark.asyncio
    async def test_get_data_success(self, csv_provider: CSVPriceProvider):
        """Test getting price data from CSV file."""
        start_time = datetime(2025, 7, 15, 0, 0, 0)
        end_time = datetime(2025, 7, 15, 23, 59, 59)

        csv_provider.set_range(start_time, end_time)
        data = await csv_provider.get_data()

        assert not data.empty
        assert PriceColumns.PRICE_DOLLAR_MWH.value in data.columns
        assert len(data) == 24  # 24 hours of data


class TestCAISOPriceProvider:
    """Test CAISO price provider functionality without API calls."""

    @pytest.fixture
    def mock_location(self) -> GeospatialLocation:
        """Create mock location in California."""
        return GeospatialLocation(latitude=37.7749, longitude=-122.4194)  # SF

    @pytest.fixture
    def mock_caiso_provider_kwargs(self) -> dict:
        """Create kwargs for CAISO provider initialization."""
        return {
            "start_date": "2025-07-15 00:00:00",
            "end_date": "2025-07-15 23:59:59",
            "organization": "TestOrg",
            "asset": "TestAsset",
            "data_type": "caiso_lmp",
        }

    @patch("app.core.simulation.caiso_data.CAISOPriceProvider._fetch_range")
    def test_caiso_provider_initialization(
        self,
        mock_fetch,
        mock_location: GeospatialLocation,
        mock_caiso_provider_kwargs: dict,
    ):
        """Test CAISO provider initialization and region inference."""
        provider = CAISOPriceProvider(
            location=mock_location, **mock_caiso_provider_kwargs
        )
        assert provider.location == mock_location
        assert provider.region is not None  # Should infer a region


class TestIESOPriceProvider:
    """Test IESO price provider functionality without API calls."""

    @pytest.fixture
    def mock_location(self) -> GeospatialLocation:
        """Create mock location in Ontario."""
        return GeospatialLocation(latitude=43.6532, longitude=-79.3832)  # Toronto

    @pytest.fixture
    def mock_ieso_provider_kwargs(self) -> dict:
        """Create kwargs for IESO provider initialization."""
        return {
            "start_date": "2025-07-15 00:00:00",
            "end_date": "2025-07-15 23:59:59",
            "organization": "TestOrg",
            "asset": "TestAsset",
            "data_type": "ieso_hoep",
        }

    @patch("app.core.simulation.ieso_data.IESOPriceProvider._fetch_range")
    def test_ieso_provider_initialization(
        self,
        mock_fetch,
        mock_location: GeospatialLocation,
        mock_ieso_provider_kwargs: dict,
    ):
        """Test IESO provider initialization."""
        provider = IESOPriceProvider(
            location=mock_location, **mock_ieso_provider_kwargs
        )
        assert provider.location == mock_location


class TestPriceProviderFactory:
    """Test price provider factory function with improved coverage."""

    @pytest.fixture
    def sample_csv_path(self) -> Path:
        """Provide path to sample CSV file."""
        return Path(__file__).parent.parent.parent / "data" / "sample_price_data.csv"

    @pytest.fixture
    def mock_location(self) -> GeospatialLocation:
        """Create mock location for testing."""
        return GeospatialLocation(latitude=37.7749, longitude=-122.4194)

    @pytest.fixture
    def mock_provider_kwargs(self) -> dict:
        """Create kwargs for provider initialization."""
        return {
            "start_date": "2025-07-15 00:00:00",
            "end_date": "2025-07-15 23:59:59",
            "organization": "TestOrg",
            "asset": "TestAsset",
            "data_type": "test_price",
        }

    def test_create_csv_provider(self, sample_csv_path: Path):
        """Test factory creates CSV provider correctly."""
        provider = create_price_provider("csv", csv_file_path=sample_csv_path)

        assert isinstance(provider, CSVPriceProvider)
        assert provider.csv_file_path == sample_csv_path

    @patch("app.core.simulation.caiso_data.CAISOPriceProvider._fetch_range")
    def test_create_caiso_provider(
        self, mock_fetch, mock_location: GeospatialLocation, mock_provider_kwargs: dict
    ):
        """Test factory creates CAISO provider correctly."""
        provider = create_price_provider(
            "caiso", location=mock_location, **mock_provider_kwargs
        )

        assert isinstance(provider, CAISOPriceProvider)
        assert provider.location == mock_location
