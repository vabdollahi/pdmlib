"""
Shared price provider primitives and facade.

This module defines:
- ElectricityDataColumns: unified column names for price data
- BasePriceProvider: common provider interface + sync bridge and standardization
- CSVPriceProvider: generic CSV-backed provider
- create_price_provider: factory that routes to CAISO/IESO/CSV
- PriceProvider facade + PriceProviderConfig convenience builder

Provider-specific implementations (CAISO, IESO) remain in their modules and
inherit from BaseProvider (cache/crawl) and BasePriceProvider (interface/helpers).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from app.core.simulation.price_regions import PriceMarketRegion
from app.core.utils.date_handling import TimeInterval
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger
from app.core.utils.storage import DataStorage

logger = get_logger("price_provider")


# -----------------------------------------------------------------------------
# Unified columns
# -----------------------------------------------------------------------------


class ElectricityDataColumns(Enum):
    """Standard column names for electricity pricing data."""

    TIMESTAMP = "timestamp"
    PRICE_DOLLAR_MWH = "price_dollar_mwh"


# -----------------------------------------------------------------------------
# Base provider interface and helpers
# -----------------------------------------------------------------------------


@runtime_checkable
class PriceProviderProtocol(Protocol):
    def get_price_data(
        self, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame: ...
    def validate_data_format(self, df: pd.DataFrame) -> bool: ...


class BasePriceProvider:
    """
    Base interface and helpers for electricity price providers.

    Providers typically also inherit from app.core.utils.caching.BaseProvider to
    gain async cache-first `get_data()` and range fetching. This base class
    offers a default sync bridge for get_price_data and column standardization.
    """

    # --- Public API ---------------------------------------------------------
    def get_price_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Default synchronous bridge that updates the provider's date range and
        invokes async `get_data()` using asyncio.run, with support for already-
        running event loops (e.g., inside pytest-asyncio) by offloading to a
        thread.
        """
        # Providers that also extend BaseProvider expose start_date/end_date
        try:
            self.start_date = start_time.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore[attr-defined]
            self.end_date = end_time.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore[attr-defined]
        except Exception:  # best-effort; still try to call get_data
            pass

        import asyncio
        import concurrent.futures

        async def _run() -> pd.DataFrame:
            try:
                get_data = getattr(self, "get_data", None)
                if get_data is None:
                    logger.warning("Provider has no get_data(); returning empty DF")
                    return pd.DataFrame()
                return await get_data()  # type: ignore[misc]
            except Exception as e:
                logger.error(f"Error fetching price data via async bridge: {e}")
                return pd.DataFrame()

        try:
            try:
                asyncio.get_running_loop()
                # In running loop: offload a one-shot asyncio.run to a thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Defer coroutine creation to the worker thread to avoid
                    # un-awaited coroutine warnings on submission failures.
                    future = executor.submit(lambda: asyncio.run(_run()))
                    return future.result()
            except RuntimeError:
                # No running loop
                return asyncio.run(_run())
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return pd.DataFrame()

    def validate_data_format(self, df: pd.DataFrame) -> bool:  # pragma: no cover
        """Generic validation: require timestamp and price when non-empty.

        Providers may override with more specific checks. Empty frames are
        considered valid to allow cache-miss / error scenarios to propagate.
        """
        if df.empty:
            return True
        ts_ok = ElectricityDataColumns.TIMESTAMP.value in df.columns or isinstance(
            df.index, pd.DatetimeIndex
        )
        price_ok = ElectricityDataColumns.PRICE_DOLLAR_MWH.value in df.columns
        return bool(ts_ok and price_ok)

    # --- Helpers ------------------------------------------------------------
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize to unified columns without dropping source columns.

        - Create/rename a timestamp column to `timestamp` when possible.
        - Create `price_dollar_mwh` from any known price columns if missing.
        """
        if df.empty:
            return df

        result = df.copy()

        # Normalize timestamp column name
        ts_aliases = ["timestamp", "datetime", "date_time", "time", "TIMESTAMP"]
        ts_col = next((c for c in ts_aliases if c in result.columns), None)
        if ts_col and ts_col != ElectricityDataColumns.TIMESTAMP.value:
            result = result.rename(
                columns={ts_col: ElectricityDataColumns.TIMESTAMP.value}
            )

        # Ensure timestamp is datetime when present
        if ElectricityDataColumns.TIMESTAMP.value in result.columns:
            result[ElectricityDataColumns.TIMESTAMP.value] = pd.to_datetime(
                result[ElectricityDataColumns.TIMESTAMP.value], errors="coerce"
            )

        # Normalize price
        price_targets = [
            ElectricityDataColumns.PRICE_DOLLAR_MWH.value,
            "price",
            "price_mwh",
            "lmp",
            "lmp_price",
            "PRICE",
            "VALUE",
            "MW",
            "hoep_cad_mwh",
        ]
        if ElectricityDataColumns.PRICE_DOLLAR_MWH.value not in result.columns:
            src = next((c for c in price_targets if c in result.columns), None)
            if src:
                result[ElectricityDataColumns.PRICE_DOLLAR_MWH.value] = pd.to_numeric(
                    result[src], errors="coerce"
                )

        return result


# -----------------------------------------------------------------------------
# CSV-backed provider (generic)
# -----------------------------------------------------------------------------


class CSVPriceProvider(BasePriceProvider):
    """Price provider that reads data from a CSV file."""

    def __init__(self, csv_file_path: Union[str, Path]):
        self.csv_file_path = Path(csv_file_path)
        self._data_cache: Optional[pd.DataFrame] = None

    def get_price_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        # For CSV we don't need async; just load and filter
        try:
            if self._data_cache is None:
                self._data_cache = self._load_csv_data()
            if self._data_cache.empty:
                return pd.DataFrame()

            df = self._data_cache

            if not isinstance(df.index, pd.DatetimeIndex):
                # Ensure index is timestamp for efficient slicing
                if ElectricityDataColumns.TIMESTAMP.value in df.columns:
                    df = df.set_index(ElectricityDataColumns.TIMESTAMP.value)
                else:
                    logger.error("CSV missing timestamp column for indexing")
                    return pd.DataFrame()

            # Filter inclusive using label-based slicing
            return df.loc[start_time:end_time].copy()
        except Exception as e:
            logger.error(f"CSV provider error: {e}")
            return pd.DataFrame()

    def _load_csv_data(self) -> pd.DataFrame:
        try:
            if not self.csv_file_path.exists():
                raise FileNotFoundError(f"CSV not found: {self.csv_file_path}")
            df = pd.read_csv(self.csv_file_path)
            df = self._standardize_columns(df)
            if not self.validate_data_format(df):
                raise ValueError("CSV file does not have expected format")
            # Sort and index by timestamp
            if ElectricityDataColumns.TIMESTAMP.value in df.columns:
                df = df.sort_values(ElectricityDataColumns.TIMESTAMP.value)
                df[ElectricityDataColumns.TIMESTAMP.value] = pd.to_datetime(
                    df[ElectricityDataColumns.TIMESTAMP.value]
                )
                df = df.set_index(ElectricityDataColumns.TIMESTAMP.value)
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return pd.DataFrame()

    def validate_data_format(self, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        cols = df.columns
        ts_ok = ElectricityDataColumns.TIMESTAMP.value in cols or isinstance(
            df.index, pd.DatetimeIndex
        )
        price_ok = ElectricityDataColumns.PRICE_DOLLAR_MWH.value in cols
        if not (ts_ok and price_ok):
            logger.error("CSV missing required columns")
            return False
        try:
            pd.to_numeric(
                df[ElectricityDataColumns.PRICE_DOLLAR_MWH.value], errors="raise"
            )
        except Exception:
            logger.error("CSV price column is not numeric")
            return False
        return True


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def create_price_provider(
    source_type: str,
    location: Optional[GeospatialLocation] = None,
    csv_file_path: Optional[Union[str, Path]] = None,
    market_region: Optional[Union[str, PriceMarketRegion]] = None,
    **kwargs,
) -> PriceProviderProtocol:
    """Create a price provider instance by source_type.

    Supported types: 'csv', 'caiso', 'ieso', 'auto'
    """
    st = source_type.lower()
    if st == "csv":
        if not csv_file_path:
            raise ValueError("CSV file path is required for CSV price provider")
        return CSVPriceProvider(csv_file_path=csv_file_path)

    if st in {"caiso", "ieso", "auto"}:
        # Resolve market region for 'auto'
        region = (
            PriceMarketRegion(market_region)
            if isinstance(market_region, str)
            else market_region
        )
        if st == "ieso" or region == PriceMarketRegion.IESO:
            # Lazy import to avoid cycles
            from app.core.simulation.ieso_data import IESOPriceProvider

            return IESOPriceProvider(location=location, **kwargs)
        # Default to CAISO
        if location is None:
            raise ValueError("Location is required for CAISO/auto price provider")
        from app.core.simulation.caiso_data import CAISOPriceProvider

        return CAISOPriceProvider(location=location, **kwargs)

    raise ValueError(
        f"Unknown source type: {source_type}. Supported: 'caiso', 'csv', 'ieso', 'auto'"
    )


class PriceProvider(BaseModel):
    """General price provider facade that routes by market region.

    This model mirrors the common BaseProvider configuration and instantiates
    the correct underlying provider (CAISO or IESO) based on `market_region`.
    """

    # Common provider configuration
    location: GeospatialLocation
    start_date: str
    end_date: str
    organization: str
    asset: str
    data_type: str = Field(default="market_price")
    interval: TimeInterval = Field(default=TimeInterval.HOURLY)
    storage: DataStorage = Field(default_factory=DataStorage)

    # Routing argument
    market_region: PriceMarketRegion = Field(default=PriceMarketRegion.CAISO)

    # Internal delegated provider (private attribute, not a Field)
    _provider: Optional[PriceProviderProtocol] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """Instantiate the underlying provider after validation."""
        if self.market_region == PriceMarketRegion.CAISO:
            logger.info("Initializing CAISO price provider")
            # Lazy import to avoid circulars
            from app.core.simulation.caiso_data import CAISOPriceProvider

            self._provider = CAISOPriceProvider(
                location=self.location,
                start_date=self.start_date,
                end_date=self.end_date,
                organization=self.organization,
                asset=self.asset,
                data_type=self.data_type,
                interval=self.interval,
                storage=self.storage,
            )
        elif self.market_region == PriceMarketRegion.IESO:
            logger.info("Initializing IESO price provider")
            # Import locally to avoid circulars on module import
            from app.core.simulation.ieso_data import IESOPriceProvider

            self._provider = IESOPriceProvider(
                location=self.location,
                start_date=self.start_date,
                end_date=self.end_date,
                organization=self.organization,
                asset=self.asset,
                data_type=self.data_type,
                interval=self.interval,
                storage=self.storage,
            )
        else:
            raise ValueError(f"Unsupported market region: {self.market_region}")

    # --- Public API (delegated) -------------------------------------------------
    def get_price_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Delegate price retrieval to the selected provider."""
        if not self._provider:
            logger.error("Underlying provider not initialized")
            return pd.DataFrame()
        return self._provider.get_price_data(start_time, end_time)

    async def get_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Delegate cache-first retrieval if supported by underlying provider."""
        provider = getattr(self, "_provider", None)
        if provider is None or not hasattr(provider, "get_data"):
            logger.warning(
                "Underlying provider does not support get_data(); using get_price_data"
            )
            # As a fallback, use the synchronous bridge method
            from app.core.utils.date_handling import parse_datetime_input

            start_dt = parse_datetime_input(self.start_date)
            end_dt = parse_datetime_input(self.end_date)
            return self.get_price_data(start_dt, end_dt)
        # Update provider dates to match wrapper in case they changed
        provider.start_date = self.start_date
        provider.end_date = self.end_date
        return await provider.get_data(force_refresh=force_refresh)

    # --- Introspection helpers -------------------------------------------------
    @property
    def underlying(self) -> Optional[PriceProviderProtocol]:
        """Access the underlying provider instance (read-only)."""
        return self._provider


class PriceProviderConfig(BaseModel):
    """Pydantic config to construct a region-specific price provider.

    This mirrors the previous main.py helper but lives in the simulation layer.
    By default, it returns the underlying market-specific provider for
    compatibility with existing interfaces.
    """

    market_region: PriceMarketRegion = Field(
        default=PriceMarketRegion.CAISO, description="Target market region"
    )
    location: GeospatialLocation = Field(description="Geospatial location of the asset")
    start_date: str
    end_date: str
    organization: str = "SolarRevenue"
    asset: str = "LMP_Data"
    data_type: str = "market_price"
    interval: TimeInterval = Field(default=TimeInterval.HOURLY)
    storage: DataStorage = Field(default_factory=DataStorage)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_provider(self) -> PriceProviderProtocol:
        # Choose data_type by region for nicer partitioning
        region_dt = (
            "caiso_lmp"
            if self.market_region == PriceMarketRegion.CAISO
            else "ieso_hoep"
        )
        return create_price_provider(
            source_type="auto",
            location=self.location,
            start_date=self.start_date,
            end_date=self.end_date,
            organization=self.organization,
            asset=self.asset,
            data_type=region_dt,
            market_region=self.market_region,
            interval=self.interval,
            storage=self.storage,
        )
