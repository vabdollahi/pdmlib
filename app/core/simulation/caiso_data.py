"""
Electricity price data providers and analysis module for CAISO.

Provider-specific logic for CAISO lives here. Shared primitives like
ElectricityDataColumns and BasePriceProvider live in price_provider.py.
"""

from datetime import datetime
from enum import Enum
from typing import ClassVar, Dict, Optional

import pandas as pd
from pydantic import Field, model_validator

from app.core.simulation.price_provider import (
    BasePriceProvider,
    ElectricityDataColumns,
)
from app.core.utils.api_clients import CAISORateLimitedClient
from app.core.utils.caching import BaseProvider
from app.core.utils.http_cache import SimpleHTTPCache
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger

logger = get_logger("price_data")

# CAISO-specific exports only
__all__ = [
    "CAISORegion",
    "CAISOMarketType",
    "CAISOClient",
    "CAISOPriceProvider",
    "CAISODataState",
]


class CAISORegion(Enum):
    """CAISO pricing nodes and regions."""

    TH_NP15_GEN_APND = "TH_NP15_GEN-APND"  # NP15 generation hub
    TH_SP15_GEN_APND = "TH_SP15_GEN-APND"  # SP15 generation hub
    TH_ZP26_GEN_APND = "TH_ZP26_GEN-APND"  # ZP26 generation hub
    DLAP_PGAE_APND = "DLAP_PGAE-APND"  # PG&E load aggregation point
    DLAP_SCE_APND = "DLAP_SCE-APND"  # SCE load aggregation point
    DLAP_SDGE_APND = "DLAP_SDGE-APND"  # SDG&E load aggregation point


class CAISOMarketType(Enum):
    """CAISO market types."""

    DAM = "DAM"  # Day-Ahead Market
    RTM = "RTM"  # Real-Time Market


class CAISOClient(CAISORateLimitedClient):
    """
    Client for fetching electricity pricing data from CAISO OASIS API.

    CAISO OASIS provides real-time and day-ahead locational marginal prices (LMPs)
    for the California electricity market.
    """

    def __init__(self, cache_ttl_seconds: int = 600, cache_namespace: str = "caiso"):
        # Base URL for CAISO OASIS API (no trailing slash to avoid double slashes)
        super().__init__(base_url="https://oasis.caiso.com/oasisapi")
        # Allow overriding cache behavior via environment variables
        # CAISO_HTTP_CACHE_TTL (seconds), CAISO_HTTP_CACHE_DISABLED ("1" to disable)
        import os

        env_ttl = os.getenv("CAISO_HTTP_CACHE_TTL")
        if env_ttl is not None:
            try:
                cache_ttl_seconds = int(env_ttl)
            except ValueError:
                logger.warning(
                    "Invalid CAISO_HTTP_CACHE_TTL='%s', using default %ss",
                    env_ttl,
                    cache_ttl_seconds,
                )

        disabled = os.getenv("CAISO_HTTP_CACHE_DISABLED", "0").strip() in {
            "1",
            "true",
            "True",
        }
        effective_ttl = 0 if disabled else cache_ttl_seconds

        # Short-lived local cache for SingleZip downloads (reduces API calls & 429s)
        self.http_cache = SimpleHTTPCache(
            ttl_seconds=effective_ttl, namespace=cache_namespace
        )

    async def get_lmp_data(
        self,
        pricing_node: str,
        start_date: str,
        end_date: str,
        market_type: str = "DAM",
    ) -> pd.DataFrame:
        """
        Fetch Locational Marginal Price (LMP) data from CAISO OASIS.

        Args:
            pricing_node: CAISO pricing node (e.g., TH_NP15_GEN-APND)
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            market_type: Market type (DAM or RTM)

        Returns:
            DataFrame with LMP pricing data
        """
        logger.info(f"Fetching CAISO LMP data for node {pricing_node}")

        # CAISO OASIS parameters for PRC_LMP; timestamps are GMT (YYYYMMDDThh:mm-0000)
        params = {
            "queryname": "PRC_LMP",
            "version": "1",
            "market_run_id": market_type,
            "node": pricing_node,
            "startdatetime": f"{start_date}T00:00-0000",
            # Inclusive end day at 23:59 for the requested range
            "enddatetime": f"{end_date}T23:59-0000",
            "resultformat": "6",  # CSV format
        }

        try:
            # Query the 'SingleZip' endpoint, which returns a ZIP containing one CSV.
            # Try cache first using a canonical key derived from endpoint + params.
            cache_key = SimpleHTTPCache.canonical_key("SingleZip", params)
            zip_bytes = self.http_cache.get(cache_key)
            if zip_bytes is None:
                zip_bytes = await self.get_bytes("SingleZip", params)
                # Best-effort cache write of the ZIP payload
                self.http_cache.set(cache_key, zip_bytes)

            # Extract first CSV file from zip
            import io
            import zipfile

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    logger.warning("CAISO ZIP contains no CSV files")
                    return pd.DataFrame()
                with zf.open(csv_names[0]) as f:
                    response_text = f.read().decode("utf-8", errors="replace")

            return self._process_csv_response(response_text, pricing_node)

        except Exception as e:
            logger.error(f"Error fetching CAISO data: {e}")
            return pd.DataFrame()

    def _process_csv_response(self, csv_text: str, pricing_node: str) -> pd.DataFrame:
        """
        Process CAISO CSV response into a standardized DataFrame.

        Args:
            csv_text: Raw CSV response from CAISO
            pricing_node: Pricing node for context

        Returns:
            Processed DataFrame with standardized columns
        """
        try:
            # CAISO CSV begins with metadata rows; locate the header
            # and parse from there
            lines = csv_text.strip().split("\n")

            # Find the header line (look for common CAISO columns)
            header_idx = -1
            for i, line in enumerate(lines):
                if (
                    "INTERVALSTARTTIME_GMT" in line
                    or "OPR_DT" in line
                    or "INTERVALSTARTTIME" in line
                ):
                    header_idx = i
                    break

            if header_idx == -1:
                logger.warning("Could not find header in CAISO CSV response")
                return pd.DataFrame()

            # Read CSV starting from the header line
            from io import StringIO

            csv_data = "\n".join(lines[header_idx:])
            df = pd.read_csv(StringIO(csv_data))

            if df.empty:
                logger.warning("Empty data from CAISO")
                return pd.DataFrame()

            # Timestamp handling: prefer INTERVALSTARTTIME_GMT; otherwise derive from
            # OPR_DT, OPR_HR, OPR_INTERVAL (5- or 15-minute intervals); as a fallback
            # accept a generic TIMESTAMP column.
            ts_series = None
            if "INTERVALSTARTTIME_GMT" in df.columns:
                ts_series = pd.to_datetime(df["INTERVALSTARTTIME_GMT"], utc=True)
            elif {
                "OPR_DT",
                "OPR_HR",
                "OPR_INTERVAL",
            }.issubset(df.columns):
                # Build timestamp from date + hour + interval start (vectorized)
                opr_dt = pd.to_datetime(df["OPR_DT"], errors="coerce", utc=True)
                opr_hr = (
                    pd.to_numeric(df["OPR_HR"], errors="coerce").fillna(0).astype(int)
                    - 1
                )
                opr_itv = (
                    pd.to_numeric(df["OPR_INTERVAL"], errors="coerce")
                    .fillna(1)
                    .astype(int)
                    - 1
                )
                # Determine step: 5 minutes when OPR_INTERVAL spans >4; else 15
                step_min = (
                    5
                    if (df["OPR_INTERVAL"].dropna().astype(int).max() or 0) > 4
                    else 15
                )
                minutes = opr_itv * step_min
                ts_series = (
                    opr_dt
                    + pd.to_timedelta(opr_hr, unit="h")
                    + pd.to_timedelta(minutes, unit="m")
                )
            elif "TIMESTAMP" in df.columns:
                ts_series = pd.to_datetime(df["TIMESTAMP"], utc=True)

            if ts_series is None:
                logger.warning("CAISO CSV missing recognizable timestamp columns")
                return pd.DataFrame()

            # Price handling for PRC_LMP: keep only LMP rows (XML_DATA_ITEM=LMP_PRC)
            # and use the numeric value column (MW preferred).
            price_series = None
            if {"LMP_TYPE", "XML_DATA_ITEM"}.issubset(df.columns):
                mask = df["LMP_TYPE"].astype(str).str.upper().eq("LMP") & df[
                    "XML_DATA_ITEM"
                ].astype(str).str.upper().eq("LMP_PRC")
                lmp_rows = df[mask]
                # Choose preferred value column
                value_col = None
                for col in ["MW", "VALUE", "LMP_PRC", "PRC", "PRICE"]:
                    if col in lmp_rows.columns:
                        value_col = col
                        break
                if value_col is not None:
                    price_series = pd.to_numeric(lmp_rows[value_col], errors="coerce")
                    # Align timestamps to the filtered rows
                    if "INTERVALSTARTTIME_GMT" in df.columns:
                        ts_series = pd.to_datetime(
                            lmp_rows["INTERVALSTARTTIME_GMT"], utc=True
                        )
                    else:
                        ts_series = ts_series.loc[lmp_rows.index]
            else:
                # Fallback heuristics if expected columns are missing
                for col in ["MW", "VALUE", "LMP_PRC", "PRC", "PRICE"]:
                    if col in df.columns:
                        price_series = pd.to_numeric(df[col], errors="coerce")
                        break

            if price_series is None or ts_series is None:
                logger.warning("CAISO CSV missing price or timestamp data")
                return pd.DataFrame()

            processed_df = pd.DataFrame(
                {
                    ElectricityDataColumns.TIMESTAMP.value: ts_series.dt.tz_convert(
                        "UTC"
                    ).dt.tz_localize(None),
                    ElectricityDataColumns.PRICE_DOLLAR_MWH.value: price_series,
                }
            )

            # Drop NaNs, sort by timestamp, and reset index
            processed_df = processed_df.dropna().sort_values(
                ElectricityDataColumns.TIMESTAMP.value
            )

            # Set timestamp as index so upstream caching (Parquet) can persist it
            processed_df.set_index(ElectricityDataColumns.TIMESTAMP.value, inplace=True)

            logger.info(
                f"Processed {len(processed_df)} CAISO data points "
                f"for node {pricing_node}"
            )
            return processed_df

        except Exception as e:
            logger.error(f"Error processing CAISO CSV response: {e}")
            return pd.DataFrame()


class CAISOPriceProvider(BaseProvider, BasePriceProvider):
    """
    CAISO-specific price provider that fetches LMP data from CAISO OASIS API.

    This provider implements the BasePriceProvider interface for CAISO data,
    with region inference capabilities.
    """

    # Regional boundaries for automatic region inference (California focus)
    REGION_BOUNDARIES: ClassVar[Dict[CAISORegion, Dict[str, float]]] = {
        CAISORegion.TH_NP15_GEN_APND: {  # Northern California
            "lat_min": 37.0,
            "lat_max": 42.0,
            "lon_min": -124.5,
            "lon_max": -120.0,
        },
        CAISORegion.TH_SP15_GEN_APND: {  # Southern California
            "lat_min": 32.5,
            "lat_max": 37.0,
            "lon_min": -122.0,
            "lon_max": -114.0,
        },
        CAISORegion.TH_ZP26_GEN_APND: {  # Central/Bay Area
            "lat_min": 36.0,
            "lat_max": 39.0,
            "lon_min": -123.0,
            "lon_max": -119.0,
        },
    }

    # Configuration
    region: Optional[CAISORegion] = Field(
        default=None, description="CAISO pricing node"
    )
    market_type: str = Field(default="DAM", description="Market type (DAM or RTM)")

    def __init__(self, location: GeospatialLocation, **kwargs):
        super().__init__(location=location, **kwargs)

    def model_post_init(self, __context):
        """Post-initialization setup."""
        super().model_post_init(__context)
        # Region inference is handled by model validators automatically

    @model_validator(mode="after")
    def validate_date_range(self) -> "CAISOPriceProvider":
        """Validate that the date range is reasonable for CAISO data."""
        try:
            # Try different date formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:
                try:
                    start = datetime.strptime(self.start_date, fmt)
                    end = datetime.strptime(self.end_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Invalid date format: {self.start_date}")

            # CAISO data is available from around 2009
            min_date = datetime(2009, 1, 1)
            if start < min_date:
                logger.warning(f"Start date {start} is before CAISO data availability")

            # Warn if requesting more than 31 days of hourly data
            if (end - start).days > 31:
                logger.warning(
                    "Requesting more than 31 days of hourly data may be slow"
                )

        except Exception as e:
            logger.warning(f"Date validation error: {e}")

        return self

    @model_validator(mode="after")
    def infer_region_from_location(self) -> "CAISOPriceProvider":
        """Infer CAISO pricing node from geographical location."""
        if self.region is None and isinstance(self.location, GeospatialLocation):
            lat, lon = self.location.latitude, self.location.longitude

            # Find the best matching region
            for region, bounds in self.REGION_BOUNDARIES.items():
                if (
                    bounds["lat_min"] <= lat <= bounds["lat_max"]
                    and bounds["lon_min"] <= lon <= bounds["lon_max"]
                ):
                    self.region = region
                    logger.info(f"Inferred CAISO region: {region.value}")
                    break

            # Default to SP15 if no match (covers most of California)
            if self.region is None:
                self.region = CAISORegion.TH_SP15_GEN_APND
                logger.info(
                    f"Location outside known regions. Defaulting to {self.region.value}"
                )

        elif self.region is None:
            # Default to SP15 if no location provided
            self.region = CAISORegion.TH_SP15_GEN_APND
            logger.info(
                f"No location or region specified. Defaulting to {self.region.value}"
            )

        return self

    # Sync get_price_data is provided by BasePriceProvider

    async def _get_lmp_data_async(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Helper method to get LMP data asynchronously."""
        try:
            return await self._fetch_range(start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching CAISO data: {e}")
            # Return empty DataFrame when API fails
            return pd.DataFrame()

    def validate_data_format(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has the expected CAISO format.

        Args:
            df: DataFrame to validate

        Returns:
            True if format is valid, False otherwise
        """
        if df.empty:
            return True

        # Check if required columns exist (allowing for index-based timestamp)
        has_timestamp = (
            ElectricityDataColumns.TIMESTAMP.value in df.columns
            or df.index.name == ElectricityDataColumns.TIMESTAMP.value
            or isinstance(df.index, pd.DatetimeIndex)
        )

        has_price = ElectricityDataColumns.PRICE_DOLLAR_MWH.value in df.columns

        if not (has_timestamp and has_price):
            logger.error("CAISO data missing required columns")
            return False

        return True

    async def _fetch_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch CAISO pricing data for the specified date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with pricing data
        """
        try:
            # Ensure region is set (should be set by validators)
            if self.region is None:
                self.region = CAISORegion.TH_SP15_GEN_APND
                logger.warning("No region set, defaulting to SP15")

            client = CAISOClient()

            # Convert dates to CAISO format (YYYYMMDD)
            # Handle both date-only (YYYY-MM-DD) and datetime formats
            start_date_only = start_date.split()[0] if " " in start_date else start_date
            end_date_only = end_date.split()[0] if " " in end_date else end_date
            start_caiso = start_date_only.replace("-", "")
            end_caiso = end_date_only.replace("-", "")

            # Fetch data from CAISO OASIS
            data = await client.get_lmp_data(
                pricing_node=self.region.value,
                start_date=start_caiso,
                end_date=end_caiso,
                market_type=self.market_type,
            )

            # If no data received, return empty DataFrame
            if data.empty:
                logger.warning("No data from CAISO API")
                return pd.DataFrame()

            logger.info(f"Fetched {len(data)} CAISO price data points")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch CAISO data: {e}")
            # Return empty DataFrame when API fails
            return pd.DataFrame()


class CAISODataState:
    """
    Represents the current state of CAISO electricity pricing data for market
    operations.

    This class provides a snapshot of electricity market conditions at a specific
    time, useful for real-time decision making and market simulations.
    """

    def __init__(
        self,
        timestamp: datetime,
        price_dollar_mwh: float,
        region: CAISORegion,
    ):
        self.timestamp = timestamp
        self.price_dollar_mwh = price_dollar_mwh
        self.region = region
        self._analysis_cache = {}

    def is_peak_hour(self) -> bool:
        """Check if current time is typically a peak pricing hour."""
        hour = self.timestamp.hour
        # Peak hours are typically 16:00-20:00 (4 PM - 8 PM)
        return 16 <= hour <= 20

    def is_off_peak_hour(self) -> bool:
        """Check if current time is typically an off-peak pricing hour."""
        hour = self.timestamp.hour
        # Off-peak hours are typically 22:00-06:00 (10 PM - 6 AM)
        return hour >= 22 or hour <= 6

    def price_category(self) -> str:
        """Categorize the current price level."""
        # These thresholds are approximate and region-dependent
        if self.price_dollar_mwh < 30:
            return "low"
        elif self.price_dollar_mwh < 60:
            return "normal"
        elif self.price_dollar_mwh < 100:
            return "high"
        else:
            return "extreme"

    def get_revenue_potential(self, energy_mwh: float) -> float:
        """Calculate potential revenue for selling given amount of energy."""
        return energy_mwh * self.price_dollar_mwh

    def to_dict(self) -> Dict:
        """Convert state to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "price_dollar_mwh": self.price_dollar_mwh,
            "is_peak_hour": self.is_peak_hour(),
            "is_off_peak_hour": self.is_off_peak_hour(),
            "price_category": self.price_category(),
        }

    @classmethod
    def from_dataframe_row(
        cls, row: pd.Series, region: CAISORegion
    ) -> "CAISODataState":
        """Create state from a DataFrame row."""
        timestamp = pd.to_datetime(row["timestamp"])
        price = float(row.get("price_dollar_mwh", 0))

        return cls(
            timestamp=timestamp,
            price_dollar_mwh=price,
            region=region,
        )
