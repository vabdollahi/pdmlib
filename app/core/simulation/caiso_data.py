"""
Electricity price data providers and analysis module.

This module provides functionality to fetch and analyze electricity pricing data from
multiple sources including CAISO OASIS API and user-provided CSV files.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar, Dict, Optional, Union

import pandas as pd
from pydantic import Field, model_validator

from app.core.utils.api_clients import CAISORateLimitedClient
from app.core.utils.caching import BaseProvider
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger

logger = get_logger("price_data")


class ElectricityDataColumns(Enum):
    """Standard column names for electricity pricing data."""

    TIMESTAMP = "timestamp"
    PRICE_USD_MWH = "price_usd_mwh"


class BasePriceProvider(ABC):
    """
    Abstract base class for electricity price data providers.

    This class defines the interface for all price data sources, whether from
    APIs like CAISO or user-provided CSV files.
    """

    @abstractmethod
    def get_price_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Get electricity price data for the specified time range.

        Args:
            start_time: Start datetime for price data
            end_time: End datetime for price data

        Returns:
            DataFrame with columns: timestamp, price_usd_mwh
        """
        pass

    @abstractmethod
    def validate_data_format(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has the expected format.

        Args:
            df: DataFrame to validate

        Returns:
            True if format is valid, False otherwise
        """
        pass

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to match ElectricityDataColumns enum.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized column names
        """
        if df.empty:
            return df

        # Map common column name variations to standard names
        column_mapping = {
            "datetime": ElectricityDataColumns.TIMESTAMP.value,
            "date_time": ElectricityDataColumns.TIMESTAMP.value,
            "time": ElectricityDataColumns.TIMESTAMP.value,
            "price": ElectricityDataColumns.PRICE_USD_MWH.value,
            "price_mwh": ElectricityDataColumns.PRICE_USD_MWH.value,
            "lmp": ElectricityDataColumns.PRICE_USD_MWH.value,
            "lmp_price": ElectricityDataColumns.PRICE_USD_MWH.value,
        }

        # Apply mapping
        df_renamed = df.rename(columns=column_mapping)

        return df_renamed


class CSVPriceProvider(BasePriceProvider):
    """
    Price provider that reads data from user-provided CSV files.

    Expects CSV files with columns: timestamp, price_usd_mwh
    """

    def __init__(self, csv_file_path: Union[str, Path]):
        """
        Initialize CSV price provider.

        Args:
            csv_file_path: Path to the CSV file containing price data
        """
        self.csv_file_path = Path(csv_file_path)
        self._data_cache = None

    def get_price_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Get price data from CSV file for the specified time range.

        Args:
            start_time: Start datetime for price data
            end_time: End datetime for price data

        Returns:
            DataFrame with price data filtered to the time range
        """
        try:
            # Load data if not cached
            if self._data_cache is None:
                self._data_cache = self._load_csv_data()

            if self._data_cache.empty:
                logger.warning("No data loaded from CSV file")
                return pd.DataFrame()

            # Filter by time range
            filtered_data = self._filter_by_time_range(
                self._data_cache, start_time, end_time
            )

            logger.info(f"Retrieved {len(filtered_data)} price data points from CSV")
            return filtered_data

        except Exception as e:
            logger.error(f"Error getting price data from CSV: {e}")
            return pd.DataFrame()

    def _load_csv_data(self) -> pd.DataFrame:
        """Load and validate CSV data."""
        try:
            if not self.csv_file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

            # Read CSV file
            df = pd.read_csv(self.csv_file_path)

            # Standardize column names
            df = self._standardize_columns(df)

            # Validate format
            if not self.validate_data_format(df):
                raise ValueError("CSV file does not have expected format")

            # Convert timestamp to datetime
            df[ElectricityDataColumns.TIMESTAMP.value] = pd.to_datetime(
                df[ElectricityDataColumns.TIMESTAMP.value]
            )

            # Sort by timestamp
            df = df.sort_values(ElectricityDataColumns.TIMESTAMP.value)

            # Set timestamp as index
            df.set_index(ElectricityDataColumns.TIMESTAMP.value, inplace=True)

            logger.info(f"Loaded {len(df)} price data points from CSV file")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return pd.DataFrame()

    def _filter_by_time_range(
        self, df: pd.DataFrame, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """Filter DataFrame by time range."""
        if df.empty:
            return df

        return df[(df.index >= start_time) & (df.index <= end_time)].copy()

    def validate_data_format(self, df: pd.DataFrame) -> bool:
        """
        Validate that the CSV DataFrame has the expected format.

        Args:
            df: DataFrame to validate

        Returns:
            True if format is valid, False otherwise
        """
        required_columns = [
            ElectricityDataColumns.TIMESTAMP.value,
            ElectricityDataColumns.PRICE_USD_MWH.value,
        ]

        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check if price column has numeric data
        try:
            pd.to_numeric(
                df[ElectricityDataColumns.PRICE_USD_MWH.value], errors="raise"
            )
        except (ValueError, TypeError):
            logger.error("Price column contains non-numeric data")
            return False

        return True


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

    def __init__(self):
        super().__init__(base_url="http://oasis.caiso.com/oasisapi/SingleZip")

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

        # CAISO OASIS API parameters for LMP data
        params = {
            "queryname": "PRC_LMP",
            "version": "1",
            "market_run_id": market_type,
            "node": pricing_node,
            "startdatetime": f"{start_date}T00:00-0000",
            "enddatetime": f"{start_date}T23:59-0000",  # Same day, just different time
            "resultformat": "6",  # CSV format
        }

        try:
            # CAISO returns CSV data, not JSON
            response_text = await self.get_text("", params)
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
            # CAISO CSV typically starts with metadata, find the actual data
            lines = csv_text.strip().split("\n")

            # Find the header line (usually contains 'INTERVALSTARTTIME_GMT')
            header_idx = -1
            for i, line in enumerate(lines):
                if "INTERVALSTARTTIME_GMT" in line or "OPR_DT" in line:
                    header_idx = i
                    break

            if header_idx == -1:
                logger.warning("Could not find header in CAISO CSV response")
                return pd.DataFrame()

            # Read CSV from the header line
            from io import StringIO

            csv_data = "\n".join(lines[header_idx:])
            df = pd.read_csv(StringIO(csv_data))

            if df.empty:
                logger.warning("Empty data from CAISO")
                return pd.DataFrame()

            # Standardize columns based on CAISO format
            processed_df = pd.DataFrame()

            # Handle different possible timestamp column names
            timestamp_col = None
            for col in ["INTERVALSTARTTIME_GMT", "OPR_DT", "TIMESTAMP"]:
                if col in df.columns:
                    timestamp_col = col
                    break

            if timestamp_col:
                processed_df[ElectricityDataColumns.TIMESTAMP.value] = pd.to_datetime(
                    df[timestamp_col]
                )

            # Handle LMP price columns
            price_col = None
            for col in ["MW", "LMP_PRC", "PRC", "PRICE"]:
                if col in df.columns:
                    price_col = col
                    break

            if price_col:
                # CAISO prices are already in $/MWh
                processed_df[ElectricityDataColumns.PRICE_USD_MWH.value] = (
                    pd.to_numeric(df[price_col], errors="coerce")
                )

            # Remove any rows with NaN values
            processed_df = processed_df.dropna()

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

    def get_price_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Get electricity price data for the specified time range.

        This method implements the BasePriceProvider interface for CAISO data.

        Args:
            start_time: Start datetime for price data
            end_time: End datetime for price data

        Returns:
            DataFrame with columns: timestamp, price_usd_mwh
        """
        # Convert datetime to string format expected by the existing method
        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")

        # Use existing async method (we'll need to handle this properly)
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to handle
                # this differently. For now, we'll create a new thread.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._get_lmp_data_async(start_date, end_date)
                    )
                    return future.result()
            else:
                return asyncio.run(self._get_lmp_data_async(start_date, end_date))
        except Exception as e:
            logger.error(f"Error getting CAISO price data: {e}")
            return pd.DataFrame()

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

        has_price = ElectricityDataColumns.PRICE_USD_MWH.value in df.columns

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


# For backward compatibility with the old module
ElectricityPriceProvider = CAISOPriceProvider


class ElectricityPriceAnalyzer:
    """Utility class for analyzing electricity pricing data and patterns."""

    @staticmethod
    def calculate_energy_value(
        pv_generation_data: pd.DataFrame,
        price_data: pd.DataFrame,
        energy_column: str = "Total AC power (W)",
    ) -> pd.DataFrame:
        """
        Calculate the economic value of energy generation based on pricing data.

        Args:
            pv_generation_data: DataFrame with energy generation data
            price_data: DataFrame with electricity pricing data
            energy_column: Name of the energy column in generation data

        Returns:
            DataFrame with combined generation, pricing, and value data
        """
        try:
            # Align data by timestamp
            if "date_time" in pv_generation_data.columns:
                gen_data = pv_generation_data.copy()
                gen_data["timestamp"] = pd.to_datetime(gen_data["date_time"])
            else:
                gen_data = pv_generation_data.copy()
                if "timestamp" not in gen_data.columns:
                    raise ValueError(
                        "Generation data must have 'date_time' or 'timestamp' column"
                    )

            if "timestamp" not in price_data.columns:
                raise ValueError("Price data must have 'timestamp' column")

            # Merge data on timestamp
            combined_data = pd.merge(gen_data, price_data, on="timestamp", how="inner")

            if combined_data.empty:
                logger.warning(
                    "No overlapping timestamps between generation and price data"
                )
                return pd.DataFrame()

            # Convert power to energy (assuming hourly data)
            if energy_column in combined_data.columns:
                # Convert W to MWh (assuming hourly intervals)
                combined_data["energy_mwh"] = combined_data[energy_column] / 1_000_000
            else:
                raise ValueError(f"Energy column '{energy_column}' not found in data")

            # Calculate energy value using CAISO price data
            if "price_usd_mwh" in combined_data.columns:
                combined_data["energy_value_usd"] = (
                    combined_data["energy_mwh"] * combined_data["price_usd_mwh"]
                )
            else:
                raise ValueError("Price data missing required price column")

            return combined_data

        except Exception as e:
            logger.error(f"Error calculating energy value: {e}")
            return pd.DataFrame()

    @staticmethod
    def calculate_economics_summary(economic_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary economics metrics from energy value data.

        Args:
            economic_data: DataFrame with energy value calculations

        Returns:
            Dictionary with summary metrics
        """
        try:
            if economic_data.empty:
                return {}

            summary = {}

            # Energy metrics
            if "energy_mwh" in economic_data.columns:
                total_energy = economic_data["energy_mwh"].sum()
                summary["total_energy_mwh"] = total_energy

            # Value metrics
            if "energy_value_usd" in economic_data.columns:
                total_value = economic_data["energy_value_usd"].sum()
                summary["total_value_usd"] = total_value

                # Average value per hour
                if len(economic_data) > 0:
                    avg_value_per_hour = total_value / len(economic_data)
                    summary["average_value_per_hour_usd"] = avg_value_per_hour

            # Price metrics
            if "price_usd_mwh" in economic_data.columns:
                avg_price = economic_data["price_usd_mwh"].mean()
                summary["average_price_usd_mwh"] = avg_price

            return summary

        except Exception as e:
            logger.error(f"Error calculating economics summary: {e}")
            return {}

    @staticmethod
    def analyze_price_patterns(
        pricing_data: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> Dict[str, float]:
        """
        Analyze electricity price patterns and volatility.

        Args:
            pricing_data: DataFrame with pricing data
            timestamp_col: Name of timestamp column

        Returns:
            Dictionary with price pattern analysis
        """
        try:
            if pricing_data.empty:
                return {}

            analysis = {}
            price_col = "price_usd_mwh"

            if price_col not in pricing_data.columns:
                return analysis

            # Basic price statistics
            prices = pricing_data[price_col]
            analysis["average_price_usd_mwh"] = prices.mean()
            analysis["min_price_usd_mwh"] = prices.min()
            analysis["max_price_usd_mwh"] = prices.max()
            analysis["price_volatility_usd_mwh"] = prices.std()

            # Time-based patterns (if timestamp available)
            if timestamp_col in pricing_data.columns:
                data_with_time = pricing_data.copy()
                data_with_time[timestamp_col] = pd.to_datetime(
                    data_with_time[timestamp_col]
                )
                data_with_time["hour"] = data_with_time[timestamp_col].dt.hour

                # Find peak price hour
                hourly_avg = data_with_time.groupby("hour")[price_col].mean()
                peak_hour = hourly_avg.idxmax()
                analysis["peak_price_hour"] = peak_hour

                # Price range
                analysis["price_range_usd_mwh"] = prices.max() - prices.min()

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing price patterns: {e}")
            return {}


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
        price_usd_mwh: float,
        region: CAISORegion,
    ):
        self.timestamp = timestamp
        self.price_usd_mwh = price_usd_mwh
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
        if self.price_usd_mwh < 30:
            return "low"
        elif self.price_usd_mwh < 60:
            return "normal"
        elif self.price_usd_mwh < 100:
            return "high"
        else:
            return "extreme"

    def get_revenue_potential(self, energy_mwh: float) -> float:
        """Calculate potential revenue for selling given amount of energy."""
        return energy_mwh * self.price_usd_mwh

    def to_dict(self) -> Dict:
        """Convert state to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "price_usd_mwh": self.price_usd_mwh,
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
        price = float(row.get("price_usd_mwh", 0))

        return cls(
            timestamp=timestamp,
            price_usd_mwh=price,
            region=region,
        )


# Factory function for creating price providers
def create_price_provider(
    source_type: str,
    location: Optional[GeospatialLocation] = None,
    csv_file_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> BasePriceProvider:
    """
    Factory function to create price providers based on source type.

    Args:
        source_type: Type of price data source ("caiso" or "csv")
        location: Geographic location (required for CAISO)
        csv_file_path: Path to CSV file (required for CSV source)
        **kwargs: Additional arguments for the provider

    Returns:
        BasePriceProvider instance

    Raises:
        ValueError: If invalid source_type or missing required parameters
    """
    if source_type.lower() == "caiso":
        if location is None:
            raise ValueError("Location is required for CAISO price provider")
        return CAISOPriceProvider(location=location, **kwargs)
    elif source_type.lower() == "csv":
        if csv_file_path is None:
            raise ValueError("CSV file path is required for CSV price provider")
        return CSVPriceProvider(csv_file_path=csv_file_path)
    else:
        raise ValueError(
            f"Unknown source type: {source_type}. Supported: 'caiso', 'csv'"
        )
