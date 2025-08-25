"""
Ontario (IESO) price provider: Hourly Ontario Energy Price (HOEP)
historical and forecast.

This provider fetches:
- Historical day-ahead HOEP XML from IESO public reports
- Pre-dispatch (forecast) HOEP XML (future delivery dates)

Output is standardized to columns: timestamp (index), price_usd_mwh.
Note: HOEP is published in $/MWh CAD; no FX conversion is applied here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

import pandas as pd

from app.core.simulation.price_provider import BasePriceProvider, ElectricityDataColumns
from app.core.utils.api_clients import CAISORateLimitedClient
from app.core.utils.caching import BaseProvider
from app.core.utils.http_cache import SimpleHTTPCache
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger

logger = get_logger("ieso_price")


# IESO public XML data URLs
DAY_AHEAD_HOEP_BASE_URL = (
    "https://reports-public.ieso.ca/public/DAHourlyOntarioZonalPrice"
)
PREDISPATCH_HOEP_BASE_URL = (
    "https://reports-public.ieso.ca/public/PredispHourlyOntarioZonalPrice"
)


class IESORateLimitedClient(CAISORateLimitedClient):
    """Rate-limited HTTP client for IESO XML data with caching."""

    def __init__(self, cache_ttl_seconds: int = 600, cache_namespace: str = "ieso"):
        # Base URL unused for full URLs but set to empty-safe base
        super().__init__(base_url="https://reports-public.ieso.ca")
        self.http_cache = SimpleHTTPCache(
            ttl_seconds=cache_ttl_seconds, namespace=cache_namespace
        )

    async def get_xml(self, url: str) -> Optional[str]:
        """
        Fetch XML content from a URL with caching and rate limiting.

        Args:
            url: URL to fetch XML from

        Returns:
            XML content as string, or None if failed
        """
        try:
            # Try cache first
            cache_key = SimpleHTTPCache.canonical_key(url, None)
            blob = self.http_cache.get(cache_key)
            if blob is not None:
                logger.debug(f"Cache hit for {url}")
                return blob.decode("utf-8", errors="replace")

            # Make HTTP request with rate limiting
            text = await self.get_text(url)
            if text:
                # Cache the response
                blob = text.encode("utf-8", errors="replace")
                self.http_cache.set(cache_key, blob)
                return text

            return None
        except Exception as e:
            logger.error(f"Failed to fetch XML from {url}: {e}")
            return None

    def parse_ieso_xml(self, xml_content: str) -> Optional[pd.DataFrame]:
        """
        Parse IESO XML content to DataFrame with columns:
        DeliveryDate, Hour, ZonalPrice.

        Args:
            xml_content: XML content as string

        Returns:
            DataFrame with hourly price data, or None if parsing failed
        """
        try:
            root = ET.fromstring(xml_content)

            # Define namespace for easier access
            ns = {"ieso": "http://www.ieso.ca/schema"}

            # Extract delivery date
            delivery_date_elem = root.find(".//ieso:DeliveryDate", ns)
            if delivery_date_elem is None:
                logger.warning("No DeliveryDate found in XML")
                return None

            delivery_date = delivery_date_elem.text

            # Extract hourly price components
            rows = []
            for component in root.findall(".//ieso:HourlyPriceComponents", ns):
                hour_elem = component.find("ieso:PricingHour", ns)
                price_elem = component.find("ieso:ZonalPrice", ns)

                if hour_elem is not None and price_elem is not None:
                    try:
                        if hour_elem.text is None or price_elem.text is None:
                            continue
                        hour = int(hour_elem.text)
                        price = float(price_elem.text)
                        rows.append(
                            {
                                "DeliveryDate": delivery_date,
                                "Hour": hour,
                                "ZonalPrice": price,
                            }
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse hour/price: {e}")
                        continue

            if not rows:
                logger.warning("No valid price data found in XML")
                return None

            return pd.DataFrame(rows)

        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing XML: {e}")
            return None


class IESOPriceProvider(BaseProvider, BasePriceProvider):
    """IESO HOEP provider with historical + forecast support."""

    @property
    def source_price_column_name(self) -> str:
        """Return the column name used by IESO XML for price data."""
        return "hoep_cad_mwh"  # IESO uses CAD currency notation

    def __init__(self, location: Optional[GeospatialLocation] = None, **kwargs):
        # Ottawa default if not provided
        super().__init__(
            location=location or GeospatialLocation(latitude=45.4, longitude=-75.7),
            **kwargs,
        )

    def get_price_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        # Bridge to BaseProvider cache path
        self.start_date = start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.end_date = end_time.strftime("%Y-%m-%d %H:%M:%S")

        import asyncio

        async def _run():
            try:
                return await self.get_data()
            except Exception as e:
                logger.error(f"IESO provider error: {e}")
                return pd.DataFrame()

        try:
            try:
                asyncio.get_running_loop()
                # If we're already in an async context, run in a worker thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Create and run coroutine within the worker thread to avoid
                    # un-awaited coroutine warnings.
                    future = executor.submit(lambda: asyncio.run(_run()))
                    return future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                return asyncio.run(_run())
        except Exception as e:
            logger.error(f"Error getting IESO price data: {e}")
            return pd.DataFrame()

    def validate_data_format(self, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        has_price = ElectricityDataColumns.PRICE_DOLLAR_MWH.value in df.columns
        return has_price

    async def _fetch_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch HOEP data by generating date range and fetching individual XML files.
        """
        try:
            client = IESORateLimitedClient()

            # Generate date range for fetching
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")

            frames = []

            # Fetch day-ahead historical data
            for date in date_range:
                date_str = date.strftime("%Y%m%d")
                da_url = (
                    f"{DAY_AHEAD_HOEP_BASE_URL}/"
                    f"PUB_DAHourlyOntarioZonalPrice_{date_str}.xml"
                )

                xml_content = await client.get_xml(da_url)
                if xml_content:
                    df = client.parse_ieso_xml(xml_content)
                    if df is not None and not df.empty:
                        frames.append(df)

            # Fetch predispatch data (current and future dates)
            current_date = pd.Timestamp.now().normalize()
            for date in date_range:
                # Only fetch predispatch for current/future dates
                if date >= current_date:
                    date_str = date.strftime("%Y%m%d")
                    pred_url = (
                        f"{PREDISPATCH_HOEP_BASE_URL}/"
                        f"PUB_PredispHourlyOntarioZonalPrice_{date_str}.xml"
                    )

                    xml_content = await client.get_xml(pred_url)
                    if xml_content:
                        df = client.parse_ieso_xml(xml_content)
                        if df is not None and not df.empty:
                            frames.append(df)

            # Also try current files without date suffix
            for base_url in [DAY_AHEAD_HOEP_BASE_URL, PREDISPATCH_HOEP_BASE_URL]:
                if "DAHourly" in base_url:
                    filename = "PUB_DAHourlyOntarioZonalPrice.xml"
                else:
                    filename = "PUB_PredispHourlyOntarioZonalPrice.xml"
                current_url = f"{base_url}/{filename}"

                xml_content = await client.get_xml(current_url)
                if xml_content:
                    df = client.parse_ieso_xml(xml_content)
                    if df is not None and not df.empty:
                        frames.append(df)

            if not frames:
                logger.warning("No IESO XML data found for date range")
                return pd.DataFrame()

            # Combine all frames
            df_all = pd.concat(frames, ignore_index=True)

            # Convert to standard format with timestamp
            df_converted = []
            for _, row in df_all.iterrows():
                try:
                    # Parse delivery date and hour
                    delivery_date = pd.to_datetime(row["DeliveryDate"])
                    hour = int(row["Hour"])
                    # IESO uses hours 1-24, convert to 0-23 for timestamp
                    hour_offset = hour - 1
                    timestamp = delivery_date + pd.Timedelta(hours=hour_offset)

                    df_converted.append(
                        {
                            ElectricityDataColumns.TIMESTAMP.value: timestamp,
                            "hoep_cad_mwh": float(row["ZonalPrice"]),
                        }
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert row: {e}")
                    continue

            if not df_converted:
                return pd.DataFrame()

            result_df = pd.DataFrame(df_converted)

            # Remove duplicates (same timestamp)
            timestamp_col = ElectricityDataColumns.TIMESTAMP.value
            result_df = result_df.drop_duplicates(subset=[timestamp_col])
            result_df = result_df.sort_values(timestamp_col)

            # Filter to requested range (inclusive)
            start = pd.to_datetime(start_date)
            # End of day
            end = (
                pd.to_datetime(end_date)
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
            result_df = result_df[
                (result_df[timestamp_col] >= start) & (result_df[timestamp_col] <= end)
            ]

            # Index by timestamp for storage
            if not result_df.empty:
                result_df.set_index(timestamp_col, inplace=True)

            # Standardize column names to unified format
            result_df = self._standardize_columns(result_df)

            logger.info(
                "Fetched %s IESO HOEP points for range %s to %s",
                len(result_df),
                start_date,
                end_date,
            )
            return result_df

        except Exception as e:
            logger.error(f"Failed to fetch IESO data: {e}")
            return pd.DataFrame()
