"""
Solar power revenue calculation using real electricity market prices.

This module provides simple revenue calculation for solar producers selling
power at real-time wholesale electricity prices (LMP) from CAISO.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core.simulation.price_provider import (
    BasePriceProvider,
    PriceColumns,
)
from app.core.simulation.pv_model import PVModel
from app.core.utils.logging import get_logger

logger = get_logger("solar_revenue")


class SolarRevenueResult(BaseModel):
    """Results from solar revenue calculation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_generation_mwh: float = Field(description="Total solar generation in MWh")
    total_revenue_dollar: float = Field(description="Total revenue in dollars")
    avg_lmp_dollar_mwh: float = Field(description="Average LMP price")
    min_lmp_dollar_mwh: float = Field(description="Minimum LMP price")
    max_lmp_dollar_mwh: float = Field(description="Maximum LMP price")
    negative_price_hours: int = Field(description="Hours with negative LMP")
    avg_revenue_per_mwh: float = Field(description="Average revenue per MWh")
    duck_curve_detected: bool = Field(
        description="Whether Duck Curve effect was detected"
    )
    hourly_data: Optional[pd.DataFrame] = Field(default=None, exclude=True)


class SolarRevenueCalculator:
    """
    Calculator for solar power revenue using real CAISO LMP prices.

    This class demonstrates the core economics of solar power:
    - Solar producers are paid the real-time Locational Marginal Price (LMP)
    - LMP can go negative during high solar production (Duck Curve effect)
    - Revenue = Generation (MW) × LMP ($/MWh) × Hours
    """

    def __init__(
        self,
        price_provider: BasePriceProvider,
        pv_model: PVModel,
    ):
        """
        Initialize the solar revenue calculator.

        Args:
            price_provider: CAISO price data provider
            pv_model: Solar PV generation model
        """
        self.price_provider = price_provider
        self.pv_model = pv_model
        logger.info("Initialized solar revenue calculator")

    async def calculate_revenue(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> SolarRevenueResult:
        """
        Calculate solar revenue from PV generation and electricity prices.

        Args:
            start_time: Optional start time for price data
            end_time: Optional end time for price data

        Returns:
            SolarRevenueResult with revenue calculations
        """
        logger.info("Starting solar revenue calculation")

        # Get LMP price data
        logger.info("Fetching LMP data")
        if start_time and end_time:
            self.price_provider.set_range(start_time, end_time)
            price_data = await self.price_provider.get_data()
        else:
            # For backward compatibility, try to get data from the provider's cache
            # This assumes the provider has been configured with dates already
            try:
                price_data = await self.price_provider.get_data()
            except Exception as e:
                logger.warning(f"Could not get price data: {e}")
                price_data = pd.DataFrame()

        if len(price_data) == 0:
            raise ValueError("No LMP price data available for revenue calculation")

        logger.info(f"Retrieved {len(price_data)} LMP data points")

        # Get solar generation data
        logger.info("Calculating solar generation")
        generation_data = await self.pv_model.run_simulation()

        if len(generation_data) == 0:
            raise ValueError(
                "No solar generation data available for revenue calculation"
            )

        logger.info(f"Generated {len(generation_data)} generation data points")

        # Align data by hour
        combined_data = self._align_data(price_data, generation_data)

        if len(combined_data) == 0:
            raise ValueError(
                "No overlapping data between prices and generation - "
                "check data alignment and time ranges"
            )

        logger.info(f"Aligned {len(combined_data)} data points for revenue calculation")

        # Calculate revenue
        return self._calculate_revenue_metrics(combined_data)

    async def calculate_instantaneous_revenue(
        self,
        power_mw: float,
        timestamp: datetime,
        interval_min: float = 60.0,
    ) -> float:
        """
        Calculate revenue for a specific power output at a given timestamp.

        This method is designed for real-time reward calculation in the power
        management environment.

        Args:
            power_mw: Power output in MW (can be negative for consumption)
            timestamp: Timestamp for price lookup
            interval_min: Time interval in minutes (default: 60 minutes)

        Returns:
            Revenue in dollars for the given power and time interval

        Raises:
            ValueError: If no price data is available
        """
        # Get price data
        price_data = await self.price_provider.get_data()

        if len(price_data) == 0:
            raise ValueError("No price data available for revenue calculation")

        # Find the price at the given timestamp
        current_price = await self._get_price_at_timestamp(price_data, timestamp)

        # Calculate revenue: Power (MW) × Price ($/MWh) × Time (hours)
        time_hours = interval_min / 60.0
        revenue = power_mw * current_price * time_hours

        logger.debug(
            f"Instantaneous revenue: {power_mw:.2f}MW × "
            f"${current_price:.2f}/MWh × {time_hours:.2f}h = ${revenue:.4f}"
        )

        return revenue

    async def _get_price_at_timestamp(
        self, price_data: pd.DataFrame, timestamp: datetime
    ) -> float:
        """
        Get electricity price at a specific timestamp.

        Raises:
            ValueError: If price data cannot be found for the timestamp
        """
        price_col = PriceColumns.PRICE_DOLLAR_MWH.value

        # Reset index to work with timestamp column
        price_data_reset = price_data.reset_index()

        if "timestamp" not in price_data_reset.columns:
            raise ValueError("No timestamp column found in price data")

        if price_col not in price_data_reset.columns:
            raise ValueError(f"Price column '{price_col}' not found in price data")

        # Find the closest timestamp
        time_diffs = abs(price_data_reset["timestamp"] - timestamp)
        closest_idx = time_diffs.idxmin()

        current_price = float(price_data_reset.at[closest_idx, price_col])

        logger.debug(f"Found price ${current_price:.2f}/MWh at timestamp {timestamp}")

        return current_price

    def _align_data(
        self, price_data: pd.DataFrame, generation_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Align price and generation data by hour."""

        # Prepare price data
        price_df = price_data.copy()

        # Ensure index is datetime
        if not isinstance(price_df.index, pd.DatetimeIndex):
            if "timestamp" in price_df.columns:
                price_df.set_index("timestamp", inplace=True)
            else:
                price_df.index = pd.to_datetime(price_df.index)

        price_df["hour"] = price_df.index.to_series().dt.floor("h")

        # Prepare generation data
        generation_df = generation_data.copy()

        # Ensure index is datetime
        if not isinstance(generation_df.index, pd.DatetimeIndex):
            if "date_time" in generation_df.columns:
                generation_df.set_index("date_time", inplace=True)
            else:
                generation_df.index = pd.to_datetime(generation_df.index)

        generation_df["hour"] = generation_df.index.to_series().dt.floor("h")

        # Find power output column
        power_col = None
        for col in generation_df.columns:
            if "power" in col.lower() or "output" in col.lower():
                power_col = col
                break

        if power_col is None:
            logger.error("No power output column found in generation data")
            logger.error(f"Available columns: {list(generation_df.columns)}")
            return pd.DataFrame()

        # Merge data
        combined = price_df.reset_index().merge(
            generation_df.reset_index(), on="hour", how="inner"
        )

        # Convert power to MW if needed (assuming input might be in W)
        combined["generation_mw"] = combined[power_col] / 1000.0
        price_col = PriceColumns.PRICE_DOLLAR_MWH.value
        combined["lmp_dollar_mwh"] = combined[price_col]

        # Calculate hourly revenue: LMP ($/MWh) × Generation (MW) × 1 hour
        combined["revenue_dollar"] = (
            combined["generation_mw"] * combined["lmp_dollar_mwh"]
        )

        return combined

    def _calculate_revenue_metrics(self, data: pd.DataFrame) -> SolarRevenueResult:
        """Calculate revenue metrics from aligned data."""

        # Basic metrics
        total_generation_mwh = data["generation_mw"].sum()
        total_revenue_dollar = data["revenue_dollar"].sum()
        avg_lmp_dollar_mwh = data["lmp_dollar_mwh"].mean()
        min_lmp_dollar_mwh = data["lmp_dollar_mwh"].min()
        max_lmp_dollar_mwh = data["lmp_dollar_mwh"].max()

        # Duck Curve analysis
        negative_price_hours = len(data[data["lmp_dollar_mwh"] < 0])
        duck_curve_detected = min_lmp_dollar_mwh < 0

        # Average revenue per MWh
        avg_revenue_per_mwh = 0.0
        if total_generation_mwh > 0:
            avg_revenue_per_mwh = total_revenue_dollar / total_generation_mwh

        logger.info(
            f"Revenue calculation complete: "
            f"${total_revenue_dollar:.2f} from {total_generation_mwh:.2f} MWh"
        )
        if duck_curve_detected:
            logger.info(
                f"Duck Curve detected: {negative_price_hours} hours with negative LMP"
            )

        return SolarRevenueResult(
            total_generation_mwh=total_generation_mwh,
            total_revenue_dollar=total_revenue_dollar,
            avg_lmp_dollar_mwh=avg_lmp_dollar_mwh,
            min_lmp_dollar_mwh=min_lmp_dollar_mwh,
            max_lmp_dollar_mwh=max_lmp_dollar_mwh,
            negative_price_hours=negative_price_hours,
            avg_revenue_per_mwh=avg_revenue_per_mwh,
            duck_curve_detected=duck_curve_detected,
            hourly_data=data,
        )

    async def get_hourly_analysis(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get detailed hourly analysis of solar generation vs LMP prices.

        Args:
            start_time: Optional start time for analysis
            end_time: Optional end time for analysis

        Returns:
            DataFrame with hourly generation, LMP, and revenue data
        """
        result = await self.calculate_revenue(start_time, end_time)

        if result.hourly_data is not None:
            # Return key columns for analysis
            return result.hourly_data[
                ["hour", "generation_mw", "lmp_dollar_mwh", "revenue_dollar"]
            ].sort_values("hour")

        return pd.DataFrame()
