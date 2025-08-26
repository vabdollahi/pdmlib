"""
Solar Power Revenue Calculator Example

This demonstrates a solar producer selling power at real wholesale
electricity prices, with automatic provider selection by market region
(CAISO for California, IESO for Ontario). It includes the Duck Curve
effect where solar can drive prices negative.
"""

import asyncio
import json
import os
from pathlib import Path

from app.core.simulation.price_provider import (
    PriceProviderConfig,
    create_price_provider,
)
from app.core.simulation.price_regions import PriceMarketRegion
from app.core.simulation.pv_model import PVModel
from app.core.simulation.pvlib_models import PVLibModel
from app.core.simulation.solar_revenue import SolarRevenueCalculator
from app.core.simulation.weather_provider import WeatherProvider
from app.core.utils.date_handling import TimeInterval
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger
from app.core.utils.storage import DataStorage

logger = get_logger("main")

## PriceProviderConfig now lives in app.core.simulation.price_provider


async def create_solar_system():
    """Set up solar farm and market data providers with auto region selection."""

    # Region override via env: MARKET_REGION=CAISO|IESO (default CAISO)
    region_name = os.getenv("MARKET_REGION", "CAISO").upper().strip()
    try:
        market_region = PriceMarketRegion(region_name)
    except Exception:
        market_region = PriceMarketRegion.CAISO

    # Location defaults by market
    if market_region == PriceMarketRegion.CAISO:
        # San Francisco Bay Area
        location = GeospatialLocation(latitude=37.7749, longitude=-122.4194)
        org_asset = ("SolarRevenue", "LMP_Data")
    else:
        # Toronto, Ontario
        location = GeospatialLocation(latitude=43.65107, longitude=-79.347015)
        org_asset = ("SolarRevenue", "HOEP_Data")

    # Centralized date configuration for consistent time ranges
    # Use a known historical date for demonstration (data should be available)
    # Using July 2025 - well into the historical archive
    from datetime import datetime

    # Define analysis period as datetime objects (single source of truth)
    # Use exact 24-hour period to avoid rounding issues
    analysis_start = datetime(2025, 7, 15, 0, 0, 0)  # Start at midnight
    analysis_end = datetime(2025, 7, 16, 0, 0, 0)  # End at midnight next day

    # Convert to string format for provider initialization
    start_date = analysis_start.strftime("%Y-%m-%d %H:%M:%S")
    end_date = analysis_end.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Analysis Date: {start_date}")
    logger.info(f"Market Region: {market_region}")

    # Auto-select price provider based on region; keep CSV fallback only for CAISO demo
    try:
        logger.info("Creating price provider (auto)...")
        cfg = PriceProviderConfig(
            market_region=market_region,
            location=location,
            start_date=start_date,
            end_date=end_date,
            organization=org_asset[0],
            asset=org_asset[1],
        )
        price_provider = cfg.create_provider()

        # Test if provider can actually get data using same datetime objects
        price_provider.set_range(analysis_start, analysis_end)
        test_data = await price_provider.get_data()

        if test_data.empty:
            raise Exception("Selected provider returned no data")

        # Provider-specific info
        from app.core.simulation.caiso_data import CAISOPriceProvider

        if isinstance(price_provider, CAISOPriceProvider):
            logger.info(f"CAISO Pricing Node: {price_provider.region}")
        logger.info("✓ Price provider ready")

    except Exception as e:
        logger.warning(f"{market_region} provider failed: {e}")
        logger.warning("Falling back to CSV test data...")
        csv_path = (
            Path(__file__).parent.parent / "tests" / "data" / "sample_price_data.csv"
        )
        price_provider = create_price_provider(
            source_type="csv", csv_file_path=csv_path
        )
        logger.info("Using CSV test data")

    # Load solar farm configuration (10 MW system)
    config_path = (
        Path(__file__).parent.parent
        / "tests"
        / "config"
        / "solar_farm_10mw_pv_only.json"
    )
    with open(config_path, "r") as f:
        pv_config = json.load(f)

    # Create weather data provider matching the selected location
    # Note: By default, fetches only GHI + temperature for API cost optimization
    # Set fetch_all_radiation=True for maximum accuracy (higher API costs)
    weather_provider = WeatherProvider(
        location=location,
        start_date=start_date,
        end_date=end_date,
        organization="SolarRevenue",
        asset="Weather",
        interval=TimeInterval.HOURLY,
        storage=DataStorage(base_path="data"),
        # fetch_all_radiation=False,  # Default: optimized (GHI + temp only)
        # fetch_all_radiation=True,   # Alternative: full radiation (higher cost)
    )

    # Create PV system model
    pvlib_model = PVLibModel(**pv_config)
    pv_model = PVModel(pv_config=pvlib_model, weather_provider=weather_provider)

    return price_provider, pv_model, analysis_start, analysis_end


async def main():
    """Calculate solar revenue using real CAISO LMP prices."""

    logger.info("Solar Power Revenue Calculator")
    logger.info("Real Wholesale Electricity Prices (LMP/HOEP)")
    logger.info("Demonstrates Duck Curve Effect")
    logger.info("=" * 55)

    try:
        # Set up solar system and market data
        price_provider, pv_model, start_time, end_time = await create_solar_system()

        # Create revenue calculator
        calculator = SolarRevenueCalculator(
            price_provider=price_provider, pv_model=pv_model
        )

        logger.info("Calculating Solar Revenue...")

        # Calculate revenue using the same date range as provider initialization
        result = await calculator.calculate_revenue(start_time, end_time)

        # Display results
        logger.info("\n" + "=" * 55)
        logger.info("SOLAR REVENUE ANALYSIS")
        logger.info("=" * 55)

        logger.info(f"Total Generation: {result.total_generation_mwh:.2f} MWh")
        logger.info(f"Total Revenue: ${result.total_revenue_usd:.2f}")
        logger.info(f"Average Revenue: ${result.avg_revenue_per_mwh:.2f}/MWh")

        logger.info("LMP Price Analysis:")
        logger.info(f"Average LMP: ${result.avg_lmp_usd_mwh:.2f}/MWh")
        logger.info(f"Minimum LMP: ${result.min_lmp_usd_mwh:.2f}/MWh")
        logger.info(f"Maximum LMP: ${result.max_lmp_usd_mwh:.2f}/MWh")

        # Duck Curve analysis
        logger.info("Duck Curve Analysis:")
        logger.info(f"Hours with negative LMP: {result.negative_price_hours}")

        if result.duck_curve_detected:
            logger.warning("Duck Curve Effect: DETECTED!")
            logger.warning("Solar production drove wholesale prices negative")
            logger.warning("Solar producer pays to deliver power during some hours")
        else:
            logger.info("No negative pricing detected")

        # Show hourly details if available
        hourly_data = await calculator.get_hourly_analysis(start_time, end_time)
        if len(hourly_data) > 0:
            logger.info("Sample Hourly Data (sorted by LMP):")
            # Show range of prices
            sample = hourly_data.sort_values("lmp_usd_mwh").head(6)
            sample_cols = ["generation_mw", "lmp_usd_mwh", "revenue_usd"]
            sample_display = sample[sample_cols].copy()
            sample_display["hour"] = sample["hour"].dt.strftime("%H:%M")
            display_cols = ["hour", "generation_mw", "lmp_usd_mwh", "revenue_usd"]
            sample_display = sample_display[display_cols]
            logger.info(sample_display.to_string(index=False))

        logger.info("Key Insights:")
        logger.info(
            "   • LMP = Locational Marginal Price (wholesale electricity price)"
        )
        logger.info("   • Revenue = Generation (MW) × LMP ($/MWh) × Hours")
        logger.info("   • Duck Curve: Midday solar can drive LMP negative")
        logger.info("   • Solar producers are price-takers in wholesale market")

        logger.info("Analysis complete!")
        return 0

    except Exception as e:
        logger.warning(f"Error: {e}")
        logger.info("Common issues:")
        logger.info("   • CAISO API rate limits (try again in a few minutes)")
        logger.info("   • Recent data may not be available yet")
        logger.info("   • Network connectivity issues")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
