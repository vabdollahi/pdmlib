"""
Solar Power Revenue Calculator Example

This demonstrates a solar producer selling power at real CAISO wholesale
electricity prices (Locational Marginal Price - LMP), including the
Duck Curve effect where solar can drive prices negative.
"""

import asyncio
import json
from pathlib import Path

from app.core.simulation.caiso_data import create_price_provider
from app.core.simulation.pv_model import PVModel
from app.core.simulation.pvlib_models import PVLibModel
from app.core.simulation.solar_revenue import SolarRevenueCalculator
from app.core.simulation.weather import WeatherProvider
from app.core.utils.date_handling import TimeInterval
from app.core.utils.location import GeospatialLocation
from app.core.utils.storage import DataStorage


async def create_solar_system():
    """Set up solar farm and market data providers."""

    # Solar farm location in California (CAISO market)
    location = GeospatialLocation(latitude=37.7749, longitude=-122.4194)

    # Use a known historical date for demonstration (data should be available)
    # Using a date from 2024 to ensure data exists in CAISO system
    start_date = "2024-07-15"  # Summer date with good solar conditions
    end_date = start_date

    print(f"Analysis Date: {start_date}")
    print("Location: San Francisco Bay Area")

    # Try CAISO LMP price provider first, with CSV fallback for testing
    try:
        print("Attempting to use CAISO API...")
        price_provider = create_price_provider(
            source_type="caiso",
            location=location,
            start_date=start_date,
            end_date=end_date,
            organization="SolarRevenue",
            asset="LMP_Data",
            data_type="caiso_lmp",
        )

        # Test if CAISO provider can actually get data
        from datetime import datetime

        test_start = datetime(2024, 7, 15)
        test_end = datetime(2024, 7, 16)
        test_data = price_provider.get_price_data(test_start, test_end)

        if test_data.empty:
            raise Exception("CAISO API returned no data")

        # Check if it's actually a CAISO provider to access region
        from app.core.simulation.caiso_data import CAISOPriceProvider

        if isinstance(price_provider, CAISOPriceProvider):
            print(f"CAISO Pricing Node: {price_provider.region}")
        print("✓ CAISO API connection successful")

    except Exception as e:
        print(f"CAISO API failed: {e}")
        print("Falling back to CSV test data...")

        # Fallback to CSV data for testing
        csv_path = (
            Path(__file__).parent.parent / "tests" / "data" / "sample_price_data.csv"
        )
        price_provider = create_price_provider(
            source_type="csv", csv_file_path=csv_path
        )
        print("✓ Using CSV test data")

    # Load solar farm configuration (10 MW system)
    config_path = (
        Path(__file__).parent.parent / "tests" / "config" / "10mw_solar_farm.json"
    )
    with open(config_path, "r") as f:
        pv_config = json.load(f)

    # Create weather data provider
    weather_provider = WeatherProvider(
        location=location,
        start_date=start_date,
        end_date=end_date,
        organization="SolarRevenue",
        asset="Weather",
        interval=TimeInterval.HOURLY,
        storage=DataStorage(base_path="data"),
    )

    # Create PV system model
    pvlib_model = PVLibModel(**pv_config)
    pv_model = PVModel(pv_config=pvlib_model, weather_provider=weather_provider)

    return price_provider, pv_model


async def main():
    """Calculate solar revenue using real CAISO LMP prices."""

    print("Solar Power Revenue Calculator")
    print("Real CAISO Wholesale Electricity Prices (LMP)")
    print("Demonstrates Duck Curve Effect")
    print("=" * 55)

    try:
        # Set up solar system and market data
        price_provider, pv_model = await create_solar_system()

        # Create revenue calculator
        calculator = SolarRevenueCalculator(
            price_provider=price_provider, pv_model=pv_model
        )

        print("\nCalculating Solar Revenue...")

        # Calculate revenue with date range
        from datetime import datetime

        start_time = datetime(2024, 7, 15)
        end_time = datetime(2024, 7, 16)
        result = await calculator.calculate_revenue(start_time, end_time)

        # Display results
        print("\n" + "=" * 55)
        print("SOLAR REVENUE ANALYSIS")
        print("=" * 55)

        print(f"Total Generation: {result.total_generation_mwh:.2f} MWh")
        print(f"Total Revenue: ${result.total_revenue_usd:.2f}")
        print(f"Average Revenue: ${result.avg_revenue_per_mwh:.2f}/MWh")

        print("\nLMP Price Analysis:")
        print(f"Average LMP: ${result.avg_lmp_usd_mwh:.2f}/MWh")
        print(f"Minimum LMP: ${result.min_lmp_usd_mwh:.2f}/MWh")
        print(f"Maximum LMP: ${result.max_lmp_usd_mwh:.2f}/MWh")

        # Duck Curve analysis
        print("\nDuck Curve Analysis:")
        print(f"Hours with negative LMP: {result.negative_price_hours}")

        if result.duck_curve_detected:
            print("Duck Curve Effect: DETECTED!")
            print("Solar production drove wholesale prices negative")
            print("Solar producer pays to deliver power during some hours")
        else:
            print("No negative pricing detected")

        # Show hourly details if available
        hourly_data = await calculator.get_hourly_analysis()
        if len(hourly_data) > 0:
            print("\nSample Hourly Data (sorted by LMP):")
            # Show range of prices
            sample = hourly_data.sort_values("lmp_usd_mwh").head(6)
            sample_cols = ["generation_mw", "lmp_usd_mwh", "revenue_usd"]
            sample_display = sample[sample_cols].copy()
            sample_display["hour"] = sample["hour"].dt.strftime("%H:%M")
            display_cols = ["hour", "generation_mw", "lmp_usd_mwh", "revenue_usd"]
            sample_display = sample_display[display_cols]
            print(sample_display.to_string(index=False))

        print("\nKey Insights:")
        print("   • LMP = Locational Marginal Price (wholesale electricity price)")
        print("   • Revenue = Generation (MW) × LMP ($/MWh) × Hours")
        print("   • Duck Curve: Midday solar can drive LMP negative")
        print("   • Solar producers are price-takers in wholesale market")

        print("\nAnalysis complete!")
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        print("\nCommon issues:")
        print("   • CAISO API rate limits (try again in a few minutes)")
        print("   • Recent data may not be available yet")
        print("   • Network connectivity issues")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
