import asyncio
import os

from app.core.simulation.weather import WeatherProvider
from app.core.utils.date_handling import TimeInterval
from app.core.utils.location import GeospatialLocation
from app.core.utils.logging import get_logger, setup_logging
from app.core.utils.storage import DataStorage

# Setup logging
setup_logging()
logger = get_logger("main")


async def main():
    """
    Main function to demonstrate the simplified data retrieval workflow.
    """
    # --- Configuration ---
    location = GeospatialLocation(latitude=52.52, longitude=13.41)
    START_DATE = "2024-01-01"
    END_DATE = "2024-03-31"
    ORGANIZATION = "SolarCorp"
    ASSET = "Berlin-PV-Plant-1"

    # --- Storage Configuration ---
    storage_path = os.getenv("STORAGE_PATH", "data")
    storage = DataStorage(base_path=storage_path)

    # --- Create the Provider ---
    # The WeatherProvider now handles all the caching logic internally
    # and supports configurable time intervals.
    weather_provider = WeatherProvider(
        location=location,
        start_date=START_DATE,
        end_date=END_DATE,
        organization=ORGANIZATION,
        asset=ASSET,
        interval=TimeInterval.HOURLY,
        storage=storage,  # Inject the storage dependency
    )

    # --- Get Data ---
    # The caller doesn't need to know if it's from cache or API.
    print(f"--- Getting weather data for {ASSET} ---")
    weather_data = await weather_provider.get_data()
    print("\n--- Data retrieved successfully. ---")
    print(weather_data.head())

    # --- Demonstrate Different Time Intervals ---
    print(f"\n--- Demonstrating different time intervals for {ASSET} ---")

    # Example with 15-minute intervals for detailed intraday analysis
    weather_provider_15min = WeatherProvider(
        location=location,
        start_date="2024-01-01",
        end_date="2024-01-01",  # Just one day for 15min demo
        organization=ORGANIZATION,
        asset=f"{ASSET}-15min",
        interval=TimeInterval.FIFTEEN_MINUTES,  # Higher resolution
        storage=storage,
    )

    print("Fetching 15-minute interval data for detailed analysis...")
    detailed_data = await weather_provider_15min.get_data()
    print(f"15-minute data points: {len(detailed_data)}")
    print(detailed_data.head(3))  # Show first 3 rows

    # Example with daily intervals for long-term trends
    weather_provider_daily = WeatherProvider(
        location=location,
        start_date=START_DATE,
        end_date=END_DATE,
        organization=ORGANIZATION,
        asset=f"{ASSET}-daily",
        interval=TimeInterval.DAILY,  # Lower resolution, good for trends
        storage=storage,
    )

    print("\nFetching daily interval data for trend analysis...")
    trend_data = await weather_provider_daily.get_data()
    print(f"Daily data points: {len(trend_data)}")
    print(trend_data.head(3))  # Show first 3 rows

    print("\n--- Workflow demonstration complete. ---")


# To run:
# uv run python -m app.main
# STORAGE_PATH="gcs://my-bucket/data" uv run python -m app.main
if __name__ == "__main__":
    asyncio.run(main())
