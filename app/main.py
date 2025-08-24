import asyncio
import os

from app.core.simulation.weather import WeatherProvider
from app.core.utils.location import GeospatialLocation
from app.core.utils.storage import DataStorage


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
    # The WeatherProvider now handles all the caching logic internally.
    weather_provider = WeatherProvider(
        location=location,
        start_date=START_DATE,
        end_date=END_DATE,
        organization=ORGANIZATION,
        asset=ASSET,
        storage=storage,  # Inject the storage dependency
    )

    # --- Get Data ---
    # The caller doesn't need to know if it's from cache or API.
    print(f"--- Getting weather data for {ASSET} ---")
    weather_data = await weather_provider.get_data()
    print("\n--- Data retrieved successfully. ---")
    print(weather_data.head())

    # --- Force a Refresh (Example) ---
    print(f"\n--- Forcing a refresh for {ASSET} to demonstrate... ---")
    refreshed_data = await weather_provider.get_data(force_refresh=True)
    print("\n--- Data refreshed successfully. ---")
    print(refreshed_data.head())

    print("\n--- Workflow demonstration complete. ---")


# To run:
# uv run python -m app.main
# STORAGE_PATH="gcs://my-bucket/data" uv run python -m app.main
if __name__ == "__main__":
    asyncio.run(main())
