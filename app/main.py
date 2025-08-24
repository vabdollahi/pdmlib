import asyncio
import os

from app.core.data.storage import DataStorage
from app.core.simulation.weather import WeatherProvider


async def main():
    """
    Main function to demonstrate the data storage and retrieval workflow.
    """
    # --- Configuration ---
    LATITUDE = 52.52
    LONGITUDE = 13.41
    START_DATE = "2024-01-01"
    END_DATE = "2024-03-31"  # Fetch a few months of data
    ORGANIZATION = "SolarCorp"
    ASSET = "Berlin-PV-Plant-1"

    # --- Storage Configuration ---
    # To use GCS: 'gcs://your-bucket-name/data'
    # To use local: 'data'
    storage_path = os.getenv("STORAGE_PATH", "data")
    storage = DataStorage(base_path=storage_path)

    # --- Workflow ---
    print(f"--- Checking for cached data for {ASSET} in {storage_path} ---")
    # 1. Try to read data from the Parquet store first
    cached_data = storage.read_data_for_range(
        organization=ORGANIZATION,
        asset=ASSET,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    if not cached_data.empty:
        print("--- Data found in cache. ---")
        print(cached_data.head())
    else:
        print("--- No data in cache. Fetching from API... ---")
        # 2. If not found, fetch it using the WeatherProvider
        weather_provider = WeatherProvider(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            start_date=START_DATE,
            end_date=END_DATE,
        )
        weather_data = await weather_provider.get_weather_data()

        print("--- Data fetched successfully. ---")
        print(weather_data.head())

        # 3. Save the newly fetched data to the Parquet store
        print(f"\n--- Saving data to Parquet store in {storage_path}... ---")
        storage.write_data(
            df=weather_data,
            organization=ORGANIZATION,
            asset=ASSET,
            latitude=LATITUDE,
            longitude=LONGITUDE,
        )

    print("\n--- Workflow demonstration complete. ---")
    print("Run this script again to see the caching in action.")
    print("To test with GCS, set the STORAGE_PATH environment variable.")


# To run:
# uv run python -m app.main
# STORAGE_PATH="gcs://my-bucket/data" uv run python -m app.main
if __name__ == "__main__":
    asyncio.run(main())
