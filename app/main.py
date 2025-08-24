import asyncio

from app.core.simulation.weather import WeatherProvider


async def main():
    """
    Main function to test the weather provider for both historical and forecast data.
    """
    # --- Test Case 1: Historical Data ---
    print("--- Fetching historical weather data... ---")
    historical_provider = WeatherProvider(
        latitude=52.52,
        longitude=13.41,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    historical_data = await historical_provider.get_weather_data()
    print("Historical Data:")
    print(historical_data.head())
    print("-" * 40)

    # --- Test Case 2: Forecast Data ---
    print("--- Fetching forecast weather data... ---")
    forecast_provider = WeatherProvider(
        latitude=52.52,
        longitude=13.41,
        start_date="2025-08-25",  # A date in the future
        end_date="2025-08-26",
    )
    forecast_data = await forecast_provider.get_weather_data()
    print("Forecast Data:")
    print(forecast_data.head())
    print("-" * 40)


# To run:
# uv run python -m app.main
if __name__ == "__main__":
    asyncio.run(main())
