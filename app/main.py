from app.core.simulation.weather import WeatherProvider

# To run:
# uv run python -m app.main
if __name__ == "__main__":
    # test weather functionality
    weather_provider = WeatherProvider(
        latitude=52.52,
        longitude=13.41,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    weather_data = weather_provider.get_weather_data()
    print(weather_data.head(20))
