import asyncio
import json
from pathlib import Path

from app.core.simulation.pv_model import PVModel
from app.core.simulation.pvlib_models import PVLibModel
from app.core.simulation.weather import WeatherProvider
from app.core.utils.date_handling import TimeInterval
from app.core.utils.location import GeospatialLocation
from app.core.utils.storage import DataStorage


async def create_pv_model_example():
    """Create a PV model using JSON configuration."""

    # Load the 10MW solar farm configuration from JSON
    config_path = (
        Path(__file__).parent.parent / "tests" / "config" / "10mw_solar_farm.json"
    )
    with open(config_path, "r") as f:
        pv_config_data = json.load(f)

    # Create location objects
    location_config = pv_config_data["location"]
    location_geo = GeospatialLocation(
        latitude=location_config["latitude"], longitude=location_config["longitude"]
    )

    # Weather configuration (full year for comprehensive analysis)
    weather_provider = WeatherProvider(
        location=location_geo,
        start_date="2025-06-15",  # Summer period for peak performance
        end_date="2025-06-17",
        organization="UtilitySolar",
        asset=location_config["name"],
        interval=TimeInterval.HOURLY,
        storage=DataStorage(base_path="data"),
    )

    # Create PV model in one shot from JSON using Pydantic's built-in parsing
    pvlib_model = PVLibModel(**pv_config_data)

    # Create main PV Model
    pv_model = PVModel(pv_config=pvlib_model, weather_provider=weather_provider)

    return pv_model


async def main():
    """Main function to create and run the PV model."""

    try:
        # Create the PV model
        pv_model = await create_pv_model_example()

        # Run the simulation
        print("\n Running PV simulation...")
        results = await pv_model.run_simulation()

        # Display results
        print("\n Simulation Results:")
        print(f"   • Data points: {len(results)}")
        start_time = results.iloc[0]["date_time"]
        end_time = results.iloc[-1]["date_time"]
        print(f"   • Time range: {start_time} to {end_time}")

        if "Total AC power (W)" in results.columns:
            max_power = results["Total AC power (W)"].max()
            avg_power = results["Total AC power (W)"].mean()
            total_energy = results["Total AC power (W)"].sum() / 1000

            print(f"   • Peak AC Power: {max_power:,.1f} W")
            print(f"   • Average AC Power: {avg_power:,.1f} W")
            print(f"   • Total Energy: {total_energy:.2f} kWh")

        # Show sample results
        print("\n Sample Results:")
        print(results.head(3).to_string(index=False))

        print("\n PV Model execution completed successfully!")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
