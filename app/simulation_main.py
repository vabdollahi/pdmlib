"""
Comprehensive power management simulation example.

This example demonstrates the full simulation workflow:
1. Configuration-based environment creation
2. Heuristic agent integration
3. Observation handling and action execution
4. Multi-step simulation loop with rewards

Example usage:
    uv run python -m app.simulation_main
"""

import numpy as np

from app.core.actors import BasicHeuristic
from app.core.environment.config import EnvironmentConfigFactory
from app.core.environment.power_management_env import PowerManagementEnvironment
from app.core.utils.logging import get_logger

logger = get_logger("simulation_main")


def create_sample_configs():
    """Create sample configuration dictionaries for testing."""

    # Weather configuration (using CSV for deterministic testing)
    weather_config = {
        "type": "csv_file",
        "name": "weather_data",
        "data": "./tests/data/sample_weather_data.csv",
    }

    forecast_weather_config = {
        "type": "csv_file",
        "name": "weather_forecast",
        "data": "./tests/data/sample_weather_data.csv",  # Same for demo
    }

    # PV system configuration
    pv_config = {
        "capacity_mw": 5.0,
        "latitude": 37.7749,  # San Francisco
        "longitude": -122.4194,
        "tilt_angle": 30.0,
        "azimuth_angle": 180.0,
        "inverter_efficiency": 0.95,
    }

    # Battery configuration
    battery_config = {
        "capacity_mwh": 11.0,
        "max_power_mw": 5.5,
        "efficiency": 0.95,
        "min_soc": 0.1,
        "max_soc": 0.9,
        "initial_soc": 0.5,
    }

    # Virtual plant configurations
    virtual_plant_config_1 = {
        "type": "pv",
        "name": "plant1",
        "plant_model": {
            "weather_config": weather_config,
            "forecast_weather_config": forecast_weather_config,
            "pv_config": pv_config,
        },
        "battery": battery_config,
        "max_ac_power_to_grid_w": 4_881_942,  # ~4.88 MW
    }

    virtual_plant_config_2 = {
        "type": "pv",
        "name": "plant2",
        "plant_model": {
            "weather_config": weather_config,
            "forecast_weather_config": forecast_weather_config,
            "pv_config": pv_config,
        },
        "battery": battery_config,
        "max_ac_power_to_grid_w": 4_881_942,
    }

    # Market/price configuration
    spot_price_config = {
        "type": "csv_file",
        "name": "price",
        "data": "./tests/data/sample_price_data.csv",
    }

    lgc_price_config = {
        "type": "csv_file",
        "name": "lgc",
        "data": "./tests/data/sample_price_data.csv",  # Same for demo
    }

    forecast_price_config = {
        "type": "csv_file",
        "name": "forecast_price",
        "data": "./tests/data/sample_price_data.csv",  # Same for demo
    }

    market_config = {
        "prices": [spot_price_config, lgc_price_config],
        "forecast_prices": [forecast_price_config],
        "ppa": None,
        "fcas": None,
        "combine_prices": False,
    }

    # Portfolio configuration
    portfolio_config = {
        "name": "sample_portfolio",
        "plants": [virtual_plant_config_1, virtual_plant_config_2],
        "market": market_config,
        "combine_plants": True,
    }

    # Environment configuration
    environment_config = {
        "portfolios": [portfolio_config],
        "start_date_time": "2025-07-15 08:00:00+00:00",
        "end_date_time": "2025-07-15 18:00:00+00:00",
        "interval_min": 60,  # 60-minute intervals (1 hour)
        "historic_data_intervals": 4,  # 4 hours of history
        "forecast_data_intervals": 8,  # 8 hours of forecast
        "power_normalization_coefficient": 1e6,  # MW to W
        "price_normalization_coefficient": 100.0,  # $/MWh normalization
        "smoothed_reward_parameter": 0.1,
        "action_tolerance_percent": 0.01,
    }

    # Simulation configuration
    simulation_config = {
        "solver": "basic_heuristic",
        "max_lookahead_steps": 8,  # 8 hours at 1-hour intervals
    }

    return environment_config, simulation_config


def main():
    """Main simulation loop demonstrating the power management system."""

    print("Power Management Simulation")
    print("=" * 60)

    # Create configurations
    environment_config, simulation_config = create_sample_configs()

    print("Configuration Summary:")
    print(f"  Time range: {environment_config['start_date_time']} to")
    print(f"              {environment_config['end_date_time']}")
    print(f"  Interval: {environment_config['interval_min']} minutes")
    print(f"  Historic intervals: {environment_config['historic_data_intervals']}")
    print(f"  Forecast intervals: {environment_config['forecast_data_intervals']}")
    print(f"  Portfolios: {len(environment_config['portfolios'])}")

    # Create environment
    print("\nCreating environment...")
    try:
        env_config = EnvironmentConfigFactory.create(environment_config)
        env = PowerManagementEnvironment(config=env_config)
        print("Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    # Display environment information
    print("\nEnvironment Information:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Observation space shape: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action space shape: {env.action_space.shape}")

    # Sample random action for demonstration
    sample_action = env.action_space.sample()
    print(f"  Sample random action: {sample_action}")

    # Create solver/actor
    print(f"\nCreating {simulation_config['solver']} actor...")
    if simulation_config["solver"] == "basic_heuristic":
        actor = BasicHeuristic(
            max_lookahead_steps=simulation_config["max_lookahead_steps"],
            charge_threshold_ratio=0.3,
            discharge_threshold_ratio=0.7,
        )
        print("BasicHeuristic actor created")
    else:
        raise ValueError(f"Unknown solver: {simulation_config['solver']}")

    # Reset environment and get initial observation
    print("\nResetting environment...")
    observation, info = env.reset()
    print(f"  Initial observation shape: {observation.shape}")
    print(f"  Initial info: {info}")

    # Run simulation steps
    print("\nRunning simulation steps...")
    total_reward = 0.0
    step = 0

    for step in range(2):  # Run 2 steps as example
        print(f"\n  Step {step + 1}:")
        print(f"    Current timestamp: {env.timestamp}")

        # Show current observation values
        print(f"    Current observation shape: {observation.shape}")
        print(f"    Observation sample (first 10 values): {observation[:10]}")
        print(
            f"    Observation range: [{observation.min():.3f}, {observation.max():.3f}]"
        )

        try:
            # Get enhanced observation for actor (this is async)
            import asyncio

            enhanced_obs = asyncio.run(
                env.observation_factory.create_enhanced_observation(env.timestamp)
            )
            print(f"    Enhanced observation available: {len(enhanced_obs)} categories")

            # Print enhanced observation details
            for category, data in enhanced_obs.items():
                item_count = len(data) if hasattr(data, "__len__") else "N/A"
                print(f"      {category}: {type(data)} with {item_count} items")

                # Show market data if available
                if category == "market" and hasattr(data, "items"):
                    for key, value in data.items():
                        if "price" in key.lower():
                            print(f"        {key}: {type(value)}")

            # Get action from actor using enhanced observations
            actor_action = actor.get_action(enhanced_obs, env.timestamp)
            print(f"    Actor decision: {len(actor_action)} portfolios")

            # For demo, show some details about actor action
            for portfolio_name, portfolio_actions in actor_action.items():
                print(f"      {portfolio_name}:")
                for plant_name, plant_action in portfolio_actions.items():
                    if isinstance(plant_action, dict):
                        ac_target = plant_action.get("ac_power_generation_target_mw", 0)
                        battery_target = plant_action.get("battery_power_target_mw", 0)
                        print(
                            f"        {plant_name}: AC={ac_target:.2f}MW, "
                            f"Battery={battery_target:.2f}MW"
                        )
                    else:
                        print(f"        {plant_name}: {plant_action}")

        except Exception as e:
            print(f"    Error getting enhanced observation or actor action: {e}")
            enhanced_obs = {}
            actor_action = {}

        # Convert actor action to environment action if needed
        if isinstance(actor_action, dict) and actor_action:
            # For demo, convert to numpy array (this would need proper mapping)
            gym_action = np.array([0.5, 0.3])  # Placeholder conversion
        else:
            gym_action = env.action_space.sample()  # Use random action as fallback

        print(f"    Gym action: {gym_action}")
        # Action space bounds would be shown here if needed

        # Step environment
        try:
            observation, reward, terminated, truncated, info = env.step(gym_action)
            total_reward += reward

            print(f"    Reward: {reward:.4f}")
            print(f"    Terminated: {terminated}, Truncated: {truncated}")
            print(f"    Info keys: {list(info.keys()) if info else 'None'}")

            # Show portfolio info if available
            if info and "Test Multi-Plant Portfolio" in info:
                portfolio_info = info["Test Multi-Plant Portfolio"]
                print(f"    Portfolio info type: {type(portfolio_info)}")
                if isinstance(portfolio_info, dict):
                    for key, value in portfolio_info.items():
                        if isinstance(value, (int, float)):
                            print(f"      {key}: {value:.3f}")
                        else:
                            print(f"      {key}: {type(value)}")

            if terminated or truncated:
                print("    Episode finished")
                break

        except Exception as e:
            print(f"    Error during step: {e}")
            break

    print("\nSimulation Summary:")
    print(f"  Total steps: {step + 1}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Average reward per step: {total_reward / (step + 1):.4f}")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
