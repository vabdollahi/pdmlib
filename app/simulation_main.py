"""
Power Management Simulation with Complete Automation.

This simulation_main.py demonstrates complete automation from JSON configuration
to simulation execution with pre-calculated power profiles, unified market data,
and agent-driven portfolio management.

Usage:
    uv run python -m app.simulation_main
    uv run python -m app.simulation_main tests/config/test_config_simple.json
"""

import asyncio
import sys
from pathlib import Path

from app.core.simulation.simulation_manager import create_simulation_from_json
from app.core.utils.logging import get_logger

logger = get_logger("simulation_main")


async def main(config_file_path: str | None = None):
    """
    Main function demonstrating complete automated simulation workflow.

    This achieves the ultimate goal: load JSON configuration, automatically create
    simulation object with validation, pre-calculate power profiles, create market
    object, and run agent-driven portfolio management simulation.
    """

    print("Power Management Simulation")
    print("=" * 50)
    print("Complete Automation: JSON → Simulation → Results")
    print("=" * 50)  # Determine configuration file
    if config_file_path:
        config_path = Path(config_file_path)
        print(f"Configuration file: {config_path}")
    else:
        # Use default test configuration with agent
        config_path = (
            Path(__file__).parent.parent
            / "tests"
            / "config"
            / "test_config_simple.json"
        )
        print(f"Using default configuration: {config_path}")

    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return

    try:
        # Step 1: Create and initialize complete simulation from JSON
        print("\nInitializing automated simulation...")
        simulation = await create_simulation_from_json(config_path)
        print("Simulation initialized successfully!")

        # Show initialization summary
        print("\nInitialization Summary:")
        print(f"   - Environment: {type(simulation.environment).__name__}")
        print(f"   - Agent: {type(simulation.agent).__name__}")

        if simulation.market and simulation.market.price_providers:
            print(
                f"   - Market data providers: {len(simulation.market.price_providers)}"
            )
        else:
            print("   - Market data providers: 0")

        if simulation.environment_config and simulation.environment_config.portfolios:
            print(f"   - Portfolios: {len(simulation.environment_config.portfolios)}")

            total_plants = sum(
                len(p.plants) for p in simulation.environment_config.portfolios
            )
            print(f"   - Total plants: {total_plants}")
        else:
            print("   - Portfolios: 0")
            print("   - Total plants: 0")

        # Step 2: Run complete automated simulation
        print("\nRunning automated simulation...")
        print("   - Pre-calculated power profiles: Ready")
        print("   - Market data pre-loaded: Ready")
        print("   - Agent ready for decision making: Ready")

        # Run simulation for the full duration of the available data
        results = await simulation.run_simulation()

        # Step 3: Display results
        print("\nSimulation Results:")
        print(f"   - Total steps executed: {results.total_steps}")
        print(f"   - Total reward: {results.total_reward:.4f}")
        print(f"   - Average reward per step: {results.average_reward_per_step:.4f}")
        print(f"   - Simulation period: {results.start_time} to {results.end_time}")

        # Step 4: Save results
        print("\nSaving results...")
        output_file = results.save_to_file()
        print(f"   - Results saved to: {output_file}")

        print("\nComplete automation workflow finished successfully!")
        print("\nAchieved Ultimate Goal:")
        print("- JSON configuration → Automatic simulation object creation")
        print("- Pydantic validation throughout pipeline")
        print("- Automatic provider creation (weather + price)")
        print("- Data loading and validation")
        print("- Pre-calculated power generation profiles")
        print("- Unified market object with price data")
        print("- Agent-driven portfolio management")
        print("- Complete results output")

    except Exception as e:
        print(f"\nSimulation failed: {e}")
        logger.error(f"Simulation error: {e}", exc_info=True)
        return


def run_sync():
    """Synchronous wrapper for the async main function."""
    config_file = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        asyncio.run(main(config_file))
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")


if __name__ == "__main__":
    run_sync()
