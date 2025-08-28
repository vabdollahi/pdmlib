"""
Unified Simulation Manager for Power Management System.

This module provides the main orchestration layer that achieves the ultimate goal:
complete automation from JSON configuration to simulation execution with
pre-calculated power profiles, unified market data, and agent-driven portfolio
management.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core.actors import Actor
from app.core.environment.config import (
    EnvironmentConfig,
    create_environment_config_from_json,
)
from app.core.environment.power_management_env import PowerManagementEnvironment
from app.core.simulation.price_provider import BasePriceProvider
from app.core.utils.logging import get_logger

logger = get_logger("simulation_manager")


class MarketData(BaseModel):
    """Unified market data container with pre-loaded price information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Market configuration
    price_providers: Dict[str, BasePriceProvider] = Field(
        description="Price providers by portfolio name"
    )

    # Pre-loaded market data
    price_data: Dict[str, pd.DataFrame] = Field(
        default_factory=dict, description="Pre-loaded price data by portfolio"
    )

    market_rules: Dict[str, Any] = Field(
        default_factory=dict, description="Market-specific rules and parameters"
    )

    async def load_market_data(self, start_time: datetime, end_time: datetime) -> None:
        """Pre-load all market data for the simulation period."""
        logger.info(f"Loading market data from {start_time} to {end_time}")

        for portfolio_name, provider in self.price_providers.items():
            try:
                # Set provider date range
                provider.set_range(start_time, end_time)

                # Load price data
                price_df = await provider.get_data()
                self.price_data[portfolio_name] = price_df

                logger.info(
                    f"Loaded {len(price_df)} price records for portfolio "
                    f"'{portfolio_name}'"
                )

            except Exception as e:
                logger.error(f"Failed to load market data for '{portfolio_name}': {e}")
                # Create empty DataFrame as fallback
                self.price_data[portfolio_name] = pd.DataFrame()

    def get_price_at_time(
        self, portfolio_name: str, timestamp: datetime
    ) -> Optional[float]:
        """Get market price for a specific portfolio at a given time."""
        if portfolio_name not in self.price_data:
            return None

        price_df = self.price_data[portfolio_name]
        if price_df.empty:
            return None

        # Ensure all timestamps are timezone-aware UTC
        ts = pd.Timestamp(timestamp, tz="UTC")

        # Ensure price_df index is also timezone-aware UTC
        if isinstance(price_df.index, pd.DatetimeIndex):
            if price_df.index.tz is None:
                price_df.index = price_df.index.tz_localize("UTC")
            elif str(price_df.index.tz) != "UTC":
                price_df.index = price_df.index.tz_convert("UTC")

        # Get price from DataFrame
        try:
            price_col = "price_dollar_mwh"  # Standard column name
            if ts in price_df.index:
                # Direct match
                if price_col in price_df.columns:
                    return float(price_df[price_col].loc[ts])
            else:
                # Use nearest neighbor
                idx_pos = price_df.index.get_indexer([ts], method="nearest")[0]
                if idx_pos != -1 and price_col in price_df.columns:
                    return float(price_df[price_col].iloc[idx_pos])
        except Exception as e:
            logger.warning(
                f"Error getting price for {portfolio_name} at {timestamp}: {e}"
            )

        return None


class SimulationResults(BaseModel):
    """Container for complete simulation results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Simulation metadata
    start_time: datetime
    end_time: datetime
    total_steps: int
    configuration_file: Optional[str] = None

    # Performance metrics
    total_reward: float = 0.0
    average_reward_per_step: float = 0.0

    # Detailed results
    step_results: list = Field(default_factory=list)
    portfolio_results: Dict[str, pd.DataFrame] = Field(default_factory=dict)
    power_profiles: Dict[str, Dict[str, pd.DataFrame]] = Field(default_factory=dict)

    # Market analysis
    market_performance: Dict[str, Any] = Field(default_factory=dict)

    def add_step_result(
        self,
        step: int,
        timestamp: datetime,
        observation: Any,
        action: Any,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Add results from a single simulation step."""
        self.step_results.append(
            {
                "step": step,
                "timestamp": timestamp.isoformat(),
                "reward": reward,
                "info": info,
            }
        )

        self.total_reward += reward

        if self.total_steps > 0:
            self.average_reward_per_step = self.total_reward / self.total_steps

    def save_to_file(self, output_path: Optional[Path] = None) -> Path:
        """Save simulation results to file."""
        if output_path is None:
            timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"simulation_results_{timestamp_str}.json")

        # Convert to serializable format
        results_dict = {
            "simulation_config": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_steps": self.total_steps,
                "total_reward": self.total_reward,
                "average_reward": self.average_reward_per_step,
            },
            "execution_steps": [],
        }

        # Convert step results to JSON-serializable format
        for step in self.step_results:
            step_dict = {}
            for key, value in step.items():
                if hasattr(value, "isoformat"):  # datetime objects
                    step_dict[key] = value.isoformat()
                elif isinstance(value, np.ndarray):  # numpy arrays
                    step_dict[key] = value.tolist()
                elif hasattr(value, "__dict__"):  # complex objects
                    step_dict[key] = str(value)
                else:
                    step_dict[key] = value
            results_dict["execution_steps"].append(step_dict)

        # Create output directory if needed
        output_dir = Path("simulation_results")
        output_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{timestamp}.json"
        output_path = output_dir / filename

        # Save to file
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info(f"Simulation results saved to {output_path}")
        return output_path


class SimulationManager(BaseModel):
    """
    Main simulation orchestrator that provides complete automation from JSON
    configuration.

    This achieves the ultimate goal: automatic simulation object creation with
    validation, provider auto-creation, data loading, power profile pre-calculation,
    market object creation, and agent-driven portfolio management.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config_file_path: Path = Field(description="Path to JSON configuration file")
    start_date_time_override: Optional[datetime] = Field(
        default=None, description="Optional start time override"
    )
    end_date_time_override: Optional[datetime] = Field(
        default=None, description="Optional end time override"
    )

    # Core components (created during initialization)
    environment_config: Optional[EnvironmentConfig] = None
    environment: Optional[PowerManagementEnvironment] = None
    agent: Optional[Actor] = None
    market: Optional[MarketData] = None

    # Simulation state
    initialized: bool = False
    results: Optional[SimulationResults] = None

    async def initialize(self) -> None:
        """
        Auto-create all components with validation.

        This method implements the complete automation pipeline:
        1. Load and validate JSON configuration
        2. Auto-create environment, agent, and providers
        3. Pre-load market data
        4. Pre-calculate power generation profiles
        5. Set up unified market object
        """
        logger.info("Initializing simulation from JSON configuration...")
        logger.info(f"Configuration file: {self.config_file_path}")

        # Step 1: Load and validate JSON configuration
        try:
            self.environment_config = create_environment_config_from_json(
                self.config_file_path,
                start_date_time=self.start_date_time_override,
                end_date_time=self.end_date_time_override,
            )
            logger.info("✓ JSON configuration loaded and validated")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

        # Step 2: Create environment (automatic validation)
        try:
            self.environment = PowerManagementEnvironment(
                config=self.environment_config
            )
            logger.info("✓ Power management environment created")
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            raise

        # Step 3: Get agent (auto-created or create fallback)
        try:
            if self.environment_config.agent:
                self.agent = self.environment_config.agent
                logger.info(f"✓ Agent auto-created: {type(self.agent).__name__}")
            else:
                # Create default BasicHeuristic as fallback
                from app.core.actors import BasicHeuristic

                self.agent = BasicHeuristic(
                    max_lookahead_steps=8,
                    charge_threshold_ratio=0.3,
                    discharge_threshold_ratio=0.7,
                    soc_buffer=0.1,
                )
                logger.info("✓ Default BasicHeuristic agent created (fallback)")
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise

        # Step 4: Create unified market data object
        try:
            await self._create_market_data()
            logger.info("✓ Market data object created")
        except Exception as e:
            logger.error(f"Failed to create market data: {e}")
            raise

        # Step 5: Pre-calculate power generation profiles
        try:
            await self._precalculate_power_profiles()
            logger.info("✓ Power generation profiles pre-calculated")
        except Exception as e:
            logger.error(f"Failed to pre-calculate power profiles: {e}")
            raise

        # Step 6: Initialize results container
        self.results = SimulationResults(
            start_time=self.environment_config.start_date_time,
            end_time=self.environment_config.end_date_time,
            total_steps=0,
            configuration_file=str(self.config_file_path),
        )

        self.initialized = True
        logger.info("🎉 Simulation initialization complete!")

    async def _create_market_data(self) -> None:
        """Create unified market data object with price providers."""
        if not self.environment_config:
            raise RuntimeError("Environment config not initialized")

        price_providers = {}

        for portfolio in self.environment_config.portfolios:
            # Extract price providers from portfolio market configuration
            # This uses the existing provider creation infrastructure
            portfolio_name = portfolio.config.name

            # Try to get the first price provider from the portfolio's market
            if hasattr(portfolio, "plants") and portfolio.plants:
                # Get price provider from first plant (they should all share the
                # same market)
                first_plant = portfolio.plants[0]
                if (
                    hasattr(first_plant, "revenue_calculator")
                    and first_plant.revenue_calculator
                    and hasattr(first_plant.revenue_calculator, "price_provider")
                ):
                    price_providers[portfolio_name] = (
                        first_plant.revenue_calculator.price_provider
                    )
                else:
                    logger.warning(
                        f"No price provider found for portfolio '{portfolio_name}'"
                    )
            else:
                logger.warning(f"No plants found in portfolio '{portfolio_name}'")

        self.market = MarketData(price_providers=price_providers)

        # Pre-load all market data
        await self.market.load_market_data(
            self.environment_config.start_date_time,
            self.environment_config.end_date_time,
        )

    async def _precalculate_power_profiles(self) -> None:
        """Pre-calculate power generation profiles for all plants."""
        if not self.environment_config:
            raise RuntimeError("Environment config not initialized")

        logger.info("Pre-calculating power generation profiles...")

        for portfolio in self.environment_config.portfolios:
            for plant in portfolio.plants:
                try:
                    # Pre-calculate PV generation profile for entire simulation period
                    await self._precalculate_plant_profile(plant)
                    logger.debug(
                        f"Pre-calculated profile for plant '{plant.config.name}'"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to pre-calculate profile for plant "
                        f"'{plant.config.name}': {e}"
                    )

    async def _precalculate_plant_profile(self, plant) -> None:
        """Pre-calculate power generation profile for a single plant."""
        if not self.environment_config:
            raise RuntimeError("Environment config not initialized")

        # Set the PV model to cache the entire simulation period
        if hasattr(plant, "pv_model"):
            # Set cached range to the full simulation period
            plant.pv_model.set_cached_range(
                self.environment_config.start_date_time,
                self.environment_config.end_date_time,
            )

            # Run simulation to populate cache
            await plant.pv_model.run_simulation(force_refresh=False)

    async def run_simulation(
        self, max_steps: Optional[int] = None
    ) -> SimulationResults:
        """
        Execute complete simulation workflow.

        This runs the full agent-driven portfolio management simulation
        using pre-calculated profiles and unified market data.
        """
        if not self.initialized:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")

        if not self.environment or not self.agent or not self.results:
            raise RuntimeError("Simulation components not properly initialized")

        logger.info("Starting simulation execution...")

        # Reset environment
        observation, info = await self.environment.reset_async()
        logger.info("Environment reset completed")

        step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if max_steps and step >= max_steps:
                logger.info(f"Reached maximum steps limit: {max_steps}")
                break

            try:
                # Get current timestamp
                current_time = self.environment.timestamp
                logger.debug(f"Step {step + 1}: {current_time}")

                # Get observation for agent
                raw_observation = await (
                    self.environment.observation_factory.create_observation(
                        current_time
                    )
                )

                # Get agent action using observations
                agent_action = self.agent.get_action(raw_observation, current_time)

                # Convert agent action to environment format
                env_action = self._convert_agent_action_to_env_action(agent_action)

                # Execute step
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = await self.environment.step_async(env_action)

                # Record results
                self.results.add_step_result(
                    step=step + 1,
                    timestamp=current_time,
                    observation=observation,
                    action=agent_action,
                    reward=reward,
                    info=info,
                )

                step += 1

                if step % 10 == 0:  # Log progress every 10 steps
                    logger.info(f"Completed step {step}, reward: {reward:.4f}")

            except Exception as e:
                logger.error(f"Error during simulation step {step + 1}: {e}")
                break

        # Finalize results
        self.results.total_steps = step
        if step > 0:
            self.results.average_reward_per_step = self.results.total_reward / step

        logger.info("🏁 Simulation execution completed!")
        logger.info(f"Total steps: {step}")
        logger.info(f"Total reward: {self.results.total_reward:.4f}")
        logger.info(
            f"Average reward per step: {self.results.average_reward_per_step:.4f}"
        )

        return self.results

    def _convert_agent_action_to_env_action(self, agent_action: Dict) -> Any:
        """Convert agent action format to environment action format."""
        import numpy as np

        if not self.environment:
            raise RuntimeError("Environment not initialized")

        # Extract power targets from agent action
        gym_action = []

        for portfolio_name, plants_dict in agent_action.items():
            if isinstance(plants_dict, dict):
                for plant_name, plant_action in plants_dict.items():
                    if isinstance(plant_action, dict):
                        # Get target power (prefer net_power_target_mw)
                        target = plant_action.get(
                            "net_power_target_mw",
                            plant_action.get("ac_power_generation_target_mw", 0.0),
                        )
                    else:
                        target = float(plant_action) if plant_action else 0.0

                    # Normalize to [-1, 1] range (simple normalization)
                    normalized = np.clip(target / 10.0, -1.0, 1.0)
                    gym_action.append(normalized)

        # Ensure correct action space size
        action_len = (
            self.environment.action_space.shape[0]
            if self.environment.action_space.shape
            else 1
        )
        while len(gym_action) < action_len:
            gym_action.append(0.0)

        return np.array(gym_action[:action_len], dtype=np.float32)


# Convenience factory function
async def create_simulation_from_json(
    config_file_path: str | Path,
    start_time_override: Optional[datetime] = None,
    end_time_override: Optional[datetime] = None,
) -> SimulationManager:
    """
    Create and initialize a complete simulation from JSON configuration.

    This is the main entry point for achieving the ultimate automation goal.
    """
    config_path = Path(config_file_path)

    simulation = SimulationManager(
        config_file_path=config_path,
        start_date_time_override=start_time_override,
        end_date_time_override=end_time_override,
    )

    await simulation.initialize()
    return simulation
