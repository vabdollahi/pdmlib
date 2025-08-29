"""
Unified Simulation Manager for Power Management System.

This module provides the main orchestration layer that achieves the ultimate goal:
complete automation from JSON configuration to simulation execution with
pre-calculated power profiles, unified market data, and agent-driven portfolio
management.
"""

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from app.core.actors import Actor, Heuristic
from app.core.environment.power_management_env import (
    EnvironmentConfig,
    PowerManagementEnvironment,
)
from app.core.simulation.price_provider import PriceProviderProtocol
from app.core.utils.logging import get_logger

logger = get_logger("simulation_manager")


class MarketData(BaseModel):
    """
    Manages market data, including electricity prices, for the simulation.

    This class acts as a unified data source for the environment, conforming
    to the PriceProviderProtocol. It can aggregate data from multiple providers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    price_providers: Dict[str, PriceProviderProtocol] = Field(
        default_factory=dict,
        description=(
            "Dictionary of price providers, keyed by portfolio name or 'default'"
        ),
    )
    _price_data: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    _start_time: datetime.datetime | None = PrivateAttr(default=None)
    _end_time: datetime.datetime | None = PrivateAttr(default=None)

    def set_range(self, start_time: datetime.datetime, end_time: datetime.datetime):
        """Set the simulation time range."""
        self._start_time = start_time
        self._end_time = end_time
        logger.info(
            f"Setting date range for market data from {start_time} to {end_time}"
        )

    async def load_market_data(
        self, start_time: datetime.datetime, end_time: datetime.datetime
    ):
        """Load price data from all providers for the given time range."""
        # For now, we'll just use the first provider.
        # A more complex implementation could merge data from multiple sources.
        if not self.price_providers:
            logger.warning("No price providers configured for MarketData.")
            return

        provider = next(iter(self.price_providers.values()))
        provider.set_range(start_time, end_time)
        data = await provider.get_data()
        if data is not None and not data.empty:
            self._price_data = data
            self.validate_data_format(self._price_data)

    async def get_data(self) -> pd.DataFrame:
        """Get all available price data for the configured time range."""
        if self._price_data.empty and self._start_time and self._end_time:
            await self.load_market_data(self._start_time, self._end_time)
        return self._price_data

    async def get_price_at_time(self, timestamp: datetime.datetime) -> float | None:
        """
        Get the electricity price at a specific timestamp.

        This method conforms to the PriceProviderProtocol. It uses the first
        available price provider.
        """
        if not self.price_providers:
            logger.warning("No price providers available in MarketData.")
            return None

        # Use the first provider to get the price
        provider = next(iter(self.price_providers.values()))
        return await provider.get_price_at_time(timestamp)

    def validate_data_format(self, df: pd.DataFrame):
        """Validate the format of the price data."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Price data index must be a DatetimeIndex.")
        if "price_dollar_mwh" not in df.columns:
            raise ValueError("Price data must include a 'price_dollar_mwh' column.")


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and datetime objects."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        return super().default(o)


@dataclass
class SimulationResults:
    """Data class for storing simulation results."""

    start_time: datetime.datetime
    end_time: datetime.datetime
    configuration_file: str
    total_reward: float = 0.0
    step_results: list[dict] = field(default_factory=list)
    portfolio_results: dict = field(default_factory=dict)
    power_profiles: dict = field(default_factory=dict)
    market_performance: dict = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        """Return the total number of steps recorded."""
        return len(self.step_results)

    @property
    def average_reward_per_step(self) -> float:
        """Calculate the average reward per step."""
        if not self.step_results:
            return 0.0
        return self.total_reward / self.total_steps

    def add_step_result(self, **kwargs):
        """Add a result for a single simulation step."""
        self.step_results.append(kwargs)
        if "reward" in kwargs:
            self.total_reward += kwargs["reward"]

    def save_to_file(self, output_dir: str | Path = "simulation_results") -> Path:
        """
        Save the complete simulation results to a JSON file.

        Args:
            output_dir: The directory where results will be saved.

        Returns:
            The path to the saved results file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create a serializable representation of the results
        serializable_results = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "configuration_file": self.configuration_file,
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "average_reward_per_step": self.average_reward_per_step,
            "step_results": [
                {
                    **step,
                    "timestamp": step["timestamp"].isoformat()
                    if isinstance(step.get("timestamp"), datetime.datetime)
                    else step.get("timestamp"),
                }
                for step in self.step_results
            ],
            "portfolio_results": self.portfolio_results,
            "power_profiles": self.power_profiles,
            "market_performance": self.market_performance,
        }

        # Generate a unique filename
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"simulation_results_{timestamp_str}.json"

        # Save to file
        with open(file_path, "w") as f:
            json.dump(serializable_results, f, indent=4, cls=NumpyJSONEncoder)

        return file_path


class SimulationManager:
    """
    Manages the complete, automated simulation workflow.

    This class integrates configuration loading, environment setup, agent control,
    and results collection into a unified, automated process. It is designed
    to be the primary entry point for running simulations.
    """

    def __init__(self, config_file_path: str | Path):
        """
        Initialize the simulation manager with the path to a JSON config file.

        Args:
            config_file_path: Path to the JSON configuration file.
        """
        self.config_file_path = Path(config_file_path)
        self.initialized = False
        self.environment_config: EnvironmentConfig | None = None
        self.environment: PowerManagementEnvironment | None = None
        self.agent: Actor | None = None
        self.market: MarketData | None = None
        self.results: SimulationResults | None = None
        self.start_time: datetime.datetime | None = None
        self.end_time: datetime.datetime | None = None

    async def _initialize(self):
        """
        Initialize the complete simulation environment from configuration.
        """
        logger.info("Initializing simulation from JSON configuration...")
        logger.info(f"Configuration file: {self.config_file_path}")

        # Step 1: Load and validate JSON configuration
        try:
            # Create environment config from JSON spec using the new factory method
            self.environment = PowerManagementEnvironment.from_json(
                self.config_file_path
            )
            self.environment_config = self.environment.config
            self.start_time = self.environment_config.start_date_time
            self.end_time = self.environment_config.end_date_time
            logger.info("✓ JSON configuration loaded and environment created")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

        # Step 2: Get agent (auto-created from configuration)
        try:
            if self.environment_config.agent:
                self.agent = self.environment_config.agent

                # If it's a Heuristic, configure it with the environment config
                if isinstance(self.agent, Heuristic):
                    self.agent.configure(self.environment_config)
                    logger.info("✓ Heuristic configured with environment config")

                logger.info(f"✓ Agent auto-created: {type(self.agent).__name__}")
            else:
                raise ValueError(
                    "No agent configuration found. Agent configuration is required."
                )
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise

        # Step 3: Create and configure market data object
        try:
            # Get price provider from the environment's configuration portfolios
            price_provider = None
            if self.environment_config.portfolios:
                for portfolio in self.environment_config.portfolios:
                    for plant in portfolio.plants:
                        if (
                            hasattr(plant, "revenue_calculator")
                            and plant.revenue_calculator
                            and hasattr(plant.revenue_calculator, "price_provider")
                        ):
                            price_provider = plant.revenue_calculator.price_provider
                            break
                    if price_provider:
                        break

            if price_provider:
                self.market = MarketData(price_providers={"default": price_provider})
                self.market.set_range(self.start_time, self.end_time)
                logger.info("✓ Market data object created")
            else:
                logger.warning("No price provider configured.")
                self.market = MarketData()

        except Exception as e:
            logger.error(f"Failed to create market data: {e}")
            raise

        # Step 4.5: Connect market data to environment
        try:
            self.environment.set_market_data(self.market)
            logger.info("✓ Market data connected to environment")
        except Exception as e:
            logger.error(f"Failed to connect market data to environment: {e}")
            raise

        # Step 5: Pre-calculate power generation profiles
        try:
            await self.pre_calculate_power_profiles()
            logger.info("✓ Power generation profiles pre-calculated")
        except Exception as e:
            logger.error(f"Failed to pre-calculate power profiles: {e}")
            raise

        # Step 6: Initialize results container
        self.results = SimulationResults(
            start_time=self.start_time,
            end_time=self.end_time,
            configuration_file=str(self.config_file_path),
        )
        logger.info("🎉 Simulation initialization complete!")
        self.initialized = True

    async def pre_calculate_power_profiles(self):
        """Pre-calculate power generation profiles for all plants."""
        if not self.environment_config or not self.start_time or not self.end_time:
            raise RuntimeError(
                "Environment must be initialized before pre-calculating power profiles."
            )

        logger.info("Pre-calculating power generation profiles...")
        for portfolio in self.environment_config.portfolios:
            for plant in portfolio.plants:
                # Set the date range on the PV model's cache and run the simulation
                # to pre-calculate the generation profile.
                plant.pv_model.set_cached_range(self.start_time, self.end_time)
                await plant.pv_model.run_simulation(force_refresh=True)

    async def run_simulation(self, max_steps: int | None = None) -> "SimulationResults":
        """
        Execute complete simulation workflow.

        This runs the full agent-driven portfolio management simulation
        using pre-calculated profiles and unified market data.
        """
        if (
            not self.initialized
            or not self.environment
            or not self.agent
            or not self.results
        ):
            raise RuntimeError(
                "Simulation must be initialized before running. "
                "Call create_simulation_from_json() first."
            )

        logger.info("Starting simulation execution...")
        observation, info = await self.environment.reset_async()
        self.results.add_step_result(
            step=0,
            timestamp=self.environment.timestamp,
            observation=observation,
            action=None,
            reward=0.0,
            info=info,
        )

        terminated = False
        truncated = False
        step_count = 0

        while not terminated and not truncated:
            action = self.agent.get_action(observation)
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = await self.environment.step_async(action)

            step_count += 1
            self.results.add_step_result(
                step=step_count,
                timestamp=self.environment.timestamp,
                observation=observation,
                action=action,
                reward=reward,
                info=info,
            )

            if max_steps and step_count >= max_steps:
                logger.info(f"Simulation reached max_steps ({max_steps}).")
                break

        logger.info("Simulation execution finished.")
        return self.results


async def create_simulation_from_json(
    config_file_path: str | Path,
) -> "SimulationManager":
    """
    Factory function to create and initialize a SimulationManager from a JSON file.

    This is the recommended way to create a simulation, as it ensures all
    components are properly initialized and connected.

    Args:
        config_file_path: Path to the JSON configuration file.

    Returns:
        An initialized SimulationManager instance.
    """
    simulation = SimulationManager(config_file_path)
    await simulation._initialize()
    return simulation
