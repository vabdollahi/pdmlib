"""
Gymnasium-compatible power management environment.
"""

import asyncio
import datetime
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from app.core.environment.actions import ActionFactory, ActionType
from app.core.environment.config import EnvironmentConfig
from app.core.environment.observations import ObservationFactory, flatten_observation
from app.core.environment.rewards import RewardFactory
from app.core.utils.logging import get_logger

logger = get_logger("power_management_env")


class PowerManagementEnvironment(gym.Env):
    """
    A gymnasium-compatible environment for power management optimization.

    This environment simulates a power plant portfolio operating in electricity
    markets, where agents learn to optimize power dispatch decisions for revenue
    maximization while managing battery storage and PV generation.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, config: EnvironmentConfig):
        """
        Initialize the power management environment.

        Args:
            config: Environment configuration including portfolios and parameters
        """
        super().__init__()
        self.config = config
        self.timestep = 0
        self.timestamp = config.start_date_time
        self.end_date_time = config.end_date_time
        self.interval = datetime.timedelta(minutes=config.interval_min)

        # Initialize factories
        self.reward_factory = RewardFactory.create_revenue_reward(
            smoothed_reward_parameter=config.smoothed_reward_parameter
        )
        self.observation_factory = ObservationFactory(
            portfolios=config.portfolios,
            historic_data_intervals=config.historic_data_intervals,
            forecast_data_intervals=config.forecast_data_intervals,
            power_normalization_coefficient=config.power_normalization_coefficient,
            price_normalization_coefficient=config.price_normalization_coefficient,
            interval_min=config.interval_min,
        )
        self.action_factory = ActionFactory(
            portfolios=config.portfolios,
            power_normalization_coefficient=config.power_normalization_coefficient,
            interval_min=config.interval_min,
            action_tolerance_percent=config.action_tolerance_percent,
        )

        # Define action and observation spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

        # Environment state
        self.state = 0
        self._last_observation = None

        logger.info(
            f"Initialized PowerManagementEnvironment with "
            f"{len(config.portfolios)} portfolios"
        )

    def _create_action_space(self) -> gym.spaces.Box:
        """Create the action space for the environment."""
        # For simplicity, use a Box space with normalized actions
        # Each portfolio can have multiple plants, each with net power target

        total_plants = sum(
            len(portfolio.plants) for portfolio in self.config.portfolios
        )

        # Action space: net power target for each plant (normalized -1 to 1)
        action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_plants,),
            dtype=np.float32,
        )

        logger.info(f"Created action space for {total_plants} plants")
        return action_space

    def _create_observation_space(self) -> gym.spaces.Box:
        """Create observation space specification for gymnasium compatibility."""
        # Avoid triggering data fetches/simulations here. Use the calculated dimension.
        calculated_dim = self._calculate_observation_dimension()
        logger.info(f"Created observation space with dimension {calculated_dim}")
        return gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(calculated_dim,),
            dtype=np.float32,
        )

    def _calculate_observation_dimension(self) -> int:
        """Calculate the observation dimension based on portfolio configuration."""
        total_dim = 0

        # Count dimensions from each portfolio
        for portfolio in self.config.portfolios:
            for plant in portfolio.plants:
                # Plant observations: PV potential, battery SOC, etc.
                # Based on ObservationFactory structure
                plant_dims = (
                    1  # PV potential
                    + len(plant.batteries)  # Battery SOC for each battery
                    + len(plant.batteries)  # Battery power for each battery
                    + 1  # Total battery SOC
                    + 1  # Plant operation mode
                    + 1  # Plant maintenance status
                )
                total_dim += plant_dims

        # Market observations (from market data)
        market_dims = (
            1  # Current price
            + 24  # Hourly forecast prices
            + 4  # Time features (hour, day, month, day_of_week)
        )
        total_dim += market_dims

        return total_dim

    def _convert_action_to_portfolio_format(self, action: np.ndarray) -> ActionType:
        """Convert flat action array to nested portfolio action format."""
        portfolio_action = {}
        action_idx = 0

        for portfolio in self.config.portfolios:
            portfolio_name = portfolio.config.name
            portfolio_action[portfolio_name] = {}

            for plant in portfolio.plants:
                plant_name = plant.config.name

                # Get normalized action for this plant
                if action_idx < len(action):
                    normalized_action = float(action[action_idx])
                else:
                    normalized_action = 0.0

                # Clamp action to valid range [-1, 1]
                normalized_action = np.clip(normalized_action, -1.0, 1.0)

                # Convert to actual power target (MW) respecting plant constraints
                # Map normalized action [-1, 1] to [min_net_power_mw, max_net_power_mw]
                min_power = plant.config.min_net_power_mw
                max_power = plant.config.max_net_power_mw

                # Linear mapping: -1 -> min_power, +1 -> max_power
                power_range = max_power - min_power
                scaled_action = (normalized_action + 1.0) * 0.5
                target_power_mw = min_power + scaled_action * power_range

                portfolio_action[portfolio_name][plant_name] = {
                    "net_power_target_mw": target_power_mw
                }

                action_idx += 1

        return portfolio_action

    async def step_async(
        self, action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step (async version).

        Args:
            action: Action to execute (normalized -1 to 1 for each plant)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Handle default action
        if action is None:
            total_plants = sum(len(p.plants) for p in self.config.portfolios)
            action = np.zeros(total_plants, dtype=np.float32)

        # Convert action format
        portfolio_action = self._convert_action_to_portfolio_format(action)

        # Execute the action
        try:
            info, rewards = await self.action_factory.execute_action(
                portfolio_action, self.timestamp
            )

            # Calculate total reward across all portfolios
            total_reward = 0.0
            for portfolio_rewards in rewards.values():
                for plant_reward in portfolio_rewards.values():
                    total_reward += plant_reward

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            info = {"error": str(e)}
            total_reward = -1.0  # Penalty for invalid actions

        # Move forward in time
        self.timestep += 1
        self.timestamp += self.interval

        # Get the new observation
        try:
            raw_observation = await self.observation_factory.create_observation(
                self.timestamp
            )
            observation = flatten_observation(raw_observation)
            self._last_observation = observation
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            # Use previous observation or zeros
            if self._last_observation is not None:
                observation = self._last_observation
            else:
                obs_shape = self.observation_space.shape
                if obs_shape is not None:
                    observation = np.zeros(obs_shape, dtype=np.float32)
                else:
                    # Fallback to calculated dimension
                    calculated_dim = self._calculate_observation_dimension()
                    observation = np.zeros(calculated_dim, dtype=np.float32)

        # Check if episode is complete
        terminated = self.timestamp >= self.end_date_time
        truncated = False  # Could add max timestep limit here

        # Add environment info
        env_info = {
            "timestamp": self.timestamp.isoformat(),
            "timestep": self.timestep,
            "total_reward": total_reward,
        }
        if isinstance(info, dict):
            info.update(env_info)
        else:
            info = env_info

        return observation, total_reward, terminated, truncated, info

    def step(
        self, action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step (sync wrapper).

        Args:
            action: Action to execute

        Returns:
            Step results
        """
        return asyncio.run(self.step_async(action))

    async def reset_async(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state (async version).

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (initial_observation, info)
        """
        # Reset time
        self.timestep = 0

        # Handle time range from options
        if options:
            start_time = options.get("start_date_time")
            end_time = options.get("end_date_time")
            battery_soc = options.get("battery_state_of_charge")

            if start_time is not None:
                self.timestamp = start_time
            else:
                self.timestamp = self.config.start_date_time

            if end_time is not None:
                self.end_date_time = end_time
            else:
                self.end_date_time = self.config.end_date_time

            # Configure trial data
            self.config.set_trial_data(
                start_date_time=self.timestamp
                - self.config.historic_data_intervals * self.interval,
                end_date_time=self.end_date_time
                + (self.config.forecast_data_intervals + 1) * self.interval,
                battery_state_of_charge=battery_soc,
            )
        else:
            self.timestamp = self.config.start_date_time
            self.end_date_time = self.config.end_date_time

        # Get initial observation
        try:
            raw_observation = await self.observation_factory.create_observation(
                self.timestamp
            )
            observation = flatten_observation(raw_observation)
            self._last_observation = observation
        except Exception as e:
            logger.error(f"Error creating initial observation: {e}")
            obs_shape = self.observation_space.shape
            if obs_shape is not None:
                observation = np.zeros(obs_shape, dtype=np.float32)
            else:
                # Fallback to calculated dimension
                calculated_dim = self._calculate_observation_dimension()
                observation = np.zeros(calculated_dim, dtype=np.float32)
            self._last_observation = observation

        info = {
            "timestamp": self.timestamp.isoformat(),
            "end_time": self.end_date_time.isoformat(),
            "timestep": self.timestep,
        }

        logger.info(f"Environment reset: {self.timestamp} to {self.end_date_time}")
        return observation, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to initial state (sync wrapper).

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            Reset results
        """
        return asyncio.run(self.reset_async(seed=seed, options=options))

    def render(self, mode: str = "human") -> None:
        """
        Render the environment state.

        Args:
            mode: Rendering mode

        Returns:
            None
        """
        if mode == "human":
            print(f"Timestep: {self.timestep}")
            print(f"Timestamp: {self.timestamp}")

            # Print portfolio states
            for portfolio in self.config.portfolios:
                print(f"\nPortfolio: {portfolio.config.name}")
                for plant in portfolio.plants:
                    print(f"  Plant: {plant.config.name}")
                    print(f"    Battery SOC: {plant.average_battery_soc:.2%}")
                    print(f"    Operation Mode: {plant.operation_mode}")

        return None

    def close(self) -> None:
        """Clean up environment resources."""
        logger.info("Environment closed")
        super().close()
