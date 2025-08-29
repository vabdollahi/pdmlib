"""
Unified power management environment with integrated configuration.
"""

import asyncio
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.core.actors import Actor
from app.core.environment.actions import ActionFactory, ActionType
from app.core.environment.observations import ObservationFactory
from app.core.environment.rewards import RewardFactory
from app.core.simulation.portfolio import PowerPlantPortfolio
from app.core.utils.logging import get_logger

logger = get_logger("power_management_env")


class EnvironmentConfig(BaseModel):
    """Configuration for the power management environment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # Time configuration
    start_date_time: datetime.datetime = Field(
        description="Start datetime for the environment"
    )
    end_date_time: datetime.datetime = Field(
        description="End datetime for the environment"
    )
    interval_min: float = Field(
        default=60.0, description="Time interval in minutes between actions"
    )

    # Portfolio and system configuration
    portfolios: List[PowerPlantPortfolio] = Field(
        description="List of power plant portfolios to manage"
    )

    # Grid purchase configuration
    max_grid_purchase_mw: float = Field(
        default=50.0, description="Maximum power that can be purchased from grid (MW)"
    )

    # Data configuration
    historic_data_intervals: int = Field(
        default=12,
        description="Number of historic intervals to include in observations",
    )
    forecast_data_intervals: int = Field(
        default=12,
        description="Number of forecast intervals to include in observations",
    )

    # Normalization parameters
    power_normalization_coefficient: float = Field(
        default=1e6, description="Power normalization factor (e.g., MW to W)"
    )
    price_normalization_coefficient: float = Field(
        default=100.0, description="Price normalization factor"
    )

    # Action constraints
    action_tolerance_percent: float = Field(
        default=0.05, description="Tolerance for action validation (0.0-1.0)"
    )

    # Reward configuration
    smoothed_reward_parameter: float = Field(
        default=0.1, description="Smoothing parameter for reward calculation (0.0-1.0)"
    )

    # Agent configuration (optional, for automatic agent creation)
    agent: Optional[Actor] = Field(
        default=None, description="Automatically created agent for environment control"
    )

    # Environment state
    _trial_start: Optional[datetime.datetime] = None
    _trial_end: Optional[datetime.datetime] = None

    def set_trial_data(
        self,
        start_date_time: datetime.datetime,
        end_date_time: datetime.datetime,
        battery_state_of_charge: Optional[float] = None,
    ) -> None:
        """Set trial data configuration."""
        self._trial_start = start_date_time
        self._trial_end = end_date_time

        # Reset battery states if specified
        if battery_state_of_charge is not None:
            for portfolio in self.portfolios:
                for plant in portfolio.plants:
                    for battery in plant.batteries:
                        battery.config.initial_soc = battery_state_of_charge
                        battery.reset_state()


def _load_json_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


class PowerManagementEnvironment(gym.Env):
    """
    A unified power management environment with integrated configuration.

    This environment simulates a power plant portfolio operating in electricity
    markets, where agents learn to optimize power dispatch decisions for revenue
    maximization while managing battery storage and PV generation.

    Supports creation from JSON files, dictionaries, or direct configuration.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    @classmethod
    def from_json(
        cls,
        config_path: Path,
        start_date_time: Optional[datetime.datetime] = None,
        end_date_time: Optional[datetime.datetime] = None,
    ) -> "PowerManagementEnvironment":
        """
        Create environment directly from JSON configuration file.

        Args:
            config_path: Path to JSON configuration file
            start_date_time: Optional override for start time
            end_date_time: Optional override for end time

        Returns:
            Configured PowerManagementEnvironment instance
        """
        config_dict = _load_json_config(config_path)

        # Apply time overrides if provided
        if start_date_time is not None:
            config_dict["start_date_time"] = start_date_time.isoformat()
        if end_date_time is not None:
            config_dict["end_date_time"] = end_date_time.isoformat()

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PowerManagementEnvironment":
        """
        Create environment from configuration dictionary.

        Args:
            config_dict: Dictionary containing environment configuration

        Returns:
            Configured PowerManagementEnvironment instance
        """
        config = cls._create_config_from_dict(config_dict)
        return cls(config)

    @classmethod
    def _create_config_from_dict(cls, config_dict: Dict[str, Any]) -> EnvironmentConfig:
        """
        Create EnvironmentConfig from dictionary specification.

        Args:
            config_dict: Dictionary containing environment configuration

        Returns:
            EnvironmentConfig instance
        """
        # Import here to avoid circular dependencies
        from app.core.environment.spec_models import EnvironmentSpec

        # Validate input spec strictly
        spec = EnvironmentSpec.model_validate(config_dict)

        # Parse time configuration
        start_date_time = datetime.datetime.fromisoformat(spec.start_date_time)
        end_date_time = datetime.datetime.fromisoformat(spec.end_date_time)
        if end_date_time <= start_date_time:
            raise ValueError("end_date_time must be after start_date_time")

        # Parse portfolios from configuration
        portfolios = []
        if spec.portfolios:
            for portfolio_config in spec.portfolios:
                portfolio = cls._create_portfolio_from_config(
                    portfolio_config, start_date_time, end_date_time
                )
                portfolios.append(portfolio)
        else:
            raise ValueError("No portfolios specified in configuration")

        # Create agent from configuration (optional)
        agent = None
        if spec.agent:
            agent = cls._create_agent_from_config(spec.agent)

        # Create EnvironmentConfig
        return EnvironmentConfig(
            start_date_time=start_date_time,
            end_date_time=end_date_time,
            interval_min=spec.interval_min,
            portfolios=portfolios,
            max_grid_purchase_mw=spec.max_grid_purchase_mw,
            historic_data_intervals=spec.historic_data_intervals,
            forecast_data_intervals=spec.forecast_data_intervals,
            power_normalization_coefficient=spec.power_normalization_coefficient,
            price_normalization_coefficient=spec.price_normalization_coefficient,
            action_tolerance_percent=spec.action_tolerance_percent,
            smoothed_reward_parameter=spec.smoothed_reward_parameter,
            agent=agent,
        )

    @classmethod
    def _create_portfolio_from_config(
        cls, portfolio_config, start_date_time, end_date_time
    ):
        """Create a portfolio from configuration."""
        # Import here to avoid circular dependencies
        from app.core.simulation.portfolio import (
            PortfolioConfiguration,
            PowerPlantPortfolio,
        )

        plants = []
        for plant_config in portfolio_config.plants:
            # Create plant using the integrated factory method
            plant = cls._create_plant_from_config(
                plant_config, start_date_time, end_date_time, portfolio_config.market
            )
            plants.append(plant)

        # Create portfolio configuration
        portfolio_config_obj = PortfolioConfiguration(
            name=portfolio_config.name,
            max_total_power_mw=portfolio_config.max_total_power_mw,
            allow_grid_purchase=portfolio_config.allow_grid_purchase,
        )

        return PowerPlantPortfolio(config=portfolio_config_obj, plants=plants)

    @classmethod
    def _create_plant_from_config(
        cls, plant_config, start_date_time, end_date_time, market_config=None
    ):
        """Create a plant from configuration dictionary."""
        from app.core.environment.spec_models import PlantSpec
        from app.core.simulation.battery_simulator import (
            BatteryConfiguration,
            LinearBatterySimulator,
        )
        from app.core.simulation.plant import PlantConfiguration, SolarBatteryPlant
        from app.core.simulation.pv_model import PVModel
        from app.core.simulation.pvlib_models import PVLibModel
        from app.core.simulation.solar_revenue import SolarRevenueCalculator
        from app.core.utils.location import GeospatialLocation

        # Validate/normalize plant spec
        if not isinstance(plant_config, PlantSpec):
            spec = PlantSpec.model_validate(plant_config)
        else:
            spec = plant_config

        # Require detailed PVLib-style config
        if not ("location" in spec.model_dump() and "pv_systems" in spec.model_dump()):
            raise ValueError(
                "Only detailed PVLib plant configuration is supported "
                "(location + pv_systems + plant_config)."
            )

        # Build geospatial location for weather provider
        location_cfg = spec.location
        latitude = (
            location_cfg["latitude"]
            if isinstance(location_cfg, dict)
            else getattr(location_cfg, "latitude", None)
        )
        longitude = (
            location_cfg["longitude"]
            if isinstance(location_cfg, dict)
            else getattr(location_cfg, "longitude", None)
        )
        if latitude is None or longitude is None:
            raise ValueError("Plant location must include latitude and longitude")
        geo = GeospatialLocation(latitude=latitude, longitude=longitude)

        # Create weather provider
        weather_provider = cls._create_weather_provider_from_config(
            market_config, geo, start_date_time, end_date_time
        )

        # Create PVLib model
        pvlib_input = {
            "location": spec.location,
            "pv_systems": spec.pv_systems,
            "physical_simulation": {
                "aoi_model": "physical",
                "spectral_model": "no_loss",
            },
        }
        pvlib_model = PVLibModel.model_validate(pvlib_input)
        pv_model = PVModel(pv_config=pvlib_model, weather_provider=weather_provider)

        # Create batteries if specified
        batteries = []
        if spec.batteries:
            for battery_data in spec.batteries:
                if not isinstance(battery_data, dict):
                    raise ValueError("Battery configuration must be an object")
                battery_cfg = BatteryConfiguration(
                    energy_capacity_mwh=battery_data["energy_capacity_mwh"],
                    max_power_mw=battery_data["max_power_mw"],
                    round_trip_efficiency=battery_data["round_trip_efficiency"],
                    initial_soc=battery_data.get("initial_soc", 0.5),
                    min_soc=battery_data.get("min_soc", 0.1),
                    max_soc=battery_data.get("max_soc", 0.9),
                )
                batteries.append(LinearBatterySimulator(config=battery_cfg))

        # Plant configuration
        if not isinstance(spec.plant_config, dict):
            raise ValueError("plant_config must be an object with plant settings")
        plant_cfg_dict = spec.plant_config
        plant_cfg = PlantConfiguration(
            name=plant_cfg_dict["name"],
            plant_id=plant_cfg_dict.get("plant_id"),
            max_net_power_mw=plant_cfg_dict["max_net_power_mw"],
            min_net_power_mw=plant_cfg_dict.get("min_net_power_mw", 0.0),
            enable_market_participation=plant_cfg_dict.get(
                "enable_market_participation", True
            ),
        )

        # Create price provider from market config
        price_provider = cls._create_price_provider_from_config(
            market_config, geo, start_date_time, end_date_time
        )

        # Create revenue calculator
        revenue_calculator = SolarRevenueCalculator(
            pv_model=pv_model, price_provider=price_provider
        )

        # Create the plant
        return SolarBatteryPlant(
            config=plant_cfg,
            pv_model=pv_model,
            batteries=batteries,
            revenue_calculator=revenue_calculator,
        )

    @classmethod
    def _create_price_provider_from_config(
        cls, market_config, location, start_date_time, end_date_time
    ):
        """Create a price provider from market configuration."""
        from app.core.simulation.provider_config import (
            create_price_provider_from_config,
        )

        if not market_config:
            raise ValueError(
                "Market configuration is required to create price provider"
            )

        # Extract price configurations
        prices_list = []
        try:
            if hasattr(market_config, "prices"):
                prices_list = market_config.prices
            elif isinstance(market_config, dict) and "prices" in market_config:
                prices_list = market_config["prices"]
        except Exception:
            prices_list = []

        if not prices_list or len(prices_list) == 0:
            raise ValueError(
                "No price configuration found. "
                "Price provider configuration is required."
            )

        price_config = prices_list[0]

        # Import config types for validation
        from app.core.simulation.provider_config import (
            CAISOPriceProviderConfig,
            CSVPriceProviderConfig,
            IESOPriceProviderConfig,
        )

        # Validate config against proper type
        if isinstance(price_config, dict):
            provider_type = price_config.get("type", "csv_file")
            config_dict = price_config
        else:
            provider_type = getattr(price_config, "type", "csv_file")
            # Convert the object to dict for validation
            if hasattr(price_config, "model_dump"):
                config_dict = price_config.model_dump()
            elif hasattr(price_config, "__dict__"):
                config_dict = {
                    k: v
                    for k, v in price_config.__dict__.items()
                    if not k.startswith("_")
                }
            else:
                config_dict = {"type": provider_type}

        try:
            if provider_type == "csv_file":
                validated_config = CSVPriceProviderConfig.model_validate(config_dict)
            elif provider_type == "caiso":
                validated_config = CAISOPriceProviderConfig.model_validate(config_dict)
            elif provider_type == "ieso":
                validated_config = IESOPriceProviderConfig.model_validate(config_dict)
            else:
                raise ValueError(f"Unknown price provider type: {provider_type}")

            return create_price_provider_from_config(
                validated_config, location, start_date_time, end_date_time
            )
        except Exception as e:
            logger.error(f"Failed to create price provider from config: {e}")
            raise ValueError(f"Price provider configuration is invalid: {e}") from e

    @classmethod
    def _create_weather_provider_from_config(
        cls, market_config, location, start_date_time, end_date_time
    ):
        """Create a weather provider from market configuration."""
        from app.core.simulation.provider_config import (
            create_weather_provider_from_config,
        )

        if not market_config:
            raise ValueError(
                "Market configuration is required to create weather provider"
            )

        # Extract weather configuration
        weather_config = None
        try:
            if hasattr(market_config, "weather"):
                weather_config = market_config.weather
            elif isinstance(market_config, dict) and "weather" in market_config:
                weather_config = market_config["weather"]
        except Exception:
            weather_config = None

        if not weather_config:
            raise ValueError(
                "No weather configuration found. "
                "Weather provider configuration is required."
            )

        # Import config types for validation
        from app.core.simulation.provider_config import (
            CSVWeatherProviderConfig,
            OpenMeteoWeatherProviderConfig,
        )

        # Validate config against proper type
        if isinstance(weather_config, dict):
            provider_type = weather_config.get("type", "csv_file")
            config_dict = weather_config
        else:
            provider_type = getattr(weather_config, "type", "csv_file")
            # Convert the object to dict for validation
            if hasattr(weather_config, "model_dump"):
                config_dict = weather_config.model_dump()
            elif hasattr(weather_config, "__dict__"):
                config_dict = {
                    k: v
                    for k, v in weather_config.__dict__.items()
                    if not k.startswith("_")
                }
            else:
                config_dict = {"type": provider_type}

        try:
            if provider_type == "csv_file":
                validated_config = CSVWeatherProviderConfig.model_validate(config_dict)
            elif provider_type == "openmeteo":
                validated_config = OpenMeteoWeatherProviderConfig.model_validate(
                    config_dict
                )
            else:
                raise ValueError(f"Unknown weather provider type: {provider_type}")

            return create_weather_provider_from_config(
                validated_config, location, start_date_time, end_date_time
            )
        except Exception as e:
            logger.error(f"Failed to create weather provider from config: {e}")
            raise ValueError(f"Weather provider configuration is invalid: {e}") from e

    @staticmethod
    def _create_agent_from_config(agent_spec) -> Optional[Actor]:
        """Create an agent from agent specification."""
        from app.core.actors import AgentConfig, create_agent_from_config
        from app.core.environment.spec_models import AgentSpec

        # Convert AgentSpec to AgentConfig and create agent
        if isinstance(agent_spec, AgentSpec):
            # Convert AgentSpec to dict, then to AgentConfig
            agent_dict = {
                "type": agent_spec.type,
                "enabled": agent_spec.enabled,
                "name": agent_spec.name,
                "parameters": agent_spec.parameters or {},
            }
        else:
            # Assume it's already a dict-like structure
            agent_dict = agent_spec

        try:
            # Create AgentConfig and then create agent
            agent_config = AgentConfig.model_validate(agent_dict)
            agent = create_agent_from_config(agent_config)

            if agent:
                logger.info(f"Successfully created agent: {agent_config.type}")
            else:
                logger.info(f"Agent creation skipped (disabled): {agent_config.type}")

            return agent

        except Exception as e:
            logger.error(f"Failed to create agent from config: {e}")
            raise ValueError(f"Invalid agent configuration: {e}") from e

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

    def set_market_data(self, market_data):
        """Set market data for the observation factory."""
        self.observation_factory.market_data = market_data
        logger.info("Market data set for observation factory")

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

    def _create_observation_space(self) -> gym.spaces.Dict:
        """Create observation space specification for gymnasium compatibility."""
        # Define the structured observation space
        return gym.spaces.Dict(
            {
                "market": gym.spaces.Dict(
                    {
                        "market_data": gym.spaces.Dict(
                            {
                                "price": gym.spaces.Box(
                                    low=-1000.0, high=1000.0, shape=(), dtype=np.float32
                                )
                            }
                        )
                    }
                ),
                "portfolios": gym.spaces.Dict(
                    {
                        # Note: This will be dynamically populated based on actual
                        # portfolios. For now, we define a flexible structure that
                        # can handle variable portfolios
                    }
                ),
            }
        )

    async def _convert_action_to_portfolio_format(
        self, action: np.ndarray
    ) -> ActionType:
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

                # Convert to actual power target (MW) respecting available generation
                # Get current available generation for this plant
                max_gen_mw, max_cons_mw = await plant.get_available_power(
                    timestamp=self.timestamp
                )

                # Map normalized action [-1, 1] to [-max_cons_mw, +max_gen_mw]
                # Positive actions = generation, negative actions = consumption
                if normalized_action >= 0:
                    # Generation: map [0, 1] to [0, max_gen_mw]
                    target_power_mw = normalized_action * max_gen_mw
                else:
                    # Consumption: map [-1, 0] to [-max_cons_mw, 0]
                    target_power_mw = normalized_action * max_cons_mw

                portfolio_action[portfolio_name][plant_name] = {
                    "net_power_target_mw": target_power_mw
                }

                action_idx += 1

        return portfolio_action

    async def step_async(
        self, action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
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
        portfolio_action = await self._convert_action_to_portfolio_format(action)

        # Execute the action
        info, rewards = await self.action_factory.execute_action(
            portfolio_action, self.timestamp
        )

        # Calculate total reward across all portfolios
        total_reward = 0.0
        for portfolio_rewards in rewards.values():
            for plant_reward in portfolio_rewards.values():
                total_reward += plant_reward

        # Move forward in time
        self.timestep += 1
        self.timestamp += self.interval

        # Get the new observation
        try:
            raw_observation = await self.observation_factory.create_observation(
                self.timestamp
            )
            observation = raw_observation
            self._last_observation = observation
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            # Use previous observation or empty dict
            if self._last_observation is not None:
                observation = self._last_observation
            else:
                # Error: Unable to create observation without proper configuration
                raise RuntimeError(
                    "Unable to create observation - no market data or "
                    "observation factory configured"
                )

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
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
            observation = raw_observation
            self._last_observation = observation
        except Exception as e:
            logger.error(f"Error creating initial observation: {e}")
            # Error: Configuration required for proper operation
            raise RuntimeError(
                f"Failed to create initial observation: {e}. "
                "Ensure market data and observation factory are properly configured."
            ) from e

        info = {
            "timestamp": self.timestamp.isoformat(),
            "end_time": self.end_date_time.isoformat(),
            "timestep": self.timestep,
        }

        logger.info(f"Environment reset: {self.timestamp} to {self.end_date_time}")
        return observation, info

    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
