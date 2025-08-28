"""
Environment configuration for power management gymnasium environment.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.actors import Actor
from app.core.simulation.portfolio import PowerPlantPortfolio
from app.core.simulation.price_provider import BasePriceProvider
from app.core.utils.logging import get_logger

logger = get_logger("environment_config")


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

    # Market data configuration
    price_provider: Optional[BasePriceProvider] = Field(
        default=None, description="Electricity price data provider"
    )

    # Grid purchase configuration
    max_grid_purchase_mw: float = Field(
        default=50.0, description="Maximum power that can be purchased from grid (MW)"
    )
    grid_purchase_enabled: bool = Field(
        default=False,
        description="Whether grid purchases are allowed for all portfolios",
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
                        battery.reset_state(battery_state_of_charge)


# --- Module-level helpers ----------------------------------------------------


def load_portfolio_config(config_path: Path) -> Dict:
    """Load configuration dictionary from JSON file."""
    import json

    with open(config_path, "r") as f:
        return json.load(f)


def create_environment_config_from_json(
    config_path: Path,
    start_date_time: Optional[datetime.datetime] = None,
    end_date_time: Optional[datetime.datetime] = None,
) -> "EnvironmentConfig":
    """Create EnvironmentConfig from a JSON file using strict spec validation.

    If start/end datetimes are provided, they override values in the JSON.
    """
    config_dict = load_portfolio_config(config_path)
    if start_date_time is not None:
        config_dict["start_date_time"] = start_date_time.isoformat()
    if end_date_time is not None:
        config_dict["end_date_time"] = end_date_time.isoformat()
    return EnvironmentConfigFactory.create(config_dict)


class EnvironmentConfigFactory:
    """Factory for creating EnvironmentConfig from dictionary specifications."""

    @staticmethod
    def create(config_dict: Dict[str, Any]) -> EnvironmentConfig:
        """
        Create EnvironmentConfig from dictionary specification.

        Args:
            config_dict: Dictionary containing environment configuration
                Example structure:
                {
                    "start_date_time": "2023-01-01 09:30:00+11:00",
                    "end_date_time": "2023-01-01 09:40:00+11:00",
                    "interval_min": 60,
                    "historic_data_intervals": 2,
                    "forecast_data_intervals": 2,
                    "portfolios": [...],  # Portfolio configurations
                    "market": {...},      # Market/price configurations
                    "smoothed_reward_parameter": 0.1,
                    "action_tolerance_percent": 0.01
                }

        Returns:
            EnvironmentConfig instance ready for PowerManagementEnvironment
        """
        try:
            # Validate input spec strictly
            from app.core.environment.spec_models import EnvironmentSpec

            spec = EnvironmentSpec.model_validate(config_dict)

            # Parse time configuration
            start_date_time = datetime.datetime.fromisoformat(spec.start_date_time)
            end_date_time = datetime.datetime.fromisoformat(spec.end_date_time)
            if end_date_time <= start_date_time:
                raise ValueError("end_date_time must be after start_date_time")

            interval_min = spec.interval_min

            # Parse portfolios from configuration
            portfolios = []
            if spec.portfolios:
                for portfolio_config in spec.portfolios:  # type: ignore[assignment]
                    portfolio = EnvironmentConfigFactory._create_portfolio_from_config(
                        portfolio_config, start_date_time, end_date_time
                    )
                    portfolios.append(portfolio)
            else:
                raise ValueError("No portfolios specified in configuration")

            # Global market at env level is optional in the new design; each
            # portfolio provides its own MarketSpec. Keep None here.
            price_provider = None

            # Create agent from configuration (optional)
            agent = None
            if spec.agent:
                agent = EnvironmentConfigFactory._create_agent_from_config(spec.agent)

            # Create environment configuration
            env_config = EnvironmentConfig(
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                interval_min=interval_min,
                portfolios=portfolios,
                price_provider=price_provider,
                historic_data_intervals=spec.historic_data_intervals,
                forecast_data_intervals=spec.forecast_data_intervals,
                power_normalization_coefficient=spec.power_normalization_coefficient,
                price_normalization_coefficient=spec.price_normalization_coefficient,
                smoothed_reward_parameter=spec.smoothed_reward_parameter,
                action_tolerance_percent=spec.action_tolerance_percent,
                max_grid_purchase_mw=spec.max_grid_purchase_mw,
                grid_purchase_enabled=spec.grid_purchase_enabled,
                agent=agent,
            )

            return env_config

        except Exception as e:
            logger.error(f"Error creating environment config: {e}")
            raise ValueError(f"Invalid environment configuration: {e}") from e

    @staticmethod
    def _create_portfolio_from_config(
        portfolio_config: Any,
        start_date_time: datetime.datetime,
        end_date_time: datetime.datetime,
    ) -> PowerPlantPortfolio:
        """Create a portfolio from configuration dictionary."""
        from app.core.environment.spec_models import PortfolioSpec
        from app.core.simulation.portfolio import PortfolioConfiguration

        # Validate spec strictly
        if not isinstance(portfolio_config, PortfolioSpec):
            spec = PortfolioSpec.model_validate(portfolio_config)
        else:
            spec = portfolio_config

        # Create portfolio configuration
        portfolio_cfg = PortfolioConfiguration(
            name=spec.name,
            max_total_power_mw=spec.max_total_power_mw,
            allow_grid_purchase=spec.allow_grid_purchase,
        )

        # Create plants from configuration
        plants = []
        if spec.plants:
            market_cfg = spec.market
            for plant_config in spec.plants:
                plant = EnvironmentConfigFactory._create_plant_from_config(
                    plant_config,
                    start_date_time,
                    end_date_time,
                    market_cfg,
                )
                plants.append(plant)

        return PowerPlantPortfolio(config=portfolio_cfg, plants=plants)

    @staticmethod
    def _create_plant_from_config(
        plant_config: Any,
        start_date_time: datetime.datetime,
        end_date_time: datetime.datetime,
        market_config: Optional[Any] = None,
    ):
        """Create a plant from configuration dictionary (strict path only)."""
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
        location_cfg = spec.location  # type: ignore[assignment]
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

        # Create weather provider using provider system
        weather_provider = (
            EnvironmentConfigFactory._create_weather_provider_from_config(
                market_config, geo, start_date_time, end_date_time
            )
        )

        # Create PVLib model using only expected keys
        pvlib_input = {
            "location": spec.location,
            "pv_systems": spec.pv_systems,
            # Add default physical simulation parameters to ensure AOI model is set
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

        # Price provider from market config (required)
        if not market_config:
            raise ValueError(
                "Market configuration is required to create price provider for plants"
            )
        price_provider = EnvironmentConfigFactory._create_price_provider_from_config(
            market_config,
            start_date_time,
            end_date_time,
        )

        # Revenue calculator
        revenue_calc = SolarRevenueCalculator(
            price_provider=price_provider, pv_model=pv_model
        )

        return SolarBatteryPlant(
            config=plant_cfg,
            pv_model=pv_model,
            batteries=batteries,
            revenue_calculator=revenue_calc,
        )

    @staticmethod
    def _create_price_provider_from_config(
        market_config: Any,
        start_date_time: datetime.datetime,
        end_date_time: datetime.datetime,
    ):
        """Create a price provider from market configuration."""
        from app.core.simulation.price_provider import CSVPriceProvider
        from app.core.simulation.provider_config import (
            create_default_price_provider,
            create_price_provider_from_config,
        )

        # Try to get price configuration
        price_config = None
        prices_list = None

        # Try spec object access
        try:
            prices_list = getattr(market_config, "prices", None)
        except Exception:
            prices_list = None

        # If not spec object, try dict access
        if prices_list is None:
            try:
                prices_list = market_config.get("prices", [])
            except Exception:
                prices_list = []

        if prices_list and len(prices_list) > 0:
            price_config = prices_list[0]
        else:
            # Fallback to default provider
            logger.warning("No price configuration found, using default CSV provider")
            return create_default_price_provider(
                location=None,  # Not used for CSV provider
                start_date_time=start_date_time,
                end_date_time=end_date_time,
            )

        # Check if it's an old CSV format or new provider config
        if hasattr(price_config, "type") or (
            isinstance(price_config, dict) and "type" in price_config
        ):
            config_type = getattr(price_config, "type", None) or price_config.get(
                "type"
            )

            if config_type == "csv_file":
                # Handle old CSV format
                data_path = getattr(price_config, "data", None) or price_config.get(
                    "data"
                )
                if not data_path:
                    raise ValueError(
                        "Price data path not provided in market configuration"
                    )

                price_provider = CSVPriceProvider(csv_file_path=data_path)
                price_provider.set_range(start_date_time, end_date_time)
                return price_provider
            else:
                # Handle new provider configurations (CAISO, IESO)
                try:
                    # Create a dummy location for non-CSV providers
                    from app.core.utils.location import GeospatialLocation

                    dummy_location = GeospatialLocation(
                        latitude=37.7749, longitude=-122.4194
                    )

                    return create_price_provider_from_config(
                        config=price_config,
                        location=dummy_location,
                        start_date_time=start_date_time,
                        end_date_time=end_date_time,
                    )
                except Exception as e:
                    logger.error(f"Failed to create price provider from config: {e}")
                    # Fallback to default
                    return create_default_price_provider(
                        location=None,
                        start_date_time=start_date_time,
                        end_date_time=end_date_time,
                    )
        else:
            # Old format or unknown, fallback to old CSV behavior
            # Try to extract data path for backwards compatibility
            data_path = None
            try:
                data_path = getattr(price_config, "data", None) or price_config.get(
                    "data"
                )
            except Exception:
                pass

            if data_path:
                price_provider = CSVPriceProvider(csv_file_path=data_path)
                price_provider.set_range(start_date_time, end_date_time)
                return price_provider
            else:
                logger.warning(
                    "Unknown price configuration format, using default CSV provider"
                )
                return create_default_price_provider(
                    location=None,
                    start_date_time=start_date_time,
                    end_date_time=end_date_time,
                )

    @staticmethod
    def _create_weather_provider_from_config(
        market_config: Any,
        location: Any,
        start_date_time: datetime.datetime,
        end_date_time: datetime.datetime,
    ):
        """Create a weather provider from market configuration using provider system."""
        from app.core.simulation.provider_config import (
            create_default_weather_provider,
            create_weather_provider_from_config,
        )
        from app.core.utils.location import GeospatialLocation

        # Extract geospatial location
        geo = location
        if not isinstance(location, GeospatialLocation):
            # Convert from dict-like to GeospatialLocation
            latitude = (
                location["latitude"]
                if isinstance(location, dict)
                else getattr(location, "latitude", None)
            )
            longitude = (
                location["longitude"]
                if isinstance(location, dict)
                else getattr(location, "longitude", None)
            )
            if latitude is None or longitude is None:
                raise ValueError("Plant location must include latitude and longitude")
            geo = GeospatialLocation(latitude=latitude, longitude=longitude)

        # Get weather configuration
        weather_config = None

        # Try spec object access
        try:
            weather_config = getattr(market_config, "weather", None)
        except Exception:
            weather_config = None

        # If not spec object, try dict access
        if weather_config is None:
            try:
                weather_config = market_config.get("weather")
            except Exception:
                weather_config = None

        if not weather_config:
            logger.warning("No weather configuration found, using default CSV provider")
            return create_default_weather_provider(
                location=geo,
                start_date_time=start_date_time,
                end_date_time=end_date_time,
            )

        # Check if it's an old CSV format or new provider config
        config_type = None
        try:
            config_type = getattr(weather_config, "type", None)
            if config_type is None and isinstance(weather_config, dict):
                config_type = weather_config.get("type")
        except Exception:
            pass

        if config_type == "csv_file":
            # Handle old CSV format
            from app.core.simulation.weather_provider import CSVWeatherProvider

            data_path = None
            try:
                data_path = getattr(weather_config, "data", None)
                if data_path is None and isinstance(weather_config, dict):
                    data_path = weather_config.get("data")
            except Exception:
                pass

            if not data_path:
                raise ValueError(
                    "Weather data path not provided in market configuration"
                )

            weather_provider = CSVWeatherProvider(location=geo, file_path=data_path)
            weather_provider.set_range(start_date_time, end_date_time)
            return weather_provider

        elif config_type in ["openmeteo"]:
            # Handle new provider configurations
            try:
                # Convert dict to proper config model
                from app.core.simulation.provider_config import (
                    OpenMeteoWeatherProviderConfig,
                )

                if isinstance(weather_config, dict):
                    config_obj = OpenMeteoWeatherProviderConfig(**weather_config)
                else:
                    # Try to convert object to dict
                    config_dict = {
                        "type": getattr(weather_config, "type", "openmeteo"),
                        "organization": getattr(
                            weather_config, "organization", "SolarRevenue"
                        ),
                        "asset": getattr(weather_config, "asset", "Weather"),
                        "data_type": getattr(weather_config, "data_type", "weather"),
                        "fetch_all_radiation": getattr(
                            weather_config, "fetch_all_radiation", False
                        ),
                    }
                    config_obj = OpenMeteoWeatherProviderConfig(**config_dict)

                return create_weather_provider_from_config(
                    config=config_obj,
                    location=geo,
                    start_date_time=start_date_time,
                    end_date_time=end_date_time,
                )
            except Exception as e:
                logger.error(f"Failed to create weather provider from config: {e}")
                # Fallback to default
                return create_default_weather_provider(
                    location=geo,
                    start_date_time=start_date_time,
                    end_date_time=end_date_time,
                )
        else:
            # Old format or unknown, try to extract data path for backwards
            # compatibility
            data_path = None
            try:
                data_path = getattr(weather_config, "data", None)
                if data_path is None and isinstance(weather_config, dict):
                    data_path = weather_config.get("data")
            except Exception:
                pass

            if data_path:
                from app.core.simulation.weather_provider import CSVWeatherProvider

                weather_provider = CSVWeatherProvider(location=geo, file_path=data_path)
                weather_provider.set_range(start_date_time, end_date_time)
                return weather_provider
            else:
                logger.warning(
                    "Unknown weather configuration format, using default CSV provider"
                )
                return create_default_weather_provider(
                    location=geo,
                    start_date_time=start_date_time,
                    end_date_time=end_date_time,
                )

    @staticmethod
    def _create_agent_from_config(agent_spec: Any) -> Optional[Actor]:
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
