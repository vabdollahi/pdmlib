"""
Observation definitions for power management environments.
"""

import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core.simulation.portfolio import PowerPlantPortfolio
from app.core.utils.logging import get_logger

logger = get_logger("environment_observations")

# Observation types
ObservationType = Dict[str, Dict[str, np.ndarray]]


class ObservationName:
    """Available observation types."""

    # Plant observations
    PV_GENERATION_POTENTIAL = "pv_generation_potential_mw"
    PV_GENERATION_CURRENT = "pv_generation_current_mw"
    BATTERY_SOC = "battery_soc"
    BATTERY_POWER_AVAILABLE = "battery_power_available_mw"
    BATTERY_ENERGY_AVAILABLE = "battery_energy_available_mwh"
    NET_POWER_CAPABILITY = "net_power_capability_mw"

    # Market observations
    CURRENT_PRICE = "current_price_dollar_mwh"
    PRICE_FORECAST = "price_forecast_dollar_mwh"
    PRICE_HISTORY = "price_history_dollar_mwh"

    # Time-based observations
    HOUR_OF_DAY = "hour_of_day"
    DAY_OF_WEEK = "day_of_week"
    SEASON = "season"


class ObservationTemplate(BaseModel):
    """Template for creating observation data instances."""

    model_config = ConfigDict(use_enum_values=True)

    data_name: str = Field(description="Name of the observation")
    shape: tuple = Field(description="Expected shape of the observation")
    normalization_factor: float = Field(
        default=1.0, description="Factor to normalize the observation"
    )

    def create_data_instance(self, values) -> "ObservationData":
        """Create an observation data instance."""
        # Convert to numpy array and normalize
        if isinstance(values, (int, float)):
            array_values = np.array([values], dtype=np.float32)
        else:
            array_values = np.array(values, dtype=np.float32)

        # Normalize
        normalized_values = array_values / self.normalization_factor

        return ObservationData(
            name=self.data_name,
            values=normalized_values,
        )


class ObservationData(BaseModel):
    """Container for observation data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Observation name")
    values: np.ndarray = Field(description="Observation values")


class ObservationFactory:
    """Factory for creating observations from power plant portfolios."""

    def __init__(
        self,
        portfolios: List[PowerPlantPortfolio],
        historic_data_intervals: int = 12,
        forecast_data_intervals: int = 12,
        power_normalization_coefficient: float = 1e6,
        price_normalization_coefficient: float = 100.0,
        interval_min: float = 5.0,
    ):
        """
        Initialize observation factory.

        Args:
            portfolios: List of power plant portfolios
            historic_data_intervals: Number of historic intervals
            forecast_data_intervals: Number of forecast intervals
            power_normalization_coefficient: Power normalization factor
            price_normalization_coefficient: Price normalization factor
            interval_min: Time interval in minutes
        """
        self.portfolios = portfolios
        self.historic_data_intervals = historic_data_intervals
        self.forecast_data_intervals = forecast_data_intervals
        self.power_normalization_coefficient = power_normalization_coefficient
        self.price_normalization_coefficient = price_normalization_coefficient
        self.interval_min = interval_min
        self.interval = datetime.timedelta(minutes=interval_min)

    async def create_observation(self, timestamp: datetime.datetime) -> ObservationType:
        """
        Create observation at a given timestamp.

        Args:
            timestamp: Current timestamp

        Returns:
            Dictionary of observations organized by portfolio and data type
        """
        observation = {}

        # Create time indices for historic and forecast data
        historical_indices = self._create_historical_indices(timestamp)
        forecast_indices = self._create_forecast_indices(timestamp)

        for portfolio in self.portfolios:
            portfolio_observation = {}

            # Get plant observations
            for plant in portfolio.plants:
                plant_observation = await self._create_plant_observations(
                    plant, timestamp, historical_indices, forecast_indices
                )
                portfolio_observation[plant.config.name] = plant_observation

            # Get market observations (placeholder - would need market integration)
            market_observation = self._create_market_observations(
                timestamp, historical_indices, forecast_indices
            )
            portfolio_observation["market"] = market_observation

            observation[portfolio.config.name] = portfolio_observation

        return observation

    def _create_historical_indices(
        self, timestamp: datetime.datetime
    ) -> pd.DatetimeIndex:
        """Create historical time indices."""
        if self.historic_data_intervals <= 0:
            return pd.DatetimeIndex(data=pd.to_datetime([]), tz=timestamp.tzinfo)

        start_time = timestamp - self.historic_data_intervals * self.interval
        end_time = timestamp - self.interval

        return pd.date_range(
            start=start_time,
            end=end_time,
            freq=f"{self.interval_min}min",
            tz=timestamp.tzinfo,
        )

    def _create_forecast_indices(
        self, timestamp: datetime.datetime
    ) -> pd.DatetimeIndex:
        """Create forecast time indices."""
        if self.forecast_data_intervals <= 0:
            return pd.DatetimeIndex(data=pd.to_datetime([]), tz=timestamp.tzinfo)

        start_time = timestamp + self.interval
        end_time = timestamp + self.forecast_data_intervals * self.interval

        return pd.date_range(
            start=start_time,
            end=end_time,
            freq=f"{self.interval_min}min",
            tz=timestamp.tzinfo,
        )

    async def _create_plant_observations(
        self,
        plant,
        timestamp: datetime.datetime,
        historical_indices: pd.DatetimeIndex,
        forecast_indices: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        """Create observations for a single plant."""
        plant_obs = {}

        try:
            # Current PV generation potential
            pv_potential = await plant.get_pv_generation_potential(timestamp)
            plant_obs[ObservationName.PV_GENERATION_POTENTIAL] = np.array(
                [pv_potential / self.power_normalization_coefficient]
            )

            # Battery state information
            if plant.batteries:
                avg_soc = plant.average_battery_soc
                total_available_charge, total_available_discharge = (
                    plant.get_battery_available_power(self.interval_min)
                )

                plant_obs[ObservationName.BATTERY_SOC] = np.array([avg_soc])
                plant_obs[ObservationName.BATTERY_POWER_AVAILABLE] = np.array(
                    [
                        total_available_discharge
                        / self.power_normalization_coefficient,
                        total_available_charge / self.power_normalization_coefficient,
                    ]
                )
                plant_obs[ObservationName.BATTERY_ENERGY_AVAILABLE] = np.array(
                    [plant.total_battery_capacity_mwh / 100.0]  # Normalize MWh
                )
            else:
                # No battery case
                plant_obs[ObservationName.BATTERY_SOC] = np.array([0.0])
                plant_obs[ObservationName.BATTERY_POWER_AVAILABLE] = np.array(
                    [0.0, 0.0]
                )
                plant_obs[ObservationName.BATTERY_ENERGY_AVAILABLE] = np.array([0.0])

            # Net power capability
            max_gen, max_cons = await plant.get_available_power(
                timestamp, self.interval_min
            )
            plant_obs[ObservationName.NET_POWER_CAPABILITY] = np.array(
                [
                    max_gen / self.power_normalization_coefficient,
                    max_cons / self.power_normalization_coefficient,
                ]
            )

            # Historic PV generation (if available)
            if len(historical_indices) > 0:
                # This would require implementing historical PV data access
                # For now, use zeros as placeholder
                historic_pv = np.zeros(len(historical_indices))
                plant_obs["pv_generation_history"] = historic_pv

            # Forecast PV generation (if available)
            if len(forecast_indices) > 0:
                # This would require implementing forecast PV data access
                # For now, use zeros as placeholder
                forecast_pv = np.zeros(len(forecast_indices))
                plant_obs["pv_generation_forecast"] = forecast_pv

        except Exception as e:
            logger.error(f"Error creating plant observations: {e}")
            # Return default observations on error
            plant_obs = self._get_default_plant_observations()

        return plant_obs

    def _create_market_observations(
        self,
        timestamp: datetime.datetime,
        historical_indices: pd.DatetimeIndex,
        forecast_indices: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        """Create market-related observations."""
        market_obs = {}

        try:
            # Time-based features
            market_obs[ObservationName.HOUR_OF_DAY] = np.array(
                [timestamp.hour / 24.0]
            )  # Normalized 0-1
            market_obs[ObservationName.DAY_OF_WEEK] = np.array(
                [timestamp.weekday() / 6.0]
            )  # Normalized 0-1

            # Season (simplified)
            month = timestamp.month
            if month in [12, 1, 2]:
                season = 0.0  # Winter
            elif month in [3, 4, 5]:
                season = 0.25  # Spring
            elif month in [6, 7, 8]:
                season = 0.5  # Summer
            else:
                season = 0.75  # Fall
            market_obs[ObservationName.SEASON] = np.array([season])

            # Placeholder price observations
            # These would need to be integrated with actual price providers
            market_obs[ObservationName.CURRENT_PRICE] = np.array([50.0 / 100.0])

            if len(historical_indices) > 0:
                market_obs[ObservationName.PRICE_HISTORY] = np.full(
                    len(historical_indices), 50.0 / 100.0
                )

            if len(forecast_indices) > 0:
                market_obs[ObservationName.PRICE_FORECAST] = np.full(
                    len(forecast_indices), 50.0 / 100.0
                )

        except Exception as e:
            logger.error(f"Error creating market observations: {e}")
            market_obs = self._get_default_market_observations()

        return market_obs

    def _get_default_plant_observations(self) -> Dict[str, np.ndarray]:
        """Get default plant observations for error cases."""
        return {
            ObservationName.PV_GENERATION_POTENTIAL: np.array([0.0]),
            ObservationName.BATTERY_SOC: np.array([0.5]),
            ObservationName.BATTERY_POWER_AVAILABLE: np.array([0.0, 0.0]),
            ObservationName.BATTERY_ENERGY_AVAILABLE: np.array([0.0]),
            ObservationName.NET_POWER_CAPABILITY: np.array([0.0, 0.0]),
        }

    def _get_default_market_observations(self) -> Dict[str, np.ndarray]:
        """Get default market observations for error cases."""
        return {
            ObservationName.HOUR_OF_DAY: np.array([0.5]),
            ObservationName.DAY_OF_WEEK: np.array([0.5]),
            ObservationName.SEASON: np.array([0.5]),
            ObservationName.CURRENT_PRICE: np.array([0.5]),
        }


def flatten_observation(observation: ObservationType) -> np.ndarray:
    """
    Flatten hierarchical observation structure to 1D array.

    Args:
        observation: Hierarchical observation dict

    Returns:
        Flattened numpy array suitable for ML models
    """
    flattened_data = []

    for portfolio_name, portfolio_data in observation.items():
        for component_name, component_data in portfolio_data.items():
            if isinstance(component_data, dict):
                for obs_name, obs_values in component_data.items():
                    # Ensure values are numpy arrays and flatten
                    if isinstance(obs_values, np.ndarray):
                        flattened_data.extend(obs_values.flatten())
                    else:
                        # Convert scalars to arrays
                        flattened_data.extend(np.array([obs_values]).flatten())
            else:
                # Handle non-dict component data
                if isinstance(component_data, np.ndarray):
                    flattened_data.extend(component_data.flatten())
                else:
                    flattened_data.extend(np.array([component_data]).flatten())

    return np.array(flattened_data, dtype=np.float32)
