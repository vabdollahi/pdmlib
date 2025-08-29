"""
Observation definitions for power management environments.
"""

import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from app.core.simulation.portfolio import PowerPlantPortfolio
from app.core.utils.logging import get_logger

logger = get_logger("environment_observations")

# Observation types
ObservationType = Dict[str, Dict[str, np.ndarray]]


class ObservationName:
    """Available observation types."""

    # Plant observations for actors
    DC_POWER_GENERATION_POTENTIAL = "dc_power_generation_potential_mw"
    AC_POWER_GENERATION_POTENTIAL = "ac_power_generation_potential_mw"
    AC_POWER_GENERATION_POTENTIAL_FORECAST = "ac_power_generation_potential_forecast_mw"
    MAX_AC_POWER_TO_GRID = "max_ac_power_to_grid_mw"
    INVERTER_DC_TO_AC_CONVERSION_EFFICIENCY = "inverter_dc_to_ac_efficiency"

    # Battery observations for actors
    BATTERY_ENERGY_CAPACITY = "battery_energy_capacity_mwh"
    BATTERY_MAX_CHARGE_POWER = "battery_max_charge_power_mw"
    BATTERY_MAX_DISCHARGE_POWER = "battery_max_discharge_power_mw"
    BATTERY_STATE_OF_CHARGE = "battery_state_of_charge"
    BATTERY_MIN_STATE_OF_CHARGE = "battery_min_state_of_charge"
    BATTERY_MAX_STATE_OF_CHARGE = "battery_max_state_of_charge"

    # Market observations for actors
    CURRENT_PRICE = "current_price_dollar_mwh"
    PRICE_FORECAST = "price_forecast_dollar_mwh"
    PRICE_HISTORY = "price_history_dollar_mwh"

    # Time-based observations
    HOUR_OF_DAY = "hour_of_day"
    DAY_OF_WEEK = "day_of_week"
    SEASON = "season"


class ObservationFactory:
    """Factory for creating observations from power plant portfolios."""

    def __init__(
        self,
        portfolios: List[PowerPlantPortfolio],
        historic_data_intervals: int = 12,
        forecast_data_intervals: int = 12,
        power_normalization_coefficient: float = 1e6,
        price_normalization_coefficient: float = 100.0,
        interval_min: float = 60.0,
        market_data=None,
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
            market_data: Market data object for price information
        """
        self.portfolios = portfolios
        self.historic_data_intervals = historic_data_intervals
        self.forecast_data_intervals = forecast_data_intervals
        self.power_normalization_coefficient = power_normalization_coefficient
        self.price_normalization_coefficient = price_normalization_coefficient
        self.interval_min = interval_min
        self.interval = datetime.timedelta(minutes=interval_min)
        self.market_data = market_data

    async def create_observation(
        self, timestamp: datetime.datetime
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Create observations formatted for actor agents.

        Args:
            timestamp: Current timestamp

        Returns:
            Dictionary in format:
            {
                "market": {
                    "market_data": {
                        "current_price_dollar_mwh": np.array([price]),
                        "price_forecast_dollar_mwh": np.array([...]),
                        ...
                    }
                },
                "portfolios": {
                    "portfolio_name": {
                        "plant_name": {
                            "dc_power_generation_potential_mw": np.array([...]),
                            ...
                        }
                    }
                }
            }
        """
        observation = {"market": {}, "portfolios": {}}

        # Create time indices for historic and forecast data
        historical_indices = self._create_historical_indices(timestamp)
        forecast_indices = self._create_forecast_indices(timestamp)

        # Create market observations
        market_obs = await self._create_market_observations(
            timestamp, historical_indices, forecast_indices
        )
        observation["market"] = {"market_data": market_obs}

        # Create portfolio observations
        for portfolio in self.portfolios:
            portfolio_obs = {}

            for plant in portfolio.plants:
                plant_obs = await self._create_plant_observations(
                    plant, timestamp, historical_indices, forecast_indices
                )
                portfolio_obs[plant.config.name] = plant_obs

            observation["portfolios"][portfolio.config.name] = portfolio_obs

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
            # Current PV generation potential (DC)
            dc_potential = await plant.get_pv_generation_potential(timestamp)
            plant_obs[ObservationName.DC_POWER_GENERATION_POTENTIAL] = np.array(
                [dc_potential / self.power_normalization_coefficient]
            )

            # AC generation potential (assuming 95% inverter efficiency as default)
            inverter_efficiency = getattr(plant.config, "inverter_efficiency", 0.95)
            ac_potential = dc_potential * inverter_efficiency
            plant_obs[ObservationName.AC_POWER_GENERATION_POTENTIAL] = np.array(
                [ac_potential / self.power_normalization_coefficient]
            )

            # Inverter efficiency
            plant_obs[ObservationName.INVERTER_DC_TO_AC_CONVERSION_EFFICIENCY] = (
                np.array([inverter_efficiency])
            )

            # Max AC power to grid (from plant configuration)
            max_ac_power = getattr(plant.config, "max_net_power_mw", ac_potential)
            plant_obs[ObservationName.MAX_AC_POWER_TO_GRID] = np.array(
                [max_ac_power / self.power_normalization_coefficient]
            )

            # AC generation forecast
            if len(forecast_indices) > 0:
                # For now, create a simple forecast based on current potential
                # In a real implementation, this would use weather forecasts
                forecast_ac = np.full(len(forecast_indices), ac_potential)
                # Add some variation for realism
                variation = np.random.normal(0, 0.1, len(forecast_indices))
                forecast_ac = forecast_ac * (1 + variation)
                forecast_ac = np.maximum(forecast_ac, 0)  # Ensure non-negative

                plant_obs[ObservationName.AC_POWER_GENERATION_POTENTIAL_FORECAST] = (
                    forecast_ac / self.power_normalization_coefficient
                )
            else:
                plant_obs[ObservationName.AC_POWER_GENERATION_POTENTIAL_FORECAST] = (
                    np.array([])
                )

            # Battery observations
            if plant.batteries:
                battery = plant.batteries[0]  # Assume single battery per plant

                # Battery capacity and power limits
                plant_obs[ObservationName.BATTERY_ENERGY_CAPACITY] = np.array(
                    [battery.config.energy_capacity_mwh]
                )
                plant_obs[ObservationName.BATTERY_MAX_CHARGE_POWER] = np.array(
                    [battery.config.max_power_mw]
                )
                plant_obs[ObservationName.BATTERY_MAX_DISCHARGE_POWER] = np.array(
                    [battery.config.max_power_mw]
                )

                # Battery state of charge
                current_soc = battery.state_of_charge
                plant_obs[ObservationName.BATTERY_STATE_OF_CHARGE] = np.array(
                    [current_soc]
                )

                # SOC limits
                plant_obs[ObservationName.BATTERY_MIN_STATE_OF_CHARGE] = np.array(
                    [battery.config.min_soc]
                )
                plant_obs[ObservationName.BATTERY_MAX_STATE_OF_CHARGE] = np.array(
                    [battery.config.max_soc]
                )
            else:
                # No battery case - set all battery observations to zero
                for battery_obs_name in [
                    ObservationName.BATTERY_ENERGY_CAPACITY,
                    ObservationName.BATTERY_MAX_CHARGE_POWER,
                    ObservationName.BATTERY_MAX_DISCHARGE_POWER,
                    ObservationName.BATTERY_STATE_OF_CHARGE,
                    ObservationName.BATTERY_MIN_STATE_OF_CHARGE,
                    ObservationName.BATTERY_MAX_STATE_OF_CHARGE,
                ]:
                    plant_obs[battery_obs_name] = np.array([0.0])

        except Exception as e:
            logger.error(f"Error creating plant observations: {e}")
            # Return empty observations on error
            plant_obs = {
                ObservationName.DC_POWER_GENERATION_POTENTIAL: np.array([0.0]),
                ObservationName.AC_POWER_GENERATION_POTENTIAL: np.array([0.0]),
                ObservationName.AC_POWER_GENERATION_POTENTIAL_FORECAST: np.array([]),
                ObservationName.MAX_AC_POWER_TO_GRID: np.array([0.0]),
                ObservationName.INVERTER_DC_TO_AC_CONVERSION_EFFICIENCY: (
                    np.array([0.0])
                ),
                ObservationName.BATTERY_ENERGY_CAPACITY: np.array([0.0]),
                ObservationName.BATTERY_MAX_CHARGE_POWER: np.array([0.0]),
                ObservationName.BATTERY_MAX_DISCHARGE_POWER: np.array([0.0]),
                ObservationName.BATTERY_STATE_OF_CHARGE: np.array([0.0]),
                ObservationName.BATTERY_MIN_STATE_OF_CHARGE: np.array([0.0]),
                ObservationName.BATTERY_MAX_STATE_OF_CHARGE: np.array([0.0]),
            }

        return plant_obs

    async def _create_market_observations(
        self,
        timestamp: datetime.datetime,
        historical_indices: pd.DatetimeIndex,
        forecast_indices: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        """Create market observations."""
        market_obs = {}

        try:
            # Get current price from market data
            if not self.market_data:
                raise ValueError("No market data available for price lookup")

            # The `market_data` object should conform to the PriceProviderProtocol
            market_price = await self.market_data.get_price_at_time(timestamp)

            if market_price is None:
                raise ValueError(f"No price data available for {timestamp}")

            current_price = market_price

            # Normalize price using configured coefficient
            normalized_price = current_price / self.price_normalization_coefficient
            market_obs[ObservationName.CURRENT_PRICE] = np.array([normalized_price])

            # Price history
            if len(historical_indices) > 0:
                # Simple historical price simulation
                hist_prices = np.full(len(historical_indices), current_price)
                # Add some variation
                variation = np.random.normal(0, 5, len(historical_indices))
                hist_prices = hist_prices + variation
                hist_prices = np.maximum(hist_prices, 0)  # Ensure non-negative

                market_obs[ObservationName.PRICE_HISTORY] = hist_prices
            else:
                market_obs[ObservationName.PRICE_HISTORY] = np.array([])

            # Price forecast
            if len(forecast_indices) > 0:
                # Simple price forecast simulation
                forecast_prices = np.full(len(forecast_indices), current_price)
                # Add trend and variation
                trend = np.linspace(0, 10, len(forecast_indices))  # Slight upward trend
                variation = np.random.normal(0, 3, len(forecast_indices))
                forecast_prices = forecast_prices + trend + variation
                forecast_prices = np.maximum(forecast_prices, 0)  # Ensure non-negative

                market_obs[ObservationName.PRICE_FORECAST] = forecast_prices
            else:
                market_obs[ObservationName.PRICE_FORECAST] = np.array([])

        except Exception as e:
            logger.error(f"Error creating market observations: {e}")
            raise RuntimeError(f"Failed to create market observations: {e}")

        return market_obs


def flatten_observation(
    observation: Dict[str, Dict[str, Dict[str, np.ndarray]]],
) -> np.ndarray:
    """
    Flatten hierarchical observation structure to 1D array.

    Args:
        observation: Hierarchical observation dict in format:
        {
            "market": { "market_data": {...} },
            "portfolios": { "portfolio_name": { "plant_name": {...} } }
        }

    Returns:
        Flattened numpy array suitable for ML models
    """
    flattened_data = []

    # Process market data
    market_data = observation.get("market", {}).get("market_data", {})
    for obs_name, obs_values in market_data.items():
        if obs_values is None:
            continue

        if isinstance(obs_values, np.ndarray):
            flattened_data.extend(obs_values.flatten())
        elif isinstance(obs_values, (int, float)):
            flattened_data.extend(np.array([obs_values]).flatten())
        else:
            try:
                flattened_data.extend(np.array([obs_values]).flatten())
            except (ValueError, TypeError):
                continue

    # Process portfolio data
    portfolios = observation.get("portfolios", {})
    for portfolio_name, portfolio_data in portfolios.items():
        for plant_name, plant_data in portfolio_data.items():
            if isinstance(plant_data, dict):
                for obs_name, obs_values in plant_data.items():
                    if obs_values is None:
                        continue

                    if isinstance(obs_values, np.ndarray):
                        flattened_data.extend(obs_values.flatten())
                    elif isinstance(obs_values, (int, float)):
                        flattened_data.extend(np.array([obs_values]).flatten())
                    else:
                        try:
                            flattened_data.extend(np.array([obs_values]).flatten())
                        except (ValueError, TypeError):
                            continue

    return np.array(flattened_data, dtype=np.float32)
