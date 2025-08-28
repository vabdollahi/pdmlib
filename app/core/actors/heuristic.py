"""
Basic heuristic actor for power management optimization.

This actor implements a simple revenue-maximizing heuristic that looks at
current and forecast prices, battery state, and generation potential to
make optimal power management decisions.
"""

import datetime
from typing import Dict, Optional

import numpy as np

from app.core.environment.actions import ActionName
from app.core.environment.observations import ObservationFactory, ObservationName
from app.core.utils.logging import get_logger

from .base import Actor

logger = get_logger("basic_heuristic")


class BasicHeuristic(Actor):
    """
    Basic heuristic actor that maximizes revenue using price forecasts
    and battery state information.
    """

    def __init__(
        self,
        max_lookahead_steps: int = 12,  # 1 hour at 5-min intervals
        charge_threshold_ratio: float = 0.3,  # Below 30th percentile
        discharge_threshold_ratio: float = 0.7,  # Above 70th percentile
        soc_buffer: float = 0.1,  # 10% SOC buffer from limits
    ):
        """
        Initialize basic heuristic actor.

        Args:
            max_lookahead_steps: Maximum number of forecast steps to consider
            charge_threshold_ratio: Price percentile below which to charge
            discharge_threshold_ratio: Price percentile above which to discharge
            soc_buffer: SOC buffer from min/max limits
        """
        self.max_lookahead_steps = max_lookahead_steps
        self.charge_threshold_ratio = charge_threshold_ratio
        self.discharge_threshold_ratio = discharge_threshold_ratio
        self.soc_buffer = soc_buffer

    def get_action(
        self,
        observation: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        timestamp: Optional[datetime.datetime] = None,
        observation_factory: Optional[ObservationFactory] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get optimal actions based on heuristic strategy.

        Args:
            observation: Enhanced observation from environment
            observation_factory: Optional observation factory for additional data
            timestamp: Current timestamp

        Returns:
            Dictionary of actions in format:
            {
                "portfolio_name": {
                    "plant_name": {
                        "target_ac_power_generation_mw": float,
                        "target_battery_power_mw": float,  # +discharge, -charge
                    }
                }
            }
        """
        try:
            actions = {}

            # Process each portfolio
            for portfolio_name, portfolio_obs in observation["portfolios"].items():
                actions[portfolio_name] = {}

                # Process each plant in the portfolio
                for plant_name, plant_obs in portfolio_obs.items():
                    plant_action = self._get_plant_action(
                        plant_obs, observation["market"]["market_data"]
                    )
                    actions[portfolio_name][plant_name] = plant_action

            return actions

        except Exception as e:
            logger.error(f"Error getting heuristic action: {e}")
            return self._get_default_action(observation)

    def _get_plant_action(
        self,
        plant_obs: Dict[str, np.ndarray],
        market_obs: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Get optimal action for a single plant.

        Args:
            plant_obs: Plant observations
            market_obs: Market observations

        Returns:
            Plant action dictionary
        """
        # Get current generation potential
        ac_potential = plant_obs.get(
            ObservationName.AC_POWER_GENERATION_POTENTIAL, np.array([0.0])
        )[0]

        # Get battery information
        battery_capacity = plant_obs.get(
            ObservationName.BATTERY_ENERGY_CAPACITY, np.array([0.0])
        )[0]

        if battery_capacity <= 0:
            # No battery - just generate at full potential
            return {
                ActionName.AC_POWER_GENERATION_TARGET.value: ac_potential,
                ActionName.BATTERY_POWER_TARGET.value: 0.0,
            }

        # Get battery state and limits
        current_soc = plant_obs.get(
            ObservationName.BATTERY_STATE_OF_CHARGE, np.array([0.5])
        )[0]

        min_soc = plant_obs.get(
            ObservationName.BATTERY_MIN_STATE_OF_CHARGE, np.array([0.1])
        )[0]

        max_soc = plant_obs.get(
            ObservationName.BATTERY_MAX_STATE_OF_CHARGE, np.array([0.9])
        )[0]

        max_battery_power = plant_obs.get(
            ObservationName.BATTERY_MAX_DISCHARGE_POWER, np.array([0.0])
        )[0]

        # Get price information
        current_price = market_obs.get(ObservationName.CURRENT_PRICE, np.array([50.0]))[
            0
        ]

        price_forecast = market_obs.get(ObservationName.PRICE_FORECAST, np.array([]))

        # Determine battery action based on price analysis
        battery_action = self._determine_battery_action(
            current_price=current_price,
            price_forecast=price_forecast,
            current_soc=current_soc,
            min_soc=min_soc + self.soc_buffer,
            max_soc=max_soc - self.soc_buffer,
            max_battery_power=max_battery_power,
        )

        # Generate at full potential (curtailment can be added later)
        generation_target = ac_potential

        return {
            ActionName.AC_POWER_GENERATION_TARGET.value: generation_target,
            ActionName.BATTERY_POWER_TARGET.value: battery_action,
        }

    def _determine_battery_action(
        self,
        current_price: float,
        price_forecast: np.ndarray,
        current_soc: float,
        min_soc: float,
        max_soc: float,
        max_battery_power: float,
    ) -> float:
        """
        Determine optimal battery action based on price analysis.

        Args:
            current_price: Current electricity price
            price_forecast: Array of forecast prices
            current_soc: Current state of charge
            min_soc: Minimum safe SOC
            max_soc: Maximum safe SOC
            max_battery_power: Maximum battery power (MW)

        Returns:
            Battery power target (positive=discharge, negative=charge)
        """
        # Check SOC limits first
        if current_soc <= min_soc:
            # Must charge - SOC too low
            return -max_battery_power * 0.5  # Moderate charging

        if current_soc >= max_soc:
            # Must discharge - SOC too high
            return max_battery_power * 0.5  # Moderate discharging

        # Use price analysis if we have forecast data
        if len(price_forecast) > 0:
            # Limit lookahead to configured maximum
            forecast_length = min(len(price_forecast), self.max_lookahead_steps)
            relevant_forecast = price_forecast[:forecast_length]

            # Include current price in analysis
            all_prices = np.concatenate([[current_price], relevant_forecast])

            # Calculate price percentiles
            charge_threshold = np.percentile(
                all_prices, self.charge_threshold_ratio * 100
            )
            discharge_threshold = np.percentile(
                all_prices, self.discharge_threshold_ratio * 100
            )

            if current_price <= charge_threshold:
                # Low price - charge battery if SOC allows
                charge_intensity = (charge_threshold - current_price) / charge_threshold
                charge_power = max_battery_power * charge_intensity * 0.8
                return -min(float(charge_power), float(max_battery_power))

            elif current_price >= discharge_threshold:
                # High price - discharge battery if SOC allows
                discharge_intensity = (
                    current_price - discharge_threshold
                ) / discharge_threshold
                discharge_power = max_battery_power * discharge_intensity * 0.8
                return min(float(discharge_power), float(max_battery_power))

        # Default: no battery action
        return 0.0

    def _get_default_action(
        self, observation: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> Dict[str, Dict[str, float]]:
        """Get default action in case of errors."""
        actions = {}

        for portfolio_name, portfolio_obs in observation["portfolios"].items():
            actions[portfolio_name] = {}

            for plant_name, plant_obs in portfolio_obs.items():
                # Default: generate at potential, no battery action
                ac_potential = plant_obs.get(
                    ObservationName.AC_POWER_GENERATION_POTENTIAL, np.array([0.0])
                )[0]

                actions[portfolio_name][plant_name] = {
                    ActionName.AC_POWER_GENERATION_TARGET.value: ac_potential,
                    ActionName.BATTERY_POWER_TARGET.value: 0.0,
                }

        return actions
