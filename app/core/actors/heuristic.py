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
from app.core.environment.observations import ObservationName
from app.core.utils.logging import get_logger

from .base import Actor

logger = get_logger("basic_heuristic")


class BasicHeuristic(Actor):
    """
    Basic heuristic actor that maximizes revenue using price forecasts
    and battery state information.

    Key features:
    - Avoids selling power to grid during negative pricing periods
    - Prioritizes battery charging from generation during negative prices
    - Uses price percentiles for charging/discharging decisions
    - Implements SOC safety buffers to protect battery health
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
    ) -> Dict[str, Dict[str, float]]:
        """
        Get optimal actions based on heuristic strategy.

        Args:
            observation: Observation from environment
            timestamp: Current timestamp (unused but kept for compatibility)

        Returns:
            Dictionary of actions in format:
            {
                "portfolio_name": {
                    "plant_name": {
                        "ac_power_generation_target_mw": float,
                        "battery_power_target_mw": float,  # +discharge, -charge
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

        # Get price information first (needed for all decisions)
        current_price = market_obs.get(ObservationName.CURRENT_PRICE, np.array([50.0]))[
            0
        ]

        # Get battery information
        battery_capacity = plant_obs.get(
            ObservationName.BATTERY_ENERGY_CAPACITY, np.array([0.0])
        )[0]

        if battery_capacity <= 0:
            # No battery - avoid selling at negative prices
            generation_target = ac_potential if current_price >= 0 else 0.0

            if current_price < 0:
                logger.info(
                    f"No battery and negative price ({current_price:.2f} $/MWh): "
                    f"Curtailing all generation"
                )

            return {
                ActionName.AC_POWER_GENERATION_TARGET.value: generation_target,
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

        # Determine generation target based on price and battery capacity
        generation_target = self._determine_generation_target(
            ac_potential=ac_potential,
            current_price=current_price,
            battery_action=battery_action,
            max_battery_power=max_battery_power,
            current_soc=current_soc,
            max_soc=max_soc - self.soc_buffer,
        )

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

        # Special case: negative prices - charge aggressively if possible
        if current_price < 0 and current_soc < max_soc:
            logger.info(
                f"Negative price ({current_price:.2f} $/MWh): "
                f"Aggressive battery charging"
            )
            return -max_battery_power * 0.9  # Aggressive charging at negative prices

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

    def _determine_generation_target(
        self,
        ac_potential: float,
        current_price: float,
        battery_action: float,
        max_battery_power: float,
        current_soc: float,
        max_soc: float,
    ) -> float:
        """
        Determine optimal generation target based on price and battery considerations.

        Args:
            ac_potential: Maximum AC generation potential (MW)
            current_price: Current electricity price ($/MWh)
            battery_action: Planned battery action (negative=charge, positive=discharge)
            max_battery_power: Maximum battery power (MW)
            current_soc: Current state of charge
            max_soc: Maximum safe SOC

        Returns:
            Generation target (MW)
        """
        # If price is negative, avoid selling power to grid
        if current_price < 0:
            # Prioritize charging battery from generation instead of selling to grid
            if battery_action < 0:  # If we're planning to charge
                # Generate only enough to charge the battery (if possible)
                charge_power = abs(battery_action)

                # Check if we can charge more (SOC limit)
                if current_soc < max_soc:
                    # Generate up to what we can use for charging,
                    # but not more than potential
                    generation_target = min(ac_potential, charge_power)
                else:
                    # Battery is full, don't generate at negative prices
                    generation_target = 0.0
            else:
                # Not charging battery and price is negative - don't generate
                generation_target = 0.0

            logger.info(
                f"Negative price ({current_price:.2f} $/MWh): "
                f"Curtailing generation to {generation_target:.2f}MW "
                f"(potential: {ac_potential:.2f}MW)"
            )
            return generation_target

        # For positive prices, generate at full potential
        return ac_potential

    def _get_default_action(
        self, observation: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> Dict[str, Dict[str, float]]:
        """Get default action in case of errors."""
        actions = {}

        try:
            # Get current price to check for negative pricing
            current_price = 50.0  # Default fallback
            if "market" in observation and "market_data" in observation["market"]:
                market_data = observation["market"]["market_data"]
                if ObservationName.CURRENT_PRICE in market_data:
                    price_array = market_data[ObservationName.CURRENT_PRICE]
                    if len(price_array) > 0:
                        current_price = float(price_array[0])

            for portfolio_name, portfolio_obs in observation["portfolios"].items():
                actions[portfolio_name] = {}

                for plant_name, plant_obs in portfolio_obs.items():
                    # Default: check if plant_obs is dict-like
                    if isinstance(plant_obs, dict):
                        ac_potential = plant_obs.get(
                            ObservationName.AC_POWER_GENERATION_POTENTIAL,
                            np.array([0.0]),
                        )[0]
                    else:
                        # Fallback if plant_obs is not a dict
                        ac_potential = 0.0

                    # Don't generate at negative prices by default
                    generation_target = ac_potential if current_price >= 0 else 0.0

                    actions[portfolio_name][plant_name] = {
                        ActionName.AC_POWER_GENERATION_TARGET.value: generation_target,
                        ActionName.BATTERY_POWER_TARGET.value: 0.0,
                    }

        except Exception as e:
            logger.warning(f"Error in default action generation: {e}")
            # Ultra-safe fallback
            actions = {}

        return actions
