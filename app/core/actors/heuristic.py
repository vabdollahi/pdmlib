"""
Heuristic actor for power management optimization.

This actor implements a proper revenue-maximizing heuristic that uses
structured observation data including available generation potential and prices
to make optimal decisions.

Designed for gym compatibility:
- Receives structured dictionary observations
- Parses observations to extract key information
- Returns numpy array actions in [-1, 1] range (environment scales to MW)
"""

import datetime
from typing import Optional

import numpy as np

from app.core.utils.logging import get_logger

from .base import Actor

logger = get_logger("heuristic")


class Heuristic(Actor):
    """
    Intelligent heuristic agent for power management.

    This agent uses structured normalized observations to extract key information:
    - Market prices (normalized by price_normalization_coefficient, unitless ~[-1,1])
    - AC generation potential (normalized, unitless [0,1])
    - Battery state of charge (normalized, unitless [0,1])

    Makes optimal decisions using unitless normalized values and returns
    actions in [-1, 1] range for the environment to scale appropriately.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the heuristic agent.

        Args:
            config: Optional configuration dictionary
            **kwargs: Configuration parameters (charge_threshold_ratio, etc.)
        """
        super().__init__()
        self.config = config or {}
        self.num_plants = 1  # Default, will be updated in configure

        # Heuristic-specific parameters (can be overridden by config or kwargs)
        self.max_lookahead_steps = kwargs.get("max_lookahead_steps", 12)
        self.charge_threshold_ratio = kwargs.get("charge_threshold_ratio", 0.3)
        self.discharge_threshold_ratio = kwargs.get("discharge_threshold_ratio", 0.7)
        self.soc_buffer = kwargs.get("soc_buffer", 0.1)

        # Additional configurable parameters
        self.max_discharge_intensity = kwargs.get("max_discharge_intensity", 0.8)
        self.max_charge_intensity = kwargs.get("max_charge_intensity", 0.3)
        self.price_trend_threshold = kwargs.get("price_trend_threshold", 0.1)
        self.strong_trend_threshold = kwargs.get("strong_trend_threshold", 0.15)
        self.min_solar_generation = kwargs.get("min_solar_generation", 0.3)
        self.default_solar_generation = kwargs.get("default_solar_generation", 0.6)

    def configure(self, config) -> None:
        """Configure the agent with environment parameters."""
        self.config = config

        # Calculate total number of plants for action sizing
        if hasattr(config, "portfolios"):
            # It's an EnvironmentConfig object
            portfolios = config.portfolios
            self.power_normalization_coefficient = (
                config.power_normalization_coefficient
            )
            self.price_normalization_coefficient = (
                config.price_normalization_coefficient
            )
        else:
            # It's a dictionary
            portfolios = config.get("portfolios", [])
            self.power_normalization_coefficient = config.get(
                "power_normalization_coefficient", 1e6
            )
            self.price_normalization_coefficient = config.get(
                "price_normalization_coefficient", 100.0
            )

        self.num_plants = 0
        for portfolio in portfolios:
            if isinstance(portfolio, dict):
                self.num_plants += len(portfolio.get("plants", []))
            else:
                self.num_plants += len(portfolio.plants)

        logger.info(f"Heuristic configured for {self.num_plants} plants")
        logger.info(f"Power normalization: {self.power_normalization_coefficient}")
        logger.info(f"Price normalization: {self.price_normalization_coefficient}")
        logger.info("Heuristic parameters:")
        logger.info(f"  - Max lookahead steps: {self.max_lookahead_steps}")
        logger.info(f"  - Charge threshold: {self.charge_threshold_ratio:.1%}")
        logger.info(f"  - Discharge threshold: {self.discharge_threshold_ratio:.1%}")
        logger.info(f"  - SOC buffer: {self.soc_buffer:.1%}")
        logger.info(f"  - Max discharge intensity: {self.max_discharge_intensity}")
        logger.info(f"  - Max charge intensity: {self.max_charge_intensity}")

    def _parse_structured_observation(self, observation: dict) -> dict:
        """
        Parse structured observation dictionary to extract key information.

        Args:
            observation: Structured observation dict from environment

        Returns:
            Dictionary with parsed values:
            - current_price: Current electricity price (normalized)
            - price_forecast: Array of future price forecasts (normalized)
            - price_history: Array of historical prices (normalized)
            - ac_generation_potential: Normalized AC generation potential [0,1]
            - battery_soc: Battery state of charge [0,1]
        """
        try:
            # Extract market data
            market_data = observation.get("market", {}).get("market_data", {})

            # Current price
            current_price = market_data.get("current_price_dollar_mwh", 0.0)
            if isinstance(current_price, np.ndarray):
                current_price = float(current_price.item())

            # Price forecast array
            price_forecast = market_data.get("price_forecast_dollar_mwh", np.array([]))
            if isinstance(price_forecast, np.ndarray) and len(price_forecast) == 0:
                # Default forecast length from config
                price_forecast = np.array([current_price] * self.max_lookahead_steps)
            elif not isinstance(price_forecast, np.ndarray):
                price_forecast = np.array([current_price] * self.max_lookahead_steps)

            # Price history array
            price_history = market_data.get("price_history_dollar_mwh", np.array([]))
            if isinstance(price_history, np.ndarray) and len(price_history) == 0:
                # Default history length from config
                price_history = np.array([current_price] * self.max_lookahead_steps)
            elif not isinstance(price_history, np.ndarray):
                price_history = np.array([current_price] * self.max_lookahead_steps)

            # Extract portfolio data - get first portfolio's first plant
            portfolios = observation.get("portfolios", {})
            if not portfolios:
                raise ValueError("No portfolio data in observation")

            # Get first portfolio
            portfolio_name = next(iter(portfolios.keys()))
            portfolio_data = portfolios[portfolio_name]

            # Get first plant
            plant_name = next(iter(portfolio_data.keys()))
            plant_data = portfolio_data[plant_name]

            # Extract normalized AC generation potential
            ac_potential_norm = plant_data.get("ac_power_generation_potential_mw", 0.0)
            if isinstance(ac_potential_norm, np.ndarray):
                ac_potential_norm = float(ac_potential_norm.item())

            # Extract battery SOC from plant data
            battery_soc = plant_data.get("battery_state_of_charge", 0.5)
            if isinstance(battery_soc, np.ndarray):
                battery_soc = float(battery_soc.item())

            logger.debug(
                f"Parsed observation - Price: {current_price:.3f} (normalized), "
                f"AC potential: {ac_potential_norm:.3f}, SOC: {battery_soc:.3f}, "
                f"Forecast length: {len(price_forecast)}, "
                f"History length: {len(price_history)}"
            )

            return {
                "current_price": float(current_price),
                "price_forecast": price_forecast,
                "price_history": price_history,
                "ac_generation_potential": float(ac_potential_norm),
                "battery_soc": float(battery_soc),
            }

        except Exception as e:
            logger.error(f"Failed to parse structured observation: {e}")
            raise

    def get_action(
        self,
        observation: dict,
        timestamp: Optional[datetime.datetime] = None,
    ) -> np.ndarray:
        """
        Get optimal actions based on structured observation data.

        Args:
            observation: Structured observation dictionary from environment
            timestamp: Current timestamp (optional)

        Returns:
            Action array with values in [-1, 1] range for each plant
        """
        try:
            # Parse the structured observation to extract key information
            obs_data = self._parse_structured_observation(observation)

            current_price = obs_data["current_price"]
            price_forecast = obs_data["price_forecast"]
            price_history = obs_data["price_history"]
            ac_potential_norm = obs_data["ac_generation_potential"]
            battery_soc = obs_data["battery_soc"]

            # Intelligent forecast-based battery management
            action = self._get_optimal_action(
                current_price,
                price_forecast,
                price_history,
                ac_potential_norm,
                battery_soc,
            )

            # Create action array for all plants
            actions = np.full(self.num_plants, action, dtype=np.float32)

            logger.info(
                f"Heuristic: price={current_price:.3f} (normalized), "
                f"gen_potential={ac_potential_norm:.3f}, "
                f"soc={battery_soc:.3f}, action={action:.3f}"
            )

            return actions

        except Exception as e:
            logger.error(f"Error getting heuristic action: {e}")
            raise

    def _get_optimal_action(
        self,
        current_price: float,
        price_forecast: np.ndarray,
        price_history: np.ndarray,
        ac_potential_norm: float,
        battery_soc: float,
    ) -> float:
        """
        Determine optimal action using forecast-based battery economics.

        Args:
            current_price: Current electricity price (normalized, unitless)
            price_forecast: Array of future price forecasts (normalized, unitless)
            price_history: Array of historical prices (normalized, unitless)
            ac_potential_norm: Normalized AC generation potential [0,1]
            battery_soc: Battery state of charge [0,1]

        Returns:
            Action value in [-1, 1] range
        """
        # Combine price data for analysis
        all_prices = np.concatenate([price_history, [current_price], price_forecast])

        # Calculate dynamic price percentiles based on configuration
        charge_threshold = np.percentile(all_prices, self.charge_threshold_ratio * 100)
        discharge_threshold = np.percentile(
            all_prices, self.discharge_threshold_ratio * 100
        )

        # Analyze price trend from forecast (limit to configured lookahead)
        forecast_window = min(len(price_forecast), self.max_lookahead_steps)
        if forecast_window >= 1:
            # Look at near-term forecast for trend analysis
            near_term_forecast = price_forecast[:forecast_window]
            avg_forecast_price = np.mean(near_term_forecast)
            price_trend = (avg_forecast_price - current_price) / max(
                abs(current_price), 0.01
            )
        else:
            price_trend = 0.0

        # Calculate SOC limits with safety buffer
        min_soc_safe = self.soc_buffer
        max_soc_safe = 1.0 - self.soc_buffer

        # Battery discharge decision (positive action = generation/discharge)
        if current_price >= discharge_threshold:
            # High current price - consider discharging battery
            if battery_soc > min_soc_safe:
                # Calculate discharge intensity based on SOC and price strength
                price_range = max((discharge_threshold - charge_threshold), 0.01)
                price_strength = (current_price - discharge_threshold) / price_range
                soc_range = max_soc_safe - min_soc_safe
                soc_availability = (battery_soc - min_soc_safe) / soc_range
                discharge_intensity = min(
                    self.max_discharge_intensity,
                    soc_availability * (0.5 + 0.5 * price_strength),
                )
                return min(discharge_intensity, ac_potential_norm)
            else:
                # Low SOC but high price - generate available solar only
                return min(0.5, ac_potential_norm)

        elif current_price <= charge_threshold:
            # Low current price - consider charging if forecast shows higher prices
            price_increase_expected = price_trend > self.price_trend_threshold
            if price_increase_expected and battery_soc < max_soc_safe:
                # Charge from grid while generating available solar
                soc_range = max_soc_safe - min_soc_safe
                soc_capacity = (max_soc_safe - battery_soc) / soc_range
                charge_intensity = min(self.max_charge_intensity, soc_capacity * 0.5)
                solar_generation = min(0.4, ac_potential_norm)
                return solar_generation - charge_intensity
            else:
                # Low price but no expected increase or battery full
                # Use available solar generation
                return ac_potential_norm

        else:
            # Medium price range - focus on solar generation with minimal battery action
            if price_trend > self.strong_trend_threshold:
                # Strong upward trend - prepare for discharge
                return min(0.7, ac_potential_norm)
            elif price_trend < -self.price_trend_threshold:
                # Downward trend - consider light charging
                if battery_soc < (min_soc_safe + max_soc_safe) / 2:  # Below midpoint
                    charge_intensity = min(0.1, (max_soc_safe - battery_soc) * 0.2)
                    solar_generation = min(
                        self.default_solar_generation, ac_potential_norm
                    )
                    return solar_generation - charge_intensity
                else:
                    return min(self.default_solar_generation, ac_potential_norm)
            else:
                # Neutral trend - balanced generation
                return ac_potential_norm
