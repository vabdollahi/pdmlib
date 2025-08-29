"""
Reward definitions for power management environments.
"""

import datetime
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from app.core.simulation.portfolio import PowerPlantPortfolio
from app.core.utils.logging import get_logger

logger = get_logger("environment_rewards")


class Reward(ABC):
    """Abstract base class for reward calculation."""

    @abstractmethod
    async def create(
        self,
        power_to_grid_mw: float,
        portfolio: PowerPlantPortfolio,
        timestamp: datetime.datetime,
        interval_min: float,
    ) -> Tuple[float, Dict]:
        """
        Calculate reward for given power dispatch.

        Args:
            power_to_grid_mw: Net power sent to grid (positive) or consumed (negative)
            portfolio: Portfolio that generated the power
            timestamp: Current timestamp
            interval_min: Time interval in minutes

        Returns:
            Tuple of (reward_value, reward_info_dict)
        """
        raise NotImplementedError


class RevenueReward(Reward):
    """Revenue-based reward calculation."""

    def __init__(self, smoothed_reward_parameter: float = 0.1):
        """
        Initialize revenue reward calculator.

        Args:
            smoothed_reward_parameter: Smoothing factor for reward (0.0-1.0)
                Higher values = less smoothing (more reactive)
                Lower values = more smoothing (more stable)
        """
        self._smoothed_reward_parameter = smoothed_reward_parameter
        self._smoothed_revenue = None

    async def create(
        self,
        power_to_grid_mw: float,
        portfolio: PowerPlantPortfolio,
        timestamp: datetime.datetime,
        interval_min: float,
    ) -> Tuple[float, Dict]:
        """Calculate revenue-based reward."""
        # Get current electricity price (placeholder implementation)
        # In a full implementation, this would integrate with the price providers
        current_price_dollar_mwh = await self._get_current_price(portfolio, timestamp)

        # Calculate instant revenue
        # Revenue = Power (MW) × Price ($/MWh) × Time (hours)
        time_hours = interval_min / 60.0
        instant_revenue = power_to_grid_mw * current_price_dollar_mwh * time_hours

        # Apply reward smoothing if configured
        if self._smoothed_reward_parameter == 1.0:
            reward = instant_revenue
        else:
            self._update_smoothed_revenue(instant_revenue)
            reward = self._smoothed_revenue

        # Additional reward components
        reward_info = {
            "instant_revenue_dollar": instant_revenue,
            "smoothed_revenue_dollar": self._smoothed_revenue or instant_revenue,
            "power_to_grid_mw": power_to_grid_mw,
            "price_dollar_mwh": current_price_dollar_mwh,
            "interval_hours": time_hours,
        }

        # Add penalty for battery degradation (simplified)
        battery_penalty = self._calculate_battery_penalty(portfolio)
        reward = (reward or 0.0) - battery_penalty
        reward_info["battery_penalty_dollar"] = battery_penalty

        logger.debug(
            f"Reward calculation: power={power_to_grid_mw:.2f}MW, "
            f"price=${current_price_dollar_mwh:.2f}/MWh, "
            f"revenue=${instant_revenue:.4f}, reward={reward:.4f}"
        )

        return reward, reward_info

    async def _get_current_price(
        self, portfolio: PowerPlantPortfolio, timestamp: datetime.datetime
    ) -> float:
        """Get current electricity price for the portfolio's market."""
        # Get price from the portfolio's plants' revenue calculators
        for plant in portfolio.plants:
            if hasattr(plant, "revenue_calculator") and plant.revenue_calculator:
                # Get price data from the plant's revenue calculator
                price_provider = plant.revenue_calculator.price_provider
                price_data = await price_provider.get_data()

                if price_data.empty:
                    raise ValueError("No price data available from revenue calculator")

                # Find price closest to current timestamp
                price_data_indexed = price_data.reset_index()
                timestamp_col = "timestamp"
                if timestamp_col not in price_data_indexed.columns:
                    raise ValueError("No timestamp column found in price data")

                # Find closest timestamp
                time_diffs = abs(price_data_indexed[timestamp_col] - timestamp)
                closest_idx = time_diffs.idxmin()
                price_col = "price_dollar_mwh"

                if price_col not in price_data_indexed.columns:
                    raise ValueError(
                        f"Price column '{price_col}' not found in price data"
                    )

                current_price = float(price_data_indexed.at[closest_idx, price_col])
                logger.debug(
                    f"Retrieved price ${current_price:.2f}/MWh from price provider"
                )
                return current_price

        raise ValueError("No revenue calculator available for price calculation")

    def _update_smoothed_revenue(self, revenue: float) -> None:
        """Update smoothed revenue using exponential moving average."""
        if self._smoothed_revenue is None:
            self._smoothed_revenue = revenue
        else:
            alpha = self._smoothed_reward_parameter
            self._smoothed_revenue = (
                alpha * revenue + (1.0 - alpha) * self._smoothed_revenue
            )

    def _calculate_battery_penalty(self, portfolio: PowerPlantPortfolio) -> float:
        """Calculate penalty for battery usage (degradation cost)."""
        total_penalty = 0.0

        for plant in portfolio.plants:
            for battery in plant.batteries:
                # Simple degradation model: cost per MWh throughput
                # This is a placeholder - real models would be more sophisticated
                degradation_cost_per_mwh = 0.05  # $/MWh throughput

                # Calculate energy throughput for this timestep only
                # Use actual power throughput rather than SOC change from initial
                # This should ideally track power flow through the battery
                # For now, use a conservative estimate based on current power capacity
                current_power_capability = battery.config.max_power_mw

                # Estimate throughput as fraction of max power usage per timestep
                # This is a placeholder - real implementation would track actual
                # power throughput per timestep
                estimated_throughput_mwh = (
                    current_power_capability * 0.1  # Conservative estimate
                )

                penalty = estimated_throughput_mwh * degradation_cost_per_mwh
                total_penalty += penalty

        return total_penalty


class RewardFactory:
    """Factory for creating reward calculators."""

    @staticmethod
    def create_revenue_reward(smoothed_reward_parameter: float = 0.1) -> RevenueReward:
        """Create a revenue-based reward calculator."""
        return RevenueReward(smoothed_reward_parameter=smoothed_reward_parameter)

    @staticmethod
    def create_profit_reward(
        operational_cost_dollar_mwh: float = 5.0,
        smoothed_reward_parameter: float = 0.1,
    ) -> "ProfitReward":
        """Create a profit-based reward calculator (revenue - costs)."""
        return ProfitReward(
            operational_cost_dollar_mwh=operational_cost_dollar_mwh,
            smoothed_reward_parameter=smoothed_reward_parameter,
        )


class ProfitReward(RevenueReward):
    """Profit-based reward that includes operational costs."""

    def __init__(
        self,
        operational_cost_dollar_mwh: float = 5.0,
        smoothed_reward_parameter: float = 0.1,
    ):
        """
        Initialize profit reward calculator.

        Args:
            operational_cost_dollar_mwh: Operational cost per MWh
            smoothed_reward_parameter: Smoothing factor for reward
        """
        super().__init__(smoothed_reward_parameter)
        self.operational_cost_dollar_mwh = operational_cost_dollar_mwh

    async def create(
        self,
        power_to_grid_mw: float,
        portfolio: PowerPlantPortfolio,
        timestamp: datetime.datetime,
        interval_min: float,
    ) -> Tuple[float, Dict]:
        """Calculate profit-based reward (revenue - operational costs)."""
        # Get base revenue reward
        revenue_reward, reward_info = await super().create(
            power_to_grid_mw, portfolio, timestamp, interval_min
        )

        # Calculate operational costs
        time_hours = interval_min / 60.0
        if power_to_grid_mw > 0:  # Only charge costs for generation
            operational_cost = (
                power_to_grid_mw * self.operational_cost_dollar_mwh * time_hours
            )
        else:
            operational_cost = 0.0

        # Profit = Revenue - Operational Costs - Battery Penalty
        profit_reward = revenue_reward - operational_cost

        # Update reward info
        reward_info["operational_cost_dollar"] = operational_cost
        reward_info["profit_dollar"] = profit_reward

        return profit_reward, reward_info
