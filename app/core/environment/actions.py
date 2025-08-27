"""
Action definitions and execution for power management environments.
"""

import datetime
from enum import Enum
from typing import Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.core.simulation.portfolio import PowerPlantPortfolio
from app.core.utils.logging import get_logger

logger = get_logger("environment_actions")

# Action types - maps to plant control parameters
ActionType = Dict[str, Dict[str, Dict[str, float]]]


class ActionName(str, Enum):
    """Available action types for plant control."""

    NET_POWER_TARGET = "net_power_target_mw"
    BATTERY_POWER_TARGET = "battery_power_target_mw"
    PV_CURTAILMENT_FACTOR = "pv_curtailment_factor"


class ActionTemplate(BaseModel):
    """Template for creating action data instances."""

    model_config = ConfigDict(use_enum_values=True)

    data_name: ActionName = Field(description="Name of the action")
    min_value: float = Field(description="Minimum allowed value")
    max_value: float = Field(description="Maximum allowed value")

    def create_data_instance(self, value: float) -> "ActionData":
        """Create an action data instance with validation."""
        # Clamp value to allowed range
        clamped_value = max(self.min_value, min(value, self.max_value))

        if clamped_value != value:
            logger.warning(
                f"Action {self.data_name.value} clamped from {value:.3f} "
                f"to {clamped_value:.3f}"
            )

        return ActionData(
            name=self.data_name,
            values=clamped_value,
        )


class ActionData(BaseModel):
    """Container for action data."""

    model_config = ConfigDict(use_enum_values=True)

    name: ActionName = Field(description="Action name")
    values: float = Field(description="Action value")


class ActionFactory:
    """Factory for creating and executing actions on power plant portfolios."""

    def __init__(
        self,
        portfolios: List[PowerPlantPortfolio],
        power_normalization_coefficient: float = 1e6,
        interval_min: float = 5.0,
        action_tolerance_percent: float = 0.05,
    ):
        """
        Initialize action factory.

        Args:
            portfolios: List of power plant portfolios
            power_normalization_coefficient: Normalization factor for power values
            interval_min: Time interval in minutes
            action_tolerance_percent: Tolerance for action validation
        """
        self.portfolios = portfolios
        self.power_normalization_coefficient = power_normalization_coefficient
        self.interval_min = interval_min
        self.action_tolerance_percent = action_tolerance_percent

        # Build action templates based on portfolio configuration
        self.action_templates = self._build_action_templates()

    def _build_action_templates(self) -> Dict[str, ActionTemplate]:
        """Build action templates based on portfolio capabilities."""
        templates = {}

        # For each portfolio, determine action ranges
        for portfolio in self.portfolios:
            # Net power action template
            max_power = portfolio.get_total_capacity()
            min_power = -max_power  # Allow negative for consumption

            templates[ActionName.NET_POWER_TARGET.value] = ActionTemplate(
                data_name=ActionName.NET_POWER_TARGET,
                min_value=min_power,
                max_value=max_power,
            )

            # Battery power template (if batteries present)
            total_battery_power = 0.0
            for plant in portfolio.plants:
                total_battery_power += plant.total_battery_power_mw

            if total_battery_power > 0:
                templates[ActionName.BATTERY_POWER_TARGET.value] = ActionTemplate(
                    data_name=ActionName.BATTERY_POWER_TARGET,
                    min_value=-total_battery_power,  # Negative for charging
                    max_value=total_battery_power,  # Positive for discharging
                )

            # PV curtailment template
            templates[ActionName.PV_CURTAILMENT_FACTOR.value] = ActionTemplate(
                data_name=ActionName.PV_CURTAILMENT_FACTOR,
                min_value=0.0,  # 0% = full curtailment
                max_value=1.0,  # 100% = no curtailment
            )

        return templates

    async def execute_action(
        self, action: ActionType, timestamp: datetime.datetime
    ) -> Tuple[Dict, Dict]:
        """
        Execute an action at a given timestamp.

        Args:
            action: Nested dict of {portfolio_name: {plant_name: {action_name: value}}}
            timestamp: Timestamp for the action

        Returns:
            Tuple of (execution_info, rewards)
        """
        portfolios_executed_action_info = {}
        portfolios_reward = {}

        for portfolio_name, portfolio_action in action.items():
            try:
                portfolio = next(
                    p for p in self.portfolios if p.config.name == portfolio_name
                )
            except StopIteration as e:
                logger.exception(f"Unrecognized portfolio name '{portfolio_name}'")
                raise RuntimeError(
                    f"Unrecognized portfolio name '{portfolio_name}'"
                ) from e

            portfolio_executed_action_info = {}
            portfolio_reward = {}

            for plant_name, plant_action in portfolio_action.items():
                try:
                    plant = next(
                        p for p in portfolio.plants if p.config.name == plant_name
                    )
                except StopIteration as e:
                    logger.exception(f"Unrecognized plant name '{plant_name}'")
                    raise RuntimeError(f"Unrecognized plant name '{plant_name}'") from e

                # Execute plant-level actions
                plant_info, plant_reward = await self._execute_plant_action(
                    plant, plant_action, timestamp
                )

                portfolio_executed_action_info[plant_name] = plant_info
                portfolio_reward[plant_name] = plant_reward

            portfolios_executed_action_info[portfolio_name] = (
                portfolio_executed_action_info
            )
            portfolios_reward[portfolio_name] = portfolio_reward

        return portfolios_executed_action_info, portfolios_reward

    async def _execute_plant_action(
        self,
        plant,
        plant_action: Dict[str, float],
        timestamp: datetime.datetime,
    ) -> Tuple[Dict, float]:
        """Execute action on a single plant."""
        # Parse actions and track what's being controlled
        net_power_target = None
        battery_power_target = None
        pv_curtailment = 1.0  # Default: no curtailment

        for action_name, action_value in plant_action.items():
            # Action values are already in MW - no denormalization needed
            if action_name == ActionName.NET_POWER_TARGET.value:
                # Action value is already in MW from environment conversion
                net_power_target = action_value
            elif action_name == ActionName.BATTERY_POWER_TARGET.value:
                # Store for potential future battery-specific control
                battery_power_target = action_value
            elif action_name == ActionName.PV_CURTAILMENT_FACTOR.value:
                pv_curtailment = action_value  # Already normalized 0-1

        # Dispatch power based on action type
        if net_power_target is not None:
            # Use plant's dispatch method with MW values
            actual_power, plant_state, valid = await plant.dispatch_power(
                net_power_target, timestamp, self.interval_min
            )
        else:
            # Manual control mode (not implemented in current plant interface)
            logger.warning(
                "Manual battery/PV control not yet implemented. "
                "Currently only supports net power targets."
            )
            actual_power = 0.0
            plant_state = {}
            valid = False

        # Log unused control parameters for debugging
        if battery_power_target is not None and net_power_target is None:
            logger.debug(
                f"Battery power target {battery_power_target:.3f}MW specified "
                "but not used (manual battery control not implemented)"
            )
        if pv_curtailment != 1.0 and net_power_target is None:
            logger.debug(
                f"PV curtailment {pv_curtailment:.1%} specified "
                "but not used (manual PV control not implemented)"
            )

        # Calculate basic reward (revenue based on power output)
        # Note: This is a simplified calculation. The main reward calculation
        # should be done by the RewardFactory which has access to actual prices
        reward = 0.0

        # For action execution, we provide a basic placeholder reward
        # The actual reward will be calculated by the environment's reward system
        if actual_power > 0:
            # Positive power = generation = positive placeholder reward
            reward = actual_power * 0.001  # Small positive value per MW
        elif actual_power < 0:
            # Negative power = consumption = small negative reward
            reward = actual_power * 0.001  # Small negative value
        else:
            # No power = no reward
            reward = 0.0

        # Scale reward to reasonable range for RL training
        reward = reward * (self.interval_min / 60.0)  # Scale by interval length

        # Normalize actual power for return (convert back to normalized scale)
        normalized_actual_power = actual_power / self.power_normalization_coefficient

        execution_info = {
            "actual_power_normalized": normalized_actual_power,
            "target_power_mw": net_power_target,
            "operation_valid": valid,
            "plant_state": plant_state,
        }

        return execution_info, reward
