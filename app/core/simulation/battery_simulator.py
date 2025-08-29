"""
Linear battery energy storage system simulation.

This module provides a linear battery model for energy storage systems that can be
integrated with solar power generation for optimized electricity market participation.
The implementation focuses on efficient power dispatch calculations with configurable
charging/discharging characteristics and state management.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Protocol

from app.core.utils.logging import get_logger

logger = get_logger("battery_simulator")


# -----------------------------------------------------------------------------
# Constants and Types
# -----------------------------------------------------------------------------

MINUTES_PER_HOUR = 60.0
SECONDS_PER_HOUR = 3600.0


class BatteryOperationMode(str, Enum):
    """Battery operation modes for different dispatch strategies."""

    CHARGING = "charging"
    DISCHARGING = "discharging"
    IDLE = "idle"


class BatteryStateColumns(str, Enum):
    """Standardized column names for battery state data."""

    TIMESTAMP = "timestamp"
    STATE_OF_CHARGE = "state_of_charge"
    POWER_MW = "power_mw"
    ENERGY_MWH = "energy_mwh"
    OPERATION_MODE = "operation_mode"
    EFFICIENCY = "efficiency"


# -----------------------------------------------------------------------------
# Battery Protocol and Base Classes
# -----------------------------------------------------------------------------


class BatteryProtocol(Protocol):
    """Protocol defining the interface for battery storage systems."""

    def dispatch_power(
        self,
        target_power_mw: float,
        interval_minutes: float = 60.0,
    ) -> Tuple[float, float, bool]:
        """
        Dispatch power to/from the battery.

        Args:
            target_power_mw: Power to dispatch (positive=discharge, negative=charge)
            interval_minutes: Time interval for the operation

        Returns:
            Tuple of (actual_power_mw, state_of_charge, operation_valid)
        """
        ...

    def get_available_power(
        self, interval_minutes: float = 60.0
    ) -> Tuple[float, float]:
        """
        Get available charging and discharging power.

        Args:
            interval_minutes: Time interval for the calculation

        Returns:
            Tuple of (max_charge_power_mw, max_discharge_power_mw)
        """
        ...


class BatteryConfiguration(BaseModel):
    """Base configuration for battery energy storage systems."""

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", use_enum_values=True
    )

    # Core battery specifications
    energy_capacity_mwh: float = Field(
        description="Total energy capacity in MWh",
        gt=0.001,  # Minimum 1 kWh
    )
    max_power_mw: float = Field(
        description="Maximum power rating in MW (for both charge and discharge)",
        gt=0.001,  # Minimum 1 kW
    )

    # State of charge constraints
    min_soc: float = Field(
        default=0.05,  # 5% minimum for battery protection
        description="Minimum state of charge (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    max_soc: float = Field(
        default=0.95,  # 95% maximum for battery protection
        description="Maximum state of charge (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    initial_soc: float = Field(
        default=0.5, description="Initial state of charge (0.0 to 1.0)", ge=0.0, le=1.0
    )

    # Efficiency parameters
    round_trip_efficiency: float = Field(
        default=0.85, description="Round-trip efficiency (0.0 to 1.0)", gt=0.0, le=1.0
    )

    @model_validator(mode="after")
    def validate_soc_bounds(self) -> "BatteryConfiguration":
        """Validate state of charge constraints."""
        if self.min_soc >= self.max_soc:
            raise ValueError("min_soc must be less than max_soc")

        if not (self.min_soc <= self.initial_soc <= self.max_soc):
            raise ValueError("initial_soc must be between min_soc and max_soc")

        return self

    @field_validator("max_power_mw")
    @classmethod
    def validate_power_rating(cls, v: float) -> float:
        """Validate power rating is reasonable."""
        if v > 1000.0:  # 1 GW limit for sanity check
            raise ValueError("Power rating exceeds reasonable limits (>1000 MW)")
        return v


# -----------------------------------------------------------------------------
# Linear Battery Implementation
# -----------------------------------------------------------------------------


class LinearBatterySimulator(BaseModel):
    """
    Linear battery energy storage system with configurable power and efficiency.

    This simulator implements a simplified linear battery model suitable for
    electricity market optimization studies. It provides realistic charging/
    discharging behavior with efficiency losses and power constraints.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    config: BatteryConfiguration = Field(description="Battery configuration")

    # Current state (mutable) - calculated fields with temporary defaults
    current_soc: float = Field(default=0.5, description="Current state of charge")
    charge_efficiency: float = Field(default=0.9, description="Charging efficiency")
    discharge_efficiency: float = Field(
        default=0.9, description="Discharging efficiency"
    )

    def model_post_init(self, __context) -> None:
        """Set initial values from configuration after validation."""
        # Set initial state from configuration
        self.current_soc = self.config.initial_soc

        # Calculate charge/discharge efficiencies from round-trip efficiency
        rt_eff = self.config.round_trip_efficiency
        # Assume symmetric losses: charge_eff * discharge_eff = round_trip_eff
        self.charge_efficiency = rt_eff**0.5
        self.discharge_efficiency = rt_eff**0.5

        logger.info(
            f"Initialized battery: {self.config.energy_capacity_mwh:.1f} MWh, "
            f"{self.config.max_power_mw:.1f} MW, RT efficiency: {rt_eff:.2%}"
        )

    @property
    def state_of_charge(self) -> float:
        """Current state of charge (0.0 to 1.0)."""
        return self.current_soc

    @property
    def stored_energy_mwh(self) -> float:
        """Current stored energy in MWh."""
        return self.current_soc * self.config.energy_capacity_mwh

    @property
    def available_capacity_mwh(self) -> float:
        """Available charging capacity in MWh."""
        return (
            self.config.max_soc - self.current_soc
        ) * self.config.energy_capacity_mwh

    @property
    def available_energy_mwh(self) -> float:
        """Available discharging energy in MWh."""
        return (
            self.current_soc - self.config.min_soc
        ) * self.config.energy_capacity_mwh

    def get_available_power(
        self, interval_minutes: float = 60.0
    ) -> Tuple[float, float]:
        """
        Calculate maximum available charging and discharging power.

        Args:
            interval_minutes: Time interval for the calculation

        Returns:
            Tuple of (max_charge_power_mw, max_discharge_power_mw)
        """
        if interval_minutes <= 0:
            raise ValueError("Interval must be positive")

        if interval_minutes < 0.1:  # Minimum 6 seconds
            raise ValueError(
                "Interval too short for battery dispatch (min 0.1 minutes)"
            )

        if interval_minutes > 1440:  # More than 24 hours
            raise ValueError("Interval too long for battery dispatch (max 24 hours)")

        interval_hours = interval_minutes / MINUTES_PER_HOUR

        # Power-limited charging (negative value indicates charging)
        power_limited_charge = -self.config.max_power_mw

        # Energy-limited charging based on available capacity
        if interval_hours > 0:
            energy_limited_charge = (
                -(self.available_capacity_mwh / interval_hours) / self.charge_efficiency
            )
        else:
            energy_limited_charge = power_limited_charge

        # Take the more restrictive limit (less negative for charging)
        max_charge_power = max(power_limited_charge, energy_limited_charge)

        # Power-limited discharging
        power_limited_discharge = self.config.max_power_mw

        # Energy-limited discharging based on available energy
        if interval_hours > 0:
            energy_limited_discharge = (
                self.available_energy_mwh / interval_hours
            ) * self.discharge_efficiency
        else:
            energy_limited_discharge = power_limited_discharge

        # Take the more restrictive limit (smaller positive for discharging)
        max_discharge_power = min(power_limited_discharge, energy_limited_discharge)

        return max_charge_power, max_discharge_power

    def dispatch_power(
        self,
        target_power_mw: float,
        interval_minutes: float = 60.0,
    ) -> Tuple[float, float, bool]:
        """
        Dispatch power to/from the battery with efficiency and constraint handling.

        Args:
            target_power_mw: Target power (positive=discharge, negative=charge)
            interval_minutes: Time interval for the operation

        Returns:
            Tuple of (actual_power_mw, new_soc, operation_valid)
        """
        if interval_minutes <= 0:
            raise ValueError("Interval must be positive")

        if interval_minutes < 0.1:  # Minimum 6 seconds
            raise ValueError(
                "Interval too short for battery dispatch (min 0.1 minutes)"
            )

        if interval_minutes > 1440:  # More than 24 hours
            raise ValueError("Interval too long for battery dispatch (max 24 hours)")

        # Check for invalid float values first (before range check)
        if not (
            isinstance(target_power_mw, (int, float))
            and target_power_mw == target_power_mw
        ):  # NaN check
            raise ValueError(
                f"Target power must be a valid number, got {target_power_mw}"
            )

        if not (-1000.0 <= target_power_mw <= 1000.0):  # Sanity check
            raise ValueError(
                f"Target power {target_power_mw} MW exceeds reasonable limits"
            )

        interval_hours = interval_minutes / MINUTES_PER_HOUR
        max_charge_power, max_discharge_power = self.get_available_power(
            interval_minutes
        )

        operation_valid = True
        actual_power_mw = target_power_mw

        if target_power_mw > 0:
            # Discharging
            if target_power_mw > max_discharge_power:
                actual_power_mw = max_discharge_power
                operation_valid = False

            # Calculate energy change considering discharge efficiency
            energy_delivered_mwh = actual_power_mw * interval_hours
            energy_from_battery_mwh = energy_delivered_mwh / self.discharge_efficiency

            # Update state of charge
            delta_soc = energy_from_battery_mwh / self.config.energy_capacity_mwh
            new_soc = max(self.current_soc - delta_soc, self.config.min_soc)

            # Adjust actual power if SOC limit was hit
            if new_soc == self.config.min_soc and new_soc != self.current_soc:
                # Recalculate actual power based on available energy
                actual_energy_from_battery = (
                    self.current_soc - self.config.min_soc
                ) * self.config.energy_capacity_mwh
                actual_energy_delivered = (
                    actual_energy_from_battery * self.discharge_efficiency
                )
                actual_power_mw = actual_energy_delivered / interval_hours
                operation_valid = False

        elif target_power_mw < 0:
            # Charging
            if target_power_mw < max_charge_power:
                actual_power_mw = max_charge_power
                operation_valid = False

            # Calculate energy change considering charge efficiency
            energy_input_mwh = abs(actual_power_mw) * interval_hours
            energy_stored_mwh = energy_input_mwh * self.charge_efficiency

            # Update state of charge
            delta_soc = energy_stored_mwh / self.config.energy_capacity_mwh
            new_soc = min(self.current_soc + delta_soc, self.config.max_soc)

            # Adjust actual power if SOC limit was hit
            if new_soc == self.config.max_soc and new_soc != self.current_soc:
                # Recalculate actual power based on available capacity
                actual_energy_stored = (
                    self.config.max_soc - self.current_soc
                ) * self.config.energy_capacity_mwh
                actual_energy_input = actual_energy_stored / self.charge_efficiency
                actual_power_mw = -actual_energy_input / interval_hours
                operation_valid = False

        else:
            # Idle operation
            new_soc = self.current_soc
            actual_power_mw = 0.0

        # Update internal state with bounds checking
        self.current_soc = max(self.config.min_soc, min(new_soc, self.config.max_soc))

        logger.debug(
            f"Battery dispatch: target={target_power_mw:.2f}MW, "
            f"actual={actual_power_mw:.2f}MW, SOC={self.current_soc:.1%}, "
            f"valid={operation_valid}"
        )

        return actual_power_mw, self.current_soc, operation_valid

    def reset_state(self, soc: Optional[float] = None) -> None:
        """
        Reset battery state to initial or specified state of charge.

        Args:
            soc: State of charge to reset to (defaults to initial_soc)
        """
        if soc is None:
            soc = self.config.initial_soc

        if not (self.config.min_soc <= soc <= self.config.max_soc):
            raise ValueError(
                f"SOC {soc:.1%} outside valid range "
                f"[{self.config.min_soc:.1%}, {self.config.max_soc:.1%}]"
            )

        self.current_soc = soc
        logger.info(f"Battery state reset to SOC: {soc:.1%}")

    def simulate_dispatch_schedule(
        self,
        power_schedule_mw: pd.Series,
        interval_minutes: float = 60.0,
    ) -> pd.DataFrame:
        """
        Simulate battery operation over a power dispatch schedule.

        Args:
            power_schedule_mw: Time series of target power values (MW)
            interval_minutes: Time interval between dispatch points

        Returns:
            DataFrame with battery state and operation results
        """
        if power_schedule_mw.empty:
            return pd.DataFrame()

        results = []
        initial_soc = self.current_soc

        try:
            for timestamp, target_power in power_schedule_mw.items():
                actual_power, soc, valid = self.dispatch_power(
                    target_power, interval_minutes
                )

                # Determine operation mode based on actual power
                power_threshold = 0.001  # 1 kW threshold for mode detection
                if actual_power > power_threshold:
                    mode = BatteryOperationMode.DISCHARGING
                    efficiency = self.discharge_efficiency
                elif actual_power < -power_threshold:
                    mode = BatteryOperationMode.CHARGING
                    efficiency = self.charge_efficiency
                else:
                    mode = BatteryOperationMode.IDLE
                    efficiency = 1.0

                results.append(
                    {
                        BatteryStateColumns.TIMESTAMP: timestamp,
                        BatteryStateColumns.STATE_OF_CHARGE: soc,
                        BatteryStateColumns.POWER_MW: actual_power,
                        BatteryStateColumns.ENERGY_MWH: (
                            soc * self.config.energy_capacity_mwh
                        ),
                        BatteryStateColumns.OPERATION_MODE: mode,
                        BatteryStateColumns.EFFICIENCY: efficiency,
                        "target_power_mw": target_power,
                        "operation_valid": valid,
                    }
                )

            df = pd.DataFrame(results)
            df.set_index(BatteryStateColumns.TIMESTAMP, inplace=True)

            logger.info(
                f"Simulated {len(df)} dispatch intervals: "
                f"SOC {initial_soc:.1%} → {self.current_soc:.1%}"
            )

            return df

        except Exception as e:
            # Reset state on error
            self.current_soc = initial_soc
            logger.error(f"Battery simulation failed: {e}")
            raise


# -----------------------------------------------------------------------------
# Factory and Helper Functions
# -----------------------------------------------------------------------------


class BatterySimulatorFactory:
    """Factory for creating battery simulator instances."""

    @staticmethod
    def create_linear_battery(
        energy_capacity_mwh: float,
        max_power_mw: float,
        round_trip_efficiency: float = 0.85,
        initial_soc: float = 0.5,
        min_soc: float = 0.05,  # Updated default
        max_soc: float = 0.95,  # Updated default
    ) -> LinearBatterySimulator:
        """
        Create a linear battery simulator with standard configuration.

        Args:
            energy_capacity_mwh: Battery energy capacity in MWh
            max_power_mw: Maximum power rating in MW
            round_trip_efficiency: Round-trip efficiency (0.0 to 1.0)
            initial_soc: Initial state of charge (0.0 to 1.0)
            min_soc: Minimum state of charge (0.0 to 1.0)
            max_soc: Maximum state of charge (0.0 to 1.0)

        Returns:
            Configured LinearBatterySimulator instance
        """
        config = BatteryConfiguration(
            energy_capacity_mwh=energy_capacity_mwh,
            max_power_mw=max_power_mw,
            round_trip_efficiency=round_trip_efficiency,
            initial_soc=initial_soc,
            min_soc=min_soc,
            max_soc=max_soc,
        )

        return LinearBatterySimulator(config=config)


# Type alias for battery types
BatterySimulator = Union[LinearBatterySimulator]
