"""
Power plant model for solar PV systems with battery energy storage.

This module provides a plant abstraction that combines PV generation with battery
storage for participation in electricity markets. The plant model handles power
dispatch optimization, state management, and revenue generation.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Protocol

from app.core.simulation.battery_simulator import LinearBatterySimulator
from app.core.simulation.pv_model import PVLibResultsColumns, PVModel
from app.core.simulation.solar_revenue import SolarRevenueCalculator
from app.core.utils.logging import get_logger

if TYPE_CHECKING:
    from app.core.utils.storage import DataStorage

logger = get_logger("plant")


# -----------------------------------------------------------------------------
# Constants and Enums
# -----------------------------------------------------------------------------

MINUTES_PER_HOUR = 60.0


class PlantOperationMode(str, Enum):
    """Plant operation modes for different market strategies."""

    OPTIMIZING = "optimizing"  # Active market participation
    MAINTENANCE = "maintenance"  # Scheduled maintenance
    EMERGENCY = "emergency"  # Emergency shutdown
    IDLE = "idle"  # Standby mode


class PlantStateColumns(str, Enum):
    """Standardized column names for plant state data."""

    TIMESTAMP = "timestamp"
    PV_GENERATION_MW = "pv_generation_mw"
    BATTERY_POWER_MW = "battery_power_mw"
    NET_POWER_MW = "net_power_mw"
    BATTERY_SOC = "battery_soc"
    OPERATION_MODE = "operation_mode"
    REVENUE_DOLLAR = "revenue_dollar"


# -----------------------------------------------------------------------------
# Plant Protocol and Base Classes
# -----------------------------------------------------------------------------


class PlantProtocol(Protocol):
    """Protocol defining the interface for power plants."""

    def dispatch_power(
        self,
        target_net_power_mw: float,
        timestamp: datetime.datetime,
        interval_minutes: float = 60.0,
    ) -> Tuple[float, Dict[str, float], bool]:
        """
        Dispatch power from the plant.

        Args:
            target_net_power_mw: Net power to dispatch to grid (MW)
            timestamp: Time of dispatch
            interval_minutes: Time interval for the operation

        Returns:
            Tuple of (actual_net_power_mw, plant_state, operation_valid)
        """
        ...

    def get_available_power(
        self, timestamp: datetime.datetime, interval_minutes: float = 60.0
    ) -> Tuple[float, float]:
        """
        Get available power generation and consumption capability.

        Args:
            timestamp: Time for availability calculation
            interval_minutes: Time interval for the calculation

        Returns:
            Tuple of (max_generation_mw, max_consumption_mw)
        """
        ...


class PlantConfiguration(BaseModel):
    """Base configuration for power plants."""

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", use_enum_values=True
    )

    # Plant identification
    name: str = Field(description="Plant identifier", min_length=1)
    plant_id: Optional[str] = Field(default=None, description="Unique plant ID")

    # Operational parameters
    max_net_power_mw: float = Field(
        description="Maximum net power to grid in MW", gt=0.001
    )
    min_net_power_mw: float = Field(
        default=0.0, description="Minimum net power to grid in MW", ge=0.0
    )

    # Market participation settings
    enable_market_participation: bool = Field(
        default=True, description="Enable electricity market participation"
    )
    maintenance_schedule: List[Tuple[datetime.datetime, datetime.datetime]] = Field(
        default_factory=list, description="Scheduled maintenance windows"
    )

    @model_validator(mode="after")
    def validate_power_limits(self) -> "PlantConfiguration":
        """Validate power limit constraints."""
        if self.min_net_power_mw >= self.max_net_power_mw:
            raise ValueError("min_net_power_mw must be less than max_net_power_mw")
        return self


# -----------------------------------------------------------------------------
# Solar Plant with Battery Storage Implementation
# -----------------------------------------------------------------------------


class SolarBatteryPlant(BaseModel):
    """
    Solar power plant with battery energy storage system.

    This plant model combines PV generation with battery storage for optimized
    electricity market participation. It handles power dispatch coordination
    between solar generation and battery charge/discharge operations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # Core components
    config: PlantConfiguration = Field(description="Plant configuration")
    pv_model: PVModel = Field(description="Solar PV generation model")
    batteries: List[LinearBatterySimulator] = Field(
        default_factory=list, description="Battery energy storage systems"
    )

    # Revenue calculation (optional for market participation)
    revenue_calculator: Optional[SolarRevenueCalculator] = Field(
        default=None, description="Revenue calculation model"
    )

    # Plant state
    _operation_mode: PlantOperationMode = PlantOperationMode.IDLE
    _current_timestamp: Optional[datetime.datetime] = None
    pv_cache_enabled: bool = Field(default=True, exclude=True)
    # Simple in-memory cache of PV potential by timestamp
    # to avoid duplicate work within a step
    # Private attribute for intra-step PV potential cache
    _pv_potential_cache: Dict[str, float] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        """Initialize plant with validation."""
        super().__init__(**data)

        # Validate max_net_power_mw against PV system capacity
        self._validate_power_capacity_consistency()

        # Enable PV caching by default with standard data storage
        if self.pv_cache_enabled:
            try:
                from app.core.utils.storage import DataStorage

                default_storage = DataStorage(base_path="data")
                organization = getattr(self.config, "organization", "SolarRevenue")
                asset = self.config.name or self.config.plant_id or "unnamed_plant"

                self.pv_model.enable_caching(
                    storage=default_storage, organization=organization, asset=asset
                )
                logger.debug(f"PV caching auto-enabled for plant '{self.config.name}'")
            except Exception as e:
                logger.warning(f"Could not auto-enable PV caching: {e}")
                self.pv_cache_enabled = False

        logger.debug(
            f"Initialized solar-battery plant '{self.config.name}' with "
            f"{len(self.batteries)} batteries"
        )

    def _validate_power_capacity_consistency(self) -> None:
        """Validate that max_net_power_mw is consistent with PV system capacity."""
        try:
            pv_capacity = self.pv_model.get_system_capacity()
            pv_ac_capacity_mw = pv_capacity["ac_capacity_w"] / 1e6

            if pv_ac_capacity_mw > 0:
                # Check if max_net_power_mw significantly exceeds PV AC capacity
                capacity_ratio = self.config.max_net_power_mw / pv_ac_capacity_mw

                if capacity_ratio > 1.1:  # Allow 10% margin for measurement differences
                    logger.warning(
                        f"Plant '{self.config.name}' max_net_power_mw "
                        f"({self.config.max_net_power_mw:.2f} MW) exceeds "
                        f"PV AC capacity ({pv_ac_capacity_mw:.2f} MW) by "
                        f"{(capacity_ratio - 1) * 100:.1f}%. "
                        f"Consider setting max_net_power_mw to match PV capacity."
                    )
                elif capacity_ratio < 0.5:  # Warn if significantly under-utilized
                    logger.info(
                        f"Plant '{self.config.name}' max_net_power_mw "
                        f"({self.config.max_net_power_mw:.2f} MW) is only "
                        f"{capacity_ratio * 100:.1f}% of PV AC capacity "
                        f"({pv_ac_capacity_mw:.2f} MW). This may limit utilization."
                    )
                else:
                    logger.debug(
                        f"Plant '{self.config.name}' power capacity is consistent: "
                        f"max_net_power_mw={self.config.max_net_power_mw:.2f} MW, "
                        f"PV AC capacity={pv_ac_capacity_mw:.2f} MW"
                    )
            else:
                logger.warning(
                    f"Plant '{self.config.name}' PV system has zero AC capacity. "
                    f"Check PV system configuration."
                )

        except Exception as e:
            logger.warning(
                f"Could not validate power capacity consistency for plant "
                f"'{self.config.name}': {e}"
            )

    def __hash__(self):
        """Make plant hashable by using its name and plant_id."""
        return hash((self.config.name, self.config.plant_id))

    def __eq__(self, other):
        """Define equality based on config name and plant_id."""
        if not isinstance(other, SolarBatteryPlant):
            return False
        return (self.config.name, self.config.plant_id) == (
            other.config.name,
            other.config.plant_id,
        )

    @property
    def operation_mode(self) -> PlantOperationMode:
        """Current plant operation mode."""
        return self._operation_mode

    @property
    def total_battery_capacity_mwh(self) -> float:
        """Total battery energy capacity across all batteries."""
        return sum(battery.config.energy_capacity_mwh for battery in self.batteries)

    @property
    def total_battery_power_mw(self) -> float:
        """Total battery power rating across all batteries."""
        return sum(battery.config.max_power_mw for battery in self.batteries)

    @property
    def average_battery_soc(self) -> float:
        """Average state of charge across all batteries."""
        if not self.batteries:
            return 0.0

        total_capacity = self.total_battery_capacity_mwh
        if total_capacity == 0:
            return 0.0

        weighted_soc = sum(
            battery.state_of_charge * battery.config.energy_capacity_mwh
            for battery in self.batteries
        )
        return weighted_soc / total_capacity

    def enable_pv_caching(self, storage: "DataStorage") -> None:
        """
        Enable PV generation caching for this plant.

        Args:
            storage: DataStorage instance for caching
        """
        organization = getattr(self.config, "organization", "default")
        asset = self.config.name or self.config.plant_id or "unnamed_plant"

        self.pv_model.enable_caching(
            storage=storage, organization=organization, asset=asset
        )
        self.pv_cache_enabled = True
        logger.debug(f"PV caching enabled for plant '{self.config.name}'")

    def disable_pv_caching(self) -> None:
        """Disable PV generation caching for this plant."""
        self.pv_model.disable_caching()
        self.pv_cache_enabled = False
        logger.info(f"PV caching disabled for plant '{self.config.name}'")

    @property
    def is_pv_caching_enabled(self) -> bool:
        """Check if PV caching is enabled."""
        return self.pv_cache_enabled

    async def get_pv_generation_potential(
        self, timestamp: datetime.datetime, force_refresh: bool = False
    ) -> float:
        """
        Get PV generation potential at given timestamp.

        Args:
            timestamp: Time for generation calculation
            force_refresh: If True, bypass cache and run fresh simulation

        Returns:
            PV generation potential in MW
        """
        try:
            # Check in-memory cache first for efficiency within same step
            if not force_refresh:
                cache_key = pd.Timestamp(timestamp).isoformat()
                if cache_key in self._pv_potential_cache:
                    return max(0.0, float(self._pv_potential_cache[cache_key]))

            # Set day-scoped cache range to optimize data fetching
            day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = timestamp.replace(hour=23, minute=0, second=0, microsecond=0)
            self.pv_model.set_cached_range(day_start, day_end)

            # Run PV simulation to get generation data (with caching support)
            pv_results = await self.pv_model.run_simulation(force_refresh=force_refresh)

            # Find the closest timestamp in results
            ts = pd.Timestamp(timestamp)

            # Ensure timezone compatibility for comparison
            if isinstance(pv_results.index, pd.DatetimeIndex):
                if pv_results.index.tz is not None and ts.tz is None:
                    # Index has timezone, timestamp doesn't - localize timestamp
                    ts = ts.tz_localize(pv_results.index.tz)
                elif pv_results.index.tz is None and ts.tz is not None:
                    # Index is naive, timestamp has timezone - convert to naive
                    ts = ts.tz_localize(None)
                elif pv_results.index.tz is not None and ts.tz is not None:
                    # Both have timezones - convert to same timezone
                    ts = ts.tz_convert(pv_results.index.tz)

            if ts in pv_results.index:
                generation_mw = (
                    pv_results[PVLibResultsColumns.AC.value].loc[ts] / 1e6
                )  # Convert W to MW
            else:
                # Use nearest neighbor interpolation
                series = pv_results[PVLibResultsColumns.AC.value]
                if not series.empty:
                    idx_pos = series.index.get_indexer([ts], method="nearest")[0]
                    if idx_pos != -1:
                        generation_mw = series.iloc[idx_pos] / 1e6
                    else:
                        generation_mw = 0.0
                else:
                    generation_mw = 0.0

            generation_mw = max(0.0, generation_mw)

            # Store in in-memory cache for reuse within same step
            cache_key = pd.Timestamp(timestamp).isoformat()
            # Keep cache bounded to prevent memory growth
            if len(self._pv_potential_cache) > 1000:
                self._pv_potential_cache.clear()
            self._pv_potential_cache[cache_key] = float(generation_mw)

            return generation_mw

        except Exception as e:
            logger.error(f"Error calculating PV generation potential: {e}")
            return 0.0

    def get_battery_available_power(
        self, interval_minutes: float = 60.0
    ) -> Tuple[float, float]:
        """
        Get total available battery power across all batteries.

        Args:
            interval_minutes: Time interval for calculation

        Returns:
            Tuple of (max_charge_power_mw, max_discharge_power_mw)
        """
        total_charge_power = 0.0
        total_discharge_power = 0.0

        for battery in self.batteries:
            try:
                charge_power, discharge_power = battery.get_available_power(
                    interval_minutes
                )
                total_charge_power += abs(charge_power)  # Charge power is negative
                total_discharge_power += discharge_power
            except Exception as e:
                logger.warning(f"Error getting battery power availability: {e}")

        return total_charge_power, total_discharge_power

    def set_operation_mode(self, mode: PlantOperationMode) -> None:
        """Set plant operation mode."""
        if mode != self._operation_mode:
            logger.info(
                f"Plant operation mode changed: {self._operation_mode} → {mode}"
            )
            self._operation_mode = mode

    def is_in_maintenance(self, timestamp: datetime.datetime) -> bool:
        """Check if plant is in maintenance mode at given timestamp."""
        for start_time, end_time in self.config.maintenance_schedule:
            if start_time <= timestamp <= end_time:
                return True
        return False

    async def get_available_power(
        self,
        timestamp: datetime.datetime,
        interval_minutes: float = 60.0,
        pv_potential_mw: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Get total available power generation and consumption capability.

        Args:
            timestamp: Time for availability calculation
            interval_minutes: Time interval for calculation

        Returns:
            Tuple of (max_generation_mw, max_consumption_mw)
        """
        # Check if plant is available
        if self.is_in_maintenance(timestamp):
            return 0.0, 0.0

        # Get PV generation potential (reuse if provided)
        if pv_potential_mw is None:
            pv_potential_mw = await self.get_pv_generation_potential(timestamp)

        # Get battery capabilities
        battery_charge_mw, battery_discharge_mw = self.get_battery_available_power(
            interval_minutes
        )

        # Maximum generation: PV + battery discharge
        max_generation_mw = min(
            pv_potential_mw + battery_discharge_mw, self.config.max_net_power_mw
        )

        # Maximum consumption: battery charging (limited by PV curtailment)
        max_consumption_mw = min(battery_charge_mw, pv_potential_mw)

        return max_generation_mw, max_consumption_mw

    async def dispatch_power(
        self,
        target_net_power_mw: float,
        timestamp: datetime.datetime,
        interval_minutes: float = 60.0,
    ) -> Tuple[float, Dict[str, float], bool]:
        """
        Dispatch power from the plant coordinating PV and battery systems.

        Args:
            target_net_power_mw: Target net power to grid (MW)
            timestamp: Time of dispatch
            interval_minutes: Time interval for operation

        Returns:
            Tuple of (actual_net_power_mw, plant_state, operation_valid)
        """
        self._current_timestamp = timestamp

        # Check maintenance mode
        if self.is_in_maintenance(timestamp):
            self.set_operation_mode(PlantOperationMode.MAINTENANCE)
            return 0.0, self._get_plant_state(timestamp, 0.0, 0.0), False

        # Validate power limits
        if not (
            self.config.min_net_power_mw
            <= target_net_power_mw
            <= self.config.max_net_power_mw
        ):
            logger.warning(
                f"Target power {target_net_power_mw:.2f} MW outside limits "
                f"[{self.config.min_net_power_mw:.2f}, "
                f"{self.config.max_net_power_mw:.2f}]"
            )

        target_net_power_mw = max(
            self.config.min_net_power_mw,
            min(target_net_power_mw, self.config.max_net_power_mw),
        )

        # Get PV generation potential
        pv_potential_mw = await self.get_pv_generation_potential(timestamp)

        # Determine PV generation and battery dispatch strategy
        if target_net_power_mw <= pv_potential_mw:
            # Target achievable with PV alone
            pv_generation_mw = target_net_power_mw
            excess_pv_mw = pv_potential_mw - target_net_power_mw

            # Try to charge batteries with excess PV, but don't force it
            # Allow for PV curtailment if batteries can't absorb excess
            battery_target_power_mw = -min(
                excess_pv_mw,
                sum(
                    battery.get_available_power(interval_minutes)[0]
                    for battery in self.batteries
                ),
            )
        else:
            # Need battery discharge to meet target
            pv_generation_mw = pv_potential_mw
            battery_target_power_mw = target_net_power_mw - pv_potential_mw

        # Dispatch battery power
        actual_battery_power_mw = 0.0
        operation_valid = True

        for battery in self.batteries:
            try:
                # Distribute power dispatch across batteries proportionally
                battery_fraction = (
                    battery.config.max_power_mw / self.total_battery_power_mw
                    if self.total_battery_power_mw > 0
                    else 0
                )
                battery_power_target = battery_target_power_mw * battery_fraction

                # Dispatch power from battery
                actual_power, new_soc, valid = battery.dispatch_power(
                    battery_power_target, interval_minutes
                )
                actual_battery_power_mw += actual_power

                if not valid:
                    operation_valid = False

            except Exception as e:
                logger.error(f"Error dispatching battery power: {e}")
                operation_valid = False

        # Calculate actual net power
        actual_net_power_mw = pv_generation_mw + actual_battery_power_mw

        # Update operation mode
        if actual_battery_power_mw > 0.001:
            self.set_operation_mode(PlantOperationMode.OPTIMIZING)
        elif actual_battery_power_mw < -0.001:
            self.set_operation_mode(PlantOperationMode.OPTIMIZING)
        else:
            self.set_operation_mode(PlantOperationMode.IDLE)

        # Create plant state
        plant_state = self._get_plant_state(
            timestamp, pv_generation_mw, actual_battery_power_mw
        )

        logger.debug(
            f"Plant dispatch: target={target_net_power_mw:.2f}MW, "
            f"actual={actual_net_power_mw:.2f}MW, "
            f"PV={pv_generation_mw:.2f}MW, battery={actual_battery_power_mw:.2f}MW, "
            f"valid={operation_valid}"
        )

        return actual_net_power_mw, plant_state, operation_valid

    def _get_plant_state(
        self,
        timestamp: datetime.datetime,
        pv_generation_mw: float,
        battery_power_mw: float,
    ) -> Dict[str, float]:
        """Get current plant state information."""
        net_power_mw = pv_generation_mw + battery_power_mw

        state = {
            PlantStateColumns.TIMESTAMP.value: timestamp,
            PlantStateColumns.PV_GENERATION_MW.value: pv_generation_mw,
            PlantStateColumns.BATTERY_POWER_MW.value: battery_power_mw,
            PlantStateColumns.NET_POWER_MW.value: net_power_mw,
            PlantStateColumns.BATTERY_SOC.value: self.average_battery_soc,
            PlantStateColumns.OPERATION_MODE.value: self.operation_mode,
        }

        # Add revenue calculation if available
        if self.revenue_calculator and self.config.enable_market_participation:
            try:
                # TODO: Implement real-time revenue calculation
                # This requires integrating with the SolarRevenueCalculator
                # For now, use placeholder but log the limitation
                state[PlantStateColumns.REVENUE_DOLLAR.value] = 0.0
                logger.debug("Revenue calculation placeholder - needs implementation")
            except Exception as e:
                logger.warning(f"Error calculating revenue: {e}")
                state[PlantStateColumns.REVENUE_DOLLAR.value] = 0.0
        else:
            state[PlantStateColumns.REVENUE_DOLLAR.value] = 0.0

        return state

    async def simulate_operation(
        self,
        power_schedule_mw: pd.Series,
        interval_minutes: float = 60.0,
    ) -> pd.DataFrame:
        """
        Simulate plant operation over a power dispatch schedule.

        Args:
            power_schedule_mw: Time series of target net power values (MW)
            interval_minutes: Time interval between dispatch points

        Returns:
            DataFrame with plant operation results
        """
        if power_schedule_mw.empty:
            return pd.DataFrame()

        results = []
        initial_battery_states = [battery.state_of_charge for battery in self.batteries]

        try:
            for ts in power_schedule_mw.index:
                target_power = power_schedule_mw.loc[ts]
                actual_power, plant_state, valid = await self.dispatch_power(
                    target_power, ts, interval_minutes
                )

                # Add operation validity to state
                plant_state["target_power_mw"] = target_power
                plant_state["operation_valid"] = valid

                results.append(plant_state)

            df = pd.DataFrame(results)
            if not df.empty:
                df.set_index(PlantStateColumns.TIMESTAMP.value, inplace=True)

            logger.info(
                f"Simulated {len(df)} dispatch intervals for plant '{self.config.name}'"
            )

            return df

        except Exception as e:
            # Reset battery states on error
            for i, battery in enumerate(self.batteries):
                if i < len(initial_battery_states):
                    try:
                        battery.reset_state(initial_battery_states[i])
                    except Exception as reset_error:
                        logger.warning(f"Failed to reset battery {i}: {reset_error}")

            logger.error(f"Plant simulation failed: {e}")
            raise

    def reset_batteries(self) -> None:
        """Reset all batteries to their initial state."""
        for battery in self.batteries:
            battery.reset_state()
        logger.info("Reset all batteries to initial state")

    def get_system_capacity(self) -> Dict[str, float]:
        """Get plant capacity information."""
        pv_capacity = self.pv_model.get_system_capacity()

        return {
            "pv_ac_capacity_mw": pv_capacity["ac_capacity_w"] / 1e6,
            "pv_dc_capacity_mw": pv_capacity["dc_capacity_w"] / 1e6,
            "battery_energy_capacity_mwh": self.total_battery_capacity_mwh,
            "battery_power_capacity_mw": self.total_battery_power_mw,
            "max_net_power_mw": self.config.max_net_power_mw,
            "dc_ac_ratio": pv_capacity["dc_ac_ratio"],
        }

    def get_recommended_max_power_mw(self) -> float:
        """
        Get recommended max_net_power_mw based on PV system AC capacity.

        Returns:
            Recommended maximum net power in MW based on PV capacity
        """
        try:
            pv_capacity = self.pv_model.get_system_capacity()
            pv_ac_capacity_mw = pv_capacity["ac_capacity_w"] / 1e6

            # Return PV AC capacity as the recommended max power
            # This ensures the plant can't exceed its physical generation capability
            return pv_ac_capacity_mw

        except Exception as e:
            logger.warning(
                f"Could not calculate recommended max power for plant "
                f"'{self.config.name}': {e}"
            )
            # Fallback to current configured value
            return self.config.max_net_power_mw

    def __add__(self, other: "SolarBatteryPlant") -> "SolarBatteryPlant":
        """
        Combine two plants into a single aggregated plant.

        Args:
            other: Another SolarBatteryPlant instance

        Returns:
            Combined SolarBatteryPlant instance
        """
        if not isinstance(other, SolarBatteryPlant):
            raise ValueError("Can only combine with another SolarBatteryPlant")

        # Create combined configuration
        combined_config = PlantConfiguration(
            name=f"{self.config.name}+{other.config.name}",
            plant_id=f"{self.config.plant_id or 'P1'}+{other.config.plant_id or 'P2'}",
            max_net_power_mw=(
                self.config.max_net_power_mw + other.config.max_net_power_mw
            ),
            min_net_power_mw=(
                self.config.min_net_power_mw + other.config.min_net_power_mw
            ),
            enable_market_participation=(
                self.config.enable_market_participation
                and other.config.enable_market_participation
            ),
            maintenance_schedule=(
                self.config.maintenance_schedule + other.config.maintenance_schedule
            ),
        )

        # Combine batteries
        combined_batteries = self.batteries + other.batteries

        # PV model combination: Create warning for now as this needs proper
        # implementation
        logger.warning(
            "PV model combination not fully implemented - using first plant's "
            "PV model. Consider implementing proper PV aggregation for "
            "accurate results."
        )

        combined_plant = SolarBatteryPlant(
            config=combined_config,
            pv_model=self.pv_model,  # TODO: Implement proper PV model combination
            batteries=combined_batteries,
            revenue_calculator=self.revenue_calculator,
        )

        logger.info(
            f"Combined plants '{self.config.name}' and '{other.config.name}' "
            f"into '{combined_config.name}'"
        )

        return combined_plant


# -----------------------------------------------------------------------------
# Factory and Helper Functions
# -----------------------------------------------------------------------------


class PlantFactory:
    """Factory for creating power plant instances."""

    @staticmethod
    def create_solar_battery_plant(
        name: str,
        pv_model: PVModel,
        batteries: List[LinearBatterySimulator],
        max_net_power_mw: Optional[float] = None,
        revenue_calculator: Optional[SolarRevenueCalculator] = None,
        plant_id: Optional[str] = None,
    ) -> SolarBatteryPlant:
        """
        Create a solar-battery plant with standard configuration.

        Args:
            name: Plant name
            pv_model: Solar PV generation model
            batteries: List of battery simulators
            max_net_power_mw: Maximum net power limit (auto-calculated if None)
            revenue_calculator: Optional revenue calculation model
            plant_id: Optional unique plant identifier

        Returns:
            Configured SolarBatteryPlant instance
        """
        # Auto-calculate max power if not provided
        if max_net_power_mw is None:
            pv_capacity = pv_model.get_system_capacity()
            total_battery_power = sum(
                battery.config.max_power_mw for battery in batteries
            )
            max_net_power_mw = (
                pv_capacity["ac_capacity_w"] / 1e6
            ) + total_battery_power

        config = PlantConfiguration(
            name=name,
            plant_id=plant_id,
            max_net_power_mw=max_net_power_mw,
        )

        return SolarBatteryPlant(
            config=config,
            pv_model=pv_model,
            batteries=batteries,
            revenue_calculator=revenue_calculator,
        )


# Type aliases
Plant = Union[SolarBatteryPlant]
