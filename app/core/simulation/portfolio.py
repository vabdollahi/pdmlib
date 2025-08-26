"""
Portfolio management for multiple power plants in electricity markets.

This module provides portfolio abstraction for managing multiple power plants
and their coordinated participation in electricity markets, including revenue
optimization and risk management.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Protocol

from app.core.simulation.plant import Plant, SolarBatteryPlant
from app.core.simulation.solar_revenue import SolarRevenueCalculator
from app.core.utils.logging import get_logger

logger = get_logger("portfolio")


# -----------------------------------------------------------------------------
# Constants and Enums
# -----------------------------------------------------------------------------


class PortfolioStrategy(str, Enum):
    """Portfolio optimization strategies for market participation."""

    REVENUE_MAXIMIZATION = "revenue_maximization"
    RISK_MINIMIZATION = "risk_minimization"
    BALANCED = "balanced"
    GRID_SERVICES = "grid_services"


class PortfolioStateColumns(str, Enum):
    """Standardized column names for portfolio state data."""

    TIMESTAMP = "timestamp"
    TOTAL_GENERATION_MW = "total_generation_mw"
    TOTAL_CONSUMPTION_MW = "total_consumption_mw"
    NET_POWER_MW = "net_power_mw"
    REVENUE_USD = "revenue_usd"
    PLANT_COUNT = "plant_count"
    STRATEGY = "strategy"


# -----------------------------------------------------------------------------
# Portfolio Protocol and Base Classes
# -----------------------------------------------------------------------------


class PortfolioProtocol(Protocol):
    """Protocol defining the interface for power plant portfolios."""

    def dispatch_power(
        self,
        target_net_power_mw: float,
        timestamp: datetime.datetime,
        interval_minutes: float = 5.0,
    ) -> Tuple[float, Dict[str, float], bool]:
        """
        Dispatch power across the portfolio.

        Args:
            target_net_power_mw: Net power to dispatch to grid (MW)
            timestamp: Time of dispatch
            interval_minutes: Time interval for the operation

        Returns:
            Tuple of (actual_net_power_mw, portfolio_state, operation_valid)
        """
        ...

    def get_available_power(
        self, timestamp: datetime.datetime, interval_minutes: float = 5.0
    ) -> Tuple[float, float]:
        """
        Get available power generation and consumption capability.

        Args:
            timestamp: Time for availability calculation
            interval_minutes: Time interval for calculation

        Returns:
            Tuple of (max_generation_mw, max_consumption_mw)
        """
        ...


class PortfolioConfiguration(BaseModel):
    """Configuration for power plant portfolios."""

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", use_enum_values=True
    )

    # Portfolio identification
    name: str = Field(description="Portfolio identifier", min_length=1)
    portfolio_id: Optional[str] = Field(default=None, description="Unique portfolio ID")

    # Market strategy
    strategy: PortfolioStrategy = Field(
        default=PortfolioStrategy.BALANCED,
        description="Portfolio optimization strategy",
    )

    # Risk management
    max_portfolio_risk: float = Field(
        default=0.2, description="Maximum portfolio risk factor", ge=0.0, le=1.0
    )
    diversification_weight: float = Field(
        default=0.3,
        description="Weight given to plant diversification",
        ge=0.0,
        le=1.0,
    )

    # Market participation
    enable_market_arbitrage: bool = Field(
        default=True, description="Enable energy market arbitrage"
    )
    enable_ancillary_services: bool = Field(
        default=False, description="Enable ancillary services participation"
    )

    # Portfolio limits
    max_total_power_mw: Optional[float] = Field(
        default=None, description="Maximum total portfolio power (MW)", gt=0
    )
    min_operating_plants: int = Field(
        default=1, description="Minimum number of operating plants", ge=1
    )


# -----------------------------------------------------------------------------
# Power Plant Portfolio Implementation
# -----------------------------------------------------------------------------


class PowerPlantPortfolio(BaseModel):
    """
    Portfolio manager for multiple power plants in electricity markets.

    This portfolio model coordinates multiple plants for optimal market
    participation, considering revenue maximization, risk management,
    and grid service obligations.

    The portfolio supports multiple dispatch strategies:
    - Revenue Maximization: Prioritizes highest revenue generating plants
    - Risk Minimization: Distributes load for diversification
    - Balanced: Combines revenue and risk considerations
    - Grid Services: Equal distribution for stability services

    Example:
        >>> portfolio = PowerPlantPortfolio(
        ...     config=PortfolioConfiguration(name="Solar Farm Portfolio"),
        ...     plants=[plant1, plant2, plant3]
        ... )
        >>> portfolio.set_strategy(PortfolioStrategy.REVENUE_MAXIMIZATION)
        >>> power, state, valid = await portfolio.dispatch_power(100.0, timestamp)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # Core components
    config: PortfolioConfiguration = Field(description="Portfolio configuration")
    plants: List[Plant] = Field(default_factory=list, description="Portfolio plants")

    # Revenue and market integration
    revenue_calculator: Optional[SolarRevenueCalculator] = Field(
        default=None, description="Portfolio revenue calculation model"
    )

    # Portfolio state
    _current_strategy: PortfolioStrategy = PortfolioStrategy.BALANCED
    _current_timestamp: Optional[datetime.datetime] = None

    def __init__(self, **data):
        """Initialize portfolio with validation."""
        super().__init__(**data)
        self._validate_portfolio()
        logger.info(
            f"Initialized portfolio '{self.config.name}' with {len(self.plants)} plants"
        )

    def _validate_portfolio(self) -> None:
        """Validate portfolio composition and constraints."""
        if len(self.plants) < self.config.min_operating_plants:
            raise ValueError(
                f"Portfolio must have at least {self.config.min_operating_plants} "
                f"plants, got {len(self.plants)}"
            )

        # Validate total capacity limits
        if self.config.max_total_power_mw is not None:
            total_capacity = self.get_total_capacity()
            if total_capacity > self.config.max_total_power_mw:
                raise ValueError(
                    f"Total portfolio capacity {total_capacity:.2f} MW exceeds "
                    f"limit {self.config.max_total_power_mw:.2f} MW"
                )

    @property
    def current_strategy(self) -> PortfolioStrategy:
        """Current portfolio optimization strategy."""
        return self._current_strategy

    @property
    def plant_count(self) -> int:
        """Number of plants in portfolio."""
        return len(self.plants)

    @property
    def operating_plant_count(self) -> int:
        """Number of currently operating plants."""
        operating_count = 0
        for plant in self.plants:
            if hasattr(plant, "operation_mode"):
                from app.core.simulation.plant import PlantOperationMode

                if plant.operation_mode not in [
                    PlantOperationMode.MAINTENANCE,
                    PlantOperationMode.EMERGENCY,
                ]:
                    operating_count += 1
            else:
                # Assume plant is operating if no operation mode
                operating_count += 1
        return operating_count

    def add_plant(self, plant: Plant) -> None:
        """Add a plant to the portfolio."""
        if plant in self.plants:
            logger.warning(f"Plant {plant.config.name} already in portfolio")
            return

        # Validate plant compatibility
        if not hasattr(plant, "config") or not hasattr(
            plant.config, "max_net_power_mw"
        ):
            raise ValueError(
                f"Plant {getattr(plant, 'name', 'unknown')} lacks required "
                "configuration"
            )

        self.plants.append(plant)
        logger.info(f"Added plant '{plant.config.name}' to portfolio")

        # Revalidate portfolio after addition
        try:
            self._validate_portfolio()
        except ValueError as e:
            # Remove plant if validation fails
            self.plants.remove(plant)
            logger.error(f"Failed to add plant due to validation error: {e}")
            raise

    def remove_plant(self, plant: Plant) -> bool:
        """Remove a plant from the portfolio."""
        if plant not in self.plants:
            logger.warning(f"Plant {plant.config.name} not in portfolio")
            return False

        self.plants.remove(plant)
        logger.info(f"Removed plant '{plant.config.name}' from portfolio")

        # Check if portfolio still meets minimum requirements
        try:
            self._validate_portfolio()
            return True
        except ValueError as e:
            # Re-add plant if removal violates constraints
            self.plants.append(plant)
            logger.error(f"Cannot remove plant: {e}")
            return False

    def get_total_capacity(self) -> float:
        """Get total portfolio power capacity in MW."""
        total_capacity = 0.0
        for plant in self.plants:
            if hasattr(plant, "config") and hasattr(plant.config, "max_net_power_mw"):
                total_capacity += plant.config.max_net_power_mw
        return total_capacity

    def get_portfolio_diversity_score(self) -> float:
        """
        Calculate portfolio diversity score based on plant characteristics.

        Returns:
            Diversity score between 0 (no diversity) and 1 (maximum diversity)
        """
        if len(self.plants) <= 1:
            return 0.0

        # Analyze plant characteristics for diversity calculation
        capacities = []
        locations = []
        technologies = []

        for plant in self.plants:
            if hasattr(plant, "config"):
                capacities.append(plant.config.max_net_power_mw)

            # Extract technology and location information if available
            if hasattr(plant, "pv_model"):
                technologies.append("solar")
                if hasattr(plant.pv_model, "pv_config"):
                    # Location diversity based on coordinates
                    coord = getattr(plant.pv_model.pv_config, "location", None)
                    if coord and hasattr(coord, "latitude"):
                        locations.append((coord.latitude, coord.longitude))

        # Calculate capacity diversity (lower coefficient of variation =
        # higher diversity)
        capacity_cv = (
            pd.Series(capacities).std() / pd.Series(capacities).mean()
            if capacities
            else 0
        )
        capacity_diversity = max(0, 1 - min(capacity_cv, 1))

        # Geographic diversity (simplified - based on coordinate spread)
        geographic_diversity = 0.5  # Placeholder - would need proper implementation

        # Technology diversity (currently all solar, so limited)
        tech_diversity = 0.1  # Placeholder for future technology mixing

        # Weighted combination
        diversity_score = (
            0.5 * capacity_diversity + 0.3 * geographic_diversity + 0.2 * tech_diversity
        )

        return min(1.0, max(0.0, diversity_score))

    def set_strategy(self, strategy: PortfolioStrategy) -> None:
        """Set portfolio optimization strategy."""
        if strategy != self._current_strategy:
            logger.info(
                f"Portfolio strategy changed: {self._current_strategy} → {strategy}"
            )
            self._current_strategy = strategy

    async def get_available_power(
        self, timestamp: datetime.datetime, interval_minutes: float = 5.0
    ) -> Tuple[float, float]:
        """
        Get total available power generation and consumption capability.

        Args:
            timestamp: Time for availability calculation
            interval_minutes: Time interval for calculation

        Returns:
            Tuple of (max_generation_mw, max_consumption_mw)
        """
        total_generation = 0.0
        total_consumption = 0.0

        for plant in self.plants:
            try:
                if hasattr(plant, "get_available_power"):
                    gen_mw, cons_mw = await plant.get_available_power(
                        timestamp, interval_minutes
                    )
                    total_generation += gen_mw
                    total_consumption += cons_mw
            except Exception as e:
                logger.warning(
                    f"Error getting power availability from plant "
                    f"{plant.config.name}: {e}"
                )

        return total_generation, total_consumption

    def _calculate_plant_allocation(
        self, target_power_mw: float, timestamp: datetime.datetime
    ) -> Dict[Plant, float]:
        """
        Calculate power allocation across plants based on strategy.

        Args:
            target_power_mw: Total target power for portfolio
            timestamp: Time of allocation

        Returns:
            Dictionary mapping plants to their allocated power targets
        """
        allocation = {}

        if not self.plants:
            return allocation

        # Strategy-based allocation logic
        if self._current_strategy == PortfolioStrategy.REVENUE_MAXIMIZATION:
            # Allocate based on expected revenue potential
            allocation = self._allocate_by_revenue_potential(target_power_mw, timestamp)

        elif self._current_strategy == PortfolioStrategy.RISK_MINIMIZATION:
            # Diversified allocation to minimize risk
            allocation = self._allocate_by_risk_distribution(target_power_mw)

        elif self._current_strategy == PortfolioStrategy.BALANCED:
            # Balanced approach considering both revenue and risk
            revenue_allocation = self._allocate_by_revenue_potential(
                target_power_mw, timestamp
            )
            risk_allocation = self._allocate_by_risk_distribution(target_power_mw)

            # Weighted combination
            for plant in self.plants:
                revenue_target = revenue_allocation.get(plant, 0.0)
                risk_target = risk_allocation.get(plant, 0.0)
                allocation[plant] = (
                    0.7 * revenue_target + 0.3 * risk_target
                )  # Favor revenue

        else:  # GRID_SERVICES
            # Equal distribution for grid stability services
            allocation = self._allocate_equally(target_power_mw)

        return allocation

    def _allocate_by_revenue_potential(
        self, target_power_mw: float, timestamp: datetime.datetime
    ) -> Dict[Plant, float]:
        """Allocate power based on revenue generation potential."""
        allocation = {}

        # Simple capacity-weighted allocation (would need more sophisticated
        # revenue modeling in practice)
        total_capacity = self.get_total_capacity()
        if total_capacity == 0:
            return allocation

        for plant in self.plants:
            if hasattr(plant, "config"):
                plant_capacity = plant.config.max_net_power_mw
                capacity_fraction = plant_capacity / total_capacity
                allocation[plant] = target_power_mw * capacity_fraction

        return allocation

    def _allocate_by_risk_distribution(
        self, target_power_mw: float
    ) -> Dict[Plant, float]:
        """Allocate power to minimize portfolio risk through diversification."""
        allocation = {}

        if not self.plants:
            return allocation

        # Risk-weighted allocation based on plant characteristics
        plant_weights = []
        total_capacity = self.get_total_capacity()

        for plant in self.plants:
            # Calculate risk weight (simplified - would need more sophisticated
            # risk modeling)
            base_weight = 1.0
            if hasattr(plant, "config") and total_capacity > 0:
                # Smaller plants get higher weights for diversification
                capacity_fraction = plant.config.max_net_power_mw / total_capacity
                # Inverse weighting for diversification (smaller plants = higher weight)
                base_weight = 1.0 / (
                    capacity_fraction + 0.1
                )  # Add small value to avoid division by zero

            plant_weights.append(base_weight)

        total_weight = sum(plant_weights)
        if total_weight == 0:
            return self._allocate_equally(target_power_mw)

        for i, plant in enumerate(self.plants):
            weight_fraction = plant_weights[i] / total_weight
            allocation[plant] = target_power_mw * weight_fraction

        return allocation

    def _allocate_equally(self, target_power_mw: float) -> Dict[Plant, float]:
        """Allocate power equally across all plants."""
        allocation = {}
        if not self.plants:
            return allocation

        power_per_plant = target_power_mw / len(self.plants)
        for plant in self.plants:
            allocation[plant] = power_per_plant

        return allocation

    async def dispatch_power(
        self,
        target_net_power_mw: float,
        timestamp: datetime.datetime,
        interval_minutes: float = 5.0,
    ) -> Tuple[float, Dict, bool]:
        """
        Dispatch power across the portfolio using optimization strategy.

        Args:
            target_net_power_mw: Target net power to grid (MW)
            timestamp: Time of dispatch
            interval_minutes: Time interval for operation

        Returns:
            Tuple of (actual_net_power_mw, portfolio_state, operation_valid)
        """
        self._current_timestamp = timestamp

        if not self.plants:
            return 0.0, self._get_portfolio_state(timestamp, 0.0, 0.0), False

        # Calculate power allocation across plants
        plant_allocations = self._calculate_plant_allocation(
            target_net_power_mw, timestamp
        )

        # Dispatch to individual plants
        total_actual_power = 0.0
        total_generation = 0.0
        total_consumption = 0.0
        all_operations_valid = True

        plant_results = {}

        for plant, target_power in plant_allocations.items():
            try:
                if hasattr(plant, "dispatch_power"):
                    actual_power, plant_state, valid = await plant.dispatch_power(
                        target_power, timestamp, interval_minutes
                    )

                    total_actual_power += actual_power
                    plant_results[plant.config.name] = {
                        "target_mw": target_power,
                        "actual_mw": actual_power,
                        "valid": valid,
                    }

                    # Extract generation/consumption details if available
                    if "pv_generation_mw" in plant_state:
                        total_generation += plant_state["pv_generation_mw"]
                    if "battery_power_mw" in plant_state:
                        battery_power = plant_state["battery_power_mw"]
                        if battery_power > 0:
                            total_generation += battery_power
                        else:
                            total_consumption += abs(battery_power)

                    if not valid:
                        all_operations_valid = False

            except Exception as e:
                logger.error(
                    f"Error dispatching power to plant {plant.config.name}: {e}"
                )
                all_operations_valid = False

        # Create portfolio state with extended information
        portfolio_state = self._get_portfolio_state(
            timestamp, total_generation, total_consumption
        )
        portfolio_state.update(
            {
                "plant_results": plant_results,
                "target_power_mw": target_net_power_mw,
                "allocation_efficiency": (
                    total_actual_power / target_net_power_mw
                    if target_net_power_mw != 0
                    else 1.0
                ),
            }
        )

        logger.debug(
            f"Portfolio dispatch: target={target_net_power_mw:.2f}MW, "
            f"actual={total_actual_power:.2f}MW, "
            f"plants={len(plant_allocations)}, valid={all_operations_valid}"
        )

        return total_actual_power, portfolio_state, all_operations_valid

    def _get_portfolio_state(
        self,
        timestamp: datetime.datetime,
        total_generation_mw: float,
        total_consumption_mw: float,
    ) -> Dict:
        """Get current portfolio state information."""
        net_power_mw = total_generation_mw - total_consumption_mw

        state = {
            PortfolioStateColumns.TIMESTAMP.value: timestamp,
            PortfolioStateColumns.TOTAL_GENERATION_MW.value: total_generation_mw,
            PortfolioStateColumns.TOTAL_CONSUMPTION_MW.value: total_consumption_mw,
            PortfolioStateColumns.NET_POWER_MW.value: net_power_mw,
            PortfolioStateColumns.PLANT_COUNT.value: len(self.plants),
            PortfolioStateColumns.STRATEGY.value: self._current_strategy,
        }

        # Add revenue calculation if available
        if self.revenue_calculator:
            try:
                # Portfolio-level revenue calculation would need implementation
                state[PortfolioStateColumns.REVENUE_USD.value] = 0.0
            except Exception as e:
                logger.warning(f"Error calculating portfolio revenue: {e}")
                state[PortfolioStateColumns.REVENUE_USD.value] = 0.0

        return state

    async def simulate_operation(
        self,
        power_schedule_mw: pd.Series,
        interval_minutes: float = 5.0,
    ) -> pd.DataFrame:
        """
        Simulate portfolio operation over a power dispatch schedule.

        Args:
            power_schedule_mw: Time series of target net power values (MW)
            interval_minutes: Time interval between dispatch points

        Returns:
            DataFrame with portfolio operation results
        """
        if power_schedule_mw.empty:
            return pd.DataFrame()

        results = []

        try:
            for ts in power_schedule_mw.index:
                target_power = power_schedule_mw.loc[ts]
                actual_power, portfolio_state, valid = await self.dispatch_power(
                    target_power, ts, interval_minutes
                )

                # Add operation validity to state
                portfolio_state["target_power_mw"] = target_power
                portfolio_state["operation_valid"] = valid

                results.append(portfolio_state)

            df = pd.DataFrame(results)
            if not df.empty:
                df.set_index(PortfolioStateColumns.TIMESTAMP.value, inplace=True)

            logger.info(
                f"Simulated {len(df)} dispatch intervals for portfolio "
                f"'{self.config.name}'"
            )

            return df

        except Exception as e:
            logger.error(f"Portfolio simulation failed: {e}")
            raise

    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Get portfolio performance and risk metrics."""
        return {
            "total_capacity_mw": self.get_total_capacity(),
            "plant_count": self.plant_count,
            "operating_plant_count": self.operating_plant_count,
            "diversity_score": self.get_portfolio_diversity_score(),
            "capacity_utilization": (
                self.operating_plant_count / self.plant_count
                if self.plant_count > 0
                else 0.0
            ),
            "risk_score": self.config.max_portfolio_risk,
        }

    def optimize_plant_mix(self) -> Dict[str, Union[str, float]]:
        """
        Analyze and recommend portfolio optimization.

        Returns:
            Dictionary with optimization recommendations
        """
        metrics = self.get_portfolio_metrics()

        recommendations = {
            "current_strategy": self._current_strategy,
            "diversity_score": metrics["diversity_score"],
            "optimization_potential": "medium",  # Placeholder
        }

        # Simple optimization logic
        if metrics["diversity_score"] < 0.3:
            recommendations["recommendation"] = "Increase portfolio diversity"
            recommendations["optimization_potential"] = "high"
        elif metrics["capacity_utilization"] < 0.8:
            recommendations["recommendation"] = "Improve plant availability"
            recommendations["optimization_potential"] = "medium"
        else:
            recommendations["recommendation"] = "Portfolio well-optimized"
            recommendations["optimization_potential"] = "low"

        return recommendations

    def reset_all_plants(self) -> None:
        """Reset all plants in the portfolio to their initial state."""
        for plant in self.plants:
            if hasattr(plant, "reset_batteries"):
                plant.reset_batteries()
        logger.info(f"Reset all {len(self.plants)} plants in portfolio")

    def __add__(self, other: "PowerPlantPortfolio") -> "PowerPlantPortfolio":
        """
        Combine two portfolios into a single merged portfolio.

        Args:
            other: Another PowerPlantPortfolio instance

        Returns:
            Combined PowerPlantPortfolio instance
        """
        if not isinstance(other, PowerPlantPortfolio):
            raise ValueError("Can only combine with another PowerPlantPortfolio")

        # Create combined configuration
        combined_config = PortfolioConfiguration(
            name=f"{self.config.name}+{other.config.name}",
            portfolio_id=(
                f"{self.config.portfolio_id or 'PF1'}+"
                f"{other.config.portfolio_id or 'PF2'}"
            ),
            strategy=self.config.strategy,  # Use first portfolio's strategy
            max_portfolio_risk=max(
                self.config.max_portfolio_risk, other.config.max_portfolio_risk
            ),
            diversification_weight=(
                self.config.diversification_weight + other.config.diversification_weight
            )
            / 2,
            enable_market_arbitrage=(
                self.config.enable_market_arbitrage
                and other.config.enable_market_arbitrage
            ),
            min_operating_plants=(
                self.config.min_operating_plants + other.config.min_operating_plants
            ),
        )

        # Combine plants
        combined_plants = self.plants + other.plants

        combined_portfolio = PowerPlantPortfolio(
            config=combined_config,
            plants=combined_plants,
            revenue_calculator=self.revenue_calculator,
        )

        logger.info(
            f"Combined portfolios '{self.config.name}' and "
            f"'{other.config.name}' into '{combined_config.name}'"
        )

        return combined_portfolio


# -----------------------------------------------------------------------------
# Factory and Helper Functions
# -----------------------------------------------------------------------------


class PortfolioFactory:
    """Factory for creating power plant portfolio instances."""

    @staticmethod
    def create_solar_portfolio(
        name: str,
        plants: List[SolarBatteryPlant],
        strategy: PortfolioStrategy = PortfolioStrategy.BALANCED,
        revenue_calculator: Optional[SolarRevenueCalculator] = None,
        portfolio_id: Optional[str] = None,
    ) -> PowerPlantPortfolio:
        """
        Create a solar plant portfolio with standard configuration.

        Args:
            name: Portfolio name
            plants: List of solar-battery plants
            strategy: Portfolio optimization strategy
            revenue_calculator: Optional revenue calculation model
            portfolio_id: Optional unique portfolio identifier

        Returns:
            Configured PowerPlantPortfolio instance
        """
        if not plants:
            raise ValueError("Portfolio must contain at least one plant")

        config = PortfolioConfiguration(
            name=name,
            portfolio_id=portfolio_id,
            strategy=strategy,
            min_operating_plants=min(len(plants), 1),
        )

        return PowerPlantPortfolio(
            config=config,
            plants=plants,
            revenue_calculator=revenue_calculator,
        )

    @staticmethod
    def create_diversified_portfolio(
        name: str,
        plants: List[Plant],
        max_risk: float = 0.2,
        portfolio_id: Optional[str] = None,
    ) -> PowerPlantPortfolio:
        """
        Create a risk-diversified portfolio.

        Args:
            name: Portfolio name
            plants: List of plants with different characteristics
            max_risk: Maximum portfolio risk level
            portfolio_id: Optional unique portfolio identifier

        Returns:
            Configured PowerPlantPortfolio with risk minimization strategy
        """
        config = PortfolioConfiguration(
            name=name,
            portfolio_id=portfolio_id,
            strategy=PortfolioStrategy.RISK_MINIMIZATION,
            max_portfolio_risk=max_risk,
            diversification_weight=0.6,  # Higher weight on diversification
        )

        return PowerPlantPortfolio(
            config=config,
            plants=plants,
        )


# Type aliases
Portfolio = Union[PowerPlantPortfolio]
