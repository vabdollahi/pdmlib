"""
Environment configuration for power management gymnasium environment.
"""

import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.simulation.portfolio import PowerPlantPortfolio
from app.core.simulation.price_provider import BasePriceProvider


class EnvironmentConfig(BaseModel):
    """Configuration for the power management environment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # Time configuration
    start_date_time: datetime.datetime = Field(
        description="Start datetime for the environment"
    )
    end_date_time: datetime.datetime = Field(
        description="End datetime for the environment"
    )
    interval_min: float = Field(
        default=5.0, description="Time interval in minutes between actions"
    )

    # Portfolio and system configuration
    portfolios: List[PowerPlantPortfolio] = Field(
        description="List of power plant portfolios to manage"
    )

    # Market data configuration
    price_provider: Optional[BasePriceProvider] = Field(
        default=None, description="Electricity price data provider"
    )

    # Grid purchase configuration
    max_grid_purchase_mw: float = Field(
        default=50.0, description="Maximum power that can be purchased from grid (MW)"
    )
    grid_purchase_enabled: bool = Field(
        default=False,
        description="Whether grid purchases are allowed for all portfolios",
    )

    # Data configuration
    historic_data_intervals: int = Field(
        default=12,
        description="Number of historic intervals to include in observations",
    )
    forecast_data_intervals: int = Field(
        default=12,
        description="Number of forecast intervals to include in observations",
    )

    # Normalization parameters
    power_normalization_coefficient: float = Field(
        default=1e6, description="Power normalization factor (e.g., MW to W)"
    )
    price_normalization_coefficient: float = Field(
        default=100.0, description="Price normalization factor"
    )

    # Action constraints
    action_tolerance_percent: float = Field(
        default=0.05, description="Tolerance for action validation (0.0-1.0)"
    )

    # Reward configuration
    smoothed_reward_parameter: float = Field(
        default=0.1, description="Smoothing parameter for reward calculation (0.0-1.0)"
    )

    # Environment state
    _trial_start: Optional[datetime.datetime] = None
    _trial_end: Optional[datetime.datetime] = None

    def set_trial_data(
        self,
        start_date_time: datetime.datetime,
        end_date_time: datetime.datetime,
        battery_state_of_charge: Optional[float] = None,
    ) -> None:
        """Set trial data configuration."""
        self._trial_start = start_date_time
        self._trial_end = end_date_time

        # Reset battery states if specified
        if battery_state_of_charge is not None:
            for portfolio in self.portfolios:
                for plant in portfolio.plants:
                    for battery in plant.batteries:
                        battery.reset_state(battery_state_of_charge)


@staticmethod
def load_portfolio_config(config_path: Path) -> Dict:
    """Load portfolio configuration from JSON file."""
    import json

    with open(config_path, "r") as f:
        return json.load(f)


@staticmethod
def create_environment_config_from_json(
    config_path: Path,
    start_date_time: datetime.datetime,
    end_date_time: datetime.datetime,
    **kwargs,
) -> EnvironmentConfig:
    """Create environment configuration from JSON portfolio config."""
    # This would need to be implemented to create portfolio from config
    # For now, raise NotImplementedError to indicate this needs integration
    # with the portfolio creation system
    raise NotImplementedError(
        "Portfolio creation from JSON config needs to be implemented. "
        "This requires integration with the main portfolio system."
    )
