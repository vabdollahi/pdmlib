"""
Strict Pydantic spec models for simulation configuration.

These are used to validate incoming configuration dictionaries from
simulation_main.py or JSON files before constructing runtime objects.
"""

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Import provider configuration models
from app.core.simulation.provider_config import (
    PriceProviderConfig,
    WeatherProviderConfig,
)


class PriceSourceCSV(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["csv_file"]
    name: str
    data: str


PriceSource = Union[PriceSourceCSV, PriceProviderConfig]


class WeatherSourceCSV(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["csv_file"]
    data: str


WeatherSource = Union[WeatherSourceCSV, WeatherProviderConfig]


class MarketSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prices: List[PriceSource] = Field(min_length=1)
    forecast_prices: Optional[List[PriceSource]] = None
    weather: WeatherSource


class AgentSpec(BaseModel):
    """Specification for automatic agent creation."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["BasicHeuristic"] = Field(
        default="BasicHeuristic", description="Type of agent to create"
    )
    parameters: Optional[dict] = Field(
        default=None, description="Agent-specific configuration parameters"
    )
    enabled: bool = Field(
        default=True, description="Whether the agent is enabled for execution"
    )
    name: Optional[str] = Field(
        default=None, description="Optional name for the agent instance"
    )


class PlantSpec(BaseModel):
    """PVLib-detailed plant spec wrapper.

    We intentionally do not re-model the full PVLib JSON here; instead
    we validate the presence of required top-level keys and pass the
    dictionary onwards to PVLibModel for deep validation.
    """

    model_config = ConfigDict(extra="forbid")

    # Store the full dict to pass into PVLibModel
    location: Any
    pv_systems: Any
    plant_config: Any
    batteries: Optional[List[Any]] = None

    @model_validator(mode="after")
    def validate_required(self) -> "PlantSpec":
        if not isinstance(self.location, dict) or not {
            "latitude",
            "longitude",
        }.issubset(self.location.keys()):
            raise ValueError("location must include latitude and longitude")
        if not isinstance(self.pv_systems, list) or len(self.pv_systems) == 0:
            raise ValueError("pv_systems must be a non-empty list")
        if not isinstance(self.plant_config, dict):
            raise ValueError("plant_config must be an object")
        return self


class PortfolioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    max_total_power_mw: Optional[float] = None
    allow_grid_purchase: bool = False
    plants: List[PlantSpec]
    market: MarketSpec


class EnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date_time: str
    end_date_time: str
    interval_min: int = 60

    historic_data_intervals: int = 12
    forecast_data_intervals: int = 12

    power_normalization_coefficient: float = 1e6
    price_normalization_coefficient: float = 100.0

    action_tolerance_percent: float = 0.01
    smoothed_reward_parameter: float = 0.1

    max_grid_purchase_mw: float = 50.0
    grid_purchase_enabled: bool = False

    portfolios: List[PortfolioSpec]

    # Agent configuration (optional)
    agent: Optional[AgentSpec] = Field(
        default=None, description="Agent configuration for automatic agent creation"
    )
