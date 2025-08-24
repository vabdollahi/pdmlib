"""
PVLib-based photovoltaic system modeling components.

This module provides configurable PV system components including mounts, arrays,
inverters, and complete system models for solar power simulation.
"""

from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from pvlib import location, modelchain, pvsystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pydantic import BaseModel, Field

from app.core.utils.logging import get_logger

logger = get_logger("pvlib_models")


@lru_cache(maxsize=128)
def _get_inverter_parameters(database: str, record: str) -> Dict[str, float]:
    """Get inverter parameters from database with caching."""
    try:
        inverters = pvsystem.retrieve_sam(database)
        return dict(inverters[record])
    except KeyError as e:
        raise ValueError(f"Inverter not found: {database}/{record}") from e


@lru_cache(maxsize=128)
def _get_module_parameters(database: str, record: str) -> Dict[str, float]:
    """Get module parameters from database with caching."""
    try:
        modules = pvsystem.retrieve_sam(database)
        module_parameters = modules[record]
        # Convert Series to dict if needed
        if hasattr(module_parameters, "to_dict"):
            return module_parameters.to_dict()
        return dict(module_parameters)
    except KeyError as e:
        raise ValueError(f"Module not found: {database}/{record}") from e


@lru_cache(maxsize=64)
def _get_temperature_model_parameters(database: str, record: str) -> Dict[str, float]:
    """Get temperature model parameters with caching."""
    try:
        temperature_model = TEMPERATURE_MODEL_PARAMETERS[database][record]
        return dict(temperature_model)
    except KeyError as e:
        raise ValueError(f"Temperature model not found: {database}/{record}") from e


class InverterParams(TypedDict):
    """Type definition for inverter parameters."""

    pdc0: float
    pac0: float


class ModuleParams(TypedDict):
    """Type definition for module parameters."""

    pdc0: float
    gamma_pdc: float


class TemperatureParams(TypedDict):
    """Type definition for temperature model parameters."""

    a: float
    b: float
    deltaT: float


class SimulationParams(TypedDict, total=False):
    """Type definition for simulation parameters."""

    aoi_model: Any
    spectral_model: Any


class PVLibResultsColumns(Enum):
    """Column names for PVLib simulation results."""

    DATE_TIME = "date_time"
    AC = "Total AC power (W)"
    DC = "Total DC power (W)"
    DC_ARRAY = "DC power (W)"


class PvLibComponentCreatorMixin(ABC):
    """Abstract mixin for PVLib component creation."""

    @abstractmethod
    def create(self) -> Any:
        """Create the corresponding PVLib component."""
        raise NotImplementedError


class MountBase(BaseModel, PvLibComponentCreatorMixin):
    """Base class for mount configuration."""

    pass


class MountFixed(MountBase):
    """Fixed mount configuration for solar panels."""

    type: Literal["fixed_mount"]
    tilt_degrees: Optional[float] = Field(
        default=None,
        ge=0,
        le=90,
        description="Surface tilt angle with respect to the horizon",
    )
    azimuth_degrees: Optional[float] = Field(
        default=None,
        ge=0,
        le=360,
        description=(
            "Azimuth angle of the module surface. "
            "North=0, East=90, South=180, West=270."
        ),
    )
    racking_model: Optional[str] = Field(default=None, description="Racking model name")
    module_height_meter: Optional[float] = Field(
        default=None, ge=0, le=50, description="Module height above ground in meters"
    )

    def create(self) -> pvsystem.FixedMount:
        """Create a PVLib FixedMount object."""
        # Filter out None values to pass only valid parameters
        params = {}
        if self.tilt_degrees is not None:
            params["surface_tilt"] = self.tilt_degrees
        if self.azimuth_degrees is not None:
            params["surface_azimuth"] = self.azimuth_degrees
        if self.racking_model is not None:
            params["racking_model"] = self.racking_model
        if self.module_height_meter is not None:
            params["module_height"] = self.module_height_meter

        mount = pvsystem.FixedMount(**params)
        logger.debug(
            f"Created FixedMount: tilt={self.tilt_degrees}°, "
            f"azimuth={self.azimuth_degrees}°"
        )
        return mount


class MountSingleAxis(MountBase):
    """Single-axis tracking mount configuration."""

    type: Literal["single_axis"]
    axis_tilt_degrees: Optional[float] = Field(
        default=None,
        ge=0,
        le=90,
        description="The tilt of the axis of rotation with respect to the horizon",
    )
    axis_azimuth_degrees: Optional[float] = Field(
        default=None,
        ge=0,
        le=360,
        description=(
            "The direction along which the axis of rotation lies, "
            "measured east of north"
        ),
    )
    max_angle_degrees: Optional[float] = Field(
        default=90,
        ge=0,
        le=180,
        description=(
            "Maximum rotation angle of the one-axis tracker from its "
            "horizontal position"
        ),
    )
    backtrack: Optional[bool] = Field(
        default=True, description="If the tracker has the capability to backtrack"
    )
    gcr: Optional[float] = Field(
        default=2.0 / 7.0,
        ge=0,
        le=1,
        description=(
            "Ground coverage ratio of a tracker system which utilizes backtracking"
        ),
    )
    cross_axis_tilt_degrees: Optional[float] = Field(
        default=0.0,
        ge=-90,
        le=90,
        description="Angle of the panel relative to the axis of rotation",
    )
    racking_model: Optional[str] = Field(default=None, description="Racking model name")
    module_height_meter: Optional[float] = Field(
        default=None, ge=0, le=50, description="Module height above ground in meters"
    )

    def create(self) -> pvsystem.SingleAxisTrackerMount:
        """Create a PVLib SingleAxisTrackerMount object."""
        # Filter out None values to pass only valid parameters
        params = {}
        if self.axis_tilt_degrees is not None:
            params["axis_tilt"] = self.axis_tilt_degrees
        if self.axis_azimuth_degrees is not None:
            params["axis_azimuth"] = self.axis_azimuth_degrees
        if self.max_angle_degrees is not None:
            params["max_angle"] = self.max_angle_degrees
        if self.backtrack is not None:
            params["backtrack"] = self.backtrack
        if self.gcr is not None:
            params["gcr"] = self.gcr
        if self.cross_axis_tilt_degrees is not None:
            params["cross_axis_tilt"] = self.cross_axis_tilt_degrees
        if self.racking_model is not None:
            params["racking_model"] = self.racking_model
        if self.module_height_meter is not None:
            params["module_height"] = self.module_height_meter

        mount = pvsystem.SingleAxisTrackerMount(**params)
        logger.debug(
            f"Created SingleAxisTrackerMount: axis_tilt={self.axis_tilt_degrees}°"
        )
        return mount


class TemperatureModel(BaseModel, PvLibComponentCreatorMixin):
    """Temperature model configuration for PV modules."""

    database: str = Field(default="sapm", description="Temperature model database name")
    record: str = Field(
        default="open_rack_glass_glass", description="Record name in the database"
    )

    def create(self) -> Dict[str, float]:
        """Create temperature model parameters."""
        try:
            temperature_model = _get_temperature_model_parameters(
                self.database, self.record
            )
            logger.debug(f"Created temperature model: {self.database}/{self.record}")
            return temperature_model
        except ValueError:
            # Re-raise ValueError from cached function
            raise


class InverterBase(BaseModel, PvLibComponentCreatorMixin):
    """Base class for inverter configuration."""

    pass


class InverterDatabase(InverterBase):
    """Inverter configuration from PVLib database."""

    count: int = Field(gt=0, description="Number of inverters")
    database: str = Field(description="Inverter model database name")
    record: str = Field(description="Record name in the database")
    _efficiency_rating_percent: Optional[float] = None

    def create(self) -> InverterParams:
        """Create inverter parameters from database."""
        try:
            inverter_parameters = _get_inverter_parameters(self.database, self.record)

            # Scale parameters by inverter count
            scaled_parameters: InverterParams = {
                "pdc0": float(inverter_parameters["Pdco"] * self.count),
                "pac0": float(inverter_parameters["Paco"] * self.count),
            }

            self._efficiency_rating_percent = (
                scaled_parameters["pac0"] / scaled_parameters["pdc0"]
            )

            logger.debug(
                f"Created inverter from database: {self.database}/"
                f"{self.record}, count={self.count}"
            )
            return scaled_parameters
        except ValueError:
            # Re-raise ValueError from cached function
            raise

    @property
    def efficiency_rating_percent(self) -> Optional[float]:
        """Get the efficiency rating percentage."""
        return self._efficiency_rating_percent


class InverterParameters(InverterBase):
    """Manual inverter configuration with explicit parameters."""

    count: int = Field(gt=0, description="Number of inverters")
    max_power_output_ac_w: float = Field(
        gt=0,
        description=(
            "Maximum AC power generation of the inverter at standard conditions (pac0)"
        ),
    )
    efficiency_rating_percent: float = Field(
        gt=0, le=1, description="DC to AC power conversion efficiency"
    )

    def create(self) -> InverterParams:
        """Create inverter parameters from explicit values."""
        pac0 = self.max_power_output_ac_w * self.count
        pdc0 = pac0 / self.efficiency_rating_percent

        inverter_parameters: InverterParams = {
            "pdc0": pdc0,
            "pac0": pac0,
        }

        logger.debug(
            f"Created inverter parameters: pac0={pac0}W, pdc0={pdc0}W, "
            f"count={self.count}"
        )
        return inverter_parameters


class ModuleBase(BaseModel, PvLibComponentCreatorMixin):
    """Base class for PV module configuration."""

    pass


class ModuleDatabase(ModuleBase):
    """PV module configuration from PVLib database."""

    count: int = Field(gt=0, description="Number of PV modules")
    name: str = Field(min_length=1, description="Name of PV modules")
    database: str = Field(description="PV modules model database name")
    record: str = Field(description="Record name in the database")

    def create(self) -> Dict[str, float]:
        """Create module parameters from database."""
        if self.database != "CECMod":
            raise NotImplementedError("Only CECMod database is currently supported.")

        try:
            module_parameters = _get_module_parameters(self.database, self.record)
            logger.debug(f"Created module from database: {self.database}/{self.record}")
            return module_parameters
        except ValueError:
            # Re-raise ValueError from cached function
            raise


class ModuleParameters(ModuleBase):
    """Manual PV module configuration with explicit parameters."""

    count: int = Field(gt=0, description="Number of PV modules")
    name: str = Field(min_length=1, description="Name of PV modules")
    nameplate_dc_rating_w: float = Field(
        gt=0,
        le=1000,
        description=(
            "Power of the modules at 1000 W/m^2 and cell reference temperature (pdc0)"
        ),
    )
    power_temperature_coefficient_per_degree_c: float = Field(
        ge=-1, le=0, description="The temperature coefficient of power (gamma_pdc)"
    )

    def create(self) -> ModuleParams:
        """Create module parameters from explicit values."""
        module_parameters: ModuleParams = {
            "pdc0": self.nameplate_dc_rating_w,
            "gamma_pdc": self.power_temperature_coefficient_per_degree_c,
        }
        logger.debug(f"Created module parameters: pdc0={self.nameplate_dc_rating_w}W")
        return module_parameters


class ArraySetup(BaseModel):
    """Configuration for PV array setup."""

    name: str = Field(min_length=1, description="Array name identifier")
    mount: Union[MountFixed, MountSingleAxis] = Field(
        discriminator="type", description="Mount configuration"
    )
    number_of_strings: int = Field(
        gt=0, le=10000, description="Number of PV strings in the array"
    )
    temperature_model: Optional[TemperatureModel] = Field(
        default=None, description="Module temperature model"
    )


class PvArray(BaseModel, PvLibComponentCreatorMixin):
    """Configuration for PV array."""

    pv_modules: Union[ModuleDatabase, ModuleParameters] = Field(
        description="PV module configuration"
    )
    array_setup: ArraySetup = Field(description="Array setup configuration")

    def create(self) -> pvsystem.Array:
        """Create a PVLib Array object."""
        temperature_params = None
        if self.array_setup.temperature_model:
            temperature_params = self.array_setup.temperature_model.create()

        array = pvsystem.Array(
            mount=self.array_setup.mount.create(),
            name=self.array_setup.name,
            module=self.pv_modules.name,
            strings=self.array_setup.number_of_strings,
            modules_per_string=self.pv_modules.count,
            module_parameters=self.pv_modules.create(),
            temperature_model_parameters=temperature_params,
        )

        logger.debug(
            f"Created array '{self.array_setup.name}' with "
            f"{self.array_setup.number_of_strings} strings"
        )
        return array


class PvSystem(BaseModel, PvLibComponentCreatorMixin):
    """Configuration for complete PV system."""

    inverters: Union[InverterDatabase, InverterParameters] = Field(
        description="Inverter configuration"
    )
    pv_arrays: List[PvArray] = Field(description="List of PV arrays")
    _max_power_output_ac_w: Optional[float] = None

    def create(self) -> pvsystem.PVSystem:
        """Create a PVLib PVSystem object."""
        self._check_module_types(self.pv_arrays)

        arrays = [array.create() for array in self.pv_arrays]
        inverter_parameters = self.inverters.create()
        self._max_power_output_ac_w = inverter_parameters["pac0"]

        pv_system = pvsystem.PVSystem(
            arrays=arrays, inverter_parameters=inverter_parameters
        )

        logger.debug(
            f"Created PV system with {len(arrays)} arrays and "
            f"{inverter_parameters['pac0']}W AC capacity"
        )
        return pv_system

    @classmethod
    def _check_module_types(cls, pv_arrays: List[PvArray]) -> None:
        """Ensure all modules are of the same type for DC model inference."""
        if not pv_arrays:
            return

        module_type = type(pv_arrays[0].pv_modules)
        for array_config in pv_arrays[1:]:
            current_module_type = type(array_config.pv_modules)
            if current_module_type != module_type:
                raise ValueError(
                    "All pv_modules within pv_arrays must have the same DC model type."
                )

    @property
    def max_power_output_ac_w(self) -> Optional[float]:
        """Get maximum AC power output in watts."""
        return self._max_power_output_ac_w


class Location(BaseModel, PvLibComponentCreatorMixin):
    """Geographic location configuration for PV system."""

    name: str = Field(min_length=1, description="Location name")
    latitude: float = Field(ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(ge=-180, le=180, description="Longitude in degrees")
    tz: str = Field(min_length=1, description="Timezone identifier")
    altitude: Optional[float] = Field(
        default=None, ge=-500, le=9000, description="Altitude in meters"
    )

    def create(self) -> location.Location:
        """Create a PVLib Location object."""
        pv_location = location.Location(
            name=self.name,
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            tz=self.tz,
        )

        logger.debug(
            f"Created location: {self.name} ({self.latitude}, {self.longitude})"
        )
        return pv_location


class PhysicalSimulation(BaseModel, PvLibComponentCreatorMixin):
    """Configuration for PVLib simulation parameters."""

    aoi_model: Optional[str] = Field(
        default="physical", description="Angle of incidence model"
    )
    spectral_model: Optional[str] = Field(
        default="no_loss", description="Model for spectral distribution"
    )

    def create(self) -> SimulationParams:
        """Create simulation parameters dictionary."""
        simulation_parameters: SimulationParams = {}
        if self.aoi_model is not None:
            simulation_parameters["aoi_model"] = self.aoi_model
        if self.spectral_model is not None:
            simulation_parameters["spectral_model"] = self.spectral_model

        logger.debug(f"Created simulation parameters: {simulation_parameters}")
        return simulation_parameters


class PVLibModel(BaseModel, PvLibComponentCreatorMixin):
    """Complete PVLib model configuration."""

    location: Location = Field(description="Geographic location")
    pv_systems: List[PvSystem] = Field(description="List of PV systems")
    physical_simulation: Optional[PhysicalSimulation] = Field(
        default=None, description="Physical simulation parameters"
    )

    def create(self) -> modelchain.ModelChain:
        """Create a PVLib ModelChain object."""
        if len(self.pv_systems) > 1:
            raise NotImplementedError("Multiple PV systems are not yet supported.")

        if not self.pv_systems:
            raise ValueError("At least one PV system must be provided.")

        pv_system = self.pv_systems[0].create()
        pv_location = self.location.create()

        simulation_params = {}
        if self.physical_simulation:
            simulation_params = dict(self.physical_simulation.create())

        # Type: ignore due to PVLib's flexible parameter acceptance
        model_chain = modelchain.ModelChain(pv_system, pv_location, **simulation_params)  # type: ignore[misc]

        logger.info(
            f"Created ModelChain for location '{self.location.name}' with "
            f"{len(pv_system.arrays)} arrays"
        )
        return model_chain
