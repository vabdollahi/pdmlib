"""
Consolidated PV/PVLib Test Suite

Streamlined test suite covering all essential PV model functionality with
minimal redundancy. Replaces test_component_validation.py,
test_integration_consolidation.py, and test_final_consolidation.py.
"""

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from app.core.simulation.pvlib_models import (
    InverterParameters,
    Location,
    ModuleParameters,
    MountFixed,
    MountSingleAxis,
    PVLibModel,
)


class TestComponentValidation:
    """Essential component validation tests - consolidated from 23 tests to 5."""

    def test_location_validation(self):
        """Test location validation (consolidates 3 parametric tests)."""
        # Valid location
        location = Location(name="Test", latitude=33.4, longitude=-112.0, tz="UTC")
        assert location.latitude == 33.4

        # Invalid latitude/longitude
        with pytest.raises(ValidationError):
            Location(name="Test", latitude=95.0, longitude=-112.0, tz="UTC")
        with pytest.raises(ValidationError):
            Location(name="Test", latitude=33.4, longitude=185.0, tz="UTC")

    def test_mount_validation(self):
        """Test mount validation (consolidates 6 parametric tests)."""
        # Valid fixed mount
        fixed = MountFixed(type="fixed_mount", tilt_degrees=30, azimuth_degrees=180)
        assert fixed.tilt_degrees == 30

        # Valid single axis mount
        tracker = MountSingleAxis(type="single_axis", axis_tilt_degrees=0, gcr=0.4)
        assert tracker.gcr == 0.4

        # Invalid values
        with pytest.raises(ValidationError):
            MountFixed(type="fixed_mount", tilt_degrees=-5, azimuth_degrees=180)

    def test_module_validation(self):
        """Test module validation (consolidates 2 skipped parametric tests)."""
        # Valid module
        module = ModuleParameters(
            count=20,
            name="Test",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        assert module.nameplate_dc_rating_w == 300.0

        # Invalid values
        with pytest.raises(ValidationError):
            ModuleParameters(
                count=0,
                name="Test",
                nameplate_dc_rating_w=300.0,
                power_temperature_coefficient_per_degree_c=-0.004,
            )

    def test_inverter_validation(self):
        """Test inverter validation (consolidates 2 skipped parametric tests)."""
        # Valid inverter
        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )
        assert inverter.efficiency_rating_percent == 0.95

        # Invalid values
        with pytest.raises(ValidationError):
            InverterParameters(
                count=1,
                max_power_output_ac_w=5000.0,
                efficiency_rating_percent=1.5,
            )

    def test_component_creation(self):
        """Test components can create PVLib objects (consolidates 8 tests)."""
        # Location
        location = Location(name="Test", latitude=33.4, longitude=-112.0, tz="UTC")
        location_obj = location.create()
        assert hasattr(location_obj, "latitude")

        # Mounts
        fixed_mount = MountFixed(
            type="fixed_mount", tilt_degrees=30, azimuth_degrees=180
        )
        fixed_obj = fixed_mount.create()
        assert hasattr(fixed_obj, "surface_tilt")

        tracker_mount = MountSingleAxis(
            type="single_axis", axis_tilt_degrees=0, gcr=0.4
        )
        tracker_obj = tracker_mount.create()
        assert hasattr(tracker_obj, "axis_tilt")

        # Module/Inverter create() returns dicts
        module = ModuleParameters(
            count=20,
            name="Test",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        module_dict = module.create()
        assert isinstance(module_dict, dict) and "pdc0" in module_dict

        inverter = InverterParameters(
            count=1,
            max_power_output_ac_w=5000.0,
            efficiency_rating_percent=0.95,
        )
        inverter_dict = inverter.create()
        assert isinstance(inverter_dict, dict) and "pac0" in inverter_dict


class TestSystemIntegration:
    """Essential system integration tests - consolidated from 12 tests to 4."""

    @pytest.fixture
    def basic_system_config(self) -> Dict[str, Any]:
        """Standard system configuration for testing."""
        return {
            "location": {
                "name": "Test_Location",
                "latitude": 33.4484,
                "longitude": -112.0740,
                "tz": "US/Arizona",
            },
            "pv_systems": [
                {
                    "pv_arrays": [
                        {
                            "pv_modules": {
                                "count": 20,
                                "name": "Test_Module",
                                "nameplate_dc_rating_w": 300.0,
                                "power_temperature_coefficient_per_degree_c": -0.004,
                            },
                            "array_setup": {
                                "name": "Test_Array",
                                "mount": {
                                    "type": "fixed_mount",
                                    "tilt_degrees": 30.0,
                                    "azimuth_degrees": 180.0,
                                },
                                "number_of_strings": 1,
                            },
                        }
                    ],
                    "inverters": {
                        "count": 1,
                        "max_power_output_ac_w": 5000.0,
                        "efficiency_rating_percent": 0.95,
                    },
                    "temperature_model": {"model": "sapm"},
                }
            ],
        }

    def test_basic_system_creation(self, basic_system_config):
        """Test basic system creation workflow."""
        pv_model = PVLibModel(**basic_system_config)

        assert pv_model.location.name == "Test_Location"
        assert len(pv_model.pv_systems) == 1
        assert len(pv_model.pv_systems[0].pv_arrays) == 1

    def test_multi_array_system(self, basic_system_config):
        """Test system with multiple arrays."""
        # Add second array with different mount type
        second_array = {
            "pv_modules": {
                "count": 16,
                "name": "Second_Module",
                "nameplate_dc_rating_w": 400.0,
                "power_temperature_coefficient_per_degree_c": -0.0035,
            },
            "array_setup": {
                "name": "Second_Array",
                "mount": {
                    "type": "single_axis",
                    "axis_tilt_degrees": 0.0,
                    "gcr": 0.35,
                },
                "number_of_strings": 8,
            },
        }

        basic_system_config["pv_systems"][0]["pv_arrays"].append(second_array)
        basic_system_config["pv_systems"][0]["inverters"]["max_power_output_ac_w"] = (
            15000.0
        )

        pv_model = PVLibModel(**basic_system_config)
        assert len(pv_model.pv_systems[0].pv_arrays) == 2

    def test_system_physics_validation(self, basic_system_config):
        """Test system produces physically reasonable results."""
        pv_model = PVLibModel(**basic_system_config)
        system = pv_model.pv_systems[0]

        # Calculate DC capacity
        array = system.pv_arrays[0]
        module_params = array.pv_modules.create()
        module_power = module_params.get("pdc0", 300.0)
        strings = array.array_setup.number_of_strings
        total_dc = module_power * array.pv_modules.count * strings

        # Get inverter capacity
        inverter_params = system.inverters.create()
        inverter_capacity = inverter_params.get("pac0", 5000.0)
        total_ac = inverter_capacity * system.inverters.count

        # Verify reasonable DC/AC ratio
        dc_ac_ratio = total_dc / total_ac
        assert 0.8 <= dc_ac_ratio <= 4.0, (
            f"DC/AC ratio {dc_ac_ratio:.2f} outside reasonable range"
        )

    def test_error_handling(self):
        """Test system error handling."""
        # Invalid location
        config = {
            "location": {
                "name": "Test",
                "latitude": 95.0,
                "longitude": -112.0,
                "tz": "UTC",
            },
            "pv_systems": [],
        }
        with pytest.raises(ValidationError):
            PVLibModel(**config)

        # Invalid module power
        config = {
            "location": {
                "name": "Test",
                "latitude": 33.4,
                "longitude": -112.0,
                "tz": "UTC",
            },
            "pv_systems": [
                {
                    "pv_arrays": [
                        {
                            "pv_modules": {
                                "count": 20,
                                "name": "Test",
                                "nameplate_dc_rating_w": -100.0,
                                "power_temperature_coefficient_per_degree_c": -0.004,
                            },
                            "array_setup": {
                                "name": "Test_Array",
                                "mount": {
                                    "type": "fixed_mount",
                                    "tilt_degrees": 30.0,
                                    "azimuth_degrees": 180.0,
                                },
                                "number_of_strings": 1,
                            },
                        }
                    ],
                    "inverters": {
                        "count": 1,
                        "max_power_output_ac_w": 5000.0,
                        "efficiency_rating_percent": 0.95,
                    },
                }
            ],
        }
        with pytest.raises(ValidationError):
            PVLibModel(**config)


class TestSystemScenarios:
    """Comprehensive system scenarios - consolidated from multiple test files."""

    @pytest.mark.parametrize(
        "config_name,dc_kw,arrays,mount_type",
        [
            ("residential", 6.0, 1, "fixed_mount"),
            ("commercial", 100.0, 2, "fixed_mount"),
            ("utility", 1000.0, 4, "single_axis"),
        ],
    )
    def test_system_scenarios(self, config_name, dc_kw, arrays, mount_type):
        """Test different system scales and configurations."""
        # Calculate module configuration
        modules_per_array = 20
        module_power = min(500.0, (dc_kw * 1000) / (arrays * modules_per_array))

        # Create mount
        if mount_type == "fixed_mount":
            mount = {
                "type": "fixed_mount",
                "tilt_degrees": 30.0,
                "azimuth_degrees": 180.0,
            }
        else:
            mount = {
                "type": "single_axis",
                "axis_tilt_degrees": 0.0,
                "gcr": 0.35,
            }

        # Calculate strings per array
        strings_per_array = max(
            1, int(dc_kw * 1000 / (arrays * modules_per_array * module_power))
        )

        # Build configuration
        config = {
            "location": {
                "name": f"{config_name}_test",
                "latitude": 33.4,
                "longitude": -112.0,
                "tz": "UTC",
            },
            "pv_systems": [
                {
                    "pv_arrays": [
                        {
                            "pv_modules": {
                                "count": modules_per_array,
                                "name": f"{config_name}_module_{i}",
                                "nameplate_dc_rating_w": module_power,
                                "power_temperature_coefficient_per_degree_c": -0.004,
                            },
                            "array_setup": {
                                "name": f"{config_name}_array_{i}",
                                "mount": mount,
                                "number_of_strings": strings_per_array,
                            },
                        }
                        for i in range(arrays)
                    ],
                    "inverters": {
                        "count": max(1, arrays),
                        "max_power_output_ac_w": (dc_kw * 1000 * 0.8) / max(1, arrays),
                        "efficiency_rating_percent": 0.95,
                    },
                    "temperature_model": {"model": "sapm"},
                }
            ],
        }

        # Test system creation and basic validation
        pv_model = PVLibModel(**config)
        assert len(pv_model.pv_systems[0].pv_arrays) == arrays
        assert pv_model.location.name == f"{config_name}_test"
