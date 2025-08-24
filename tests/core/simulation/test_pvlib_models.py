"""
Tests for PVLib models and components.

This module tests the improved PVLib integration including TypedDict returns,
validation constraints, caching functionality, and component creation.
"""

import pytest
from pvlib import location, pvsystem
from pydantic import ValidationError

from app.core.simulation.pvlib_models import (
    ArraySetup,
    InverterDatabase,
    InverterParameters,
    Location,
    ModuleDatabase,
    ModuleParameters,
    MountFixed,
    MountSingleAxis,
    PhysicalSimulation,
    PvArray,
    PVLibModel,
    PvSystem,
    TemperatureModel,
    _get_inverter_parameters,
    _get_module_parameters,
    _get_temperature_model_parameters,
)


class TestMountValidation:
    """Test mount configuration validation with realistic ranges."""

    def test_fixed_mount_valid_angles(self):
        """Test that valid angles are accepted for fixed mount."""
        mount = MountFixed(type="fixed_mount", tilt_degrees=30.0, azimuth_degrees=180.0)
        assert mount.tilt_degrees == 30.0
        assert mount.azimuth_degrees == 180.0

    def test_fixed_mount_invalid_tilt_high(self):
        """Test that tilt angles above 90 degrees are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MountFixed(type="fixed_mount", tilt_degrees=95.0)
        assert "less than or equal to 90" in str(exc_info.value)

    def test_fixed_mount_invalid_tilt_negative(self):
        """Test that negative tilt angles are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MountFixed(type="fixed_mount", tilt_degrees=-5.0)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_fixed_mount_invalid_azimuth_high(self):
        """Test that azimuth angles above 360 degrees are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MountFixed(type="fixed_mount", azimuth_degrees=370.0)
        assert "less than or equal to 360" in str(exc_info.value)

    def test_fixed_mount_invalid_module_height(self):
        """Test that unrealistic module heights are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MountFixed(type="fixed_mount", module_height_meter=60.0)
        assert "less than or equal to 50" in str(exc_info.value)

    def test_single_axis_valid_configuration(self):
        """Test valid single-axis tracker configuration."""
        mount = MountSingleAxis(
            type="single_axis",
            axis_tilt_degrees=0.0,
            axis_azimuth_degrees=180.0,
            max_angle_degrees=60.0,
            gcr=0.4,
        )
        assert mount.axis_tilt_degrees == 0.0
        assert mount.gcr == 0.4

    def test_single_axis_invalid_gcr(self):
        """Test that invalid ground coverage ratio is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MountSingleAxis(type="single_axis", gcr=1.5)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_single_axis_invalid_max_angle(self):
        """Test that invalid max angle is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MountSingleAxis(type="single_axis", max_angle_degrees=200.0)
        assert "less than or equal to 180" in str(exc_info.value)


class TestInverterConfiguration:
    """Test inverter configuration and TypedDict returns."""

    def test_inverter_parameters_valid_config(self):
        """Test valid inverter parameters configuration."""
        inverter = InverterParameters(
            count=2,
            max_power_output_ac_w=5000.0,
            efficiency_rating_percent=0.95,
        )
        assert inverter.count == 2
        assert inverter.max_power_output_ac_w == 5000.0
        assert inverter.efficiency_rating_percent == 0.95

    def test_inverter_parameters_invalid_count(self):
        """Test that zero or negative inverter count is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            InverterParameters(
                count=0, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
            )
        assert "greater than 0" in str(exc_info.value)

    def test_inverter_parameters_invalid_power(self):
        """Test that zero or negative power is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            InverterParameters(
                count=1, max_power_output_ac_w=0.0, efficiency_rating_percent=0.95
            )
        assert "greater than 0" in str(exc_info.value)

    def test_inverter_parameters_invalid_efficiency_high(self):
        """Test that efficiency above 100% is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            InverterParameters(
                count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=1.2
            )
        assert "less than or equal to 1" in str(exc_info.value)

    def test_inverter_parameters_create_returns_typed_dict(self):
        """Test that inverter parameters create method returns TypedDict."""
        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )
        result = inverter.create()

        # Check return type structure
        assert isinstance(result, dict)
        assert "pdc0" in result
        assert "pac0" in result
        assert isinstance(result["pdc0"], float)
        assert isinstance(result["pac0"], float)

        # Check calculations
        expected_pac0 = 5000.0 * 1  # max_power * count
        expected_pdc0 = expected_pac0 / 0.95  # pac0 / efficiency
        assert result["pac0"] == expected_pac0
        assert abs(result["pdc0"] - expected_pdc0) < 0.01

    def test_inverter_database_valid_config(self):
        """Test valid database inverter configuration."""
        inverter = InverterDatabase(
            count=1,
            database="CECInverter",
            record="ABB__MICRO_0_25_I_OUTD_US_208__208V_",
        )
        assert inverter.count == 1
        assert inverter.database == "CECInverter"

    def test_inverter_database_invalid_count(self):
        """Test that invalid count is rejected for database inverter."""
        with pytest.raises(ValidationError) as exc_info:
            InverterDatabase(
                count=-1,
                database="CECInverter",
                record="ABB__MICRO_0_25_I_OUTD_US_208__208V_",
            )
        assert "greater than 0" in str(exc_info.value)


class TestModuleConfiguration:
    """Test module configuration and validation."""

    def test_module_parameters_valid_config(self):
        """Test valid module parameters configuration."""
        module = ModuleParameters(
            count=20,
            name="Test Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        assert module.count == 20
        assert module.nameplate_dc_rating_w == 300.0

    def test_module_parameters_invalid_count(self):
        """Test that invalid module count is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModuleParameters(
                count=0,
                name="Test",
                nameplate_dc_rating_w=300.0,
                power_temperature_coefficient_per_degree_c=-0.004,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_module_parameters_invalid_power_high(self):
        """Test that unrealistic module power is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModuleParameters(
                count=1,
                name="Test",
                nameplate_dc_rating_w=1200.0,  # Too high
                power_temperature_coefficient_per_degree_c=-0.004,
            )
        assert "less than or equal to 1000" in str(exc_info.value)

    def test_module_parameters_invalid_temp_coeff(self):
        """Test that invalid temperature coefficient is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModuleParameters(
                count=1,
                name="Test",
                nameplate_dc_rating_w=300.0,
                power_temperature_coefficient_per_degree_c=0.5,  # Should be negative
            )
        assert "less than or equal to 0" in str(exc_info.value)

    def test_module_parameters_empty_name(self):
        """Test that empty module name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModuleParameters(
                count=1,
                name="",
                nameplate_dc_rating_w=300.0,
                power_temperature_coefficient_per_degree_c=-0.004,
            )
        assert "at least 1 character" in str(exc_info.value)

    def test_module_parameters_create_returns_typed_dict(self):
        """Test that module parameters create method returns ModuleParams TypedDict."""
        module = ModuleParameters(
            count=1,
            name="Test Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        result = module.create()

        # Check return type structure
        assert isinstance(result, dict)
        assert "pdc0" in result
        assert "gamma_pdc" in result
        assert result["pdc0"] == 300.0
        assert result["gamma_pdc"] == -0.004

    def test_module_database_valid_config(self):
        """Test valid database module configuration."""
        module = ModuleDatabase(
            count=20,
            name="Test Module",
            database="CECMod",
            record="Canadian_Solar_Inc__CS5P_220M",
        )
        assert module.count == 20
        assert module.database == "CECMod"


class TestLocationValidation:
    """Test location configuration validation."""

    def test_location_valid_coordinates(self):
        """Test valid location coordinates."""
        location = Location(
            name="Test Location",
            latitude=34.0522,
            longitude=-118.2437,
            tz="America/Los_Angeles",
            altitude=71.0,
        )
        assert location.latitude == 34.0522
        assert location.longitude == -118.2437

    def test_location_invalid_latitude_high(self):
        """Test that latitude above 90 degrees is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Location(
                name="Test",
                latitude=95.0,
                longitude=-118.0,
                tz="UTC",
            )
        assert "less than or equal to 90" in str(exc_info.value)

    def test_location_invalid_longitude_low(self):
        """Test that longitude below -180 degrees is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Location(
                name="Test",
                latitude=34.0,
                longitude=-185.0,
                tz="UTC",
            )
        assert "greater than or equal to -180" in str(exc_info.value)

    def test_location_invalid_altitude(self):
        """Test that unrealistic altitude is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Location(
                name="Test",
                latitude=34.0,
                longitude=-118.0,
                tz="UTC",
                altitude=10000.0,  # Too high
            )
        assert "less than or equal to 9000" in str(exc_info.value)

    def test_location_empty_name(self):
        """Test that empty location name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Location(
                name="",
                latitude=34.0,
                longitude=-118.0,
                tz="UTC",
            )
        assert "at least 1 character" in str(exc_info.value)


class TestArraySetupValidation:
    """Test array setup configuration validation."""

    def test_array_setup_valid_config(self):
        """Test valid array setup configuration."""
        mount = MountFixed(type="fixed_mount", tilt_degrees=30.0)
        array_setup = ArraySetup(name="Test Array", mount=mount, number_of_strings=100)
        assert array_setup.name == "Test Array"
        assert array_setup.number_of_strings == 100

    def test_array_setup_invalid_string_count_zero(self):
        """Test that zero string count is rejected."""
        mount = MountFixed(type="fixed_mount")
        with pytest.raises(ValidationError) as exc_info:
            ArraySetup(name="Test", mount=mount, number_of_strings=0)
        assert "greater than 0" in str(exc_info.value)

    def test_array_setup_invalid_string_count_high(self):
        """Test that unrealistic string count is rejected."""
        mount = MountFixed(type="fixed_mount")
        with pytest.raises(ValidationError) as exc_info:
            ArraySetup(name="Test", mount=mount, number_of_strings=15000)
        assert "less than or equal to 10000" in str(exc_info.value)

    def test_array_setup_empty_name(self):
        """Test that empty array name is rejected."""
        mount = MountFixed(type="fixed_mount")
        with pytest.raises(ValidationError) as exc_info:
            ArraySetup(name="", mount=mount, number_of_strings=10)
        assert "at least 1 character" in str(exc_info.value)


class TestSimulationParametersTyping:
    """Test simulation parameters and TypedDict returns."""

    def test_physical_simulation_create_returns_typed_dict(self):
        """Test that physical simulation create method returns TypedDict."""
        sim = PhysicalSimulation(aoi_model="physical", spectral_model="no_loss")
        result = sim.create()

        # Check return type structure
        assert isinstance(result, dict)
        assert "aoi_model" in result
        assert "spectral_model" in result
        assert result["aoi_model"] == "physical"
        assert result["spectral_model"] == "no_loss"

    def test_physical_simulation_none_values(self):
        """Test that None values are properly handled."""
        sim = PhysicalSimulation(aoi_model=None, spectral_model="no_loss")
        result = sim.create()

        assert "spectral_model" in result
        assert result["spectral_model"] == "no_loss"
        # aoi_model should not be in result if None
        assert "aoi_model" not in result or result["aoi_model"] is None


class TestCachingFunctionality:
    """Test caching functions for performance optimization."""

    def test_get_temperature_model_parameters_valid(self):
        """Test cached temperature model parameter retrieval."""
        result = _get_temperature_model_parameters("sapm", "open_rack_glass_glass")
        assert isinstance(result, dict)
        assert "a" in result or "c0" in result  # Different models have different keys

    def test_get_temperature_model_parameters_invalid(self):
        """Test cached temperature model parameter retrieval with invalid input."""
        with pytest.raises(ValueError) as exc_info:
            _get_temperature_model_parameters("invalid_db", "invalid_record")
        assert "Temperature model not found" in str(exc_info.value)

    def test_get_inverter_parameters_cache_consistency(self):
        """Test that cached inverter parameters are consistent across calls."""
        try:
            result1 = _get_inverter_parameters(
                "CECInverter", "ABB__MICRO_0_25_I_OUTD_US_208__208V_"
            )
            result2 = _get_inverter_parameters(
                "CECInverter", "ABB__MICRO_0_25_I_OUTD_US_208__208V_"
            )

            # Results should be identical (cached)
            assert result1 == result2
            assert isinstance(result1, dict)
            assert "Paco" in result1
            assert "Pdco" in result1
        except ValueError:
            # Skip if specific inverter not available in test environment
            pytest.skip("Specific inverter model not available in test environment")

    def test_get_module_parameters_cache_consistency(self):
        """Test that cached module parameters are consistent across calls."""
        try:
            result1 = _get_module_parameters("CECMod", "Canadian_Solar_Inc__CS5P_220M")
            result2 = _get_module_parameters("CECMod", "Canadian_Solar_Inc__CS5P_220M")

            # Results should be identical (cached)
            assert result1 == result2
            assert isinstance(result1, dict)
        except ValueError:
            # Skip if specific module not available in test environment
            pytest.skip("Specific module model not available in test environment")


class TestComponentCreation:
    """Test actual PVLib component creation."""

    def test_fixed_mount_creation(self):
        """Test that fixed mount creates proper PVLib object."""
        mount = MountFixed(type="fixed_mount", tilt_degrees=30.0, azimuth_degrees=180.0)
        pvlib_mount = mount.create()
        assert isinstance(pvlib_mount, pvsystem.FixedMount)

    def test_single_axis_mount_creation(self):
        """Test that single axis mount creates proper PVLib object."""
        mount = MountSingleAxis(type="single_axis", axis_tilt_degrees=0.0)
        pvlib_mount = mount.create()
        assert isinstance(pvlib_mount, pvsystem.SingleAxisTrackerMount)

    def test_location_creation(self):
        """Test that location creates proper PVLib object."""
        loc = Location(
            name="Test Location",
            latitude=34.0522,
            longitude=-118.2437,
            tz="America/Los_Angeles",
        )
        pvlib_location = loc.create()
        assert isinstance(pvlib_location, location.Location)
        assert pvlib_location.latitude == 34.0522

    def test_temperature_model_creation(self):
        """Test that temperature model creates proper parameters."""
        temp_model = TemperatureModel(database="sapm", record="open_rack_glass_glass")
        result = temp_model.create()
        assert isinstance(result, dict)
        assert len(result) > 0  # Should have temperature parameters


class TestPvSystemValidation:
    """Test PV system validation logic."""

    def test_pv_system_module_type_consistency_check(self):
        """Test that PV system enforces module type consistency."""
        mount = MountFixed(type="fixed_mount")
        array_setup = ArraySetup(name="Test", mount=mount, number_of_strings=10)

        # Create arrays with different module types
        module1 = ModuleParameters(
            count=20,
            name="Module1",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        module2 = ModuleDatabase(
            count=20,
            name="Module2",
            database="CECMod",
            record="Canadian_Solar_Inc__CS5P_220M",
        )

        array1 = PvArray(pv_modules=module1, array_setup=array_setup)
        array2 = PvArray(pv_modules=module2, array_setup=array_setup)

        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )

        # This should raise an error due to mixed module types
        # Note: The validation occurs during PV system initialization
        pv_system = PvSystem(inverters=inverter, pv_arrays=[array1, array2])

        # The validation should happen when we try to create the system
        with pytest.raises(ValueError) as exc_info:
            pv_system.create()
        assert "same DC model type" in str(exc_info.value)

    def test_pv_system_valid_configuration(self):
        """Test that valid PV system configuration is accepted."""
        mount = MountFixed(type="fixed_mount")
        array_setup = ArraySetup(name="Test", mount=mount, number_of_strings=10)

        module = ModuleParameters(
            count=20,
            name="Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        array = PvArray(pv_modules=module, array_setup=array_setup)

        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )

        pv_system = PvSystem(inverters=inverter, pv_arrays=[array])
        assert len(pv_system.pv_arrays) == 1
        assert (
            pv_system.max_power_output_ac_w is None
        )  # Not set until create() is called


class TestPVLibModelConfiguration:
    """Test complete PVLib model configuration."""

    def test_pvlib_model_valid_configuration(self):
        """Test that valid PVLib model configuration is accepted."""
        location = Location(
            name="Test Location",
            latitude=34.0522,
            longitude=-118.2437,
            tz="America/Los_Angeles",
        )

        mount = MountFixed(type="fixed_mount", tilt_degrees=30.0)
        array_setup = ArraySetup(name="Test Array", mount=mount, number_of_strings=10)
        module = ModuleParameters(
            count=20,
            name="Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        array = PvArray(pv_modules=module, array_setup=array_setup)
        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )
        pv_system = PvSystem(inverters=inverter, pv_arrays=[array])

        pvlib_model = PVLibModel(location=location, pv_systems=[pv_system])
        assert len(pvlib_model.pv_systems) == 1
        assert pvlib_model.location.name == "Test Location"

    def test_pvlib_model_empty_systems(self):
        """Test that PVLib model with no systems is rejected."""
        location = Location(
            name="Test Location",
            latitude=34.0522,
            longitude=-118.2437,
            tz="America/Los_Angeles",
        )

        with pytest.raises(ValueError) as exc_info:
            pvlib_model = PVLibModel(location=location, pv_systems=[])
            pvlib_model.create()  # This should trigger the validation
        assert "At least one PV system must be provided" in str(exc_info.value)

    def test_pvlib_model_multiple_systems_not_supported(self):
        """Test that multiple PV systems are not yet supported."""
        location = Location(
            name="Test Location",
            latitude=34.0522,
            longitude=-118.2437,
            tz="America/Los_Angeles",
        )

        # Create two identical systems
        mount = MountFixed(type="fixed_mount")
        array_setup = ArraySetup(name="Test", mount=mount, number_of_strings=10)
        module = ModuleParameters(
            count=20,
            name="Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        array = PvArray(pv_modules=module, array_setup=array_setup)
        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )
        system1 = PvSystem(inverters=inverter, pv_arrays=[array])
        system2 = PvSystem(inverters=inverter, pv_arrays=[array])

        pvlib_model = PVLibModel(location=location, pv_systems=[system1, system2])

        with pytest.raises(NotImplementedError) as exc_info:
            pvlib_model.create()
        assert "Multiple PV systems are not yet supported" in str(exc_info.value)
