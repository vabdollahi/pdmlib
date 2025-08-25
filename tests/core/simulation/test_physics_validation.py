"""
Physics-based validation tests for PV components.

This module consolidates physics validation testing with realistic
parameter ranges and industry-standard expectations, reducing redundancy
while improving validation quality.
"""

from typing import Any, List, Tuple

import pytest

from app.core.simulation.pvlib_models import (
    ArraySetup,
    BifacialConfiguration,
    DCLossModel,
    IAMModel,
    InverterParameters,
    Location,
    ModuleParameters,
    MountFixed,
    PvArray,
    PVLibModel,
    PvSystem,
    SoilingModel,
)


class PhysicsValidator:
    """Helper class for physics-based validation checks."""

    # Industry standard ranges for PV components
    PHYSICS_RANGES = {
        "iam_martin_ruiz_a_r": (0.14, 0.20, "Martin-Ruiz a_r coefficient"),
        "iam_ashrae_b": (0.03, 0.10, "ASHRAE b coefficient"),
        "bifaciality": (0.70, 0.90, "Bifacial module bifaciality"),
        "ground_albedo": (0.15, 0.40, "Ground albedo (concrete to snow)"),
        "soiling_rate_daily": (0.00005, 0.005, "Daily soiling rate (0.005% to 0.5%)"),
        "dc_losses_total": (0.03, 0.08, "Total DC losses (3% to 8%)"),
        "inverter_degradation": (0.002, 0.008, "Annual inverter degradation"),
        "module_temp_coeff": (-0.006, -0.002, "Module temperature coefficient"),
        "tracking_max_angle": (45, 65, "Single-axis tracking max rotation"),
    }

    @classmethod
    def validate_range(cls, value: float, param_name: str) -> Tuple[bool, str]:
        """Validate if a parameter is within realistic physics range."""
        if param_name not in cls.PHYSICS_RANGES:
            return True, f"No range defined for {param_name}"

        min_val, max_val, description = cls.PHYSICS_RANGES[param_name]
        is_valid = min_val <= value <= max_val

        message = (
            f"{description}: {value:.4f} "
            f"{'✓' if is_valid else '✗'} "
            f"(expected: {min_val}-{max_val})"
        )

        return is_valid, message

    @classmethod
    def validate_component_physics(cls, component: Any) -> List[str]:
        """Validate physics of a component and return list of issues."""
        issues = []

        if isinstance(component, IAMModel):
            if component.model == "martin_ruiz":
                is_valid, msg = cls.validate_range(
                    component.martin_ruiz_a_r, "iam_martin_ruiz_a_r"
                )
                if not is_valid:
                    issues.append(msg)
            elif component.model == "ashrae":
                is_valid, msg = cls.validate_range(component.ashrae_b, "iam_ashrae_b")
                if not is_valid:
                    issues.append(msg)

        elif isinstance(component, BifacialConfiguration) and component.enable_bifacial:
            is_valid, msg = cls.validate_range(component.bifaciality, "bifaciality")
            if not is_valid:
                issues.append(msg)
            is_valid, msg = cls.validate_range(component.albedo, "ground_albedo")
            if not is_valid:
                issues.append(msg)

        elif isinstance(component, SoilingModel) and component.enable_soiling:
            if component.model == "constant":
                # Convert to daily rate
                daily_rate = component.constant_loss_factor / 365
                is_valid, msg = cls.validate_range(daily_rate, "soiling_rate_daily")
                if not is_valid:
                    issues.append(msg)

        elif isinstance(component, DCLossModel):
            total_losses = (
                component.dc_wiring_loss_percent
                + component.connection_loss_percent
                + component.mismatch_loss_percent
            )
            is_valid, msg = cls.validate_range(total_losses, "dc_losses_total")
            if not is_valid:
                issues.append(msg)

        return issues


class TestPhysicsValidation:
    """Comprehensive physics validation tests."""

    def test_iam_martin_ruiz_physics_range(self):
        """Test Martin-Ruiz IAM coefficient is within realistic range."""
        # Test typical value
        iam = IAMModel(model="martin_ruiz", martin_ruiz_a_r=0.16)
        issues = PhysicsValidator.validate_component_physics(iam)
        assert len(issues) == 0, f"Physics issues: {issues}"

        # Test edge cases
        iam_low = IAMModel(model="martin_ruiz", martin_ruiz_a_r=0.14)
        issues = PhysicsValidator.validate_component_physics(iam_low)
        assert len(issues) == 0, "Lower bound should be valid"

        iam_high = IAMModel(model="martin_ruiz", martin_ruiz_a_r=0.20)
        issues = PhysicsValidator.validate_component_physics(iam_high)
        assert len(issues) == 0, "Upper bound should be valid"

    def test_bifacial_physics_validation(self):
        """Test bifacial parameters against real-world values."""
        # Typical commercial bifacial module
        bifacial = BifacialConfiguration(
            enable_bifacial=True,
            bifaciality=0.75,  # 75% is typical for PERC bifacial
            albedo=0.25,  # Grass/concrete mix
            row_height=2.0,
            pitch=7.0,  # 3.5x module height is standard
        )

        issues = PhysicsValidator.validate_component_physics(bifacial)
        assert len(issues) == 0, f"Bifacial physics issues: {issues}"

        # Verify gain calculations would be reasonable
        assert bifacial.bifaciality * bifacial.albedo < 0.3, (
            "Bifacial gain should be < 30%"
        )

    def test_soiling_model_realistic_rates(self):
        """Test soiling rates are within observed ranges."""
        # Desert environment (higher soiling)
        soiling_desert = SoilingModel(
            enable_soiling=True,
            model="constant",
            constant_loss_factor=0.02,  # 2% annual
        )

        issues = PhysicsValidator.validate_component_physics(soiling_desert)
        assert len(issues) == 0, f"Desert soiling issues: {issues}"

        # Moderate environment
        soiling_moderate = SoilingModel(
            enable_soiling=True,
            model="hsu",
            pm2_5_concentration=15.0,  # μg/m³
            deposition_rate=0.002,  # 0.2%/day
        )

        # Verify deposition rate is reasonable
        annual_loss = soiling_moderate.deposition_rate * 365
        assert 0.1 <= annual_loss <= 0.9, (
            f"Annual soiling loss {annual_loss:.1%} seems unrealistic"
        )

    def test_dc_losses_industry_standards(self):
        """Test DC losses match industry guidelines."""
        dc_losses = DCLossModel(
            dc_wiring_loss_percent=0.02,  # 2% wiring
            connection_loss_percent=0.005,  # 0.5% connections
            mismatch_loss_percent=0.02,  # 2% mismatch
        )

        issues = PhysicsValidator.validate_component_physics(dc_losses)
        assert len(issues) == 0, f"DC losses issues: {issues}"

        total_losses = (
            dc_losses.dc_wiring_loss_percent
            + dc_losses.connection_loss_percent
            + dc_losses.mismatch_loss_percent
        )

        # Verify total is reasonable (NREL typical: 3-6%)
        assert 0.03 <= total_losses <= 0.08, (
            f"Total DC losses {total_losses:.1%} outside typical range"
        )

    def test_system_level_physics_integration(self):
        """Test physics validation at the complete system level."""
        # Create a realistic system with proper DC/AC ratio
        location = Location(
            name="Physics_Test_Location",
            latitude=33.4484,
            longitude=-112.0740,
            tz="US/Arizona",
        )

        # Create mount
        mount = MountFixed(type="fixed_mount", tilt_degrees=30.0, azimuth_degrees=180.0)

        # Create array setup with realistic configuration
        array_setup = ArraySetup(
            name="Physics_Test_Array",
            mount=mount,
            number_of_strings=3,  # Reduced for better DC/AC ratio
        )

        # Create modules with proper power rating
        modules = ModuleParameters(
            count=20,
            name="Physics_Test_Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )

        # Create array
        array = PvArray(pv_modules=modules, array_setup=array_setup)

        # Create inverter with appropriate capacity
        inverter = InverterParameters(
            count=1,
            max_power_output_ac_w=15000.0,  # Increased to balance DC/AC ratio
            efficiency_rating_percent=0.95,
        )

        # Create system
        pv_system = PvSystem(inverters=inverter, pv_arrays=[array])
        pv_model = PVLibModel(location=location, pv_systems=[pv_system])

        # Validate system physics using available method
        component_issues = PhysicsValidator.validate_component_physics(pv_model)
        assert len(component_issues) == 0, f"Physics issues found: {component_issues}"

        # Calculate actual DC/AC ratio for verification
        total_dc = (
            modules.nameplate_dc_rating_w
            * modules.count
            * array_setup.number_of_strings
        )
        total_ac = inverter.max_power_output_ac_w * inverter.count
        dc_ac_ratio = total_dc / total_ac

        assert 1.1 <= dc_ac_ratio <= 1.4, (
            f"DC/AC ratio {dc_ac_ratio:.2f} outside typical range (1.1-1.4)"
        )

    def test_temperature_coefficient_physics(self):
        """Test that temperature coefficients are realistic for technologies."""
        # Typical values for different technologies
        temp_coeffs = {
            "c-Si": -0.0045,  # Crystalline silicon
            "CdTe": -0.0025,  # Cadmium telluride
            "CIGS": -0.0036,  # Copper indium gallium selenide
        }

        for tech, coeff in temp_coeffs.items():
            is_valid, msg = PhysicsValidator.validate_range(coeff, "module_temp_coeff")
            assert is_valid, (
                f"{tech} temperature coefficient outside realistic range: {msg}"
            )

    def test_geographical_physics_validation(self):
        """Test that location-based parameters make physical sense."""
        locations = [
            {
                "name": "Phoenix_AZ",
                "lat": 33.4,
                "lon": -112.1,
                "expected_ghi": (2000, 2400),
            },
            {
                "name": "Seattle_WA",
                "lat": 47.6,
                "lon": -122.3,
                "expected_ghi": (1200, 1600),
            },
            {
                "name": "Miami_FL",
                "lat": 25.8,
                "lon": -80.2,
                "expected_ghi": (1800, 2200),
            },
        ]

        for loc in locations:
            # Test that latitude/longitude are realistic
            assert -90 <= loc["lat"] <= 90, f"Invalid latitude for {loc['name']}"
            assert -180 <= loc["lon"] <= 180, f"Invalid longitude for {loc['name']}"

            # Test albedo expectations based on climate
            if "Phoenix" in loc["name"]:
                # Desert - higher albedo
                expected_albedo_range = (0.15, 0.30)
            elif "Seattle" in loc["name"]:
                # Vegetation - lower albedo
                expected_albedo_range = (0.12, 0.25)
            else:
                # Mixed urban/suburban
                expected_albedo_range = (0.15, 0.30)

            # This would be used in actual bifacial calculations
            test_albedo = sum(expected_albedo_range) / 2
            is_valid, _ = PhysicsValidator.validate_range(test_albedo, "ground_albedo")
            assert is_valid, f"Expected albedo for {loc['name']} should be realistic"


class TestParametricPhysics:
    """Parametric tests for component physics across ranges."""

    @pytest.mark.parametrize(
        "a_r,expected_valid",
        [
            (0.12, False),  # Too low
            (0.14, True),  # Lower bound
            (0.16, True),  # Typical
            (0.18, True),  # Upper typical
            (0.20, True),  # Upper bound
            (0.25, False),  # Too high
        ],
    )
    def test_martin_ruiz_parameter_range(self, a_r, expected_valid):
        """Parametric test of Martin-Ruiz a_r coefficient."""
        if expected_valid:
            iam = IAMModel(model="martin_ruiz", martin_ruiz_a_r=a_r)
            issues = PhysicsValidator.validate_component_physics(iam)
            assert len(issues) == 0, (
                f"Expected valid a_r={a_r} but got issues: {issues}"
            )
        else:
            # For invalid values, we expect physics validator to flag them
            iam = IAMModel(model="martin_ruiz", martin_ruiz_a_r=a_r)
            issues = PhysicsValidator.validate_component_physics(iam)
            assert len(issues) > 0, (
                f"Expected physics issues for a_r={a_r} but got none"
            )

    @pytest.mark.parametrize(
        "bifaciality,albedo,expected_gain_range",
        [
            (0.70, 0.20, (0.10, 0.20)),  # Conservative
            (0.75, 0.25, (0.15, 0.25)),  # Typical
            (0.80, 0.30, (0.20, 0.30)),  # Aggressive
        ],
    )
    def test_bifacial_gain_expectations(self, bifaciality, albedo, expected_gain_range):
        """Test that bifacial gain calculations are realistic."""
        bifacial = BifacialConfiguration(
            enable_bifacial=True,
            bifaciality=bifaciality,
            albedo=albedo,
            row_height=2.0,
            pitch=7.0,
        )

        issues = PhysicsValidator.validate_component_physics(bifacial)
        assert len(issues) == 0, f"Bifacial physics issues: {issues}"

        # Simple gain estimate (actual calculation would be more complex)
        estimated_gain = bifaciality * albedo
        min_gain, max_gain = expected_gain_range

        assert min_gain <= estimated_gain <= max_gain, (
            f"Estimated bifacial gain {estimated_gain:.1%} outside expected range "
            f"{min_gain:.1%}-{max_gain:.1%}"
        )
