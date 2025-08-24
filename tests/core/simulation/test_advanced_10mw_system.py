"""
Test suite for advanced 10MW PV system with all new features.

This module tests the complete advanced PV system configuration including
bifacial modules, advanced IAM models, soiling/snow modeling, shading analysis,
and DC loss modeling.
"""

import json
from pathlib import Path

import pytest

from app.core.simulation.pvlib_models import PVLibModel


class TestAdvanced10MWSystem:
    """Test advanced 10MW solar farm configuration."""

    @pytest.fixture
    def advanced_10mw_config(self):
        """Load the advanced 10MW solar farm configuration."""
        config_path = (
            Path(__file__).parent.parent.parent
            / "config"
            / "advanced_10mw_solar_farm.json"
        )
        with open(config_path) as f:
            return json.load(f)

    def test_advanced_10mw_pv_model_creation(self, advanced_10mw_config):
        """Test creating PV model from advanced JSON configuration."""
        # Create PV model using one-shot JSON parsing
        pv_model = PVLibModel(**advanced_10mw_config)

        # Verify basic structure
        assert pv_model.location.name == "Advanced_10MW_Solar_Farm_Demo"
        assert len(pv_model.pv_systems) == 1
        assert len(pv_model.pv_systems[0].pv_arrays) == 4

    def test_advanced_inverter_configuration(self, advanced_10mw_config):
        """Test advanced inverter model configuration."""
        pv_model = PVLibModel(**advanced_10mw_config)
        system = pv_model.pv_systems[0]

        # Check advanced inverter model
        assert system.advanced_inverter_model is not None
        assert system.advanced_inverter_model.ac_model == "pvwatts"
        assert system.advanced_inverter_model.enable_degradation is True
        assert system.advanced_inverter_model.degradation_rate_per_year == 0.005

    def test_bifacial_configuration_block1(self, advanced_10mw_config):
        """Test bifacial configuration in Block 1."""
        pv_model = PVLibModel(**advanced_10mw_config)
        block1 = pv_model.pv_systems[0].pv_arrays[0]

        # Check bifacial configuration
        bifacial = block1.array_setup.bifacial_config
        assert bifacial is not None
        assert bifacial.enable_bifacial is True
        assert bifacial.bifaciality == 0.75
        assert bifacial.row_height == 2.0
        assert bifacial.pitch == 7.0
        assert bifacial.albedo == 0.25

    def test_iam_models_variety(self, advanced_10mw_config):
        """Test different IAM models across arrays."""
        pv_model = PVLibModel(**advanced_10mw_config)
        arrays = pv_model.pv_systems[0].pv_arrays

        # Block 1: Martin-Ruiz
        assert arrays[0].array_setup.iam_model is not None
        assert arrays[0].array_setup.iam_model.model == "martin_ruiz"
        assert arrays[0].array_setup.iam_model.martin_ruiz_a_r == 0.16

        # Block 2: ASHRAE
        assert arrays[1].array_setup.iam_model is not None
        assert arrays[1].array_setup.iam_model.model == "ashrae"
        assert arrays[1].array_setup.iam_model.ashrae_b == 0.05

        # Block 3: Physical (default)
        assert arrays[2].array_setup.iam_model is not None
        assert arrays[2].array_setup.iam_model.model == "physical"

        # Block 4: Schlick
        assert arrays[3].array_setup.iam_model is not None
        assert arrays[3].array_setup.iam_model.model == "schlick"

    def test_soiling_models_variety(self, advanced_10mw_config):
        """Test different soiling models across arrays."""
        pv_model = PVLibModel(**advanced_10mw_config)
        arrays = pv_model.pv_systems[0].pv_arrays

        # Block 1: Hsu model
        soiling1 = arrays[0].array_setup.soiling_model
        assert soiling1.enable_soiling is True
        assert soiling1.model == "hsu"
        assert soiling1.cleaning_threshold == 0.5
        assert soiling1.pm2_5_concentration == 15.0

        # Block 2: Constant model
        soiling2 = arrays[1].array_setup.soiling_model
        assert soiling2.enable_soiling is True
        assert soiling2.model == "constant"
        assert soiling2.constant_loss_factor == 0.02

        # Block 3: Kimber model
        soiling3 = arrays[2].array_setup.soiling_model
        assert soiling3.enable_soiling is True
        assert soiling3.model == "kimber"
        assert soiling3.deposition_rate == 0.002

    def test_snow_models(self, advanced_10mw_config):
        """Test snow models in cold-climate arrays."""
        pv_model = PVLibModel(**advanced_10mw_config)
        arrays = pv_model.pv_systems[0].pv_arrays

        # Block 1: NREL snow model
        snow1 = arrays[0].array_setup.snow_model
        assert snow1.enable_snow_modeling is True
        assert snow1.model == "nrel"
        assert snow1.temp_threshold_c == 2.0

        # Block 3: Townsend snow model
        snow3 = arrays[2].array_setup.snow_model
        assert snow3.enable_snow_modeling is True
        assert snow3.model == "townsend"
        assert snow3.snow_density == 300.0
        assert snow3.slide_angle_deg == 30.0

    def test_shading_models(self, advanced_10mw_config):
        """Test shading models."""
        pv_model = PVLibModel(**advanced_10mw_config)
        arrays = pv_model.pv_systems[0].pv_arrays

        # Block 1: Self-shading
        shading1 = arrays[0].array_setup.shading_model
        assert shading1.enable_self_shading is True
        assert shading1.pitch == 7.0
        assert shading1.row_height == 2.0

        # Block 4: Near-field shading with obstacles
        shading4 = arrays[3].array_setup.shading_model
        assert shading4.enable_near_shading is True
        assert len(shading4.obstacles) == 2
        assert shading4.obstacles[0]["azimuth"] == 180.0
        assert shading4.obstacles[0]["elevation"] == 20.0

    def test_dc_loss_models(self, advanced_10mw_config):
        """Test DC loss models across arrays."""
        pv_model = PVLibModel(**advanced_10mw_config)
        arrays = pv_model.pv_systems[0].pv_arrays

        # Block 1: Advanced DC losses with ohmic modeling
        dc_loss1 = arrays[0].array_setup.dc_loss_model
        assert dc_loss1.enable_ohmic_losses is True
        assert dc_loss1.resistance_per_string_ohm == 0.3
        assert dc_loss1.voltage_dependent is False

        # Block 2: Basic DC losses only
        dc_loss2 = arrays[1].array_setup.dc_loss_model
        assert dc_loss2.enable_ohmic_losses is False
        assert dc_loss2.dc_wiring_loss_percent == 0.015

    def test_advanced_spectral_model(self, advanced_10mw_config):
        """Test advanced spectral modeling configuration."""
        pv_model = PVLibModel(**advanced_10mw_config)

        # Check advanced spectral model
        spectral = pv_model.physical_simulation.advanced_spectral_model
        assert spectral is not None
        assert spectral.model == "first_solar"
        assert spectral.module_type == "crystalline_silicon"
        assert spectral.first_solar_module == "FS-6420"
        assert spectral.precipitable_water is False

    def test_system_capacity_calculation_with_losses(self, advanced_10mw_config):
        """Test system capacity calculation includes all loss models."""
        pv_model = PVLibModel(**advanced_10mw_config)

        # Create the model to trigger capacity calculation
        model_chain = pv_model.create()

        # Verify model chain was created successfully
        assert model_chain is not None
        assert hasattr(model_chain, "system")
        assert len(model_chain.system.arrays) == 4

    def test_mount_type_variety(self, advanced_10mw_config):
        """Test variety of mount types and configurations."""
        pv_model = PVLibModel(**advanced_10mw_config)
        arrays = pv_model.pv_systems[0].pv_arrays

        # Block 1 & 2: Fixed mount
        assert arrays[0].array_setup.mount.type == "fixed_mount"
        assert arrays[1].array_setup.mount.type == "fixed_mount"
        assert arrays[0].array_setup.mount.tilt_degrees == 30.0

        # Block 3 & 4: Single-axis tracking
        assert arrays[2].array_setup.mount.type == "single_axis"
        assert arrays[3].array_setup.mount.type == "single_axis"
        assert arrays[2].array_setup.mount.max_angle_degrees == 55.0
        assert arrays[2].array_setup.mount.gcr == 0.35

    def test_temperature_model_variety(self, advanced_10mw_config):
        """Test different temperature models."""
        pv_model = PVLibModel(**advanced_10mw_config)
        arrays = pv_model.pv_systems[0].pv_arrays

        # Different temperature models for different mounting
        assert (
            arrays[0].array_setup.temperature_model.record == "close_mount_glass_glass"
        )
        assert arrays[1].array_setup.temperature_model.record == "open_rack_glass_glass"
        assert (
            arrays[2].array_setup.temperature_model.record == "close_mount_glass_glass"
        )
        assert arrays[3].array_setup.temperature_model.record == "open_rack_glass_glass"

    def test_model_creation_functionality(self, advanced_10mw_config):
        """Test that advanced models can create their configurations."""
        pv_model = PVLibModel(**advanced_10mw_config)

        # Test IAM model creation
        iam_config = pv_model.pv_systems[0].pv_arrays[0].array_setup.iam_model.create()
        assert iam_config == "martin_ruiz"

        # Test bifacial model creation
        bifacial_config = (
            pv_model.pv_systems[0].pv_arrays[0].array_setup.bifacial_config.create()
        )
        assert bifacial_config["bifaciality"] == 0.75

        # Test soiling model creation
        soiling_config = (
            pv_model.pv_systems[0].pv_arrays[0].array_setup.soiling_model.create()
        )
        assert soiling_config["model"] == "hsu"

        # Test DC loss model creation
        dc_loss_config = (
            pv_model.pv_systems[0].pv_arrays[0].array_setup.dc_loss_model.create()
        )
        assert dc_loss_config["basic_losses"] == 0.045  # 2% + 0.5% + 2% = 4.5%
        assert "ohmic_losses" in dc_loss_config

    def test_backwards_compatibility_preservation(self, advanced_10mw_config):
        """Test that advanced features don't break basic functionality."""
        # Load and compare with basic configuration
        basic_config_path = (
            Path(__file__).parent.parent.parent / "config" / "10mw_solar_farm.json"
        )
        with open(basic_config_path) as f:
            basic_config = json.load(f)

        # Both should create valid models
        basic_model = PVLibModel(**basic_config)
        advanced_model = PVLibModel(**advanced_10mw_config)

        # Basic structure should be similar
        assert len(basic_model.pv_systems) == len(advanced_model.pv_systems)
        assert len(basic_model.pv_systems[0].pv_arrays) == len(
            advanced_model.pv_systems[0].pv_arrays
        )


if __name__ == "__main__":
    pytest.main([__file__])
