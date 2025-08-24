"""
Test suite for advanced PVLib modeling components.

This module tests the new advanced PVLib features including IAM models,
bifacial configurations, soiling models, snow models, shading models,
DC loss models, and advanced spectral models.
"""

import pytest
from pydantic import ValidationError

from app.core.simulation.pvlib_models import (
    AdvancedInverterModel,
    BifacialConfiguration,
    DCLossModel,
    IAMModel,
    ShadingModel,
    SnowModel,
    SoilingModel,
    SpectralModel,
)


class TestIAMModel:
    """Test Incidence Angle Modifier models."""

    def test_iam_model_physical_default(self):
        """Test default physical IAM model."""
        iam_model = IAMModel()
        assert iam_model.model == "physical"
        assert iam_model.create() == "physical"
        assert iam_model.get_iam_parameters() is None

    def test_iam_model_ashrae(self):
        """Test ASHRAE IAM model with parameters."""
        iam_model = IAMModel(model="ashrae", ashrae_b=0.1)
        assert iam_model.model == "ashrae"
        assert iam_model.create() == "ashrae"
        assert iam_model.get_iam_parameters() == {"b": 0.1}

    def test_iam_model_martin_ruiz(self):
        """Test Martin-Ruiz IAM model with parameters."""
        iam_model = IAMModel(model="martin_ruiz", martin_ruiz_a_r=0.2)
        assert iam_model.model == "martin_ruiz"
        assert iam_model.create() == "martin_ruiz"
        assert iam_model.get_iam_parameters() == {"a_r": 0.2}

    def test_iam_model_invalid_ashrae_b(self):
        """Test validation of ASHRAE b parameter."""
        with pytest.raises(ValidationError):
            IAMModel(model="ashrae", ashrae_b=1.0)  # > 0.5


class TestBifacialConfiguration:
    """Test bifacial PV module configuration."""

    def test_bifacial_disabled_by_default(self):
        """Test bifacial modeling disabled by default."""
        bifacial = BifacialConfiguration()
        assert not bifacial.enable_bifacial
        assert bifacial.create() == {}

    def test_bifacial_enabled_configuration(self):
        """Test bifacial configuration when enabled."""
        bifacial = BifacialConfiguration(
            enable_bifacial=True,
            bifaciality=0.8,
            row_height=2.0,
            row_width=2.5,
            pitch=7.0,
            albedo=0.3,
        )
        config = bifacial.create()

        assert config["bifaciality"] == 0.8
        assert config["row_height"] == 2.0
        assert config["row_width"] == 2.5
        assert config["pitch"] == 7.0
        assert config["albedo"] == 0.3

    def test_bifacial_with_hub_height(self):
        """Test bifacial configuration with hub height."""
        bifacial = BifacialConfiguration(enable_bifacial=True, hub_height=1.2)
        config = bifacial.create()
        assert config["hub_height"] == 1.2

    def test_bifacial_validation_errors(self):
        """Test validation of bifacial parameters."""
        with pytest.raises(ValidationError):
            BifacialConfiguration(bifaciality=1.5)  # > 1.0


class TestSoilingModel:
    """Test soiling loss models."""

    def test_soiling_disabled_by_default(self):
        """Test soiling modeling disabled by default."""
        soiling = SoilingModel()
        assert not soiling.enable_soiling
        assert soiling.create() is None

    def test_soiling_constant_model(self):
        """Test constant soiling loss model."""
        soiling = SoilingModel(
            enable_soiling=True, model="constant", constant_loss_factor=0.05
        )
        config = soiling.create()

        assert config["model"] == "constant"
        assert config["loss_factor"] == 0.05

    def test_soiling_hsu_model(self):
        """Test Hsu soiling model."""
        soiling = SoilingModel(
            enable_soiling=True,
            model="hsu",
            cleaning_threshold=0.8,
            tilt_factor=1.2,
            pm2_5_concentration=20.0,
        )
        config = soiling.create()

        assert config["model"] == "hsu"
        assert config["cleaning_threshold"] == 0.8
        assert config["tilt_factor"] == 1.2
        assert config["pm2_5"] == 20.0

    def test_soiling_kimber_model(self):
        """Test Kimber soiling model."""
        soiling = SoilingModel(
            enable_soiling=True,
            model="kimber",
            deposition_rate=0.003,
            cleaning_threshold_kimber=1.5,
        )
        config = soiling.create()

        assert config["model"] == "kimber"
        assert config["deposition_rate"] == 0.003
        assert config["cleaning_threshold"] == 1.5


class TestSnowModel:
    """Test snow coverage models."""

    def test_snow_disabled_by_default(self):
        """Test snow modeling disabled by default."""
        snow = SnowModel()
        assert not snow.enable_snow_modeling
        assert snow.create() is None

    def test_snow_nrel_model(self):
        """Test NREL snow model."""
        snow = SnowModel(
            enable_snow_modeling=True,
            model="nrel",
            temp_threshold_c=1.0,
            tilt_factor=1.1,
        )
        config = snow.create()

        assert config["model"] == "nrel"
        assert config["temp_threshold"] == 1.0
        assert config["tilt_factor"] == 1.1

    def test_snow_townsend_model(self):
        """Test Townsend snow model."""
        snow = SnowModel(
            enable_snow_modeling=True,
            model="townsend",
            snow_density=350.0,
            slide_angle_deg=25.0,
        )
        config = snow.create()

        assert config["model"] == "townsend"
        assert config["snow_density"] == 350.0
        assert config["slide_angle"] == 25.0


class TestShadingModel:
    """Test shading models."""

    def test_shading_disabled_by_default(self):
        """Test shading modeling disabled by default."""
        shading = ShadingModel()
        assert not shading.enable_self_shading
        assert not shading.enable_near_shading
        assert shading.create() is None

    def test_self_shading_model(self):
        """Test self-shading model configuration."""
        shading = ShadingModel(
            enable_self_shading=True, pitch=6.0, row_height=1.5, row_width=2.0
        )
        config = shading.create()

        assert "self_shading" in config
        assert config["self_shading"]["pitch"] == 6.0
        assert config["self_shading"]["row_height"] == 1.5
        assert config["self_shading"]["row_width"] == 2.0

    def test_near_shading_model(self):
        """Test near-field shading model."""
        obstacles = [
            {"azimuth": 180.0, "elevation": 30.0, "distance": 50.0},
            {"azimuth": 90.0, "elevation": 15.0, "distance": 100.0},
        ]

        shading = ShadingModel(enable_near_shading=True, obstacles=obstacles)
        config = shading.create()

        assert "near_shading" in config
        assert config["near_shading"]["obstacles"] == obstacles


class TestDCLossModel:
    """Test DC circuit loss models."""

    def test_dc_loss_basic_only(self):
        """Test basic DC losses without ohmic modeling."""
        dc_losses = DCLossModel(
            dc_wiring_loss_percent=0.03,
            connection_loss_percent=0.01,
            mismatch_loss_percent=0.025,
        )
        config = dc_losses.create()

        assert config["basic_losses"] == 0.065  # 3% + 1% + 2.5%
        assert "ohmic_losses" not in config

    def test_dc_loss_with_ohmic(self):
        """Test DC losses with ohmic resistance modeling."""
        dc_losses = DCLossModel(
            enable_ohmic_losses=True,
            resistance_per_string_ohm=0.5,
            voltage_dependent=True,
        )
        config = dc_losses.create()

        assert "ohmic_losses" in config
        assert config["ohmic_losses"]["resistance"] == 0.5
        assert config["ohmic_losses"]["voltage_dependent"] is True


class TestSpectralModel:
    """Test advanced spectral models."""

    def test_spectral_model_no_loss_default(self):
        """Test default no_loss spectral model."""
        spectral = SpectralModel()
        assert spectral.model == "no_loss"
        assert spectral.create() == "no_loss"
        assert spectral.get_spectral_parameters() is None

    def test_spectral_model_first_solar(self):
        """Test First Solar spectral model."""
        spectral = SpectralModel(model="first_solar", first_solar_module="FS-6420")
        assert spectral.create() == "first_solar"
        assert spectral.get_spectral_parameters() == {"module": "FS-6420"}

    def test_spectral_model_with_module_type(self):
        """Test spectral model with module technology type."""
        spectral = SpectralModel(model="sapm", module_type="crystalline_silicon")
        assert spectral.create() == "sapm"
        assert spectral.get_spectral_parameters() == {
            "module_type": "crystalline_silicon"
        }


class TestAdvancedInverterModel:
    """Test advanced inverter models."""

    def test_advanced_inverter_pvwatts_default(self):
        """Test default PVWatts AC model."""
        inverter = AdvancedInverterModel()
        assert inverter.ac_model == "pvwatts"
        assert inverter.create() == "pvwatts"
        assert not inverter.enable_degradation
        assert inverter.get_degradation_parameters() is None

    def test_advanced_inverter_with_degradation(self):
        """Test inverter model with degradation."""
        inverter = AdvancedInverterModel(
            enable_degradation=True, degradation_rate_per_year=0.007
        )
        degradation = inverter.get_degradation_parameters()
        assert degradation == {"degradation_rate": 0.007}

    def test_advanced_inverter_adr_model(self):
        """Test ADR (performance ratio) inverter model."""
        inverter = AdvancedInverterModel(
            ac_model="adr",
            enable_adr=True,
            reference_irradiance=1200.0,
            reference_temperature=20.0,
        )
        assert inverter.create() == "adr"
        adr_params = inverter.get_adr_parameters()
        assert adr_params == {
            "reference_irradiance": 1200.0,
            "reference_temperature": 20.0,
        }


class TestAdvancedPVLibIntegration:
    """Test integration of advanced models with main PVLib model."""

    def test_advanced_models_in_array_setup(self):
        """Test that ArraySetup accepts advanced models."""
        from app.core.simulation.pvlib_models import ArraySetup, MountFixed

        # Create advanced models
        iam_model = IAMModel(model="ashrae", ashrae_b=0.08)
        bifacial_config = BifacialConfiguration(enable_bifacial=True, bifaciality=0.75)
        soiling_model = SoilingModel(enable_soiling=True, model="hsu")

        # Create array setup with advanced models
        array_setup = ArraySetup(
            name="Advanced Test Array",
            mount=MountFixed(
                type="fixed_mount", tilt_degrees=30.0, azimuth_degrees=180.0
            ),
            number_of_strings=100,
            iam_model=iam_model,
            bifacial_config=bifacial_config,
            soiling_model=soiling_model,
        )

        assert array_setup.iam_model.model == "ashrae"
        assert array_setup.bifacial_config.enable_bifacial is True
        assert array_setup.soiling_model.enable_soiling is True

    def test_advanced_spectral_in_physical_simulation(self):
        """Test advanced spectral model in PhysicalSimulation."""
        from app.core.simulation.pvlib_models import PhysicalSimulation

        advanced_spectral = SpectralModel(
            model="first_solar", first_solar_module="FS-6420"
        )

        simulation = PhysicalSimulation(
            aoi_model="martin_ruiz", advanced_spectral_model=advanced_spectral
        )

        params = simulation.create()
        assert params["aoi_model"] == "martin_ruiz"
        assert params["spectral_model"] == "first_solar"

    def test_backwards_compatibility(self):
        """Test that existing configurations still work."""
        from app.core.simulation.pvlib_models import PhysicalSimulation

        # Old-style configuration should still work
        simulation = PhysicalSimulation(aoi_model="physical", spectral_model="no_loss")

        params = simulation.create()
        assert params["aoi_model"] == "physical"
        assert params["spectral_model"] == "no_loss"


if __name__ == "__main__":
    pytest.main([__file__])
