"""
Tests for PV simulation model functionality.

This module tests the PVModel class including validation, simulation execution,
result processing, and capacity calculations.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from app.core.simulation.pv_model import PVModel
from app.core.simulation.pvlib_models import (
    ArraySetup,
    InverterParameters,
    Location,
    ModuleParameters,
    MountFixed,
    PvArray,
    PVLibModel,
    PvSystem,
)
from app.core.simulation.weather import WeatherProvider
from app.core.utils.date_handling import TimeInterval
from app.core.utils.location import GeospatialLocation


class TestPVModelValidation:
    """Test PV model validation functionality."""

    @pytest.fixture
    def sample_weather_provider(self):
        """Create a sample weather provider for testing."""
        location = GeospatialLocation(latitude=34.0522, longitude=-118.2437)
        return WeatherProvider(
            location=location,
            start_date="2023-06-15",
            end_date="2023-06-16",
            organization="test_org",
            asset="test_asset",
            interval=TimeInterval.HOURLY,
        )

    @pytest.fixture
    def sample_pv_config(self):
        """Create a sample PV configuration for testing."""
        pv_config = {
            "location": {
                "name": "Test Location",
                "latitude": 34.0522,
                "longitude": -118.2437,
                "tz": "America/Los_Angeles",
            },
            "pv_systems": [
                {
                    "inverters": {
                        "count": 1,
                        "max_power_output_ac_w": 5000.0,
                        "efficiency_rating_percent": 0.95,
                    },
                    "pv_arrays": [
                        {
                            "pv_modules": {
                                "count": 20,
                                "name": "Test Module",
                                "nameplate_dc_rating_w": 300.0,
                                "power_temperature_coefficient_per_degree_c": -0.004,
                            },
                            "array_setup": {
                                "name": "Test Array",
                                "mount": {
                                    "type": "fixed_mount",
                                    "tilt_degrees": 30.0,
                                },
                                "number_of_strings": 10,
                            },
                        }
                    ],
                }
            ],
        }

        return PVLibModel(**pv_config)

    def test_pv_model_valid_configuration(
        self, sample_pv_config, sample_weather_provider
    ):
        """Test that valid PV model configuration is accepted."""
        pv_model = PVModel(
            pv_config=sample_pv_config, weather_provider=sample_weather_provider
        )
        assert pv_model.pv_config is not None
        assert pv_model.weather_provider is not None

    def test_pv_model_json_like_configuration(self, sample_weather_provider):
        """Test PV model creation using JSON-like configuration format."""
        # Complete PV configuration in nested JSON-like format
        # (similar to other project structure)
        pv_config = {
            "location": {
                "name": "Test Solar Farm",
                "latitude": 34.0522,
                "longitude": -118.2437,
                "tz": "America/Los_Angeles",
                "altitude": 71,
            },
            "pv_systems": [
                {
                    "inverters": {
                        "count": 2,
                        "max_power_output_ac_w": 4000.0,
                        "efficiency_rating_percent": 0.96,
                    },
                    "pv_arrays": [
                        {
                            "pv_modules": {
                                "count": 20,
                                "name": "Test Module 400W",
                                "nameplate_dc_rating_w": 400.0,
                                "power_temperature_coefficient_per_degree_c": -0.0035,
                            },
                            "array_setup": {
                                "name": "Main Array",
                                "mount": {
                                    "type": "fixed_mount",
                                    "tilt_degrees": 25.0,
                                    "azimuth_degrees": 180.0,
                                },
                                "number_of_strings": 50,
                            },
                        }
                    ],
                }
            ],
            "physical_simulation": {
                "aoi_model": "physical",
                "spectral_model": "no_loss",
            },
        }

        # Create PV model in one shot from JSON using Pydantic's built-in parsing
        pvlib_config = PVLibModel(**pv_config)

        # Create final PV model
        pv_model = PVModel(
            pv_config=pvlib_config, weather_provider=sample_weather_provider
        )

        # Verify the configuration matches the JSON structure
        assert pv_model.pv_config is not None
        assert pv_model.weather_provider is not None
        assert pv_model.pv_config.location.name == "Test Solar Farm"
        assert len(pv_model.pv_config.pv_systems) == 1
        assert len(pv_model.pv_config.pv_systems[0].pv_arrays) == 1
        assert (
            pv_model.pv_config.pv_systems[0].pv_arrays[0].array_setup.name
            == "Main Array"
        )
        assert pv_model.pv_config.physical_simulation is not None
        assert pv_model.pv_config.physical_simulation.aoi_model == "physical"

    def test_pv_model_validation_empty_systems(self, sample_weather_provider):
        """Test that PV model validation rejects empty systems."""
        location = Location(
            name="Test Location",
            latitude=34.0522,
            longitude=-118.2437,
            tz="America/Los_Angeles",
        )
        pv_config = PVLibModel(location=location, pv_systems=[])

        with pytest.raises(ValidationError) as exc_info:
            PVModel(pv_config=pv_config, weather_provider=sample_weather_provider)
        assert "At least one PV system must be configured" in str(exc_info.value)

    def test_pv_model_validation_empty_arrays(self, sample_weather_provider):
        """Test that PV model validation rejects systems with no arrays."""
        location = Location(
            name="Test Location",
            latitude=34.0522,
            longitude=-118.2437,
            tz="America/Los_Angeles",
        )
        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )
        pv_system = PvSystem(inverters=inverter, pv_arrays=[])
        pv_config = PVLibModel(location=location, pv_systems=[pv_system])

        with pytest.raises(ValidationError) as exc_info:
            PVModel(pv_config=pv_config, weather_provider=sample_weather_provider)
        assert "PV system 0 must have at least one array" in str(exc_info.value)


class TestPVModelSimulation:
    """Test PV model simulation execution."""

    @pytest.fixture
    def sample_pv_model(self):
        """Create a sample PV model for testing."""
        location_geo = GeospatialLocation(latitude=34.0522, longitude=-118.2437)
        weather_provider = WeatherProvider(
            location=location_geo,
            start_date="2023-06-15",
            end_date="2023-06-16",
            organization="test_org",
            asset="test_asset",
            interval=TimeInterval.HOURLY,
        )

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
            name="Test Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        array = PvArray(pv_modules=module, array_setup=array_setup)
        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )
        pv_system = PvSystem(inverters=inverter, pv_arrays=[array])
        pv_config = PVLibModel(location=location, pv_systems=[pv_system])

        return PVModel(pv_config=pv_config, weather_provider=weather_provider)

    @pytest.fixture
    def mock_weather_data(self):
        """Create mock weather data for testing."""
        dates = pd.date_range("2023-06-15", periods=24, freq="h")
        return pd.DataFrame(
            {
                "ghi": [0] * 6
                + [100, 300, 500, 700, 800, 900, 800, 700, 500, 300, 100]
                + [0] * 7,
                "dni": [0] * 6
                + [150, 400, 600, 800, 850, 900, 850, 800, 600, 400, 150]
                + [0] * 7,
                "dhi": [0] * 6
                + [50, 100, 150, 200, 250, 300, 250, 200, 150, 100, 50]
                + [0] * 7,
                "temp_air": [15] * 24,
            },
            index=dates,
        )

    @patch("app.core.simulation.weather.WeatherProvider.get_data")
    @patch("app.core.simulation.pv_model.modelchain.ModelChain")
    async def test_run_simulation_success(
        self, mock_model_chain, mock_get_data, sample_pv_model, mock_weather_data
    ):
        """Test successful simulation execution."""
        # Mock the weather provider's get_data method
        mock_get_data.return_value = mock_weather_data

        # Mock the model chain and results
        mock_chain_instance = MagicMock()
        mock_model_chain.return_value = mock_chain_instance

        # Mock AC results
        mock_ac_result = MagicMock()
        mock_ac_result.index = mock_weather_data.index
        mock_ac_result.values = [100.0] * len(mock_weather_data)

        # Mock DC results
        mock_dc_result = [MagicMock()]
        mock_dc_result[0].values = [120.0] * len(mock_weather_data)

        # Set up results
        mock_chain_instance.results.ac = mock_ac_result
        mock_chain_instance.results.dc = mock_dc_result
        mock_chain_instance.system.arrays = [MagicMock(name="Test Array")]

        # Run simulation
        results = await sample_pv_model.run_simulation()

        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(mock_weather_data)
        assert "Total AC power (W)" in results.columns
        assert "date_time" in results.columns

        # Verify get_data was called
        mock_get_data.assert_called_once()

    @patch("app.core.simulation.weather.WeatherProvider.get_data")
    @patch("app.core.simulation.pv_model.modelchain.ModelChain")
    async def test_run_simulation_no_results(
        self, mock_model_chain, mock_get_data, sample_pv_model, mock_weather_data
    ):
        """Test simulation execution when no results are generated."""
        # Mock the weather provider's get_data method
        mock_get_data.return_value = mock_weather_data

        # Mock the model chain with no results
        mock_chain_instance = MagicMock()
        mock_model_chain.return_value = mock_chain_instance
        mock_chain_instance.results = None

        # Run simulation and expect error
        with pytest.raises(RuntimeError) as exc_info:
            await sample_pv_model.run_simulation()
        assert "PVLib simulation failed - no results generated" in str(exc_info.value)

    @patch("app.core.simulation.weather.WeatherProvider.get_data")
    @patch("app.core.simulation.pv_model.modelchain.ModelChain")
    async def test_run_simulation_no_ac_results(
        self, mock_model_chain, mock_get_data, sample_pv_model, mock_weather_data
    ):
        """Test simulation execution when no AC results are generated."""
        # Mock the weather provider's get_data method
        mock_get_data.return_value = mock_weather_data

        # Mock the model chain with results but no AC
        mock_chain_instance = MagicMock()
        mock_model_chain.return_value = mock_chain_instance
        mock_chain_instance.results = MagicMock()
        mock_chain_instance.results.ac = None

        # Run simulation and expect error
        with pytest.raises(RuntimeError) as exc_info:
            await sample_pv_model.run_simulation()
        assert "PVLib simulation failed - no AC results generated" in str(
            exc_info.value
        )


class TestPVModelResultProcessing:
    """Test PV model result processing for different DC models."""

    def create_mock_model_chain(self, dc_model_name="pvwatts", has_dc=True):
        """Create a mock model chain for testing."""
        """Create a mock model chain for testing."""
        mock_chain = MagicMock()

        # Mock DC model
        if dc_model_name and dc_model_name.strip():
            mock_dc_model = MagicMock()
            mock_dc_model.__name__ = dc_model_name
            mock_chain.dc_model = mock_dc_model
        else:
            mock_chain.dc_model = None

        # Mock AC results
        dates = pd.date_range("2023-06-15", periods=24, freq="h")
        mock_ac_result = MagicMock()
        mock_ac_result.index = dates
        mock_ac_result.values = [100.0] * 24

        # Mock DC results if requested
        if has_dc:
            mock_dc_result = [MagicMock()]
            mock_dc_result[0].values = [120.0] * 24
            mock_chain.results.dc = mock_dc_result
            mock_chain.system.arrays = [MagicMock(name="Test Array")]
        else:
            mock_chain.results.dc = None

        mock_chain.results.ac = mock_ac_result
        return mock_chain

    def test_get_dc_model_name(self, sample_pv_model):
        """Test DC model name extraction."""
        # Test with valid DC model
        mock_chain = self.create_mock_model_chain("pvwatts")
        result = sample_pv_model._get_dc_model_name(mock_chain)
        assert result == "pvwatts"

        # Test with no DC model
        mock_chain = self.create_mock_model_chain("")
        result = sample_pv_model._get_dc_model_name(mock_chain)
        assert result == "unknown"

    def test_process_pv_watts_results(self, sample_pv_model):
        """Test processing PV Watts model results."""
        mock_chain = self.create_mock_model_chain("pvwatts")
        results = sample_pv_model._process_pv_watts_results(mock_chain)

        assert isinstance(results, pd.DataFrame)
        assert "Total AC power (W)" in results.columns
        assert "date_time" in results.columns
        assert len(results) == 24

    def test_process_cec_results(self, sample_pv_model):
        """Test processing CEC model results."""
        mock_chain = self.create_mock_model_chain("cec")

        # Add p_mp attribute for CEC model
        mock_chain.results.ac.p_mp = MagicMock()
        mock_chain.results.ac.p_mp.values = [100.0] * 24

        results = sample_pv_model._process_cec_results(mock_chain)

        assert isinstance(results, pd.DataFrame)
        assert "Total AC power (W)" in results.columns
        assert "date_time" in results.columns
        assert len(results) == 24

    def test_process_default_results(self, sample_pv_model):
        """Test processing unknown model results."""
        mock_chain = self.create_mock_model_chain("unknown_model")
        results = sample_pv_model._process_default_results(mock_chain)

        assert isinstance(results, pd.DataFrame)
        assert "Total AC power (W)" in results.columns
        assert "date_time" in results.columns
        assert len(results) == 24

    def test_create_minimal_results(self, sample_pv_model):
        """Test creation of minimal results when processing fails."""
        mock_chain = MagicMock()
        mock_ac_result = MagicMock()
        dates = pd.date_range("2023-06-15", periods=24, freq="h")
        mock_ac_result.index = dates
        mock_chain.results.ac = mock_ac_result

        results = sample_pv_model._create_minimal_results(mock_chain)

        assert isinstance(results, pd.DataFrame)
        assert "Total AC power (W)" in results.columns
        assert "date_time" in results.columns
        assert len(results) == 24
        assert all(results["Total AC power (W)"] == 0.0)

    @pytest.fixture
    def sample_pv_model(self):
        """Create a sample PV model for testing."""
        location_geo = GeospatialLocation(latitude=34.0522, longitude=-118.2437)
        weather_provider = WeatherProvider(
            location=location_geo,
            start_date="2023-06-15",
            end_date="2023-06-16",
            organization="test_org",
            asset="test_asset",
            interval=TimeInterval.HOURLY,
        )

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
            name="Test Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        array = PvArray(pv_modules=module, array_setup=array_setup)
        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )
        pv_system = PvSystem(inverters=inverter, pv_arrays=[array])
        pv_config = PVLibModel(location=location, pv_systems=[pv_system])

        return PVModel(pv_config=pv_config, weather_provider=weather_provider)


class TestPVModelCapacityCalculation:
    """Test PV model system capacity calculations."""

    def test_get_system_capacity_single_system(self):
        """Test capacity calculation for single system."""
        location_geo = GeospatialLocation(latitude=34.0522, longitude=-118.2437)
        weather_provider = WeatherProvider(
            location=location_geo,
            start_date="2023-06-15",
            end_date="2023-06-16",
            organization="test_org",
            asset="test_asset",
            interval=TimeInterval.HOURLY,
        )

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
            name="Test Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        array = PvArray(pv_modules=module, array_setup=array_setup)
        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )
        pv_system = PvSystem(inverters=inverter, pv_arrays=[array])
        pv_config = PVLibModel(location=location, pv_systems=[pv_system])

        pv_model = PVModel(pv_config=pv_config, weather_provider=weather_provider)
        capacity = pv_model.get_system_capacity()

        # Expected DC capacity: 300W * 20 modules * 10 strings = 60,000W
        expected_dc = 300.0 * 20 * 10
        assert capacity["dc_capacity_w"] == expected_dc
        assert (
            capacity["ac_capacity_w"] == 0
        )  # AC capacity only available after create()
        assert capacity["dc_ac_ratio"] == 0  # Will be 0 since AC capacity is 0

    def test_get_system_capacity_multiple_arrays(self):
        """Test capacity calculation for system with multiple arrays."""
        location_geo = GeospatialLocation(latitude=34.0522, longitude=-118.2437)
        weather_provider = WeatherProvider(
            location=location_geo,
            start_date="2023-06-15",
            end_date="2023-06-16",
            organization="test_org",
            asset="test_asset",
            interval=TimeInterval.HOURLY,
        )

        location = Location(
            name="Test Location",
            latitude=34.0522,
            longitude=-118.2437,
            tz="America/Los_Angeles",
        )

        # Create two arrays with different configurations
        mount1 = MountFixed(type="fixed_mount", tilt_degrees=30.0)
        array_setup1 = ArraySetup(name="Array 1", mount=mount1, number_of_strings=10)
        module1 = ModuleParameters(
            count=20,
            name="Module 1",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        array1 = PvArray(pv_modules=module1, array_setup=array_setup1)

        mount2 = MountFixed(type="fixed_mount", tilt_degrees=25.0)
        array_setup2 = ArraySetup(name="Array 2", mount=mount2, number_of_strings=15)
        module2 = ModuleParameters(
            count=25,
            name="Module 2",
            nameplate_dc_rating_w=250.0,
            power_temperature_coefficient_per_degree_c=-0.003,
        )
        array2 = PvArray(pv_modules=module2, array_setup=array_setup2)

        inverter = InverterParameters(
            count=1, max_power_output_ac_w=10000.0, efficiency_rating_percent=0.95
        )
        pv_system = PvSystem(inverters=inverter, pv_arrays=[array1, array2])
        pv_config = PVLibModel(location=location, pv_systems=[pv_system])

        pv_model = PVModel(pv_config=pv_config, weather_provider=weather_provider)
        capacity = pv_model.get_system_capacity()

        # Expected DC capacity:
        # Array 1: 300W * 20 modules * 10 strings = 60,000W
        # Array 2: 250W * 25 modules * 15 strings = 93,750W
        # Total: 153,750W
        expected_dc = (300.0 * 20 * 10) + (250.0 * 25 * 15)
        assert capacity["dc_capacity_w"] == expected_dc
        assert capacity["ac_capacity_w"] == 0
        assert capacity["dc_ac_ratio"] == 0

    def test_get_dc_column_name(self):
        """Test DC column name generation."""
        location_geo = GeospatialLocation(latitude=34.0522, longitude=-118.2437)
        weather_provider = WeatherProvider(
            location=location_geo,
            start_date="2023-06-15",
            end_date="2023-06-16",
            organization="test_org",
            asset="test_asset",
            interval=TimeInterval.HOURLY,
        )

        location = Location(
            name="Test Location",
            latitude=34.0522,
            longitude=-118.2437,
            tz="America/Los_Angeles",
        )

        mount = MountFixed(type="fixed_mount")
        array_setup = ArraySetup(name="Test Array", mount=mount, number_of_strings=10)
        module = ModuleParameters(
            count=20,
            name="Test Module",
            nameplate_dc_rating_w=300.0,
            power_temperature_coefficient_per_degree_c=-0.004,
        )
        array = PvArray(pv_modules=module, array_setup=array_setup)
        inverter = InverterParameters(
            count=1, max_power_output_ac_w=5000.0, efficiency_rating_percent=0.95
        )
        pv_system = PvSystem(inverters=inverter, pv_arrays=[array])
        pv_config = PVLibModel(location=location, pv_systems=[pv_system])

        pv_model = PVModel(pv_config=pv_config, weather_provider=weather_provider)

        column_name = pv_model.get_dc_column_name("Test Array")
        assert column_name == "Test Array DC power (W)"
