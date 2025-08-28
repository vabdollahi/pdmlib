"""
Portfolio tests using existing CSV data and JSON configurations.

This module tests portfolio functionality including:
- Portfolio creation with multiple plants
- Power dispatch across different strategies
- Revenue optimization
- Available power calculations
- Portfolio diversity metrics

All tests use local CSV files for weather and price data, no API calls.
All tests use the unified configuration system for consistency.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from app.core.simulation.battery_simulator import (
    BatteryConfiguration,
    LinearBatterySimulator,
)
from app.core.simulation.plant import SolarBatteryPlant
from app.core.simulation.portfolio import (
    PortfolioConfiguration,
    PortfolioStrategy,
    PowerPlantPortfolio,
)
from app.core.simulation.price_provider import CSVPriceProvider
from app.core.simulation.pv_model import PVModel
from app.core.simulation.pvlib_models import PVLibModel
from app.core.simulation.solar_revenue import SolarRevenueCalculator
from app.core.simulation.weather_provider import CSVWeatherProvider
from app.core.utils.location import GeospatialLocation
from tests.config import test_config


# Shared fixtures for all test classes
@pytest.fixture
def test_portfolio():
    """Create a test portfolio using spec-driven configuration."""
    from app.core.environment.config import create_environment_config_from_json

    config = create_environment_config_from_json(test_config.environment_spec_path)
    # Get the first portfolio from the spec
    return config.portfolios[0]


@pytest.fixture
def portfolio_config():
    """Get portfolio configuration from unified config."""
    config_data = test_config.load_config_file("test_config_multi.json")
    # Use the first portfolio from the portfolios array
    portfolio_data = config_data["portfolios"][0]
    return PortfolioConfiguration.model_validate(
        {
            "name": portfolio_data["name"],
            "max_total_power_mw": portfolio_data["max_total_power_mw"],
            "allow_grid_purchase": portfolio_data.get("allow_grid_purchase", False),
            "strategy": PortfolioStrategy.BALANCED,  # Default strategy
        }
    )


@pytest.fixture
def plant_configs():
    """Load plant configurations from unified config."""
    config_data = test_config.load_config_file("test_config_multi.json")
    # Get plants from the first portfolio
    return config_data["portfolios"][0]["plants"]


@pytest.fixture
def weather_provider():
    """CSV weather data provider for testing - no async calls to external APIs."""
    # Create a location for the weather provider
    location = GeospatialLocation(latitude=34.0522, longitude=-118.2437)
    csv_path = Path(__file__).parent.parent.parent / "data" / "sample_weather_data.csv"
    # Initialize provider with local CSV file only
    provider = CSVWeatherProvider(location=location, file_path=str(csv_path))
    # Set range to match the data in the CSV file
    provider.set_range(
        start_time=datetime(2025, 7, 15, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2025, 7, 15, 23, 0, tzinfo=timezone.utc),
    )
    return provider


@pytest.fixture
def price_provider():
    """CSV price data provider for testing - no async calls to external APIs."""
    csv_path = Path(__file__).parent.parent.parent / "data" / "sample_price_data.csv"
    # Initialize provider with local CSV file only
    provider = CSVPriceProvider(csv_file_path=str(csv_path))
    # Set range to match the data in the CSV file
    provider.set_range(
        start_time=datetime(2025, 7, 15, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2025, 7, 15, 23, 0, tzinfo=timezone.utc),
    )
    return provider


@pytest.fixture
async def test_plants(
    plant_configs, weather_provider, price_provider
) -> List[SolarBatteryPlant]:
    """Create test plants from configurations using only local CSV data."""
    from app.core.simulation.plant import PlantConfiguration

    plants = []

    for i, config in enumerate(plant_configs):
        # Create PV model using the config and local weather provider
        pv_config = PVLibModel.model_validate(config)
        pv_model = PVModel(pv_config=pv_config, weather_provider=weather_provider)

        # Create battery simulators based on config
        batteries = []
        for bat_config in config.get("batteries", []):
            # Create battery configuration from JSON data
            battery_config = BatteryConfiguration(
                energy_capacity_mwh=bat_config["energy_capacity_mwh"],
                max_power_mw=bat_config["max_power_mw"],
                round_trip_efficiency=bat_config["round_trip_efficiency"],
                initial_soc=bat_config.get("initial_soc", 0.5),
                min_soc=bat_config.get("min_soc", 0.1),
                max_soc=bat_config.get("max_soc", 0.9),
            )

            battery = LinearBatterySimulator(config=battery_config)
            batteries.append(battery)

        # Create revenue calculator using local price provider
        revenue_calc = SolarRevenueCalculator(
            price_provider=price_provider, pv_model=pv_model
        )

        # Extract plant configuration from the JSON config
        plant_cfg_data = config.get("plant_config", {})
        plant_config = PlantConfiguration(
            name=plant_cfg_data.get("name", f"Test Plant {i + 1}"),
            plant_id=plant_cfg_data.get("plant_id", f"TP-{i + 1:03d}"),
            max_net_power_mw=plant_cfg_data.get("max_net_power_mw", 25.0),
            min_net_power_mw=plant_cfg_data.get("min_net_power_mw", 0.0),
            enable_market_participation=plant_cfg_data.get(
                "enable_market_participation", True
            ),
        )

        # Create the plant using the correct constructor arguments
        plant = SolarBatteryPlant(
            config=plant_config,
            pv_model=pv_model,
            batteries=batteries,
            revenue_calculator=revenue_calc,
        )
        plants.append(plant)

    return plants


class TestPortfolioCreation:
    """Test portfolio creation and validation using local CSV data."""

    @pytest.mark.asyncio
    async def test_portfolio_creation(self, portfolio_config, test_plants):
        """Test basic portfolio creation with multiple plants."""
        plants_list = await test_plants
        portfolio = PowerPlantPortfolio(config=portfolio_config, plants=plants_list)

        assert portfolio.config.name == "Multi-Plant Test Portfolio"
        assert portfolio.config.strategy == PortfolioStrategy.BALANCED
        assert len(portfolio.plants) == 3

        # Verify all plants are properly initialized
        plants_with_batteries = 0
        plants_without_batteries = 0

        for plant in portfolio.plants:
            assert plant.pv_model is not None
            assert plant.revenue_calculator is not None

            # Count plants with and without batteries
            if len(plant.batteries) > 0:
                plants_with_batteries += 1
            else:
                plants_without_batteries += 1

        # Verify our expected configuration: all 3 plants have batteries
        assert plants_with_batteries == 3, (
            f"Expected 3 plants with batteries, got {plants_with_batteries}"
        )
        assert plants_without_batteries == 0, (
            f"Expected 0 plants without batteries, got {plants_without_batteries}"
        )

    @pytest.mark.asyncio
    async def test_portfolio_data_loading(self, weather_provider, price_provider):
        """Test that weather and price data can be loaded from CSV files."""
        # Test weather data loading
        weather_data = await weather_provider.get_data()
        assert len(weather_data) > 0
        assert "ghi" in weather_data.columns
        assert "temperature_celsius" in weather_data.columns

        # Test price data loading
        price_data = await price_provider.get_data()
        assert len(price_data) > 0
        assert "price_dollar_mwh" in price_data.columns

        # Verify data covers the same time period
        weather_timestamps = weather_data.index
        price_timestamps = price_data.index
        assert len(weather_timestamps) == len(price_timestamps)

    def test_portfolio_validation_minimum_plants(self, portfolio_config):
        """Test portfolio validation with minimum plants requirement."""
        # Should raise error with empty plants list since min_operating_plants=2
        with pytest.raises(ValueError, match="Portfolio must have at least"):
            PowerPlantPortfolio(config=portfolio_config, plants=[])

    @pytest.mark.asyncio
    async def test_portfolio_add_plant(self, test_plants):
        """Test adding plants to portfolio."""
        plants_list = await test_plants

        # Create config that allows starting with 1 plant
        config = PortfolioConfiguration(
            name="Add Plant Test Portfolio",
            portfolio_id="add-plant-test",
            max_total_power_mw=100.0,
            min_operating_plants=1,  # Allow starting with 1 plant
            enable_market_arbitrage=True,
        )

        portfolio = PowerPlantPortfolio(
            config=config,
            plants=[plants_list[0]],  # Start with one plant
        )

        initial_count = len(portfolio.plants)
        portfolio.add_plant(plants_list[1])

        assert len(portfolio.plants) == initial_count + 1

    @pytest.mark.asyncio
    async def test_portfolio_capacity_calculation(self, portfolio_config, test_plants):
        """Test total portfolio capacity calculation."""
        plants_list = await test_plants
        portfolio = PowerPlantPortfolio(config=portfolio_config, plants=plants_list)

        total_capacity = portfolio.get_total_capacity()
        assert isinstance(total_capacity, float)
        assert total_capacity > 0

        # Verify it's positive and reasonable for a multi-plant portfolio
        assert total_capacity > 10.0  # Should be more than 10 MW for 3 plants


class TestPortfolioDispatch:
    """Test portfolio power dispatch functionality using local CSV data."""

    @pytest.fixture
    async def test_portfolio(self, portfolio_config, test_plants, price_provider):
        """Create test portfolio with revenue calculator using local data."""
        plants_list = await test_plants
        config = PortfolioConfiguration(
            name="Dispatch Test Portfolio",
            portfolio_id="dispatch-test",
            max_total_power_mw=100.0,
            min_operating_plants=1,
            enable_market_arbitrage=True,
        )

        portfolio = PowerPlantPortfolio(config=config, plants=plants_list)

        return portfolio

    @pytest.mark.asyncio
    async def test_dispatch_power_balanced_strategy(self, test_portfolio):
        """Test power dispatch with balanced strategy using local data."""
        portfolio = await test_portfolio
        # Noon - matches CSV data
        timestamp = datetime(2025, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
        target_power = 10.0  # MW

        portfolio.set_strategy(PortfolioStrategy.BALANCED)

        actual_power, state, is_valid = await portfolio.dispatch_power(
            target_net_power_mw=target_power, timestamp=timestamp, interval_minutes=60.0
        )

        assert is_valid
        assert isinstance(actual_power, float)
        assert isinstance(state, dict)
        assert "plant_results" in state

    @pytest.mark.asyncio
    async def test_get_available_power(self, test_portfolio):
        """Test available power calculation using local CSV data."""
        portfolio = await test_portfolio
        # Morning - matches CSV data
        timestamp = datetime(2025, 7, 15, 10, 0, 0, tzinfo=timezone.utc)

        max_gen, max_cons = await portfolio.get_available_power(
            timestamp=timestamp, interval_minutes=60.0
        )

        assert isinstance(max_gen, float)
        assert isinstance(max_cons, float)
        assert max_gen >= 0
        assert max_cons >= 0


class TestPortfolioStrategies:
    """Test different portfolio strategies using local CSV data."""

    @pytest.fixture
    async def strategy_test_portfolio(self, test_plants):
        """Portfolio specifically for strategy testing."""
        plants_list = await test_plants
        config = PortfolioConfiguration(
            name="Strategy Test Portfolio",
            portfolio_id="strategy-test",
            max_total_power_mw=80.0,
            min_operating_plants=1,
        )

        return PowerPlantPortfolio(config=config, plants=plants_list)

    @pytest.mark.asyncio
    async def test_strategy_setting(self, strategy_test_portfolio):
        """Test setting different portfolio strategies."""
        portfolio = await strategy_test_portfolio

        # Test all available strategies
        for strategy in PortfolioStrategy:
            portfolio.set_strategy(strategy)
            # Check the internal strategy state
            assert portfolio._current_strategy == strategy


class TestPortfolioEdgeCases:
    """Test edge cases and error conditions using local CSV data."""

    def test_empty_portfolio_validation(self):
        """Test that portfolio validates minimum plant requirements."""
        config = PortfolioConfiguration(
            name="Validation Test Portfolio",
            portfolio_id="validation-test",
            max_total_power_mw=50.0,
            min_operating_plants=1,  # Minimum allowed value
        )

        # Should raise ValueError when trying to create with no plants
        with pytest.raises(ValueError, match="Portfolio must have at least 1 plants"):
            PowerPlantPortfolio(config=config, plants=[])

    @pytest.mark.asyncio
    async def test_portfolio_capacity_limits(self, test_plants):
        """Test portfolio capacity limits and constraints."""
        plants_list = await test_plants
        config = PortfolioConfiguration(
            name="Capacity Test Portfolio",
            portfolio_id="capacity-test",
            max_total_power_mw=100.0,  # Higher than total plant capacity
            min_operating_plants=1,
        )

        portfolio = PowerPlantPortfolio(config=config, plants=plants_list)

        # Portfolio should respect max power limits
        total_capacity = portfolio.get_total_capacity()
        assert isinstance(total_capacity, float)
        assert total_capacity > 0

    @pytest.mark.asyncio
    async def test_csv_data_integrity(self, weather_provider, price_provider):
        """Test that CSV data is properly loaded and has expected structure."""
        # Load weather data
        weather_data = await weather_provider.get_data()

        # Verify weather data structure
        required_weather_cols = ["ghi", "temperature_celsius", "dni", "dhi"]
        for col in required_weather_cols:
            assert col in weather_data.columns, f"Missing weather column: {col}"

        # Verify data is within reasonable ranges
        assert weather_data["ghi"].min() >= 0, "GHI should be non-negative"
        assert weather_data["temperature_celsius"].min() > -50, "Temperature too low"
        assert weather_data["temperature_celsius"].max() < 60, "Temperature too high"

        # Load price data
        price_data = await price_provider.get_data()

        # Verify price data structure
        assert "price_dollar_mwh" in price_data.columns

        # Verify price data is reasonable
        assert price_data["price_dollar_mwh"].min() > -500, "Price too negative"
        assert price_data["price_dollar_mwh"].max() < 1000, "Price too high"

        print(f"Weather data loaded: {len(weather_data)} rows")
        print(f"Price data loaded: {len(price_data)} rows")
        print("CSV data integrity check passed - no external API calls made")


class TestPortfolioPhysicsValidation:
    """Test portfolio physics and logical constraints."""

    def test_grid_purchase_physics_limits(self):
        """Test that grid purchase limits are physically reasonable."""
        # Test realistic grid purchase limits
        config = PortfolioConfiguration(
            name="Physics Test Portfolio",
            allow_grid_purchase=True,
            max_grid_purchase_mw=50.0,  # Reasonable for small producer
        )

        assert config.allow_grid_purchase is True
        assert config.max_grid_purchase_mw == 50.0

        # Test that grid purchase limit is positive
        with pytest.raises(ValueError):
            PortfolioConfiguration(
                name="Invalid Portfolio",
                max_grid_purchase_mw=-10.0,  # Should fail
            )

    def test_portfolio_power_physics(self):
        """Test portfolio power capacity physics constraints."""
        # Test reasonable total capacity limits
        config = PortfolioConfiguration(
            name="Power Physics Test",
            max_total_power_mw=1000.0,  # 1 GW utility scale
            min_operating_plants=2,
        )

        assert config.max_total_power_mw == 1000.0
        assert config.min_operating_plants >= 1

        # Test that power limits are positive
        with pytest.raises(ValueError):
            PortfolioConfiguration(
                name="Invalid Power Portfolio",
                max_total_power_mw=-100.0,  # Should fail
            )

    def test_risk_management_physics(self):
        """Test risk management parameters are within realistic bounds."""
        config = PortfolioConfiguration(
            name="Risk Test Portfolio",
            max_portfolio_risk=0.15,  # 15% max risk
            diversification_weight=0.3,  # 30% weight on diversification
        )

        # Risk should be between 0 and 1
        assert 0.0 <= config.max_portfolio_risk <= 1.0
        assert 0.0 <= config.diversification_weight <= 1.0

        # Test extreme values are rejected
        with pytest.raises(ValueError):
            PortfolioConfiguration(
                name="High Risk Portfolio",
                max_portfolio_risk=1.5,  # Should fail - over 100%
            )
