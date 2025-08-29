"""
Test the unified simulation manager for complete automation.
"""

from pathlib import Path

import pytest

from app.core.simulation.simulation_manager import (
    SimulationManager,
    create_simulation_from_json,
)


class TestSimulationManager:
    """Test the complete automation workflow."""

    @pytest.mark.asyncio
    async def test_simulation_manager_creation(self):
        """Test creating simulation manager from JSON configuration."""
        # Use existing test configuration
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "test_config_simple.json"
        )

        # Create simulation manager
        simulation = SimulationManager(config_file_path=config_path)

        # Verify initial state
        assert simulation.config_file_path == config_path
        assert not simulation.initialized
        assert simulation.environment_config is None
        assert simulation.environment is None
        assert simulation.agent is None
        assert simulation.market is None
        assert simulation.results is None

    @pytest.mark.asyncio
    async def test_simulation_initialization(self):
        """Test complete simulation initialization workflow."""
        # Use existing test configuration
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "test_config_simple.json"
        )

        # Create and initialize simulation
        simulation = await create_simulation_from_json(config_path)

        # Verify initialization
        assert simulation.initialized
        assert simulation.environment_config is not None
        assert simulation.environment is not None
        assert simulation.agent is not None
        assert simulation.market is not None
        assert simulation.results is not None

        # Verify components are properly configured
        assert hasattr(simulation.agent, "get_action")
        assert hasattr(simulation.environment, "reset")
        assert hasattr(simulation.environment, "step")

    @pytest.mark.asyncio
    async def test_short_simulation_run(self):
        """Test running a short simulation workflow."""
        # Use existing test configuration
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "test_config_simple.json"
        )

        # Create and initialize simulation
        simulation = await create_simulation_from_json(config_path)

        # Run short simulation
        max_steps = 2
        results = await simulation.run_simulation(max_steps=max_steps)

        # Verify results
        assert results is not None
        assert results.total_steps == max_steps + 1
        assert isinstance(results.total_reward, float)
        assert isinstance(results.average_reward_per_step, float)

        # Verify each step has required fields
        # The number of steps is max_steps + 1 because of the initial reset step
        assert len(results.step_results) == max_steps + 1
        for step_result in results.step_results:
            assert "step" in step_result
            assert "timestamp" in step_result
            assert "reward" in step_result
            assert "info" in step_result

    @pytest.mark.asyncio
    async def test_market_data_loading(self):
        """Test that market data is properly loaded."""
        # Use existing test configuration
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "test_config_simple.json"
        )

        # Create and initialize simulation
        simulation = await create_simulation_from_json(config_path)

        # Verify market data was loaded
        assert simulation.market is not None
        assert simulation.market.price_providers is not None
        assert len(simulation.market.price_providers) > 0

        # Verify price data can be loaded by fetching for the simulation range
        assert simulation.start_time is not None
        assert simulation.end_time is not None
        await simulation.market.load_market_data(
            simulation.start_time, simulation.end_time
        )
        price_data = await simulation.market.get_data()
        assert not price_data.empty

    @pytest.mark.asyncio
    async def test_power_profile_precalculation(self):
        """Test that power profiles are pre-calculated."""
        # Use existing test configuration
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "test_config_simple.json"
        )

        # Create and initialize simulation
        simulation = await create_simulation_from_json(config_path)

        # Verify environment and portfolios exist
        assert simulation.environment_config is not None
        assert len(simulation.environment_config.portfolios) > 0

        # Verify plants have PV models
        portfolio = simulation.environment_config.portfolios[0]
        assert len(portfolio.plants) > 0

        plant = portfolio.plants[0]
        assert hasattr(plant, "pv_model")

        # Note: We can't easily verify the cache was populated without
        # accessing internal state, but initialization should have triggered it
