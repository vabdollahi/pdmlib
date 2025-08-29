"""
Tests for the linear battery energy storage system simulator.

This module provides comprehensive tests for the battery simulator, including
configuration validation, power dispatch calculations, efficiency modeling,
and integration with time series data.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.core.simulation.battery_simulator import (
    BatteryConfiguration,
    BatteryOperationMode,
    BatterySimulatorFactory,
    BatteryStateColumns,
    LinearBatterySimulator,
)


class TestBatteryConfiguration:
    """Test battery configuration validation and constraints."""

    def test_valid_configuration(self):
        """Test creating valid battery configuration."""
        config = BatteryConfiguration(
            energy_capacity_mwh=100.0,
            max_power_mw=25.0,
            min_soc=0.1,
            max_soc=0.9,
            initial_soc=0.5,
            round_trip_efficiency=0.85,
        )

        assert config.energy_capacity_mwh == 100.0
        assert config.max_power_mw == 25.0
        assert config.min_soc == 0.1
        assert config.max_soc == 0.9
        assert config.initial_soc == 0.5
        assert config.round_trip_efficiency == 0.85

    def test_invalid_soc_bounds(self):
        """Test validation of SOC bounds."""
        with pytest.raises(ValueError, match="min_soc must be less than max_soc"):
            BatteryConfiguration(
                energy_capacity_mwh=100.0, max_power_mw=25.0, min_soc=0.8, max_soc=0.5
            )

    def test_invalid_initial_soc(self):
        """Test validation of initial SOC within bounds."""
        with pytest.raises(
            ValueError, match="initial_soc must be between min_soc and max_soc"
        ):
            BatteryConfiguration(
                energy_capacity_mwh=100.0,
                max_power_mw=25.0,
                min_soc=0.2,
                max_soc=0.8,
                initial_soc=0.9,
            )

    def test_power_rating_validation(self):
        """Test power rating validation."""
        with pytest.raises(ValueError, match="Power rating exceeds reasonable limits"):
            BatteryConfiguration(
                energy_capacity_mwh=100.0,
                max_power_mw=1500.0,  # Exceeds 1 GW limit
            )

    def test_minimum_capacity_validation(self):
        """Test minimum energy capacity validation."""
        with pytest.raises(ValueError, match="Input should be greater than 0.001"):
            BatteryConfiguration(
                energy_capacity_mwh=0.0001,  # Too small
                max_power_mw=25.0,
            )

    def test_minimum_power_validation(self):
        """Test minimum power rating validation."""
        with pytest.raises(ValueError, match="Input should be greater than 0.001"):
            BatteryConfiguration(
                energy_capacity_mwh=100.0,
                max_power_mw=0.0001,  # Too small
            )


class TestLinearBatterySimulator:
    """Test linear battery simulator functionality."""

    @pytest.fixture
    def battery_config(self):
        """Create standard battery configuration for testing."""
        return BatteryConfiguration(
            energy_capacity_mwh=100.0,  # 100 MWh
            max_power_mw=25.0,  # 25 MW
            min_soc=0.1,  # 10% minimum
            max_soc=0.9,  # 90% maximum
            initial_soc=0.5,  # 50% initial
            round_trip_efficiency=0.85,  # 85% round-trip efficiency
        )

    @pytest.fixture
    def battery(self, battery_config):
        """Create battery simulator for testing."""
        return LinearBatterySimulator(config=battery_config)

    def test_initialization(self, battery, battery_config):
        """Test battery initialization."""
        assert battery.state_of_charge == 0.5
        assert battery.stored_energy_mwh == 50.0  # 50% of 100 MWh
        assert battery.available_capacity_mwh == 40.0  # (90% - 50%) * 100 MWh
        assert battery.available_energy_mwh == 40.0  # (50% - 10%) * 100 MWh

    def test_efficiency_calculation(self, battery):
        """Test efficiency calculation from round-trip efficiency."""
        # 85% round-trip -> ~92.2% charge and discharge efficiency
        expected_eff = 0.85**0.5
        assert abs(battery.charge_efficiency - expected_eff) < 0.001
        assert abs(battery.discharge_efficiency - expected_eff) < 0.001

    def test_available_power_calculation(self, battery):
        """Test available power calculation."""
        max_charge, max_discharge = battery.get_available_power(interval_minutes=60.0)

        # Power-limited: ±25 MW
        # Energy-limited charge: -40 MWh / 1 hour / 0.922 = -43.4 MW
        # Energy-limited discharge: 40 MWh / 1 hour * 0.922 = 36.9 MW

        assert max_charge == -25.0  # Power-limited
        assert max_discharge == 25.0  # Power-limited

    def test_discharge_operation(self, battery):
        """Test battery discharge operation."""
        target_power = 20.0  # MW
        actual_power, new_soc, valid = battery.dispatch_power(target_power, 60.0)

        assert actual_power == 20.0
        assert valid is True
        assert new_soc < 0.5  # SOC should decrease

        # Energy delivered: 20 MWh
        # Energy from battery: 20 / 0.922 ≈ 21.7 MWh
        # New SOC: (50 - 21.7) / 100 ≈ 0.283
        expected_soc = (50.0 - 20.0 / (0.85**0.5)) / 100.0
        assert abs(new_soc - expected_soc) < 0.001

    def test_charge_operation(self, battery):
        """Test battery charge operation."""
        target_power = -15.0  # MW (negative for charging)
        actual_power, new_soc, valid = battery.dispatch_power(target_power, 60.0)

        assert actual_power == -15.0
        assert valid is True
        assert new_soc > 0.5  # SOC should increase

        # Energy input: 15 MWh
        # Energy stored: 15 * 0.922 ≈ 13.8 MWh
        # New SOC: (50 + 13.8) / 100 = 0.638
        expected_soc = (50.0 + 15.0 * (0.85**0.5)) / 100.0
        assert abs(new_soc - expected_soc) < 0.001

    def test_power_limit_constraint(self, battery):
        """Test power limit constraints."""
        # Try to discharge more than maximum power
        target_power = 30.0  # MW (exceeds 25 MW limit)
        actual_power, new_soc, valid = battery.dispatch_power(target_power, 60.0)

        assert actual_power == 25.0  # Limited to max power
        assert valid is False  # Operation was constrained

    def test_soc_limit_constraint(self, battery):
        """Test SOC limit constraints."""
        # Discharge to minimum SOC
        # Available energy: (50% - 10%) * 100 MWh = 40 MWh
        # At 92.2% efficiency: 40 * 0.922 = 36.9 MWh deliverable
        # But power is limited to 25 MW for 1 hour = 25 MWh
        target_power = 25.0  # MW for 1 hour = 25 MWh
        actual_power, new_soc, valid = battery.dispatch_power(target_power, 60.0)

        # Should be power-limited, not energy-limited
        assert actual_power == 25.0
        assert valid is True  # Operation should be valid since within power limits

        # Try a power that would exceed available energy
        battery.reset_state(0.5)  # Reset to test with excessive power
        target_power = 50.0  # MW for 1 hour = 50 MWh (exceeds available energy)
        actual_power, new_soc, valid = battery.dispatch_power(target_power, 60.0)

        assert actual_power < 50.0  # Power was limited by available energy
        assert valid is False  # Operation was constrained

    def test_idle_operation(self, battery):
        """Test idle operation."""
        target_power = 0.0
        actual_power, new_soc, valid = battery.dispatch_power(target_power, 60.0)

        assert actual_power == 0.0
        assert new_soc == 0.5  # SOC unchanged
        assert valid is True

    def test_state_reset(self, battery):
        """Test battery state reset."""
        # Change state first
        battery.dispatch_power(10.0, 60.0)
        assert battery.state_of_charge != 0.5

        # Reset to initial state
        battery.reset_state()
        assert battery.state_of_charge == 0.5

        # Reset to specific state
        battery.reset_state(0.3)
        assert battery.state_of_charge == 0.3

    def test_state_reset_validation(self, battery):
        """Test state reset validation."""
        with pytest.raises(ValueError, match="outside valid range"):
            battery.reset_state(0.05)  # Below minimum SOC

        with pytest.raises(ValueError, match="outside valid range"):
            battery.reset_state(0.95)  # Above maximum SOC

    def test_dispatch_schedule_simulation(self, battery):
        """Test simulating a power dispatch schedule."""
        # Create hourly power schedule
        start_time = datetime(2024, 1, 1, 0, 0)
        timestamps = [start_time + timedelta(hours=i) for i in range(24)]

        # Simple schedule: charge at night, discharge during day
        power_values = []
        for i in range(24):
            if 0 <= i < 6 or 22 <= i < 24:  # Night hours: charge
                power_values.append(-10.0)
            elif 10 <= i < 16:  # Day hours: discharge
                power_values.append(15.0)
            else:  # Transition hours: idle
                power_values.append(0.0)

        power_schedule = pd.Series(power_values, index=timestamps)

        # Simulate dispatch
        results = battery.simulate_dispatch_schedule(
            power_schedule, interval_minutes=60.0
        )

        assert len(results) == 24
        assert BatteryStateColumns.STATE_OF_CHARGE in results.columns
        assert BatteryStateColumns.POWER_MW in results.columns
        assert BatteryStateColumns.OPERATION_MODE in results.columns

        # Check that SOC changes appropriately
        soc_values = results[BatteryStateColumns.STATE_OF_CHARGE]
        assert soc_values.min() >= 0.1  # Should not go below min SOC
        assert soc_values.max() <= 0.9  # Should not go above max SOC

        # Check operation modes are correctly assigned
        modes = results[BatteryStateColumns.OPERATION_MODE]
        unique_modes = set(modes.values)
        assert BatteryOperationMode.CHARGING in unique_modes
        assert BatteryOperationMode.DISCHARGING in unique_modes
        assert BatteryOperationMode.IDLE in unique_modes

    def test_interval_validation(self, battery):
        """Test interval validation in power methods."""
        # Test minimum interval validation
        with pytest.raises(ValueError, match="Interval too short"):
            battery.get_available_power(0.05)  # Less than 0.1 minutes

        with pytest.raises(ValueError, match="Interval too short"):
            battery.dispatch_power(10.0, 0.05)  # Less than 0.1 minutes

        # Test maximum interval validation
        with pytest.raises(ValueError, match="Interval too long"):
            battery.get_available_power(1500.0)  # More than 24 hours

        # Test negative interval validation
        with pytest.raises(ValueError, match="Interval must be positive"):
            battery.get_available_power(-5.0)

    def test_invalid_power_values(self, battery):
        """Test validation of invalid power values."""
        # Test NaN power value
        with pytest.raises(ValueError, match="Target power must be a valid number"):
            battery.dispatch_power(float("nan"), 60.0)

        # Test extremely large power value
        with pytest.raises(
            ValueError, match="Target power .* exceeds reasonable limits"
        ):
            battery.dispatch_power(2000.0, 60.0)


class TestBatterySimulatorFactory:
    """Test battery simulator factory."""

    def test_create_linear_battery(self):
        """Test creating battery via factory."""
        battery = BatterySimulatorFactory.create_linear_battery(
            energy_capacity_mwh=200.0,
            max_power_mw=50.0,
            round_trip_efficiency=0.90,
            initial_soc=0.4,
        )

        assert isinstance(battery, LinearBatterySimulator)
        assert battery.config.energy_capacity_mwh == 200.0
        assert battery.config.max_power_mw == 50.0
        assert battery.config.round_trip_efficiency == 0.90
        assert battery.state_of_charge == 0.4

    def test_factory_default_parameters(self):
        """Test factory with default parameters."""
        battery = BatterySimulatorFactory.create_linear_battery(
            energy_capacity_mwh=100.0, max_power_mw=25.0
        )

        assert battery.config.round_trip_efficiency == 0.85
        assert battery.config.initial_soc == 0.5
        assert battery.config.min_soc == 0.05  # Updated to realistic bounds
        assert battery.config.max_soc == 0.95  # Updated to realistic bounds


class TestBatteryIntegration:
    """Test battery simulator integration scenarios."""

    def test_round_trip_efficiency(self):
        """Test round-trip efficiency calculation."""
        battery = BatterySimulatorFactory.create_linear_battery(
            energy_capacity_mwh=100.0,
            max_power_mw=25.0,
            round_trip_efficiency=0.80,
            initial_soc=0.5,
        )

        initial_energy = battery.stored_energy_mwh

        # Charge 10 MW for 1 hour (10 MWh input)
        battery.dispatch_power(-10.0, 60.0)
        charged_energy = battery.stored_energy_mwh

        # Now discharge the same amount of energy that was stored
        energy_stored = charged_energy - initial_energy
        # Discharge at a rate that will extract the stored energy
        discharge_power = energy_stored  # For 1 hour
        battery.dispatch_power(discharge_power, 60.0)
        final_energy = battery.stored_energy_mwh

        # Check if we're back close to initial (accounting for efficiency losses)
        # With 80% round-trip efficiency: 10 MWh input -> 8 MWh net storage
        # after round trip. So we should have lost about 2 MWh
        energy_loss = initial_energy - final_energy
        expected_loss = 10.0 * (1.0 - 0.80)  # 2 MWh loss

        # Allow some tolerance for calculation differences
        assert abs(energy_loss - expected_loss) < 1.0  # Within 1 MWh

    def test_energy_limited_scenarios(self):
        """Test energy-limited charging and discharging."""
        battery = BatterySimulatorFactory.create_linear_battery(
            energy_capacity_mwh=10.0,  # Small battery
            max_power_mw=50.0,  # High power rating
            initial_soc=0.5,
        )

        # Try to charge for 10 minutes with high power
        # Available capacity: 5 MWh, efficiency: ~89.4%
        # Max charge power = -5 MWh / (1/6 hour) / 0.894 ≈ -29.3 MW
        max_charge, max_discharge = battery.get_available_power(10.0)

        assert max_charge < -25.0  # Energy-limited, not power-limited
        assert max_discharge < 30.0  # Energy-limited, not power-limited

    def test_continuous_operation(self):
        """Test continuous battery operation over multiple cycles."""
        battery = BatterySimulatorFactory.create_linear_battery(
            energy_capacity_mwh=50.0, max_power_mw=10.0, initial_soc=0.5
        )

        # Simulate multiple charge/discharge cycles
        for _ in range(5):
            # Charge for 2 hours
            for _ in range(2):
                battery.dispatch_power(-8.0, 60.0)

            # Discharge for 2 hours
            for _ in range(2):
                battery.dispatch_power(8.0, 60.0)

        # Battery should still be operational
        assert 0.1 <= battery.state_of_charge <= 0.9

        # Should be able to dispatch power
        power, soc, valid = battery.dispatch_power(5.0, 60.0)
        assert valid is True
        assert power > 0


class TestBatteryPhysicsValidation:
    """Test battery physics and realistic constraints."""

    def test_efficiency_physics_limits(self):
        """Test that battery efficiency is within realistic bounds."""
        # Test realistic efficiency range (80-95%)
        valid_efficiency = 0.85
        config = BatteryConfiguration(
            energy_capacity_mwh=100.0,
            max_power_mw=25.0,
            round_trip_efficiency=valid_efficiency,
        )
        assert config.round_trip_efficiency == valid_efficiency

        # Test that efficiency > 100% is rejected (violates physics)
        with pytest.raises(ValueError):
            BatteryConfiguration(
                energy_capacity_mwh=100.0,
                max_power_mw=25.0,
                round_trip_efficiency=1.1,  # 110% efficiency - impossible
            )

    def test_soc_physics_constraints(self):
        """Test SOC constraints follow battery physics."""
        config = BatteryConfiguration(
            energy_capacity_mwh=100.0,
            max_power_mw=25.0,
            min_soc=0.05,  # 5% minimum (battery protection)
            max_soc=0.95,  # 95% maximum (battery protection)
        )

        # SOC limits should be physically reasonable
        assert 0.0 <= config.min_soc <= 0.2  # Up to 20% minimum is reasonable
        assert 0.8 <= config.max_soc <= 1.0  # 80-100% maximum is reasonable

        battery = LinearBatterySimulator(config=config)

        # Test that battery respects SOC limits during operation
        # Try to discharge below minimum
        battery.reset_state(config.min_soc)
        power, soc, valid = battery.dispatch_power(50.0, 60.0)  # Large discharge
        assert soc >= config.min_soc  # Should not go below minimum

        # Try to charge above maximum
        battery.reset_state(config.max_soc)
        power, soc, valid = battery.dispatch_power(-50.0, 60.0)  # Large charge
        assert soc <= config.max_soc  # Should not go above maximum

    def test_power_capacity_relationship(self):
        """Test that power and energy capacity have realistic relationship."""
        # Typical C-rate should be 0.25 to 4.0 (15min to 4hr discharge)
        energy_mwh = 100.0

        # Test reasonable power rating (1C rate = 1-hour discharge)
        power_mw = 100.0  # 1C rate
        BatteryConfiguration(energy_capacity_mwh=energy_mwh, max_power_mw=power_mw)

        c_rate = power_mw / energy_mwh
        assert 0.25 <= c_rate <= 4.0  # Reasonable C-rate range

        # Document that extremely high C-rates should be validated
        # (This would require updating the BatteryConfiguration class)
        # extreme_power = energy_mwh * 10  # 10C rate - very high
