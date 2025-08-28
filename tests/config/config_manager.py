"""
Simplified test configuration utilities.

Provides minimal utilities for test configuration loading and data access.
All object creation should use the spec-driven EnvironmentConfigFactory approach.
"""

import json
from pathlib import Path
from typing import Dict


class TestConfigManager:
    """Simplified configuration manager for tests - utilities only."""

    def __init__(self):
        """Initialize the config manager."""
        self.config_dir = Path(__file__).parent
        self.data_dir = Path(__file__).parent.parent / "data"

    def load_config_file(self, filename: str) -> Dict:
        """Load a JSON configuration file."""
        config_path = self.config_dir / filename
        with open(config_path, "r") as f:
            return json.load(f)

    @property
    def test_weather_csv_path(self) -> str:
        """Get the path to test weather CSV file."""
        return str(self.data_dir / "sample_weather_data.csv")

    @property
    def test_price_csv_path(self) -> str:
        """Get the path to test price CSV file."""
        return str(self.data_dir / "sample_price_data.csv")

    @property
    def environment_spec_path(self) -> Path:
        """Get the path to the environment spec JSON file."""
        return self.config_dir / "test_config_multi.json"


# Global instance for easy access
test_config = TestConfigManager()
