"""
Pytest configuration file for the test suite.

This file defines shared fixtures and hooks for the test suite.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"
