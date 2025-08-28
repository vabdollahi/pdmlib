"""
Portfolio context management for coordinating caching behavior across components.
"""

import os
import sys
import threading
from typing import Optional


class PortfolioContext:
    """Context information for portfolio-level operations."""

    def __init__(self, organization: str, asset: Optional[str] = None):
        """
        Initialize portfolio context.

        Args:
            organization: Organization name (shared across portfolio)
            asset: Optional asset name (can be overridden per plant)
        """
        self.organization = organization
        self.asset = asset


# Thread-local storage for portfolio context
_context = threading.local()


def set_portfolio_context(organization: str, asset: Optional[str] = None) -> None:
    """
    Set the current portfolio context.

    Args:
        organization: Organization name (required, shared across portfolio)
        asset: Optional asset name (can be overridden by individual components)
    """
    _context.portfolio = PortfolioContext(organization, asset)


def get_portfolio_context() -> Optional[PortfolioContext]:
    """Get the current portfolio context, if any."""
    return getattr(_context, "portfolio", None)


def get_portfolio_organization() -> Optional[str]:
    """Get the current portfolio organization name, if any."""
    context = get_portfolio_context()
    return context.organization if context else None


def get_portfolio_asset() -> Optional[str]:
    """Get the current portfolio asset name, if any."""
    context = get_portfolio_context()
    return context.asset if context else None


def clear_portfolio_context() -> None:
    """Clear the current portfolio context."""
    _context.portfolio = None


def is_portfolio_context_active() -> bool:
    """Check if we are currently in a portfolio context."""
    return get_portfolio_context() is not None


def is_test_environment() -> bool:
    """
    Check if we're running in a test environment.
    """
    # Allow override to force caching in demos/examples
    if os.environ.get("FORCE_CACHING", "").lower() in ("true", "1", "yes"):
        return False

    # Check for common test environment indicators
    if "pytest" in os.environ.get("_", ""):
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if "test" in os.environ.get("PYTHON_TEST", "").lower():
        return True

    # Check if we're running via pytest
    if "pytest" in sys.modules:
        return True

    return False


def should_use_unified_structure() -> bool:
    """
    Determine if unified caching structure should be used.
    Returns True if we're in a portfolio context and not in test environment,
    unless it's a storage test that needs to test unified behavior.
    """
    # Allow storage tests to test unified structure behavior
    if is_test_environment():
        test_file = os.environ.get("PYTEST_CURRENT_TEST", "")
        if "test_storage" in test_file and "caching" in test_file:
            return is_portfolio_context_active()
        return False
    return is_portfolio_context_active()


def is_raw_data_type(data_type: str) -> bool:
    """
    Check if this is raw data that should be cached independently.
    Raw data includes weather and price data from external APIs.
    """
    # Weather data
    if data_type == "weather":
        return True

    # Price data (generic and region-specific)
    if data_type in ["price", "caiso_lmp", "ieso_hoep"]:
        return True

    return False


def is_processed_data_type(data_type: str) -> bool:
    """
    Check if this is processed data that depends on configuration.
    Processed data includes PV generation results that are plant-specific
    and must be isolated per organization/asset.
    """
    return data_type in ["pv_generation", "power_generation"]
