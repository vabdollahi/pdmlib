"""
Price market regions supported by the library.
"""

from enum import Enum


class PriceMarketRegion(str, Enum):
    """Top-level market regions for price providers."""

    CAISO = "CAISO"  # California ISO
    IESO = "IESO"  # Ontario IESO
