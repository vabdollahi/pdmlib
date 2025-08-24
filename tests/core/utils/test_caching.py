"""
Tests for the generic caching utility.
"""

import pandas as pd

from app.core.utils.caching import _find_missing_date_ranges


def test_find_missing_ranges_no_gaps():
    """Tests that no gaps are found when all dates are present."""
    existing_dates = pd.to_datetime(
        pd.date_range("2024-01-01", "2024-01-10 23:00:00", freq="h")
    )
    gaps = _find_missing_date_ranges("2024-01-01", "2024-01-10", existing_dates)
    assert not gaps


def test_find_missing_ranges_complete_gap():
    """Tests that a single gap is found when no data exists."""
    existing_dates = pd.DatetimeIndex([])
    gaps = _find_missing_date_ranges("2024-01-01", "2024-01-05", existing_dates)
    assert gaps == [("2024-01-01", "2024-01-05")]


def test_find_missing_ranges_gap_at_start():
    """Tests that a gap at the beginning of the range is found."""
    existing_dates = pd.to_datetime(
        pd.date_range("2024-01-03", "2024-01-05 23:00:00", freq="h")
    )
    gaps = _find_missing_date_ranges("2024-01-01", "2024-01-05", existing_dates)
    assert gaps == [("2024-01-01", "2024-01-02")]


def test_find_missing_ranges_gap_at_end():
    """Tests that a gap at the end of the range is found."""
    existing_dates = pd.to_datetime(
        pd.date_range("2024-01-01", "2024-01-03 23:00:00", freq="h")
    )
    gaps = _find_missing_date_ranges("2024-01-01", "2024-01-05", existing_dates)
    assert gaps == [("2024-01-04", "2024-01-05")]


def test_find_missing_ranges_gap_in_middle():
    """Tests that a gap in the middle of the range is found."""
    dates1 = pd.to_datetime(
        pd.date_range("2024-01-01", "2024-01-02 23:00:00", freq="h")
    )
    dates2 = pd.to_datetime(
        pd.date_range("2024-01-05", "2024-01-06 23:00:00", freq="h")
    )
    existing_dates = dates1.union(dates2)
    gaps = _find_missing_date_ranges("2024-01-01", "2024-01-06", existing_dates)
    assert gaps == [("2024-01-03", "2024-01-04")]
