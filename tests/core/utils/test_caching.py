"""
Tests for the generic caching utility.
"""

from datetime import datetime
from typing import List, Tuple

import pandas as pd

from app.core.utils.date_handling import TimeInterval, find_missing_intervals


def _find_missing_date_ranges(
    start_date: str,
    end_date: str,
    existing_dates: pd.DatetimeIndex,
    interval: TimeInterval = TimeInterval.HOURLY,
) -> List[Tuple[str, str]]:
    """
    Legacy wrapper function for tests to maintain compatibility.
    Converts string dates to datetime objects and back to strings.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59
    )

    missing_intervals = find_missing_intervals(
        start_dt, end_dt, existing_dates, interval
    )

    # Convert datetime tuples back to string format for test compatibility
    gaps = []
    for start, end in missing_intervals:
        start_gap = start.strftime("%Y-%m-%d")
        end_gap = end.strftime("%Y-%m-%d")
        gaps.append((start_gap, end_gap))

    return gaps


def test_find_missing_ranges_no_gaps():
    """Tests that no gaps are found when all dates are present."""
    existing_dates = pd.to_datetime(
        pd.date_range("2025-01-01", "2025-01-10 23:00:00", freq="h")
    )
    gaps = _find_missing_date_ranges("2025-01-01", "2025-01-10", existing_dates)
    assert not gaps


def test_find_missing_ranges_complete_gap():
    """Tests that a single gap is found when no data exists."""
    existing_dates = pd.DatetimeIndex([])
    gaps = _find_missing_date_ranges("2025-01-01", "2025-01-05", existing_dates)
    assert gaps == [("2025-01-01", "2025-01-05")]


def test_find_missing_ranges_gap_at_start():
    """Tests that a gap at the beginning of the range is found."""
    existing_dates = pd.to_datetime(
        pd.date_range("2025-01-03", "2025-01-05 23:00:00", freq="h")
    )
    gaps = _find_missing_date_ranges("2025-01-01", "2025-01-05", existing_dates)
    assert gaps == [("2025-01-01", "2025-01-02")]


def test_find_missing_ranges_gap_at_end():
    """Tests that a gap at the end of the range is found."""
    existing_dates = pd.to_datetime(
        pd.date_range("2025-01-01", "2025-01-03 23:00:00", freq="h")
    )
    gaps = _find_missing_date_ranges("2025-01-01", "2025-01-05", existing_dates)
    assert gaps == [("2025-01-04", "2025-01-05")]


def test_find_missing_ranges_gap_in_middle():
    """Tests that a gap in the middle of the range is found."""
    dates1 = pd.to_datetime(
        pd.date_range("2025-01-01", "2025-01-02 23:00:00", freq="h")
    )
    dates2 = pd.to_datetime(
        pd.date_range("2025-01-05", "2025-01-06 23:00:00", freq="h")
    )
    existing_dates = dates1.union(dates2)
    gaps = _find_missing_date_ranges("2025-01-01", "2025-01-06", existing_dates)
    assert gaps == [("2025-01-03", "2025-01-04")]
