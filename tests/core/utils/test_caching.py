"""
Tests for the generic caching utility.
"""

from datetime import datetime

import pandas as pd

from app.core.utils.date_handling import find_missing_intervals


def test_find_missing_ranges_no_gaps():
    """Tests that no gaps are found when all dates are present."""
    existing_dates = pd.to_datetime(
        pd.date_range("2025-01-01", "2025-01-10 23:00:00", freq="h")
    )
    start_dt = datetime.strptime("2025-01-01", "%Y-%m-%d")
    end_dt = datetime.strptime("2025-01-10", "%Y-%m-%d").replace(
        hour=23, minute=59, second=59
    )
    gaps = find_missing_intervals(start_dt, end_dt, existing_dates)
    assert not gaps


def test_find_missing_ranges_complete_gap():
    """Tests that a single gap is found when no data exists."""
    existing_dates = pd.DatetimeIndex([])
    start_dt = datetime.strptime("2025-01-01", "%Y-%m-%d")
    end_dt = datetime.strptime("2025-01-05", "%Y-%m-%d").replace(
        hour=23, minute=59, second=59
    )
    gaps = find_missing_intervals(start_dt, end_dt, existing_dates)
    assert len(gaps) == 1
    gap_start, gap_end = gaps[0]
    assert gap_start.strftime("%Y-%m-%d") == "2025-01-01"
    assert gap_end.strftime("%Y-%m-%d") == "2025-01-05"


def test_find_missing_ranges_gap_at_start():
    """Tests that a gap at the beginning of the range is found."""
    existing_dates = pd.to_datetime(
        pd.date_range("2025-01-03", "2025-01-05 23:00:00", freq="h")
    )
    start_dt = datetime.strptime("2025-01-01", "%Y-%m-%d")
    end_dt = datetime.strptime("2025-01-05", "%Y-%m-%d").replace(
        hour=23, minute=59, second=59
    )
    gaps = find_missing_intervals(start_dt, end_dt, existing_dates)
    assert len(gaps) == 1
    gap_start, gap_end = gaps[0]
    assert gap_start.strftime("%Y-%m-%d") == "2025-01-01"
    assert gap_end.strftime("%Y-%m-%d") == "2025-01-02"


def test_find_missing_ranges_gap_at_end():
    """Tests that a gap at the end of the range is found."""
    existing_dates = pd.to_datetime(
        pd.date_range("2025-01-01", "2025-01-03 23:00:00", freq="h")
    )
    start_dt = datetime.strptime("2025-01-01", "%Y-%m-%d")
    end_dt = datetime.strptime("2025-01-05", "%Y-%m-%d").replace(
        hour=23, minute=59, second=59
    )
    gaps = find_missing_intervals(start_dt, end_dt, existing_dates)
    assert len(gaps) == 1
    gap_start, gap_end = gaps[0]
    assert gap_start.strftime("%Y-%m-%d") == "2025-01-04"
    assert gap_end.strftime("%Y-%m-%d") == "2025-01-05"


def test_find_missing_ranges_gap_in_middle():
    """Tests that a gap in the middle of the range is found."""
    existing_dates = pd.to_datetime(
        list(pd.date_range("2025-01-01", "2025-01-02 23:00:00", freq="h"))
        + list(pd.date_range("2025-01-05", "2025-01-06 23:00:00", freq="h"))
    )
    start_dt = datetime.strptime("2025-01-01", "%Y-%m-%d")
    end_dt = datetime.strptime("2025-01-06", "%Y-%m-%d").replace(
        hour=23, minute=59, second=59
    )
    gaps = find_missing_intervals(start_dt, end_dt, existing_dates)
    assert len(gaps) == 1
    gap_start, gap_end = gaps[0]
    assert gap_start.strftime("%Y-%m-%d") == "2025-01-03"
    assert gap_end.strftime("%Y-%m-%d") == "2025-01-04"
