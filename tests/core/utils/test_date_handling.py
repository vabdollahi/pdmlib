"""
Essential tests for date and datetime handling utilities.
"""

from datetime import datetime, timezone

import pytest

from app.core.utils.date_handling import (
    TimeInterval,
    format_for_storage_key,
    is_same_day,
    normalize_date_range,
    parse_datetime_input,
    round_datetime_to_interval,
    split_datetime_range_by_days,
)


class TestDateTimeParsing:
    """Essential datetime parsing tests."""

    def test_parse_date_string(self):
        """Test parsing date-only string."""
        result = parse_datetime_input("2025-01-01")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 1

    def test_parse_datetime_string(self):
        """Test parsing datetime string."""
        result = parse_datetime_input("2025-01-01 12:30:00")
        assert result.year == 2025
        assert result.hour == 12
        assert result.minute == 30

    def test_parse_datetime_object(self):
        """Test parsing datetime object."""
        input_dt = datetime(2025, 1, 1, 12, 0, 0)
        result = parse_datetime_input(input_dt)
        # Function returns UTC timezone-aware datetime
        expected = input_dt.replace(tzinfo=timezone.utc)
        assert result == expected

    def test_parse_invalid_input_raises_error(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError):
            parse_datetime_input("invalid-date")


class TestDateTimeUtilities:
    """Essential datetime utility tests."""

    def test_normalize_date_range(self):
        """Test normalizing date range."""
        start, end = normalize_date_range("2025-01-01", "2025-01-02")
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start < end

    def test_split_single_day_range(self):
        """Test splitting single day range."""
        start = datetime(2025, 1, 1, 10, 0, 0)
        end = datetime(2025, 1, 1, 20, 0, 0)
        result = split_datetime_range_by_days(start, end)
        assert len(result) == 1
        assert result[0] == (start, end)

    def test_split_multi_day_range(self):
        """Test splitting multi-day range."""
        start = datetime(2025, 1, 1, 10, 0, 0)
        end = datetime(2025, 1, 3, 20, 0, 0)
        result = split_datetime_range_by_days(start, end)
        assert len(result) == 3

    def test_is_same_day(self):
        """Test same day comparison."""
        dt1 = datetime(2025, 1, 1, 10, 0, 0)
        dt2 = datetime(2025, 1, 1, 20, 0, 0)
        dt3 = datetime(2025, 1, 2, 10, 0, 0)

        assert is_same_day(dt1, dt2)
        assert not is_same_day(dt1, dt3)

    def test_format_for_storage_key(self):
        """Test storage key formatting."""
        dt = datetime(2025, 1, 1, 12, 30, 0)
        key = format_for_storage_key(dt)
        assert isinstance(key, str)
        assert "2025" in key


class TestTimeIntervalRounding:
    """Essential time interval rounding tests."""

    def test_round_to_hour(self):
        """Test rounding to hour."""
        dt = datetime(2025, 1, 1, 12, 45, 30)
        rounded = round_datetime_to_interval(dt, TimeInterval.HOURLY)
        assert rounded.minute == 0
        assert rounded.second == 0

    def test_round_to_day(self):
        """Test rounding to day."""
        dt = datetime(2025, 1, 1, 12, 45, 30)
        rounded = round_datetime_to_interval(dt, TimeInterval.DAILY)
        assert rounded.hour == 0
        assert rounded.minute == 0
        assert rounded.second == 0
