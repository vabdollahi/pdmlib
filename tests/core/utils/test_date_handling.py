"""
Tests for the enhanced date and datetime handling utilities.
"""

from datetime import date, datetime, time

import pandas as pd
import pytest

from app.core.utils.date_handling import (
    TimeInterval,
    format_for_storage_key,
    is_same_day,
    normalize_date_range,
    parse_datetime_input,
    round_datetime_range_to_interval,
    round_datetime_to_interval,
    split_datetime_range_by_days,
)


class TestParseDatetimeInput:
    """Tests for parse_datetime_input function."""

    def test_parse_date_string(self):
        """Test parsing date-only string."""
        result = parse_datetime_input("2024-01-01")
        expected = datetime(2024, 1, 1, 0, 0, 0, tzinfo=pd.Timestamp.utcnow().tz)
        assert result.replace(tzinfo=None) == expected.replace(tzinfo=None)

    def test_parse_datetime_string_with_seconds(self):
        """Test parsing full datetime string with seconds."""
        result = parse_datetime_input("2024-01-01 14:30:45")
        expected = datetime(2024, 1, 1, 14, 30, 45, tzinfo=pd.Timestamp.utcnow().tz)
        assert result.replace(tzinfo=None) == expected.replace(tzinfo=None)

    def test_parse_datetime_string_without_seconds(self):
        """Test parsing datetime string without seconds."""
        result = parse_datetime_input("2024-01-01 14:30")
        expected = datetime(2024, 1, 1, 14, 30, 0, tzinfo=pd.Timestamp.utcnow().tz)
        assert result.replace(tzinfo=None) == expected.replace(tzinfo=None)

    def test_parse_date_object(self):
        """Test parsing date object."""
        input_date = date(2024, 1, 1)
        result = parse_datetime_input(input_date)
        expected = datetime(2024, 1, 1, 0, 0, 0, tzinfo=pd.Timestamp.utcnow().tz)
        assert result.replace(tzinfo=None) == expected.replace(tzinfo=None)

    def test_parse_datetime_object(self):
        """Test parsing datetime object."""
        input_dt = datetime(2024, 1, 1, 14, 30, 45)
        result = parse_datetime_input(input_dt)
        expected = datetime(2024, 1, 1, 14, 30, 45, tzinfo=pd.Timestamp.utcnow().tz)
        assert result.replace(tzinfo=None) == expected.replace(tzinfo=None)

    def test_parse_with_custom_default_time(self):
        """Test parsing date with custom default time."""
        custom_time = time(12, 30, 0)
        result = parse_datetime_input("2024-01-01", default_time=custom_time)
        expected = datetime(2024, 1, 1, 12, 30, 0, tzinfo=pd.Timestamp.utcnow().tz)
        assert result.replace(tzinfo=None) == expected.replace(tzinfo=None)

    def test_parse_invalid_string_raises_error(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Unable to parse datetime input"):
            parse_datetime_input("invalid-date")

    def test_parse_invalid_type_raises_error(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported datetime input type"):
            parse_datetime_input(123)


class TestNormalizeDateRange:
    """Tests for normalize_date_range function."""

    def test_normalize_date_strings(self):
        """Test normalizing date-only strings."""
        start, end = normalize_date_range("2024-01-01", "2024-01-02")

        assert start.hour == 0 and start.minute == 0 and start.second == 0
        assert end.hour == 23 and end.minute == 59 and end.second == 59

    def test_normalize_datetime_strings(self):
        """Test normalizing datetime strings."""
        start, end = normalize_date_range("2024-01-01 10:30", "2024-01-01 16:45")

        assert start.hour == 10 and start.minute == 30
        assert end.hour == 16 and end.minute == 45

    def test_normalize_mixed_inputs(self):
        """Test normalizing mixed date and datetime inputs."""
        start, end = normalize_date_range("2024-01-01", "2024-01-01 12:00")

        assert start.hour == 0 and start.minute == 0
        assert end.hour == 12 and end.minute == 0

    def test_normalize_invalid_range_raises_error(self):
        """Test that invalid range (start >= end) raises ValueError."""
        with pytest.raises(
            ValueError, match="Start date/time must be before end date/time"
        ):
            normalize_date_range("2024-01-02", "2024-01-01")


class TestSplitDatetimeRangeByDays:
    """Tests for split_datetime_range_by_days function."""

    def test_split_single_day_range(self):
        """Test splitting a range within a single day."""
        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 16, 0)

        chunks = split_datetime_range_by_days(start, end)

        assert len(chunks) == 1
        assert chunks[0][0] == start
        assert chunks[0][1] == end

    def test_split_multi_day_range(self):
        """Test splitting a range across multiple days."""
        start = datetime(2024, 1, 1, 14, 30)
        end = datetime(2024, 1, 3, 10, 15)

        chunks = split_datetime_range_by_days(start, end)

        assert len(chunks) == 3

        # First day: 14:30 to 23:59:59
        assert chunks[0][0] == datetime(2024, 1, 1, 14, 30)
        assert chunks[0][1].hour == 23 and chunks[0][1].minute == 59

        # Second day: 00:00 to 23:59:59 (full day)
        assert chunks[1][0].hour == 0 and chunks[1][0].minute == 0
        assert chunks[1][1].hour == 23 and chunks[1][1].minute == 59

        # Third day: 00:00 to 10:15
        assert chunks[2][0].hour == 0 and chunks[2][0].minute == 0
        assert chunks[2][1] == datetime(2024, 1, 3, 10, 15)

    def test_split_exact_day_boundary(self):
        """Test splitting at exact day boundaries."""
        start = datetime(2024, 1, 1, 0, 0)
        end = datetime(2024, 1, 2, 0, 0)

        chunks = split_datetime_range_by_days(start, end)

        assert len(chunks) == 2
        assert chunks[0][0] == start
        assert chunks[1][1] == end


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_same_day_true(self):
        """Test is_same_day returns True for same day."""
        dt1 = datetime(2024, 1, 1, 10, 0)
        dt2 = datetime(2024, 1, 1, 20, 0)
        assert is_same_day(dt1, dt2) is True

    def test_is_same_day_false(self):
        """Test is_same_day returns False for different days."""
        dt1 = datetime(2024, 1, 1, 23, 59)
        dt2 = datetime(2024, 1, 2, 0, 1)
        assert is_same_day(dt1, dt2) is False

    def test_format_for_storage_key(self):
        """Test formatting datetime for storage keys."""
        dt = datetime(2024, 1, 1, 14, 30, 45)
        result = format_for_storage_key(dt)
        assert result == "20240101_143045"


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing date-only usage."""

    def test_existing_date_strings_work(self):
        """Test that existing date string usage continues to work."""
        # This mimics how dates are currently used in the codebase
        start_str = "2024-01-01"
        end_str = "2024-01-10"

        start_dt = parse_datetime_input(start_str)
        end_dt = parse_datetime_input(end_str, default_time=time(23, 59, 59))

        # Should produce the same behavior as current system
        assert start_dt.hour == 0 and start_dt.minute == 0
        assert end_dt.hour == 23 and end_dt.minute == 59

        # Should cover full days
        date_range = pd.date_range(start_dt, end_dt, freq="h")
        assert len(date_range) == 240  # 10 full days * 24 hours

    def test_normalize_matches_current_behavior(self):
        """Test that normalize_date_range matches current full-day behavior."""
        start, end = normalize_date_range("2024-01-01", "2024-01-10")

        # Should start at beginning of first day
        assert start == datetime(2024, 1, 1, 0, 0, 0, tzinfo=start.tzinfo)

        # Should end at end of last day
        assert end.hour == 23 and end.minute == 59 and end.second == 59


class TestRoundDatetimeToInterval:
    """Tests for round_datetime_to_interval function."""

    def test_round_to_five_minutes_down(self):
        """Test rounding down to 5-minute intervals."""
        dt = datetime(2024, 1, 1, 10, 1, 23)
        result = round_datetime_to_interval(dt, TimeInterval.FIVE_MINUTES, "down")
        expected = datetime(2024, 1, 1, 10, 0, 0)
        assert result.replace(tzinfo=None) == expected

    def test_round_to_five_minutes_up(self):
        """Test rounding up to 5-minute intervals."""
        dt = datetime(2024, 1, 1, 10, 1, 23)
        result = round_datetime_to_interval(dt, TimeInterval.FIVE_MINUTES, "up")
        expected = datetime(2024, 1, 1, 10, 5, 0)
        assert result.replace(tzinfo=None) == expected

    def test_round_to_fifteen_minutes_down(self):
        """Test rounding down to 15-minute intervals."""
        dt = datetime(2024, 1, 1, 10, 7, 30)
        result = round_datetime_to_interval(dt, TimeInterval.FIFTEEN_MINUTES, "down")
        expected = datetime(2024, 1, 1, 10, 0, 0)
        assert result.replace(tzinfo=None) == expected

    def test_round_to_fifteen_minutes_up(self):
        """Test rounding up to 15-minute intervals."""
        dt = datetime(2024, 1, 1, 10, 7, 30)
        result = round_datetime_to_interval(dt, TimeInterval.FIFTEEN_MINUTES, "up")
        expected = datetime(2024, 1, 1, 10, 15, 0)
        assert result.replace(tzinfo=None) == expected

    def test_round_to_hourly_down(self):
        """Test rounding down to hourly intervals."""
        dt = datetime(2024, 1, 1, 10, 30, 45)
        result = round_datetime_to_interval(dt, TimeInterval.HOURLY, "down")
        expected = datetime(2024, 1, 1, 10, 0, 0)
        assert result.replace(tzinfo=None) == expected

    def test_round_to_hourly_up(self):
        """Test rounding up to hourly intervals."""
        dt = datetime(2024, 1, 1, 10, 30, 45)
        result = round_datetime_to_interval(dt, TimeInterval.HOURLY, "up")
        expected = datetime(2024, 1, 1, 11, 0, 0)
        assert result.replace(tzinfo=None) == expected

    def test_round_to_daily_down(self):
        """Test rounding down to daily intervals."""
        dt = datetime(2024, 1, 1, 14, 30, 45)
        result = round_datetime_to_interval(dt, TimeInterval.DAILY, "down")
        expected = datetime(2024, 1, 1, 0, 0, 0)
        assert result.replace(tzinfo=None) == expected

    def test_round_to_daily_up(self):
        """Test rounding up to daily intervals."""
        dt = datetime(2024, 1, 1, 14, 30, 45)
        result = round_datetime_to_interval(dt, TimeInterval.DAILY, "up")
        expected = datetime(2024, 1, 2, 0, 0, 0)
        assert result.replace(tzinfo=None) == expected

    def test_round_already_at_boundary(self):
        """Test rounding when already at interval boundary."""
        dt = datetime(2024, 1, 1, 10, 0, 0)

        # Should remain the same when rounding down
        result_down = round_datetime_to_interval(dt, TimeInterval.FIVE_MINUTES, "down")
        assert result_down.replace(tzinfo=None) == dt

        # Should remain the same when rounding up
        result_up = round_datetime_to_interval(dt, TimeInterval.FIVE_MINUTES, "up")
        assert result_up.replace(tzinfo=None) == dt

    def test_round_near_day_boundary(self):
        """Test rounding near day boundaries."""
        dt = datetime(2024, 1, 1, 23, 57, 0)
        result = round_datetime_to_interval(dt, TimeInterval.FIVE_MINUTES, "up")
        expected = datetime(2024, 1, 2, 0, 0, 0)
        assert result.replace(tzinfo=None) == expected

    def test_round_preserves_timezone(self):
        """Test that timezone is preserved during rounding."""
        tz = pd.Timestamp.utcnow().tz
        dt = datetime(2024, 1, 1, 10, 1, 23, tzinfo=tz)
        result = round_datetime_to_interval(dt, TimeInterval.FIVE_MINUTES, "down")
        assert result.tzinfo == tz

    def test_round_invalid_direction(self):
        """Test that invalid direction raises ValueError."""
        dt = datetime(2024, 1, 1, 10, 1, 23)
        with pytest.raises(ValueError, match="direction must be"):
            round_datetime_to_interval(dt, TimeInterval.FIVE_MINUTES, "invalid")


class TestRoundDatetimeRangeToInterval:
    """Tests for round_datetime_range_to_interval function."""

    def test_round_range_five_minutes(self):
        """Test rounding a datetime range to 5-minute intervals."""
        start = datetime(2024, 1, 1, 10, 1, 23)
        end = datetime(2024, 1, 1, 14, 47, 15)

        rounded_start, rounded_end = round_datetime_range_to_interval(
            start, end, TimeInterval.FIVE_MINUTES
        )

        assert rounded_start.replace(tzinfo=None) == datetime(2024, 1, 1, 10, 0, 0)
        assert rounded_end.replace(tzinfo=None) == datetime(2024, 1, 1, 14, 50, 0)

    def test_round_range_hourly(self):
        """Test rounding a datetime range to hourly intervals."""
        start = datetime(2024, 1, 1, 10, 1, 23)
        end = datetime(2024, 1, 1, 14, 47, 15)

        rounded_start, rounded_end = round_datetime_range_to_interval(
            start, end, TimeInterval.HOURLY
        )

        assert rounded_start.replace(tzinfo=None) == datetime(2024, 1, 1, 10, 0, 0)
        assert rounded_end.replace(tzinfo=None) == datetime(2024, 1, 1, 15, 0, 0)

    def test_round_range_daily(self):
        """Test rounding a datetime range to daily intervals."""
        start = datetime(2024, 1, 1, 10, 1, 23)
        end = datetime(2024, 1, 1, 14, 47, 15)

        rounded_start, rounded_end = round_datetime_range_to_interval(
            start, end, TimeInterval.DAILY
        )

        assert rounded_start.replace(tzinfo=None) == datetime(2024, 1, 1, 0, 0, 0)
        assert rounded_end.replace(tzinfo=None) == datetime(2024, 1, 2, 0, 0, 0)

    def test_round_range_expands_correctly(self):
        """Test that range is expanded to ensure no data is missed."""
        # Start at 10:01 should round down to 10:00
        # End at 10:02 should round up to 10:05
        start = datetime(2024, 1, 1, 10, 1, 0)
        end = datetime(2024, 1, 1, 10, 2, 0)

        rounded_start, rounded_end = round_datetime_range_to_interval(
            start, end, TimeInterval.FIVE_MINUTES
        )

        assert rounded_start.replace(tzinfo=None) == datetime(2024, 1, 1, 10, 0, 0)
        assert rounded_end.replace(tzinfo=None) == datetime(2024, 1, 1, 10, 5, 0)


class TestNormalizeDateRangeWithInterval:
    """Tests for normalize_date_range with interval parameter."""

    def test_normalize_with_five_minute_interval(self):
        """Test normalize_date_range with 5-minute interval."""
        start, end = normalize_date_range(
            "2024-01-01 10:01:23",
            "2024-01-01 14:47:15",
            interval=TimeInterval.FIVE_MINUTES,
        )

        assert start.hour == 10 and start.minute == 0 and start.second == 0
        assert end.hour == 14 and end.minute == 50 and end.second == 0

    def test_normalize_with_hourly_interval(self):
        """Test normalize_date_range with hourly interval."""
        start, end = normalize_date_range(
            "2024-01-01 10:01:23", "2024-01-01 14:47:15", interval=TimeInterval.HOURLY
        )

        assert start.hour == 10 and start.minute == 0 and start.second == 0
        assert end.hour == 15 and end.minute == 0 and end.second == 0

    def test_normalize_with_daily_interval(self):
        """Test normalize_date_range with daily interval."""
        start, end = normalize_date_range(
            "2024-01-01 10:01:23", "2024-01-01 14:47:15", interval=TimeInterval.DAILY
        )

        assert start.hour == 0 and start.minute == 0 and start.second == 0
        assert end.hour == 0 and end.minute == 0 and end.second == 0
        assert end.date() == datetime(2024, 1, 2).date()

    def test_normalize_without_interval_unchanged(self):
        """Test that behavior is unchanged when no interval provided."""
        start_without, end_without = normalize_date_range(
            "2024-01-01 10:01:23", "2024-01-01 14:47:15"
        )

        start_with_none, end_with_none = normalize_date_range(
            "2024-01-01 10:01:23", "2024-01-01 14:47:15", interval=None
        )

        assert start_without == start_with_none
        assert end_without == end_with_none

    def test_normalize_date_strings_with_interval(self):
        """Test normalize with date-only strings and interval."""
        start, end = normalize_date_range(
            "2024-01-01", "2024-01-02", interval=TimeInterval.FIVE_MINUTES
        )

        # Should still start at beginning of day and end at start of next day
        assert start.hour == 0 and start.minute == 0 and start.second == 0
        assert start.date() == datetime(2024, 1, 1).date()

        # End should round up from 23:59:59 to next day 00:00:00
        assert end.hour == 0 and end.minute == 0 and end.second == 0
        assert end.date() == datetime(2024, 1, 3).date()


class TestIntervalRoundingExamples:
    """Tests with specific examples from the user requirements."""

    def test_user_example_five_minutes(self):
        """Test the exact example from user: 2024-01-01 10:01:23 with 5min interval."""
        start_input = "2024-01-01 10:01:23"
        interval = TimeInterval.FIVE_MINUTES

        start, _ = normalize_date_range(start_input, start_input, interval=interval)

        # Should be rounded down to 2024-01-01 10:00:00
        expected = datetime(2024, 1, 1, 10, 0, 0)
        assert start.replace(tzinfo=None) == expected

    def test_various_minute_boundaries(self):
        """Test rounding to various minute interval boundaries."""
        test_cases = [
            # (input_time, interval, expected_start_minute)
            ("2024-01-01 10:01:23", TimeInterval.FIVE_MINUTES, 0),
            ("2024-01-01 10:03:45", TimeInterval.FIVE_MINUTES, 0),
            ("2024-01-01 10:07:12", TimeInterval.FIVE_MINUTES, 5),
            ("2024-01-01 10:13:56", TimeInterval.FIVE_MINUTES, 10),
            ("2024-01-01 10:02:30", TimeInterval.FIFTEEN_MINUTES, 0),
            ("2024-01-01 10:08:45", TimeInterval.FIFTEEN_MINUTES, 0),
            ("2024-01-01 10:17:23", TimeInterval.FIFTEEN_MINUTES, 15),
            ("2024-01-01 10:31:15", TimeInterval.FIFTEEN_MINUTES, 30),
        ]

        for time_str, interval, expected_minute in test_cases:
            start, _ = normalize_date_range(time_str, time_str, interval=interval)
            assert start.minute == expected_minute, (
                f"Failed for {time_str} with {interval}: "
                f"got {start.minute}, expected {expected_minute}"
            )

    def test_end_time_rounding_up(self):
        """Test that end times are correctly rounded up."""
        test_cases = [
            # (input_time, interval, expected_end_minute)
            ("2024-01-01 10:01:23", TimeInterval.FIVE_MINUTES, 5),
            ("2024-01-01 10:03:45", TimeInterval.FIVE_MINUTES, 5),
            ("2024-01-01 10:07:12", TimeInterval.FIVE_MINUTES, 10),
            ("2024-01-01 10:02:30", TimeInterval.FIFTEEN_MINUTES, 15),
            ("2024-01-01 10:17:23", TimeInterval.FIFTEEN_MINUTES, 30),
        ]

        for time_str, interval, expected_minute in test_cases:
            _, end = normalize_date_range(time_str, time_str, interval=interval)
            assert end.minute == expected_minute, (
                f"Failed for {time_str} with {interval}: "
                f"got {end.minute}, expected {expected_minute}"
            )
