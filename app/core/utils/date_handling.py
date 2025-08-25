"""
Enhanced date and datetime handling utilities for the PDM library.

This module provides flexible parsing of date and datetime inputs,
supporting both date-only and full datetime specifications, and
configurable time intervals for data fetching.
"""

from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional, Tuple, Union

import pandas as pd

DateTimeInput = Union[str, date, datetime]


class TimeInterval(str, Enum):
    """
    Supported time intervals for data fetching from any data source.

    These intervals determine the frequency of data points
    when fetching data from APIs, databases, or other sources.
    """

    FIVE_MINUTES = "5min"
    FIFTEEN_MINUTES = "15min"
    HOURLY = "1h"
    DAILY = "1d"

    @property
    def pandas_frequency(self) -> str:
        """
        Get the pandas frequency string for this interval.

        Returns:
            Pandas frequency string (e.g., '5T', '15T', 'h', 'D')
        """
        frequency_map = {
            self.FIVE_MINUTES: "5min",  # 5 minutes
            self.FIFTEEN_MINUTES: "15min",  # 15 minutes
            self.HOURLY: "h",  # 1 hour
            self.DAILY: "D",  # 1 day
        }
        return frequency_map[self]

    @property
    def minutes(self) -> int:
        """
        Get the interval duration in minutes.

        Returns:
            Number of minutes for this interval
        """
        minutes_map = {
            self.FIVE_MINUTES: 5,
            self.FIFTEEN_MINUTES: 15,
            self.HOURLY: 60,
            self.DAILY: 1440,  # 24 hours * 60 minutes
        }
        return minutes_map[self]

    @property
    def display_name(self) -> str:
        """
        Get a user-friendly display name for this interval.

        Returns:
            Human-readable interval name
        """
        display_map = {
            self.FIVE_MINUTES: "5 minutes",
            self.FIFTEEN_MINUTES: "15 minutes",
            self.HOURLY: "1 hour",
            self.DAILY: "1 day",
        }
        return display_map[self]

    @classmethod
    def default(cls) -> "TimeInterval":
        """Get the default time interval (hourly)."""
        return cls.HOURLY


def parse_datetime_input(
    dt_input: DateTimeInput, default_time: Optional[time] = None
) -> datetime:
    """
    Parse various datetime input formats into a datetime object.

    Args:
        dt_input: Input in various formats:
            - "YYYY-MM-DD" (date string)
            - "YYYY-MM-DD HH:MM:SS" (datetime string)
            - "YYYY-MM-DD HH:MM" (datetime string without seconds)
            - date object
            - datetime object
        default_time: Default time to use for date-only inputs.
                     If None, uses 00:00:00 for start times and 23:59:59 for end times.

    Returns:
        A datetime object with timezone set to UTC.

    Examples:
        >>> parse_datetime_input("2025-01-01")
        datetime(2025, 1, 1, 0, 0, tzinfo=UTC)

        >>> parse_datetime_input("2025-01-01 14:30:00")
        datetime(2025, 1, 1, 14, 30, tzinfo=UTC)

        >>> parse_datetime_input("2025-01-01 14:30")
        datetime(2025, 1, 1, 14, 30, tzinfo=UTC)
    """
    if isinstance(dt_input, datetime):
        # Already a datetime, ensure it's UTC
        if dt_input.tzinfo is None:
            return dt_input.replace(tzinfo=pd.Timestamp.utcnow().tz)
        return dt_input.astimezone(pd.Timestamp.utcnow().tz)

    elif isinstance(dt_input, date):
        # Convert date to datetime with default time
        if default_time is None:
            default_time = time(0, 0, 0)  # midnight
        dt = datetime.combine(dt_input, default_time)
        return dt.replace(tzinfo=pd.Timestamp.utcnow().tz)

    elif isinstance(dt_input, str):
        # Parse string input
        dt_input = dt_input.strip()

        # Try different datetime formats
        formats = [
            "%Y-%m-%d %H:%M:%S",  # Full datetime with seconds
            "%Y-%m-%d %H:%M",  # Datetime without seconds
            "%Y-%m-%d",  # Date only
        ]

        for fmt in formats:
            try:
                parsed_dt = datetime.strptime(dt_input, fmt)

                # If it's date-only format and we have a default time, use it
                if fmt == "%Y-%m-%d" and default_time is not None:
                    parsed_dt = parsed_dt.replace(
                        hour=default_time.hour,
                        minute=default_time.minute,
                        second=default_time.second,
                    )

                return parsed_dt.replace(tzinfo=pd.Timestamp.utcnow().tz)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse datetime input: {dt_input}")

    else:
        raise TypeError(f"Unsupported datetime input type: {type(dt_input)}")


def normalize_date_range(
    start_input: DateTimeInput,
    end_input: DateTimeInput,
    interval: Optional[TimeInterval] = None,
) -> Tuple[datetime, datetime]:
    """
    Normalize start and end date inputs into datetime objects.

    For date-only inputs:
    - Start date defaults to 00:00:00
    - End date defaults to 23:59:59

    If an interval is provided, the datetime range will be rounded to
    interval boundaries:
    - Start time is rounded down (floor) to ensure no data is missed
    - End time is rounded up (ceil) to ensure complete coverage

    Args:
        start_input: Start date/time in various formats
        end_input: End date/time in various formats
        interval: Optional time interval to round the range to

    Returns:
        Tuple of (start_datetime, end_datetime) in UTC

    Examples:
        >>> start, end = normalize_date_range("2025-01-01", "2025-01-02")
        >>> print(start)  # 2025-01-01 00:00:00+00:00
        >>> print(end)    # 2025-01-02 23:59:59+00:00

        >>> start, end = normalize_date_range("2025-01-01 14:30", "2025-01-01 16:45")
        >>> print(start)  # 2025-01-01 14:30:00+00:00
        >>> print(end)    # 2025-01-01 16:45:00+00:00

        >>> # With interval rounding
        >>> start, end = normalize_date_range(
        ...     "2025-01-01 10:01:23", "2025-01-01 14:47:15",
        ...     TimeInterval.FIVE_MINUTES
        ... )
        >>> print(start)  # 2025-01-01 10:00:00+00:00
        >>> print(end)    # 2025-01-01 14:50:00+00:00
    """
    # For start times, default to beginning of day
    start_dt = parse_datetime_input(start_input, default_time=time(0, 0, 0))

    # For end times, check if it's a date-only input or if start==end datetime
    is_end_date_only = isinstance(end_input, str) and len(end_input.strip()) == 10
    is_same_datetime = start_input == end_input and not is_end_date_only

    if is_end_date_only:
        # Date-only format, default to end of day
        end_dt = parse_datetime_input(end_input, default_time=time(23, 59, 59))
    elif is_same_datetime:
        # Same datetime input for start and end - parse as-is, handled by rounding
        end_dt = parse_datetime_input(end_input, default_time=time(0, 0, 0))
    else:
        # Different datetime inputs - parse as-is
        end_dt = parse_datetime_input(end_input, default_time=time(23, 59, 59))

    # Apply interval rounding if specified - ensures proper start < end relationship
    if interval is not None:
        start_dt, end_dt = round_datetime_range_to_interval(start_dt, end_dt, interval)

    # Final validation - only check if no interval rounding was applied
    if interval is None and start_dt >= end_dt:
        raise ValueError("Start date/time must be before end date/time")

    return start_dt, end_dt


def parse_date_string(date_str: str) -> date:
    """
    Parse a date string that can be in either YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.

    Args:
        date_str: Date string to parse

    Returns:
        date object

    Examples:
        >>> parse_date_string("2025-01-01")
        datetime.date(2025, 1, 1)

        >>> parse_date_string("2025-01-01 10:30:00")
        datetime.date(2025, 1, 1)
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").date()
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%d").date()


def datetime_to_api_string(dt: datetime) -> str:
    """
    Convert a datetime object to the API string format.

    Args:
        dt: Datetime object

    Returns:
        String in YYYY-MM-DD format for date-only API calls,
        or YYYY-MM-DD HH:MM:SS format for datetime API calls
    """
    # For now, Open-Meteo API only accepts date format
    # But this function allows future extension to datetime APIs
    return dt.strftime("%Y-%m-%d")


def format_for_storage_key(dt: datetime) -> str:
    """
    Format datetime for use in storage keys/paths.

    Args:
        dt: Datetime object

    Returns:
        String safe for use in file paths and storage keys
    """
    return dt.strftime("%Y%m%d_%H%M%S")


def is_same_day(dt1: datetime, dt2: datetime) -> bool:
    """
    Check if two datetime objects are on the same day.

    Args:
        dt1: First datetime
        dt2: Second datetime

    Returns:
        True if both datetimes are on the same calendar day
    """
    return dt1.date() == dt2.date()


def split_datetime_range_by_days(
    start_dt: datetime, end_dt: datetime
) -> list[Tuple[datetime, datetime]]:
    """
    Split a datetime range into daily chunks.

    This is useful for APIs that work on daily boundaries
    but we want to support sub-daily ranges.

    Args:
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        List of (start, end) datetime tuples for each day

    Examples:
        >>> chunks = split_datetime_range_by_days(
        ...     datetime(2025, 1, 1, 14, 0),
        ...     datetime(2025, 1, 3, 10, 0)
        ... )
        >>> len(chunks)  # 3 days
        3
    """
    chunks = []
    current_start = start_dt

    while current_start < end_dt:
        # End of current day or the actual end time, whichever is earlier
        day_end = datetime.combine(current_start.date(), time(23, 59, 59)).replace(
            tzinfo=current_start.tzinfo
        )

        current_end = min(day_end, end_dt)
        chunks.append((current_start, current_end))

        # Start of next day using timedelta
        next_day_start = datetime.combine(current_start.date(), time(0, 0, 0)).replace(
            tzinfo=current_start.tzinfo
        ) + timedelta(days=1)

        # If end_dt is exactly at the start of a new day, create a chunk for that
        if (
            end_dt.time() == time(0, 0, 0)
            and end_dt.date() == next_day_start.date()
            and current_end == day_end
        ):
            chunks.append((next_day_start, end_dt))
            break

        current_start = next_day_start

    return chunks


def create_date_range_with_interval(
    start_dt: datetime, end_dt: datetime, interval: TimeInterval = TimeInterval.HOURLY
) -> pd.DatetimeIndex:
    """
    Create a pandas DatetimeIndex with the specified interval.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        interval: Time interval for the range

    Returns:
        DatetimeIndex with timestamps at the specified interval

    Examples:
        >>> start = datetime(2025, 1, 1, 0, 0)
        >>> end = datetime(2025, 1, 1, 2, 0)
        >>> range_5min = create_date_range_with_interval(
        ...     start, end, TimeInterval.FIVE_MINUTES
        ... )
        >>> len(range_5min)  # 2 hours * 12 intervals per hour + 1
        25

        >>> range_hourly = create_date_range_with_interval(
        ...     start, end, TimeInterval.HOURLY
        ... )
        >>> len(range_hourly)  # 2 hours + 1
        3
    """
    # Convert to UTC if the inputs have timezone info, otherwise use UTC
    if start_dt.tzinfo is not None:
        start_utc = start_dt.astimezone(pd.Timestamp.utcnow().tz)
        end_utc = end_dt.astimezone(pd.Timestamp.utcnow().tz)
    else:
        start_utc = start_dt.replace(tzinfo=pd.Timestamp.utcnow().tz)
        end_utc = end_dt.replace(tzinfo=pd.Timestamp.utcnow().tz)

    return pd.date_range(start=start_utc, end=end_utc, freq=interval.pandas_frequency)


def find_missing_intervals(
    start_dt: datetime,
    end_dt: datetime,
    existing_dates: pd.DatetimeIndex,
    interval: TimeInterval = TimeInterval.HOURLY,
) -> list[Tuple[datetime, datetime]]:
    """
    Find missing time intervals in existing data.

    This is an enhanced version of _find_missing_date_ranges that works
    with configurable time intervals instead of just hourly data.

    Args:
        start_dt: Required start datetime
        end_dt: Required end datetime
        existing_dates: DatetimeIndex of dates that are already available
        interval: Time interval to check for gaps

    Returns:
        List of tuples containing start and end datetime of missing periods

    Examples:
        >>> start = datetime(2025, 1, 1, 0, 0)
        >>> end = datetime(2025, 1, 1, 1, 0)
        >>> existing = pd.DatetimeIndex([datetime(2025, 1, 1, 0, 30)])
        >>> gaps = find_missing_intervals(
        ...     start, end, existing, TimeInterval.FIFTEEN_MINUTES
        ... )
        >>> len(gaps)  # Should find gaps at 0:00, 0:15, 0:45, 1:00
        4
    """
    # Create the required range with the specified interval
    required_range = create_date_range_with_interval(start_dt, end_dt, interval)

    # Ensure existing_dates is timezone-aware (UTC) to match required_range
    if existing_dates.tz is None:
        existing_dates = existing_dates.tz_localize("UTC")
    else:
        existing_dates = existing_dates.tz_convert("UTC")

    # Find missing timestamps
    missing_dates = required_range.difference(existing_dates)

    if missing_dates.empty:
        return []

    # Find contiguous blocks of missing dates
    # A new block starts when the time difference is greater than the interval
    missing_series = missing_dates.to_series()
    interval_seconds = interval.minutes * 60
    time_diff_threshold = interval_seconds + 1  # Allow 1 second tolerance

    contiguous_blocks = (
        missing_series.diff().dt.total_seconds().gt(time_diff_threshold)
    ).cumsum()
    groups = missing_series.groupby(contiguous_blocks)

    gaps = []
    for _, group in groups:
        start_gap = group.index.min()
        end_gap = group.index.max()
        gaps.append((start_gap.to_pydatetime(), end_gap.to_pydatetime()))

    return gaps


def estimate_data_points(
    start_dt: datetime, end_dt: datetime, interval: TimeInterval = TimeInterval.HOURLY
) -> int:
    """
    Estimate the number of data points for a given time range and interval.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        interval: Time interval

    Returns:
        Estimated number of data points

    Examples:
        >>> start = datetime(2025, 1, 1, 0, 0)
        >>> end = datetime(2025, 1, 2, 0, 0)  # 24 hours
        >>> estimate_data_points(start, end, TimeInterval.HOURLY)
        25  # 24 hours + 1 for inclusive end

        >>> estimate_data_points(start, end, TimeInterval.FIVE_MINUTES)
        289  # 24 hours * 12 intervals per hour + 1
    """
    time_diff = end_dt - start_dt
    total_minutes = time_diff.total_seconds() / 60
    intervals = int(total_minutes / interval.minutes) + 1  # +1 for inclusive end
    return intervals


def validate_interval_for_range(
    start_dt: datetime,
    end_dt: datetime,
    interval: TimeInterval,
    max_points: int = 10000,
) -> bool:
    """
    Validate that a time interval is reasonable for a given date range.

    This helps prevent accidentally requesting too much data (e.g., 5-minute
    intervals for a year of data would be ~105,000 data points).

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        interval: Time interval
        max_points: Maximum allowed data points

    Returns:
        True if the combination is reasonable, False otherwise

    Raises:
        ValueError: If the combination would result in too many data points
    """
    estimated_points = estimate_data_points(start_dt, end_dt, interval)

    if estimated_points > max_points:
        time_span = end_dt - start_dt
        raise ValueError(
            f"Requested interval '{interval.display_name}' for time span "
            f"of {time_span.days} days would result in {estimated_points:,} "
            f"data points, which exceeds the maximum of {max_points:,}. "
            f"Consider using a larger interval or shorter time range."
        )

    return True


def round_datetime_to_interval(
    dt: datetime, interval: TimeInterval, direction: str = "down"
) -> datetime:
    """
    Round a datetime to the nearest time interval boundary.

    Args:
        dt: Datetime to round
        interval: Time interval to round to
        direction: 'down' (floor) or 'up' (ceil) or 'nearest'

    Returns:
        Rounded datetime

    Examples:
        >>> dt = datetime(2025, 1, 1, 10, 1, 23)
        >>> round_datetime_to_interval(dt, TimeInterval.FIVE_MINUTES, "down")
        datetime(2025, 1, 1, 10, 0, 0)

        >>> round_datetime_to_interval(dt, TimeInterval.HOURLY, "down")
        datetime(2025, 1, 1, 10, 0, 0)

        >>> round_datetime_to_interval(dt, TimeInterval.DAILY, "down")
        datetime(2025, 1, 1, 0, 0, 0)
    """
    if direction not in ["down", "up", "nearest"]:
        raise ValueError("direction must be 'down', 'up', or 'nearest'")

    # Store original timezone
    original_tz = dt.tzinfo

    if interval == TimeInterval.DAILY:
        # Round to start/end of day
        if direction == "down":
            rounded = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif direction == "up":
            if dt.time() == time(0, 0, 0):
                # Already at start of day
                rounded = dt
            else:
                # Move to start of next day
                rounded = (dt + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
        else:  # nearest
            start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            if (dt - start_of_day) < (end_of_day - dt):
                rounded = start_of_day
            else:
                rounded = end_of_day

    elif interval == TimeInterval.HOURLY:
        # Round to start/end of hour
        if direction == "down":
            rounded = dt.replace(minute=0, second=0, microsecond=0)
        elif direction == "up":
            if dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
                # Already at start of hour
                rounded = dt
            else:
                # Move to start of next hour
                rounded = (dt + timedelta(hours=1)).replace(
                    minute=0, second=0, microsecond=0
                )
        else:  # nearest
            start_of_hour = dt.replace(minute=0, second=0, microsecond=0)
            end_of_hour = start_of_hour + timedelta(hours=1)
            if (dt - start_of_hour) < (end_of_hour - dt):
                rounded = start_of_hour
            else:
                rounded = end_of_hour

    else:
        # For minute-based intervals (5min, 15min)
        interval_minutes = interval.minutes

        # Round minutes to the interval boundary
        total_minutes = dt.hour * 60 + dt.minute

        if direction == "down":
            rounded_minutes = (total_minutes // interval_minutes) * interval_minutes
        elif direction == "up":
            is_at_boundary = (
                dt.second == 0
                and dt.microsecond == 0
                and dt.minute % interval_minutes == 0
            )
            if is_at_boundary:
                # Already at interval boundary
                rounded_minutes = total_minutes
            else:
                # Move to next interval boundary
                rounded_minutes = (
                    (total_minutes // interval_minutes) + 1
                ) * interval_minutes
        else:  # nearest
            lower_boundary = (total_minutes // interval_minutes) * interval_minutes
            upper_boundary = lower_boundary + interval_minutes

            if (total_minutes - lower_boundary) < (upper_boundary - total_minutes):
                rounded_minutes = lower_boundary
            else:
                rounded_minutes = upper_boundary

        # Convert back to hours and minutes
        rounded_hours = rounded_minutes // 60
        rounded_mins = rounded_minutes % 60

        # Handle day overflow
        if rounded_hours >= 24:
            rounded = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                days=1
            )
        else:
            rounded = dt.replace(
                hour=rounded_hours, minute=rounded_mins, second=0, microsecond=0
            )

    # Preserve timezone
    if original_tz is not None:
        rounded = rounded.replace(tzinfo=original_tz)

    return rounded


def round_datetime_range_to_interval(
    start_dt: datetime, end_dt: datetime, interval: TimeInterval
) -> Tuple[datetime, datetime]:
    """
    Round both start and end datetimes to time interval boundaries.

    Start time is rounded down (floor) to ensure it doesn't miss any data.
    End time is rounded up (ceil) to ensure complete coverage.

    If start and end times are the same or very close, end time will be
    rounded to at least one interval ahead of start time.

    Args:
        start_dt: Start datetime to round
        end_dt: End datetime to round
        interval: Time interval to round to

    Returns:
        Tuple of (rounded_start, rounded_end)

    Examples:
        >>> start = datetime(2025, 1, 1, 10, 1, 23)
        >>> end = datetime(2025, 1, 1, 14, 47, 15)
        >>> round_datetime_range_to_interval(start, end, TimeInterval.FIVE_MINUTES)
        (datetime(2025, 1, 1, 10, 0, 0), datetime(2025, 1, 1, 14, 50, 0))

        >>> round_datetime_range_to_interval(start, end, TimeInterval.HOURLY)
        (datetime(2025, 1, 1, 10, 0, 0), datetime(2025, 1, 1, 15, 0, 0))
    """
    # Round start down to ensure we don't miss any data
    rounded_start = round_datetime_to_interval(start_dt, interval, "down")

    # Round end up to ensure complete coverage
    rounded_end = round_datetime_to_interval(end_dt, interval, "up")

    # Special case: if start and end round to the same time, advance end by one interval
    if rounded_start >= rounded_end:
        # Add one interval to the rounded end time
        if interval == TimeInterval.DAILY:
            rounded_end = rounded_end + timedelta(days=1)
        elif interval == TimeInterval.HOURLY:
            rounded_end = rounded_end + timedelta(hours=1)
        else:
            # For minute-based intervals
            rounded_end = rounded_end + timedelta(minutes=interval.minutes)

    return rounded_start, rounded_end


def get_optimal_interval_for_range(
    start_dt: datetime, end_dt: datetime, max_points: int = 10000
) -> TimeInterval:
    """
    Suggest an optimal time interval for a given date range.

    This function suggests the finest granularity interval that won't
    exceed the maximum number of data points.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        max_points: Maximum allowed data points

    Returns:
        Recommended TimeInterval

    Examples:
        >>> start = datetime(2025, 1, 1)
        >>> end = datetime(2025, 1, 2)  # 1 day
        >>> get_optimal_interval_for_range(start, end)
        TimeInterval.FIVE_MINUTES  # Fine granularity for short range

        >>> end = datetime(2025, 12, 31)  # 1 year
        >>> get_optimal_interval_for_range(start, end)
        TimeInterval.DAILY  # Coarse granularity for long range
    """
    # Try intervals from finest to coarsest
    intervals_by_granularity = [
        TimeInterval.FIVE_MINUTES,
        TimeInterval.FIFTEEN_MINUTES,
        TimeInterval.HOURLY,
        TimeInterval.DAILY,
    ]

    for interval in intervals_by_granularity:
        estimated_points = estimate_data_points(start_dt, end_dt, interval)
        if estimated_points <= max_points:
            return interval

    # If even daily intervals exceed the limit, still return daily
    # The validation function will catch this later
    return TimeInterval.DAILY
