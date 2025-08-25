"""
Central logging configuration for the PDM library.

This module provides a centralized logging setup with appropriate levels
for different components of the system.
"""

import logging
import os
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    # ANSI color codes
    COLORS = {
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record):
        """Format the log record with colors for warnings and errors."""
        # Get the original formatted message
        formatted = super().format(record)

        # Add color for specific levels
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS["RESET"]
            return f"{color}{formatted}{reset}"

        return formatted


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Configure centralized logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses LOG_LEVEL environment variable or defaults to INFO.
        format_string: Custom format string for log messages.
        include_timestamp: Whether to include timestamp in log messages.
        use_colors: Whether to use colored output for warnings and errors.

    Returns:
        Configured logger instance.
    """
    # Determine log level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Create format string
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Use colored formatter if requested
    if use_colors:
        formatter = ColoredFormatter(format_string)
    else:
        formatter = logging.Formatter(format_string)

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Get application logger
    logger = logging.getLogger("pdmlib")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: The name of the module/component requesting the logger.

    Returns:
        Logger instance with the appropriate hierarchy.
    """
    # Ensure base logger is configured
    if not logging.getLogger("pdmlib").handlers:
        setup_logging()

    return logging.getLogger(f"pdmlib.{name}")
