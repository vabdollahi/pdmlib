"""
Central logging configuration for the PDM library.

This module provides a centralized logging setup with appropriate levels
for different components of the system.
"""

import logging
import os
import sys
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Configure centralized logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses LOG_LEVEL environment variable or defaults to INFO.
        format_string: Custom format string for log messages.
        include_timestamp: Whether to include timestamp in log messages.

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

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level),
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

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
