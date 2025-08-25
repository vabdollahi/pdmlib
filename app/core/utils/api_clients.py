"""
Generic HTTP API client utilities for data providers.

This module provides reusable HTTP client functionality that can be used
across different data providers (weather, price, etc.).
"""

from typing import Any, Dict, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed

from app.core.utils.logging import get_logger

logger = get_logger("api_client")


class BaseAPIClient:
    """
    Base class for HTTP API clients with retry logic and error handling.
    """

    def __init__(self, base_url: str, retry_count: int = 3, retry_wait: int = 1):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the API
            retry_count: Number of retry attempts
            retry_wait: Seconds to wait between retries
        """
        self.base_url = base_url.rstrip("/")
        self.retry_count = retry_count
        self.retry_wait = retry_wait

    def _create_retry_decorator(self):
        """Create a retry decorator with configured parameters."""
        return retry(
            stop=stop_after_attempt(self.retry_count), wait=wait_fixed(self.retry_wait)
        )

    async def get_json(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a GET request and return JSON response.

        Args:
            endpoint: API endpoint (without base URL) or full URL
            params: Query parameters
            headers: HTTP headers

        Returns:
            JSON response as dictionary

        Raises:
            aiohttp.ClientError: If the request fails
        """

        @self._create_retry_decorator()
        async def _make_request():
            # Check if endpoint is a full URL or just an endpoint
            if endpoint.startswith(("http://", "https://")):
                url = endpoint
            else:
                url = f"{self.base_url}/{endpoint.lstrip('/')}"

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        url, params=params, headers=headers
                    ) as response:
                        response.raise_for_status()
                        return await response.json()
                except aiohttp.ClientError as e:
                    logger.error(f"Error fetching data from {url}: {e}")
                    raise

        return await _make_request()
