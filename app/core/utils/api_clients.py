"""
Generic HTTP API client utilities for data providers.

This module provides reusable HTTP client functionality that can be used
across different data providers (weather, price, etc.).
"""

import asyncio
import time
from typing import Any, Dict, Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

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

    async def get_text(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Make a GET request and return text response.

        Args:
            endpoint: API endpoint (without base URL) or full URL
            params: Query parameters
            headers: HTTP headers

        Returns:
            Text response as string

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
                        return await response.text()
                except aiohttp.ClientError as e:
                    logger.error(f"Error fetching data from {url}: {e}")
                    raise

        return await _make_request()

    async def get_bytes(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """
        Make a GET request and return raw bytes.

        Args:
            endpoint: API endpoint (without base URL) or full URL
            params: Query parameters
            headers: HTTP headers

        Returns:
            Raw response bytes

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
                        return await response.read()
                except aiohttp.ClientError as e:
                    logger.error(f"Error fetching data from {url}: {e}")
                    raise

        return await _make_request()


class CAISORateLimitedClient(BaseAPIClient):
    """
    API client with rate limiting specifically designed for CAISO OASIS API.

    Implements CAISO's official rate limits:
    - Global limit: 30 requests per second (1800 per minute)
    - Service-specific enforcement periods of 5-30 seconds
    - Enhanced retry logic with exponential backoff for 429 errors
    """

    def __init__(
        self,
        base_url: str,
        requests_per_second: float = 25.0,  # Conservative: 25/30 limit
        retry_count: int = 2,
        initial_retry_wait: int = 2,
        max_retry_wait: int = 120,
    ):
        """
        Initialize the CAISO rate-limited API client.

        Args:
            base_url: Base URL for the API
            requests_per_second: Max requests per second (default: 25, under 30 limit)
            retry_count: Number of retry attempts (default: 2)
            initial_retry_wait: Initial retry wait time in seconds (default: 2)
            max_retry_wait: Maximum retry wait time in seconds (default: 120)
        """
        super().__init__(base_url, retry_count, initial_retry_wait)

        # Rate limiting configuration based on CAISO limits
        self.requests_per_second = requests_per_second
        # Minimum seconds between requests
        self.min_interval = 1.0 / requests_per_second
        self.initial_retry_wait = initial_retry_wait
        self.max_retry_wait = max_retry_wait

        # Track request timing
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock()

    def _create_retry_decorator(self):
        """Create a retry decorator with exponential backoff for rate limits."""
        return retry(
            stop=stop_after_attempt(self.retry_count),
            wait=wait_exponential(
                multiplier=self.initial_retry_wait, max=self.max_retry_wait
            ),
            retry=retry_if_exception_type(
                (aiohttp.ClientResponseError, aiohttp.ClientError)
            ),
            reraise=True,
        )

    async def _enforce_rate_limit(self):
        """Enforce rate limiting by waiting if necessary."""
        async with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.debug(
                    f"CAISO rate limiting: waiting {wait_time:.3f}s before next request"
                )
                await asyncio.sleep(wait_time)

            self._last_request_time = time.time()

    async def _handle_rate_limit_error(self, response: aiohttp.ClientResponse) -> None:
        """Handle 429 rate limit errors with appropriate waiting."""
        if response.status == 429:
            # CAISO service-level enforcement periods are 5-30 seconds
            # Wait 30 seconds to be safe
            rate_limit_wait = 30
            logger.warning(
                f"CAISO rate limit hit (429), waiting {rate_limit_wait}s "
                f"for enforcement period reset"
            )
            await asyncio.sleep(rate_limit_wait)

    async def get_json(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a rate-limited GET request and return JSON response.

        Args:
            endpoint: API endpoint (without base URL) or full URL
            params: Query parameters
            headers: HTTP headers

        Returns:
            JSON response as dictionary

        Raises:
            aiohttp.ClientError: If the request fails after all retries
        """

        @self._create_retry_decorator()
        async def _make_request():
            await self._enforce_rate_limit()

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
                        # Handle rate limiting before raising for status
                        await self._handle_rate_limit_error(response)
                        response.raise_for_status()
                        return await response.json()

                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        logger.warning(f"CAISO rate limit error (429): {e}")
                    else:
                        logger.error(
                            f"HTTP error {e.status} fetching data from {url}: {e}"
                        )
                    raise
                except aiohttp.ClientError as e:
                    logger.error(f"Error fetching data from {url}: {e}")
                    raise

        return await _make_request()

    async def get_text(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Make a rate-limited GET request and return text response.

        Args:
            endpoint: API endpoint (without base URL) or full URL
            params: Query parameters
            headers: HTTP headers

        Returns:
            Text response as string

        Raises:
            aiohttp.ClientError: If the request fails after all retries
        """

        @self._create_retry_decorator()
        async def _make_request():
            await self._enforce_rate_limit()

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
                        # Handle rate limiting before raising for status
                        await self._handle_rate_limit_error(response)
                        response.raise_for_status()
                        return await response.text()

                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        logger.warning(f"CAISO rate limit error (429): {e}")
                    else:
                        logger.error(
                            f"HTTP error {e.status} fetching data from {url}: {e}"
                        )
                    raise
                except aiohttp.ClientError as e:
                    logger.error(f"Error fetching data from {url}: {e}")
                    raise

        return await _make_request()

    async def get_bytes(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """
        Make a rate-limited GET request and return raw bytes.

        Args:
            endpoint: API endpoint (without base URL) or full URL
            params: Query parameters
            headers: HTTP headers

        Returns:
            Raw response bytes

        Raises:
            aiohttp.ClientError: If the request fails after all retries
        """

        @self._create_retry_decorator()
        async def _make_request():
            await self._enforce_rate_limit()

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
                        await self._handle_rate_limit_error(response)
                        response.raise_for_status()
                        return await response.read()

                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        logger.warning(f"CAISO rate limit error (429): {e}")
                    else:
                        logger.error(
                            f"HTTP error {e.status} fetching data from {url}: {e}"
                        )
                    raise
                except aiohttp.ClientError as e:
                    logger.error(f"Error fetching data from {url}: {e}")
                    raise

        return await _make_request()
