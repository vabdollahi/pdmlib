"""
Tests for the API client utilities.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.utils.api_clients import BaseAPIClient


class TestBaseAPIClient:
    """Tests for BaseAPIClient class."""

    @pytest.fixture
    def client(self):
        """Create a test API client."""
        return BaseAPIClient("https://api.example.com", retry_count=2, retry_wait=1)

    @pytest.mark.asyncio
    async def test_get_json_success(self, client):
        """Test successful JSON request."""
        mock_response_data = {"test": "data"}

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            # Use Mock for raise_for_status to avoid coroutine warning
            mock_response.raise_for_status = Mock(return_value=None)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await client.get_json("/test", {"param": "value"})

            assert result == mock_response_data
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_json_with_full_url(self, client):
        """Test JSON request with full URL."""
        mock_response_data = {"test": "data"}

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            # Use Mock for raise_for_status to avoid coroutine warning
            mock_response.raise_for_status = Mock(return_value=None)
            mock_get.return_value.__aenter__.return_value = mock_response

            full_url = "https://different-api.com/endpoint"
            result = await client.get_json(full_url, {"param": "value"})

            assert result == mock_response_data
            # Should use the full URL, not client base URL
            call_args = mock_get.call_args
            if len(call_args[1]) > 0 and "url" in call_args[1]:
                called_url = call_args[1]["url"]
            else:
                called_url = call_args[0][0]
            assert full_url in str(called_url)

    @pytest.mark.asyncio
    async def test_get_json_basic_functionality(self, client):
        """Test basic JSON request functionality."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Test successful response
            mock_response = AsyncMock()
            mock_response.json.return_value = {"success": True}
            # Use Mock for raise_for_status to avoid coroutine warning
            mock_response.raise_for_status = Mock(return_value=None)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await client.get_json("/test")

            assert result == {"success": True}
            assert mock_get.call_count == 1

    @pytest.mark.asyncio
    async def test_get_json_with_headers(self, client):
        """Test JSON request with custom headers."""
        mock_response_data = {"test": "data"}
        custom_headers = {"Authorization": "Bearer token"}

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            # Use Mock for raise_for_status to avoid coroutine warning
            mock_response.raise_for_status = Mock(return_value=None)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await client.get_json("/test", headers=custom_headers)

            assert result == mock_response_data
            # Verify headers were passed
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["headers"] == custom_headers
