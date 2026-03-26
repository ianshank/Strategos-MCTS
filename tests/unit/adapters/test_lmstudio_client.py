"""
Tests for LM Studio LLM client adapter.

Tests the local LLM server client including health checks,
model listing, error handling, retry logic, and streaming.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.adapters.llm.exceptions import (
    LLMClientError,
    LLMConnectionError,
    LLMResponseParseError,
    LLMServerError,
    LLMStreamError,
    LLMTimeoutError,
)
from src.adapters.llm.lmstudio_client import LMStudioClient


@pytest.fixture
def client():
    """Create LMStudioClient with test configuration."""
    return LMStudioClient(
        model="test-local-model",
        base_url="http://localhost:1234/v1",
        timeout=30.0,
        max_retries=2,
    )


@pytest.fixture
def mock_success_response():
    """Standard successful chat completion response."""
    return {
        "id": "chatcmpl-local-123",
        "object": "chat.completion",
        "model": "test-local-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from local model"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.unit
class TestLMStudioClientInit:
    """Tests for LMStudioClient initialization."""

    def test_default_init(self):
        """Test client initializes with defaults."""
        client = LMStudioClient()
        assert client.model == "local-model"
        assert client.timeout == 300.0
        assert client.max_retries == 2
        assert client.api_key == "not-required"

    def test_custom_init(self):
        """Test client initializes with custom parameters."""
        client = LMStudioClient(
            model="my-model",
            base_url="http://myhost:5000/v1",
            timeout=60.0,
            max_retries=3,
        )
        assert client.model == "my-model"
        assert client.base_url == "http://myhost:5000/v1"
        assert client.timeout == 60.0
        assert client.max_retries == 3

    def test_env_var_base_url(self, monkeypatch):
        """Test base URL from environment variable."""
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://remote:8080/v1")
        client = LMStudioClient()
        assert client.base_url == "http://remote:8080/v1"

    def test_explicit_base_url_overrides_env(self, monkeypatch):
        """Test explicit base_url takes precedence over env var."""
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://env-host:8080/v1")
        client = LMStudioClient(base_url="http://explicit:1234/v1")
        assert client.base_url == "http://explicit:1234/v1"

    def test_api_key_optional(self):
        """Test API key is not required for local server."""
        client = LMStudioClient(api_key=None)
        assert client.api_key == "not-required"

    def test_custom_api_key_accepted(self):
        """Test custom API key is stored when provided."""
        client = LMStudioClient(api_key="custom-local-key")
        assert client.api_key == "custom-local-key"


@pytest.mark.unit
class TestLMStudioClientGetClient:
    """Tests for HTTP client creation."""

    @pytest.mark.asyncio
    async def test_creates_client_on_first_call(self, client):
        """Test HTTP client is created lazily."""
        assert client._client is None
        http_client = await client._get_client()
        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)
        await client.close()

    @pytest.mark.asyncio
    async def test_reuses_existing_client(self, client):
        """Test HTTP client is reused on subsequent calls."""
        client1 = await client._get_client()
        client2 = await client._get_client()
        assert client1 is client2
        await client.close()

    @pytest.mark.asyncio
    async def test_auth_header_when_key_provided(self):
        """Test auth header is added when API key is provided."""
        c = LMStudioClient(api_key="test-key-123")
        http_client = await c._get_client()
        assert http_client.headers.get("authorization") == "Bearer test-key-123"
        await c.close()

    @pytest.mark.asyncio
    async def test_no_auth_header_when_not_required(self, client):
        """Test no auth header for default 'not-required' key."""
        http_client = await client._get_client()
        assert "authorization" not in http_client.headers
        await http_client.aclose()


@pytest.mark.unit
class TestLMStudioHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test health check returns True when server responds 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.check_health()
        assert result is True
        mock_http.get.assert_called_once_with("/models")

    @pytest.mark.asyncio
    async def test_health_check_server_down(self, client):
        """Test health check returns False when server unreachable."""
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_non_200(self, client):
        """Test health check returns False for non-200 status."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.check_health()
        assert result is False


@pytest.mark.unit
class TestLMStudioListModels:
    """Tests for model listing."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, client):
        """Test listing models from server."""
        model_data = {"data": [{"id": "model-1"}, {"id": "model-2"}]}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = model_data

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        models = await client.list_models()
        assert len(models) == 2
        assert models[0]["id"] == "model-1"

    @pytest.mark.asyncio
    async def test_list_models_server_error(self, client):
        """Test empty list on server error."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        models = await client.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_connection_error(self, client):
        """Test empty list on connection error."""
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_http.is_closed = False
        client._client = mock_http

        models = await client.list_models()
        assert models == []


@pytest.mark.unit
class TestLMStudioErrorHandling:
    """Tests for error response handling."""

    def test_server_error_500(self, client):
        """Test 500 error raises LLMServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": {"message": "Internal error"}}

        with pytest.raises(LLMServerError):
            client._handle_error_response(mock_response)

    def test_server_error_503(self, client):
        """Test 503 error raises LLMServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service unavailable"
        mock_response.json.return_value = {"error": {"message": "Service unavailable"}}

        with pytest.raises(LLMServerError):
            client._handle_error_response(mock_response)

    def test_client_error_400(self, client):
        """Test 400 error raises LLMClientError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_response.json.return_value = {"error": {"message": "Bad request"}}

        with pytest.raises(LLMClientError):
            client._handle_error_response(mock_response)

    def test_error_with_unparseable_json(self, client):
        """Test error handling when response body is not JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Plain text error"
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)

        with pytest.raises(LLMServerError):
            client._handle_error_response(mock_response)


@pytest.mark.unit
class TestLMStudioGenerate:
    """Tests for text generation."""

    @pytest.mark.asyncio
    async def test_generate_with_prompt(self, client, mock_success_response):
        """Test generation with simple prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_success_response

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.generate(prompt="Hello")
        assert result.text == "Hello from local model"
        assert result.total_tokens == 15
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_messages(self, client, mock_success_response):
        """Test generation with message list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_success_response

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        messages = [{"role": "user", "content": "Hello"}]
        result = await client.generate(messages=messages)
        assert result.text == "Hello from local model"

    @pytest.mark.asyncio
    async def test_generate_payload_includes_optional_params(self, client, mock_success_response):
        """Test that max_tokens, stop, and kwargs are included in payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_success_response

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        await client.generate(
            prompt="Hello",
            max_tokens=100,
            stop=["END"],
            top_p=0.9,
        )

        call_payload = mock_http.post.call_args[1]["json"]
        assert call_payload["max_tokens"] == 100
        assert call_payload["stop"] == ["END"]
        assert call_payload["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_generate_retry_on_timeout(self, client, mock_success_response):
        """Test retry logic on timeout."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_success_response

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=[
                httpx.TimeoutException("Timeout"),
                mock_response,
            ]
        )
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.generate(prompt="Hello")
        assert result.text == "Hello from local model"
        assert mock_http.post.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_retry_on_connection_error(self, client, mock_success_response):
        """Test retry logic on connection error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_success_response

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=[
                httpx.ConnectError("Connection refused"),
                mock_response,
            ]
        )
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.generate(prompt="Hello")
        assert result.text == "Hello from local model"

    @pytest.mark.asyncio
    async def test_generate_exhausts_retries(self, client):
        """Test LLMConnectionError after all retries exhausted."""
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(LLMConnectionError):
            await client.generate(prompt="Hello")

        assert mock_http.post.call_count == client.max_retries

    @pytest.mark.asyncio
    async def test_generate_timeout_exhausts_retries(self, client):
        """Test LLMTimeoutError after all retries exhausted."""
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(LLMTimeoutError):
            await client.generate(prompt="Hello")

    @pytest.mark.asyncio
    async def test_generate_no_retry_on_client_error(self, client):
        """Test client errors are not retried."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_response.json.return_value = {"error": {"message": "Bad request"}}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(LLMClientError):
            await client.generate(prompt="Hello")

        # Client errors should not be retried
        assert mock_http.post.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_parse_error(self, client):
        """Test LLMResponseParseError on malformed response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "not json"
        mock_response.json.return_value = {"unexpected": "format"}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises((LLMResponseParseError, KeyError)):
            await client.generate(prompt="Hello")


@pytest.mark.unit
class TestLMStudioClose:
    """Tests for client cleanup."""

    @pytest.mark.asyncio
    async def test_close_client(self, client):
        """Test close properly closes HTTP client."""
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.aclose = AsyncMock()
        client._client = mock_http

        await client.close()
        mock_http.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_noop_when_no_client(self, client):
        """Test close is safe when no client exists."""
        assert client._client is None
        await client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_noop_when_already_closed(self, client):
        """Test close is safe when client already closed."""
        mock_http = AsyncMock()
        mock_http.is_closed = True
        client._client = mock_http

        await client.close()
        mock_http.aclose.assert_not_called()
