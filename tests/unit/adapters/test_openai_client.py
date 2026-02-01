"""
Unit tests for OpenAI LLM client adapter.

Tests initialization, response handling, error handling, circuit breaker,
rate limiting, and retry logic.

Based on: NEXT_STEPS_PLAN.md Phase 2.1
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_api_key() -> str:
    """Provide a mock API key for testing."""
    return "sk-test-key-for-unit-testing-only-12345"


@pytest.fixture
def mock_httpx_response() -> MagicMock:
    """Create a mock successful HTTP response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1699000000,
        "model": "gpt-4-turbo-preview",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }
    return response


@pytest.fixture
def mock_tool_call_response() -> MagicMock:
    """Create a mock response with tool calls."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "id": "chatcmpl-test456",
        "object": "chat.completion",
        "created": 1699000000,
        "model": "gpt-4-turbo-preview",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        },
    }
    return response


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestOpenAIClientInitialization:
    """Tests for OpenAI client initialization."""

    def test_client_initializes_with_api_key(self, mock_api_key):
        """Test client initializes successfully with API key."""
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key=mock_api_key)

        assert client.api_key == mock_api_key
        assert client.model == "gpt-4-turbo-preview"
        assert client.base_url == "https://api.openai.com/v1"

    def test_client_initializes_with_custom_model(self, mock_api_key):
        """Test client accepts custom model."""
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key=mock_api_key, model="gpt-3.5-turbo")

        assert client.model == "gpt-3.5-turbo"

    def test_client_initializes_with_custom_base_url(self, mock_api_key):
        """Test client accepts custom base URL."""
        from src.adapters.llm.openai_client import OpenAIClient

        custom_url = "https://custom.api.example.com/v1"
        client = OpenAIClient(api_key=mock_api_key, base_url=custom_url)

        assert client.base_url == custom_url

    def test_client_raises_error_without_api_key(self):
        """Test client raises error when no API key provided."""
        from src.adapters.llm.exceptions import LLMAuthenticationError
        from src.adapters.llm.openai_client import OpenAIClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMAuthenticationError, match="API key not provided"):
                OpenAIClient()

    def test_client_uses_env_var_for_api_key(self):
        """Test client uses OPENAI_API_KEY from environment."""
        from src.adapters.llm.openai_client import OpenAIClient

        env_key = "sk-env-test-key-12345"
        with patch.dict("os.environ", {"OPENAI_API_KEY": env_key}):
            client = OpenAIClient()
            assert client.api_key == env_key

    def test_client_accepts_organization_id(self, mock_api_key):
        """Test client accepts organization ID."""
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(
            api_key=mock_api_key,
            organization="org-test123",
        )

        assert client.organization == "org-test123"

    def test_client_initializes_circuit_breaker(self, mock_api_key):
        """Test client initializes circuit breaker with custom settings."""
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(
            api_key=mock_api_key,
            circuit_breaker_threshold=10,
            circuit_breaker_reset=120.0,
        )

        assert client.circuit_breaker.failure_threshold == 10
        assert client.circuit_breaker.reset_timeout == 120.0


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_starts_closed(self):
        """Test circuit breaker starts in closed state."""
        from src.adapters.llm.openai_client import CircuitBreaker

        cb = CircuitBreaker()

        assert cb.state == "closed"
        assert cb.can_execute() is True

    def test_circuit_opens_after_threshold_failures(self):
        """Test circuit opens after reaching failure threshold."""
        from src.adapters.llm.openai_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"
        assert cb.can_execute() is False

    def test_circuit_resets_on_success(self):
        """Test circuit resets failure count on success."""
        from src.adapters.llm.openai_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb.failure_count == 0
        assert cb.state == "closed"

    def test_circuit_half_open_after_reset_timeout(self):
        """Test circuit transitions to half-open after reset timeout."""
        from src.adapters.llm.openai_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        # Wait for reset timeout
        import time

        time.sleep(0.15)

        # Should be able to execute (transition to half-open)
        assert cb.can_execute() is True
        assert cb.state == "half-open"

    def test_circuit_closes_on_success_in_half_open(self):
        """Test circuit closes on success while half-open."""
        from src.adapters.llm.openai_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for reset timeout
        import time

        time.sleep(0.15)

        # Transition to half-open
        cb.can_execute()

        # Record success
        cb.record_success()

        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_get_reset_time_returns_remaining_time(self):
        """Test get_reset_time returns correct remaining time."""
        from src.adapters.llm.openai_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, reset_timeout=10.0)

        cb.record_failure()

        reset_time = cb.get_reset_time()

        assert 9.0 <= reset_time <= 10.0


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestTokenBucketRateLimiter:
    """Tests for token bucket rate limiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_immediate_request(self):
        """Test rate limiter allows immediate request when tokens available."""
        from src.adapters.llm.base import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(rate_per_minute=60)

        wait_time = await limiter.acquire()

        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_rate_limiter_stats_tracking(self):
        """Test rate limiter tracks stats correctly."""
        from src.adapters.llm.base import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(rate_per_minute=60)

        await limiter.acquire()
        stats = limiter.stats

        assert "rate_limit_waits" in stats
        assert "total_rate_limit_wait_time" in stats
        assert "current_tokens" in stats


# =============================================================================
# Response Parsing Tests
# =============================================================================


class TestResponseParsing:
    """Tests for response parsing."""

    def test_llm_response_dataclass_properties(self):
        """Test LLMResponse dataclass property accessors."""
        from src.adapters.llm.base import LLMResponse

        response = LLMResponse(
            text="Test response",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            model="gpt-4",
            finish_reason="stop",
        )

        assert response.text == "Test response"
        assert response.total_tokens == 30
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20
        assert response.model == "gpt-4"

    def test_llm_response_defaults(self):
        """Test LLMResponse defaults for missing usage data."""
        from src.adapters.llm.base import LLMResponse

        response = LLMResponse(text="Test")

        assert response.total_tokens == 0
        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0

    def test_tool_call_dataclass(self):
        """Test ToolCall dataclass."""
        from src.adapters.llm.base import ToolCall

        tool = ToolCall(
            id="call_abc123",
            name="get_weather",
            arguments={"location": "NYC"},
        )

        assert tool.id == "call_abc123"
        assert tool.name == "get_weather"
        assert tool.arguments == {"location": "NYC"}
        assert tool.type == "function"  # default

    def test_llm_tool_response_with_tool_calls(self):
        """Test LLMToolResponse with tool calls."""
        from src.adapters.llm.base import LLMToolResponse, ToolCall

        tool_calls = [
            ToolCall(id="1", name="func1", arguments={}),
            ToolCall(id="2", name="func2", arguments={}),
        ]

        response = LLMToolResponse(
            text="",
            tool_calls=tool_calls,
        )

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "func1"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_authentication_error(self):
        """Test LLMAuthenticationError structure."""
        from src.adapters.llm.exceptions import LLMAuthenticationError

        error = LLMAuthenticationError("openai", "Invalid API key")

        assert error.provider == "openai"
        assert error.status_code == 401
        assert "Invalid API key" in str(error)

    def test_rate_limit_error_with_retry_after(self):
        """Test LLMRateLimitError includes retry_after."""
        from src.adapters.llm.exceptions import LLMRateLimitError

        error = LLMRateLimitError("openai", retry_after=30.0)

        assert error.retry_after == 30.0
        assert error.status_code == 429

    def test_context_length_error_with_details(self):
        """Test LLMContextLengthError with token counts."""
        from src.adapters.llm.exceptions import LLMContextLengthError

        error = LLMContextLengthError("openai", token_count=50000, max_tokens=32000)

        assert "50000" in str(error)
        assert "32000" in str(error)

    def test_model_not_found_error(self):
        """Test LLMModelNotFoundError with model name."""
        from src.adapters.llm.exceptions import LLMModelNotFoundError

        error = LLMModelNotFoundError("openai", "gpt-5-ultra")

        assert "gpt-5-ultra" in str(error)
        assert error.status_code == 404

    def test_server_error(self):
        """Test LLMServerError with status code."""
        from src.adapters.llm.exceptions import LLMServerError

        error = LLMServerError("openai", 503, "Service unavailable")

        assert error.status_code == 503
        assert "Service unavailable" in str(error)

    @pytest.mark.asyncio
    async def test_client_handles_401_response(self, mock_api_key):
        """Test client raises LLMAuthenticationError on 401."""
        from src.adapters.llm.exceptions import LLMAuthenticationError
        from src.adapters.llm.openai_client import OpenAIClient

        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 401
        error_response.json.return_value = {"error": {"message": "Invalid API key"}}
        error_response.text = "Invalid API key"

        client = OpenAIClient(api_key=mock_api_key)

        with pytest.raises(LLMAuthenticationError):
            client._handle_error_response(error_response)

    @pytest.mark.asyncio
    async def test_client_handles_429_response(self, mock_api_key):
        """Test client raises LLMRateLimitError on 429."""
        from src.adapters.llm.exceptions import LLMRateLimitError
        from src.adapters.llm.openai_client import OpenAIClient

        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 429
        error_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        error_response.headers = {"Retry-After": "30"}
        error_response.text = "Rate limit exceeded"

        client = OpenAIClient(api_key=mock_api_key)

        with pytest.raises(LLMRateLimitError) as exc_info:
            client._handle_error_response(error_response)

        assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_client_handles_500_response(self, mock_api_key):
        """Test client raises LLMServerError on 500+."""
        from src.adapters.llm.exceptions import LLMServerError
        from src.adapters.llm.openai_client import OpenAIClient

        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 503
        error_response.json.return_value = {"error": {"message": "Server overloaded"}}
        error_response.text = "Server overloaded"

        client = OpenAIClient(api_key=mock_api_key)

        with pytest.raises(LLMServerError):
            client._handle_error_response(error_response)


# =============================================================================
# Generate Method Tests
# =============================================================================


class TestGenerateMethod:
    """Tests for the generate method."""

    @pytest.mark.asyncio
    async def test_generate_returns_llm_response(
        self,
        mock_api_key,
        mock_httpx_response,
    ):
        """Test generate returns LLMResponse on success."""
        from src.adapters.llm.base import LLMResponse
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key=mock_api_key)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_httpx_response
            mock_get_client.return_value = mock_http_client

            response = await client.generate(prompt="Hello, world!")

            assert isinstance(response, LLMResponse)
            assert response.text == "This is a test response."
            assert response.total_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_with_messages(
        self,
        mock_api_key,
        mock_httpx_response,
    ):
        """Test generate accepts messages list."""
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key=mock_api_key)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_httpx_response
            mock_get_client.return_value = mock_http_client

            messages = [{"role": "user", "content": "Hello"}]
            response = await client.generate(messages=messages)

            assert response.text == "This is a test response."

    @pytest.mark.asyncio
    async def test_generate_with_temperature(
        self,
        mock_api_key,
        mock_httpx_response,
    ):
        """Test generate passes temperature parameter."""
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key=mock_api_key)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_httpx_response
            mock_get_client.return_value = mock_http_client

            await client.generate(prompt="Test", temperature=0.5)

            # Verify temperature was passed
            call_args = mock_http_client.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_respects_circuit_breaker(
        self,
        mock_api_key,
    ):
        """Test generate respects circuit breaker state."""
        from src.adapters.llm.exceptions import CircuitBreakerOpenError
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(
            api_key=mock_api_key,
            circuit_breaker_threshold=1,
        )

        # Open the circuit breaker
        client.circuit_breaker.record_failure()

        with pytest.raises(CircuitBreakerOpenError):
            await client.generate(prompt="Test")


# =============================================================================
# Tool Calling Tests
# =============================================================================


class TestToolCalling:
    """Tests for tool/function calling support."""

    @pytest.mark.asyncio
    async def test_generate_with_tool_calls_response(
        self,
        mock_api_key,
        mock_tool_call_response,
    ):
        """Test generate handles tool call responses."""
        from src.adapters.llm.base import LLMToolResponse
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key=mock_api_key)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_tool_call_response
            mock_get_client.return_value = mock_http_client

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                    },
                }
            ]

            response = await client.generate(prompt="What's the weather?", tools=tools)

            assert isinstance(response, LLMToolResponse)
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "get_weather"


# =============================================================================
# Connection Error Tests
# =============================================================================


class TestConnectionErrors:
    """Tests for connection error handling."""

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_timeout_error(self, mock_api_key):
        """Test timeout raises LLMTimeoutError."""
        from src.adapters.llm.exceptions import LLMTimeoutError
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key=mock_api_key, max_retries=1)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_get_client.return_value = mock_http_client

            with pytest.raises(LLMTimeoutError):
                await client.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_connection_error_raises_llm_connection_error(self, mock_api_key):
        """Test connection error raises LLMConnectionError."""
        from src.adapters.llm.exceptions import LLMConnectionError
        from src.adapters.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key=mock_api_key, max_retries=1)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.side_effect = httpx.ConnectError("Connection failed")
            mock_get_client.return_value = mock_http_client

            with pytest.raises(LLMConnectionError):
                await client.generate(prompt="Test")
