"""
OpenAI-compatible LLM client adapter.

Implements the LLMClient protocol for OpenAI API (and compatible APIs).
Includes retry logic, circuit breaker pattern, and streaming support.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMClient, LLMResponse, LLMToolResponse, ToolCall
from .exceptions import (
    CircuitBreakerOpenError,
    LLMAuthenticationError,
    LLMClientError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMInvalidRequestError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
    LLMStreamError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker implementation for resilience."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if reset timeout has passed
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.state = "half-open"
                self.half_open_calls = 0
                return True
            return False

        if self.state == "half-open":
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record successful request."""
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
        elif self.state == "closed":
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "half-open" or self.failure_count >= self.failure_threshold:
            self.state = "open"

    def get_reset_time(self) -> float:
        """Get time until circuit resets."""
        if self.state != "open":
            return 0.0
        elapsed = time.time() - self.last_failure_time
        return max(0, self.reset_timeout - elapsed)


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client with retry logic and circuit breaker.

    Features:
    - Exponential backoff retry for transient errors
    - Circuit breaker to prevent cascading failures
    - Streaming support
    - Structured error handling
    - Tool/function calling support
    """

    PROVIDER_NAME = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4-turbo-preview"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        organization: str | None = None,
        # Circuit breaker settings
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset: float = 60.0,
        # Rate limiting
        rate_limit_per_minute: int | None = None,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4-turbo-preview)
            base_url: API base URL (default: https://api.openai.com/v1)
            timeout: Request timeout in seconds
            max_retries: Max retry attempts for transient errors
            organization: Optional organization ID
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_reset: Seconds before circuit resets
            rate_limit_per_minute: Rate limit for requests per minute (None to disable)
        """
        import os

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMAuthenticationError(self.PROVIDER_NAME, "API key not provided and OPENAI_API_KEY not set")

        super().__init__(
            api_key=api_key,
            model=model or self.DEFAULT_MODEL,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            rate_limit_per_minute=rate_limit_per_minute,
        )

        self.organization = organization
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset,
        )

        # Initialize async HTTP client
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Convert HTTP error responses to appropriate exceptions."""
        status_code = response.status_code

        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text

        if status_code == 401:
            raise LLMAuthenticationError(self.PROVIDER_NAME, error_message)
        elif status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_after_float = float(retry_after) if retry_after else None
            raise LLMRateLimitError(self.PROVIDER_NAME, retry_after=retry_after_float, message=error_message)
        elif status_code == 402:
            raise LLMQuotaExceededError(self.PROVIDER_NAME, error_message)
        elif status_code == 404:
            raise LLMModelNotFoundError(self.PROVIDER_NAME, self.model)
        elif status_code == 400:
            if "context_length" in error_message.lower():
                raise LLMContextLengthError(self.PROVIDER_NAME)
            raise LLMInvalidRequestError(self.PROVIDER_NAME, error_message)
        elif status_code >= 500:
            raise LLMServerError(self.PROVIDER_NAME, status_code, error_message)
        else:
            raise LLMClientError(error_message, self.PROVIDER_NAME, status_code=status_code)

    def _make_retry_decorator(self):
        """Create retry decorator with exponential backoff."""
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((LLMRateLimitError, LLMServerError, LLMConnectionError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """
        Generate a response from OpenAI.

        Args:
            messages: Chat messages in OpenAI format
            prompt: Simple string prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            tools: Tool definitions for function calling
            stream: If True, returns AsyncIterator
            stop: Stop sequences
            **kwargs: Additional OpenAI parameters (top_p, presence_penalty, etc.)

        Returns:
            LLMResponse or AsyncIterator[str] for streaming
        """
        # Apply rate limiting before proceeding
        await self._apply_rate_limit()

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpenError(
                self.PROVIDER_NAME,
                self.circuit_breaker.failure_count,
                self.circuit_breaker.get_reset_time(),
            )

        if stream:
            return await self._generate_stream(
                messages=messages,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stop=stop,
                **kwargs,
            )
        else:
            return await self._generate_non_stream(
                messages=messages,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stop=stop,
                **kwargs,
            )

    async def _generate_non_stream(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming generation with retry logic."""

        @self._make_retry_decorator()
        async def _request():
            client = await self._get_client()

            # Build request payload
            payload = {
                "model": self.model,
                "messages": self._build_messages(messages, prompt),
                "temperature": temperature,
            }

            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if stop:
                payload["stop"] = stop
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = kwargs.pop("tool_choice", "auto")

            # Add any additional kwargs
            payload.update(kwargs)

            try:
                response = await client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as e:
                raise LLMTimeoutError(self.PROVIDER_NAME, self.timeout) from e
            except httpx.ConnectError as e:
                raise LLMConnectionError(self.PROVIDER_NAME, self.base_url) from e

            if response.status_code != 200:
                self._handle_error_response(response)

            return response

        try:
            response = await _request()
            self.circuit_breaker.record_success()
        except Exception:
            self.circuit_breaker.record_failure()
            raise

        # Parse response
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]

            usage = data.get("usage", {})
            finish_reason = choice.get("finish_reason", "stop")

            # Check for tool calls
            llm_response: LLMResponse | LLMToolResponse
            if "tool_calls" in message:
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in message["tool_calls"]
                ]
                llm_response = LLMToolResponse(
                    text=message.get("content", ""),
                    usage=usage,
                    model=data.get("model", self.model),
                    raw_response=data,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                )
            else:
                llm_response = LLMResponse(
                    text=message.get("content", ""),
                    usage=usage,
                    model=data.get("model", self.model),
                    raw_response=data,
                    finish_reason=finish_reason,
                )

            self._update_stats(llm_response)
            return llm_response

        except (KeyError, json.JSONDecodeError) as e:
            raise LLMResponseParseError(self.PROVIDER_NAME, response.text) from e

    async def _generate_stream(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming generation."""

        client = await self._get_client()

        # Build request payload
        payload = {
            "model": self.model,
            "messages": self._build_messages(messages, prompt),
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        # Note: tools with streaming have limited support
        if tools:
            payload["tools"] = tools

        payload.update(kwargs)

        async def stream_generator() -> AsyncIterator[str]:
            try:
                async with client.stream("POST", "/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        # Read the full response for error handling
                        await response.aread()
                        self._handle_error_response(response)

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except (json.JSONDecodeError, KeyError):
                                continue

                self.circuit_breaker.record_success()

            except httpx.TimeoutException as e:
                self.circuit_breaker.record_failure()
                raise LLMTimeoutError(self.PROVIDER_NAME, self.timeout) from e
            except httpx.ConnectError as e:
                self.circuit_breaker.record_failure()
                raise LLMConnectionError(self.PROVIDER_NAME, self.base_url) from e
            except Exception as e:
                self.circuit_breaker.record_failure()
                if isinstance(e, LLMClientError):
                    raise
                raise LLMStreamError(self.PROVIDER_NAME, str(e)) from e

        return stream_generator()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
