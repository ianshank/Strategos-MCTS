"""
Base LLM client interface for provider-agnostic model access.

This module defines the protocol and data structures for LLM clients,
enabling seamless switching between providers (OpenAI, Anthropic, LM Studio, etc.)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from src.utils.time_utils import utc_now

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str
    usage: dict = field(default_factory=dict)
    model: str = ""
    raw_response: Any = None
    finish_reason: str = "stop"
    created_at: datetime = field(default_factory=utc_now)

    @property
    def total_tokens(self) -> int:
        """Total tokens used in request/response."""
        return int(self.usage.get("total_tokens", 0))

    @property
    def prompt_tokens(self) -> int:
        """Tokens used in prompt."""
        return int(self.usage.get("prompt_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        """Tokens used in completion."""
        return int(self.usage.get("completion_tokens", 0))


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM."""

    id: str
    name: str
    arguments: dict
    type: str = "function"


@dataclass
class LLMToolResponse(LLMResponse):
    """Response containing tool calls."""

    tool_calls: list[ToolCall] = field(default_factory=list)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for controlling request rates.

    This implementation uses a token bucket algorithm where:
    - Tokens are added at a fixed rate (rate_per_second)
    - Each request consumes one token
    - If no tokens available, caller waits until one becomes available
    """

    def __init__(self, rate_per_minute: int = 60):
        """
        Initialize the rate limiter.

        Args:
            rate_per_minute: Maximum requests allowed per minute
        """
        self.rate_per_second = rate_per_minute / 60.0
        self.max_tokens = float(rate_per_minute)
        self.tokens = self.max_tokens
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._wait_count = 0
        self._total_wait_time = 0.0

    async def acquire(self) -> float:
        """
        Acquire a token, waiting if necessary.

        Returns:
            Time spent waiting (0.0 if no wait was needed)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill

            # Refill tokens based on elapsed time
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate_per_second)
            self.last_refill = now

            wait_time = 0.0
            if self.tokens < 1:
                # Calculate how long to wait for one token
                wait_time = (1 - self.tokens) / self.rate_per_second
                self._wait_count += 1
                self._total_wait_time += wait_time

                # Release lock during sleep to allow other operations
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()

                # After sleeping, update time and set tokens to 0
                self.last_refill = time.monotonic()
                self.tokens = 0
            else:
                self.tokens -= 1

            return wait_time

    @property
    def stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "rate_limit_waits": self._wait_count,
            "total_rate_limit_wait_time": self._total_wait_time,
            "current_tokens": self.tokens,
        }


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for LLM clients.

    This protocol defines the interface that all LLM provider adapters must implement.
    Using Protocol allows for structural subtyping (duck typing) while maintaining
    type safety.
    """

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
        Generate a response from the LLM.

        Args:
            messages: List of message dicts in OpenAI format [{"role": "...", "content": "..."}]
            prompt: Simple string prompt (converted to single user message)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions for function calling
            stream: If True, returns AsyncIterator[str] for streaming
            stop: Stop sequences
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse if stream=False, AsyncIterator[str] if stream=True

        Raises:
            LLMClientError: Base exception for all client errors
        """
        ...


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Provides common functionality and enforces the interface contract.
    All concrete implementations should inherit from this class.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "default",
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        rate_limit_per_minute: int | None = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for authentication
            model: Model identifier
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_per_minute: Rate limit (requests per minute), None to disable
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._request_count = 0
        self._total_tokens_used = 0
        self._rate_limited_requests = 0
        logger.debug("LLM client initialized: model=%s, timeout=%.1f, max_retries=%d", model, timeout, max_retries)

        # Initialize rate limiter if configured
        if rate_limit_per_minute is not None and rate_limit_per_minute > 0:
            self._rate_limiter: TokenBucketRateLimiter | None = TokenBucketRateLimiter(
                rate_per_minute=rate_limit_per_minute
            )
        else:
            self._rate_limiter = None

    @abstractmethod
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
        """Generate a response from the LLM."""
        pass

    def _build_messages(
        self,
        messages: list[dict] | None = None,
        prompt: str | None = None,
    ) -> list[dict]:
        """
        Build message list from either messages or prompt.

        Args:
            messages: Pre-formatted message list
            prompt: Simple string prompt

        Returns:
            List of message dicts

        Raises:
            ValueError: If neither messages nor prompt provided
        """
        if messages is not None:
            return messages
        elif prompt is not None:
            return [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

    def _update_stats(self, response: LLMResponse) -> None:
        """Update internal statistics."""
        self._request_count += 1
        self._total_tokens_used += response.total_tokens
        logger.debug(
            "LLM request #%d: model=%s, tokens=%d, finish=%s",
            self._request_count,
            response.model,
            response.total_tokens,
            response.finish_reason,
        )

    async def _apply_rate_limit(self) -> None:
        """
        Apply rate limiting if configured.

        Waits if necessary to comply with rate limits.
        Tracks rate-limited requests in metrics.
        """
        if self._rate_limiter is not None:
            wait_time = await self._rate_limiter.acquire()
            if wait_time > 0:
                self._rate_limited_requests += 1
                logger.warning(
                    "Rate limited: waited %.2fs (total rate-limited: %d)", wait_time, self._rate_limited_requests
                )

    @property
    def stats(self) -> dict:
        """Get client statistics."""
        base_stats = {
            "request_count": self._request_count,
            "total_tokens_used": self._total_tokens_used,
            "rate_limited_requests": self._rate_limited_requests,
        }

        # Include rate limiter stats if available
        if self._rate_limiter is not None:
            base_stats.update(self._rate_limiter.stats)

        return base_stats

    async def close(self) -> None:  # noqa: B027
        """Clean up resources. Override in subclasses if needed."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
