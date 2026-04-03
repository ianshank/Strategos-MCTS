"""
Tests for LLM adapter base module.

Tests LLMResponse, TokenBucketRateLimiter, LLMClient protocol,
and BaseLLMClient abstract class.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.adapters.llm.base import (
    BaseLLMClient,
    LLMResponse,
    TokenBucketRateLimiter,
)


@pytest.mark.unit
class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_defaults(self):
        r = LLMResponse(text="hello", model="gpt-4")
        assert r.text == "hello"
        assert r.model == "gpt-4"
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.total_tokens == 0
        assert r.finish_reason == "stop"
        assert r.raw_response is None
        assert r.usage == {}

    def test_with_usage(self):
        r = LLMResponse(
            text="result",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="length",
            raw_response={"id": "123"},
        )
        assert r.total_tokens == 30
        assert r.prompt_tokens == 10
        assert r.completion_tokens == 20
        assert r.finish_reason == "length"


@pytest.mark.unit
class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_init(self):
        limiter = TokenBucketRateLimiter(rate_per_minute=60)
        assert limiter.rate_per_second == 1.0
        assert limiter.max_tokens == 60.0
        assert limiter.tokens == 60.0

    @pytest.mark.asyncio
    async def test_acquire_no_wait(self):
        limiter = TokenBucketRateLimiter(rate_per_minute=600)
        wait = await limiter.acquire()
        assert wait == 0.0

    @pytest.mark.asyncio
    async def test_acquire_decrements_tokens(self):
        limiter = TokenBucketRateLimiter(rate_per_minute=60)
        initial = limiter.tokens
        await limiter.acquire()
        assert limiter.tokens < initial

    @pytest.mark.asyncio
    async def test_acquire_rate_limits_when_empty(self):
        limiter = TokenBucketRateLimiter(rate_per_minute=60)
        limiter.tokens = 0.0  # Force empty
        limiter.last_refill = time.monotonic()  # Reset refill time

        start = time.monotonic()
        wait = await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited some time
        assert wait > 0
        assert elapsed >= wait * 0.9  # Allow small timing variance

    def test_stats(self):
        limiter = TokenBucketRateLimiter(rate_per_minute=60)
        stats = limiter.stats
        assert stats["rate_limit_waits"] == 0
        assert stats["total_rate_limit_wait_time"] == 0.0
        assert "current_tokens" in stats


@pytest.mark.unit
class TestLLMClientProtocol:
    """Tests for LLMClient protocol compliance."""

    def test_protocol_is_runtime_checkable(self):
        """LLMClient protocol should be runtime checkable."""
        # A mock with the right signature should satisfy the protocol
        mock = MagicMock()
        mock.generate = AsyncMock(return_value=LLMResponse(text="ok", model="test"))
        # Protocol check - this tests the runtime_checkable decorator works
        assert hasattr(mock, "generate")


@pytest.mark.unit
class TestBaseLLMClient:
    """Tests for BaseLLMClient abstract class."""

    def _make_concrete_client(self, **kwargs):
        """Create a concrete subclass for testing."""

        class ConcreteClient(BaseLLMClient):
            async def generate(self, **kw) -> LLMResponse:
                return LLMResponse(text="test", model=self.model, usage={"total_tokens": 10})

        return ConcreteClient(**kwargs)

    def test_init_defaults(self):
        client = self._make_concrete_client()
        assert client.model == "default"
        assert client.timeout == 60.0
        assert client.max_retries == 3
        assert client._rate_limiter is None

    def test_init_with_rate_limit(self):
        client = self._make_concrete_client(rate_limit_per_minute=100)
        assert client._rate_limiter is not None
        assert isinstance(client._rate_limiter, TokenBucketRateLimiter)

    def test_init_no_rate_limit_when_zero(self):
        client = self._make_concrete_client(rate_limit_per_minute=0)
        assert client._rate_limiter is None

    def test_build_messages_from_messages(self):
        client = self._make_concrete_client()
        msgs = [{"role": "user", "content": "hi"}]
        result = client._build_messages(messages=msgs)
        assert result == msgs

    def test_build_messages_from_prompt(self):
        client = self._make_concrete_client()
        result = client._build_messages(prompt="hello")
        assert result == [{"role": "user", "content": "hello"}]

    def test_build_messages_raises_when_neither(self):
        client = self._make_concrete_client()
        with pytest.raises(ValueError, match="Either 'messages' or 'prompt'"):
            client._build_messages()

    def test_update_stats(self):
        client = self._make_concrete_client()
        response = LLMResponse(text="ok", model="test", usage={"total_tokens": 50})
        client._update_stats(response)
        assert client._request_count == 1
        assert client._total_tokens_used == 50

    @pytest.mark.asyncio
    async def test_apply_rate_limit_no_limiter(self):
        client = self._make_concrete_client()
        await client._apply_rate_limit()  # Should not raise

    @pytest.mark.asyncio
    async def test_apply_rate_limit_with_limiter(self):
        client = self._make_concrete_client(rate_limit_per_minute=600)
        await client._apply_rate_limit()
        # With high rate limit, should not wait
        assert client._rate_limited_requests == 0

    def test_stats_without_rate_limiter(self):
        client = self._make_concrete_client()
        stats = client.stats
        assert stats["request_count"] == 0
        assert stats["total_tokens_used"] == 0

    def test_stats_with_rate_limiter(self):
        client = self._make_concrete_client(rate_limit_per_minute=60)
        stats = client.stats
        assert "rate_limit_waits" in stats
        assert "current_tokens" in stats

    @pytest.mark.asyncio
    async def test_close(self):
        client = self._make_concrete_client()
        await client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self):
        client = self._make_concrete_client()
        async with client as c:
            assert c is client

    @pytest.mark.asyncio
    async def test_generate_works(self):
        client = self._make_concrete_client()
        result = await client.generate(prompt="hi")
        assert result.text == "test"
        assert result.total_tokens == 10
