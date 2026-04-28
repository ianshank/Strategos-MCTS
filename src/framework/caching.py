"""
Query-level caching for expensive operations.

Features:
- TTL-based expiration
- LRU eviction
- Configurable per-operation
- Thread-safe async operations
- No hardcoded values - all configurable via settings

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 9
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from src.config.settings import get_settings
from src.observability.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL tracking."""

    value: Any
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0  # Default from settings
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Statistics for cache operations."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/metrics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "size": self.size,
            "hit_rate": round(self.hit_rate, 4),
        }


class QueryCache:
    """
    Async-safe query result cache with TTL and LRU eviction.

    All configuration comes from settings or constructor parameters,
    no hardcoded values.

    Example:
        >>> cache = QueryCache(max_size=1000, default_ttl=300)
        >>> result = await cache.get_or_compute("key", expensive_fn)

    Thread Safety:
        Uses asyncio.Lock for thread-safe access in async contexts.
    """

    def __init__(
        self,
        max_size: int | None = None,
        default_ttl: float | None = None,
        name: str = "default",
    ):
        """
        Initialize cache with configurable parameters.

        Args:
            max_size: Maximum number of entries (defaults to settings)
            default_ttl: Default TTL in seconds (defaults to 300s)
            name: Cache name for logging/metrics
        """
        settings = get_settings()
        self.max_size = max_size or settings.MCTS_CACHE_SIZE_LIMIT
        self.default_ttl = default_ttl or 300.0
        self.name = name
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
        self._logger = logger.getChild(f"QueryCache.{name}")

    def _compute_key(self, *args: Any, **kwargs: Any) -> str:
        """
        Generate deterministic cache key from arguments.

        Uses SHA256 hash for consistent, collision-resistant keys.
        """
        key_data = f"{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def get(self, key: str) -> tuple[bool, Any]:
        """
        Get value from cache if exists and not expired.

        Args:
            key: Cache key

        Returns:
            Tuple of (found, value). If not found or expired, returns (False, None).
        """
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired:
                    entry.hits += 1
                    self._stats.hits += 1
                    # Move to end for LRU (most recently used)
                    self._cache.move_to_end(key)
                    self._logger.debug(
                        "Cache hit",
                        extra={"key": key[:8], "hits": entry.hits},
                    )
                    return True, entry.value
                else:
                    # Entry expired, remove it
                    del self._cache[key]
                    self._stats.expirations += 1
                    self._logger.debug(
                        "Cache expired",
                        extra={"key": key[:8], "age": entry.age_seconds},
                    )

            self._stats.misses += 1
            self._stats.size = len(self._cache)
            return False, None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        """
        Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override (uses default if not provided)
        """
        async with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._stats.evictions += 1
                self._logger.debug(
                    "Cache eviction",
                    extra={"evicted_key": evicted_key[:8]},
                )

            self._cache[key] = CacheEntry(
                value=value,
                ttl_seconds=ttl or self.default_ttl,
            )
            self._stats.size = len(self._cache)

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        ttl: float | None = None,
    ) -> T:
        """
        Get from cache or compute and cache result.

        This is the primary method for cache-through pattern.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached (can be async)
            ttl: Optional TTL for this entry

        Returns:
            Cached or computed value
        """
        found, value = await self.get(key)
        if found:
            return value  # type: ignore[no-any-return]

        # Compute outside lock to avoid blocking other operations
        start_time = time.time()
        if asyncio.iscoroutinefunction(compute_fn):
            result = await compute_fn()
        else:
            result = compute_fn()

        compute_time = (time.time() - start_time) * 1000
        self._logger.debug(
            "Cache compute",
            extra={"key": key[:8], "compute_ms": round(compute_time, 2)},
        )

        await self.set(key, result, ttl)
        return result  # type: ignore[no-any-return]

    async def invalidate(self, key: str) -> bool:
        """
        Remove a specific entry from cache.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Remove entries matching a pattern prefix.

        Args:
            pattern: Prefix pattern to match

        Returns:
            Number of entries removed
        """
        async with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(pattern)]
            for key in keys_to_remove:
                del self._cache[key]
            self._stats.size = len(self._cache)
            return len(keys_to_remove)

    async def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            self._logger.info("Cache cleared", extra={"cleared_entries": count})
            return count

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics as dictionary."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = CacheStats(size=len(self._cache))


class LLMResponseCache(QueryCache):
    """
    Specialized cache for LLM responses.

    Includes token-aware key generation and cost tracking.
    """

    def __init__(
        self,
        max_size: int | None = None,
        default_ttl: float | None = None,
    ):
        """Initialize LLM response cache."""
        super().__init__(max_size=max_size, default_ttl=default_ttl, name="llm")
        self._token_savings = 0

    def compute_key_from_messages(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
    ) -> str:
        """
        Compute cache key from LLM request parameters.

        Only caches deterministic requests (temperature=0).
        """
        # Only cache deterministic requests
        if temperature > 0:
            return ""

        key_data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }
        import json

        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:24]

    def record_token_savings(self, tokens: int) -> None:
        """Record token savings from cache hit."""
        self._token_savings += tokens

    @property
    def token_savings(self) -> int:
        """Get total token savings from cache hits."""
        return self._token_savings


class MCTSSimulationCache(QueryCache):
    """
    Specialized cache for MCTS simulation results.

    Optimized for high-throughput, short-TTL caching.
    """

    def __init__(
        self,
        max_size: int | None = None,
        default_ttl: float | None = None,
    ):
        """Initialize MCTS simulation cache."""
        settings = get_settings()
        super().__init__(
            max_size=max_size or settings.MCTS_CACHE_SIZE_LIMIT,
            default_ttl=default_ttl or 600.0,  # 10 minutes for simulations
            name="mcts",
        )

    def compute_key_from_state(self, state_hash: str, action: str | None = None) -> str:
        """
        Compute cache key from MCTS state.

        Args:
            state_hash: Pre-computed state hash
            action: Optional action for state-action pairs
        """
        if action:
            key_data = f"{state_hash}:{action}"
        else:
            key_data = state_hash
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]


# Global cache instances - lazily initialized
_query_cache: QueryCache | None = None
_llm_cache: LLMResponseCache | None = None
_mcts_cache: MCTSSimulationCache | None = None


def get_query_cache() -> QueryCache:
    """Get or create global query cache."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(name="query")
    return _query_cache


def get_llm_cache() -> LLMResponseCache:
    """Get or create LLM response cache."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMResponseCache()
    return _llm_cache


def get_mcts_cache() -> MCTSSimulationCache:
    """Get or create MCTS simulation cache."""
    global _mcts_cache
    if _mcts_cache is None:
        _mcts_cache = MCTSSimulationCache()
    return _mcts_cache


def reset_caches() -> None:
    """Reset all global caches. Useful for testing."""
    global _query_cache, _llm_cache, _mcts_cache
    _query_cache = None
    _llm_cache = None
    _mcts_cache = None


__all__ = [
    "QueryCache",
    "LLMResponseCache",
    "MCTSSimulationCache",
    "CacheEntry",
    "CacheStats",
    "get_query_cache",
    "get_llm_cache",
    "get_mcts_cache",
    "reset_caches",
]
