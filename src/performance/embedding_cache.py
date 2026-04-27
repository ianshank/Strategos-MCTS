"""
Embedding Cache for Feature Extraction.

Provides high-performance caching for embedding computations with:
- LRU eviction strategy
- Thread-safe access
- Batch processing support
- Memory usage monitoring
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.observability.logging import get_logger

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

logger = get_logger(__name__)


@dataclass
class EmbeddingCacheConfig:
    """Configuration for embedding cache."""

    # Cache sizing
    max_entries: int = 10000
    max_memory_mb: float = 500.0  # Maximum memory usage

    # Embedding dimensions
    embedding_dim: int = 768  # Default BERT/Sentence-Transformer dimension

    # TTL (time-to-live) in seconds
    ttl_seconds: float = 3600.0  # 1 hour default

    # Performance tuning
    batch_size: int = 32
    enable_statistics: bool = True

    # Memory estimation
    bytes_per_float: int = 4  # float32

    def estimated_memory_per_entry_bytes(self) -> int:
        """Estimate memory per cache entry."""
        # Embedding array + key overhead + metadata
        return self.embedding_dim * self.bytes_per_float + 100

    def max_entries_for_memory(self) -> int:
        """Calculate max entries based on memory limit."""
        bytes_limit = int(self.max_memory_mb * 1024 * 1024)
        entry_size = self.estimated_memory_per_entry_bytes()
        return min(self.max_entries, bytes_limit // entry_size)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    embedding: Any  # numpy array
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > ttl_seconds


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    # Timing
    total_lookup_time_ms: float = 0.0
    total_compute_time_ms: float = 0.0

    # Memory
    entries_count: int = 0
    estimated_memory_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": self.hit_rate,
            "total_lookup_time_ms": self.total_lookup_time_ms,
            "total_compute_time_ms": self.total_compute_time_ms,
            "entries_count": self.entries_count,
            "estimated_memory_mb": self.estimated_memory_mb,
        }


class EmbeddingCache:
    """
    High-performance LRU cache for embeddings.

    Features:
    - Thread-safe access
    - LRU eviction
    - TTL-based expiration
    - Batch processing support
    - Memory monitoring
    """

    def __init__(self, config: EmbeddingCacheConfig | None = None):
        """
        Initialize embedding cache.

        Args:
            config: Cache configuration (uses defaults if not provided)
        """
        self.config = config or EmbeddingCacheConfig()
        self._max_entries = self.config.max_entries_for_memory()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._compute_fn: Callable[[str], Any] | None = None

        logger.debug(
            "Initialized embedding cache with max_entries=%d, embedding_dim=%d",
            self._max_entries,
            self.config.embedding_dim,
        )

    def set_compute_function(self, fn: Callable[[str], Any]) -> None:
        """
        Set the function to compute embeddings on cache miss.

        Args:
            fn: Function that takes a string and returns an embedding array
        """
        self._compute_fn = fn

    def _compute_hash(self, text: str) -> str:
        """Compute hash key for text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def get(self, text: str) -> Any | None:
        """
        Get embedding from cache.

        Args:
            text: Input text

        Returns:
            Cached embedding or None if not found
        """
        start_time = time.perf_counter()
        key = self._compute_hash(text)

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if entry.is_expired(self.config.ttl_seconds):
                    del self._cache[key]
                    self._stats.expirations += 1
                    self._stats.misses += 1
                    return None

                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.access_count += 1
                entry.last_accessed = time.time()

                self._stats.hits += 1
                self._stats.total_lookup_time_ms += (time.perf_counter() - start_time) * 1000
                return entry.embedding

            self._stats.misses += 1
            self._stats.total_lookup_time_ms += (time.perf_counter() - start_time) * 1000
            return None

    def put(self, text: str, embedding: Any) -> None:
        """
        Store embedding in cache.

        Args:
            text: Input text
            embedding: Computed embedding
        """
        key = self._compute_hash(text)

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_entries:
                oldest_key, _ = self._cache.popitem(last=False)
                self._stats.evictions += 1
                logger.debug("Evicted cache entry: %s", oldest_key[:8])

            self._cache[key] = CacheEntry(embedding=embedding)
            self._update_stats()

    def get_or_compute(self, text: str) -> Any:
        """
        Get embedding from cache or compute it.

        Args:
            text: Input text

        Returns:
            Embedding (from cache or freshly computed)

        Raises:
            ValueError: If no compute function is set
        """
        # Try cache first
        embedding = self.get(text)
        if embedding is not None:
            return embedding

        # Compute on miss
        if self._compute_fn is None:
            raise ValueError("No compute function set. Call set_compute_function() first.")

        start_time = time.perf_counter()
        embedding = self._compute_fn(text)
        self._stats.total_compute_time_ms += (time.perf_counter() - start_time) * 1000

        # Cache the result
        self.put(text, embedding)
        return embedding

    def get_batch(self, texts: list[str]) -> list[Any]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embeddings (in same order as input)
        """
        results = []
        to_compute = []
        to_compute_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            embedding = self.get(text)
            if embedding is not None:
                results.append(embedding)
            else:
                results.append(None)  # Placeholder
                to_compute.append(text)
                to_compute_indices.append(i)

        # Batch compute missing embeddings
        if to_compute and self._compute_fn is not None:
            start_time = time.perf_counter()

            # Compute in batches
            for batch_start in range(0, len(to_compute), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(to_compute))
                batch_texts = to_compute[batch_start:batch_end]

                # Compute embeddings for batch
                for j, text in enumerate(batch_texts):
                    embedding = self._compute_fn(text)
                    idx = to_compute_indices[batch_start + j]
                    results[idx] = embedding
                    self.put(text, embedding)

            self._stats.total_compute_time_ms += (time.perf_counter() - start_time) * 1000

        return results

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._update_stats()
            logger.debug("Cleared embedding cache")

    def _update_stats(self) -> None:
        """Update cache statistics."""
        self._stats.entries_count = len(self._cache)
        entry_size = self.config.estimated_memory_per_entry_bytes()
        self._stats.estimated_memory_mb = (len(self._cache) * entry_size) / (1024 * 1024)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            self._update_stats()
            return self._stats.to_dict()

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Current cache hit rate."""
        return self._stats.hit_rate


def create_embedding_cache(
    *,
    max_entries: int = 10000,
    max_memory_mb: float = 500.0,
    embedding_dim: int = 768,
    ttl_seconds: float = 3600.0,
    compute_fn: Callable[[str], Any] | None = None,
    **kwargs: Any,
) -> EmbeddingCache:
    """
    Factory function to create an embedding cache.

    Args:
        max_entries: Maximum number of entries
        max_memory_mb: Maximum memory usage in MB
        embedding_dim: Embedding vector dimension
        ttl_seconds: Time-to-live for entries in seconds
        compute_fn: Optional function to compute embeddings on cache miss
        **kwargs: Additional configuration options

    Returns:
        Configured EmbeddingCache instance
    """
    config = EmbeddingCacheConfig(
        max_entries=max_entries,
        max_memory_mb=max_memory_mb,
        embedding_dim=embedding_dim,
        ttl_seconds=ttl_seconds,
        **kwargs,
    )

    cache = EmbeddingCache(config)

    if compute_fn is not None:
        cache.set_compute_function(compute_fn)

    return cache
