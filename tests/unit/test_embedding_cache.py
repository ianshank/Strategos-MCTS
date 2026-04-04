"""
Tests for embedding cache module.

Tests EmbeddingCacheConfig, CacheEntry, CacheStats,
EmbeddingCache operations, and factory function.
"""

import time

import numpy as np
import pytest

from src.performance.embedding_cache import (
    CacheEntry,
    CacheStats,
    EmbeddingCache,
    EmbeddingCacheConfig,
    create_embedding_cache,
)


@pytest.mark.unit
class TestEmbeddingCacheConfig:
    def test_defaults(self):
        cfg = EmbeddingCacheConfig()
        assert cfg.max_entries == 10000
        assert cfg.embedding_dim == 768
        assert cfg.ttl_seconds == 3600.0

    def test_estimated_memory_per_entry(self):
        cfg = EmbeddingCacheConfig(embedding_dim=768, bytes_per_float=4)
        size = cfg.estimated_memory_per_entry_bytes()
        assert size == 768 * 4 + 100

    def test_max_entries_for_memory(self):
        cfg = EmbeddingCacheConfig(max_entries=100000, max_memory_mb=1.0, embedding_dim=100)
        limit = cfg.max_entries_for_memory()
        assert limit < 100000  # Memory limits it


@pytest.mark.unit
class TestCacheEntry:
    def test_not_expired(self):
        entry = CacheEntry(embedding=np.zeros(10))
        assert not entry.is_expired(3600.0)

    def test_expired(self):
        entry = CacheEntry(embedding=np.zeros(10), created_at=time.time() - 100)
        assert entry.is_expired(50.0)


@pytest.mark.unit
class TestCacheStats:
    def test_hit_rate_empty(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate(self):
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == pytest.approx(0.75)

    def test_to_dict(self):
        stats = CacheStats(hits=5, misses=2, evictions=1)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 2
        assert d["evictions"] == 1
        assert "hit_rate" in d


@pytest.mark.unit
class TestEmbeddingCache:
    def test_init_default(self):
        cache = EmbeddingCache()
        assert cache.size == 0
        assert cache.hit_rate == 0.0

    def test_put_and_get(self):
        cache = EmbeddingCache()
        emb = np.array([1.0, 2.0, 3.0])
        cache.put("hello", emb)
        result = cache.get("hello")
        assert result is not None
        assert np.array_equal(result, emb)

    def test_get_miss(self):
        cache = EmbeddingCache()
        assert cache.get("nonexistent") is None

    def test_cache_hit_tracking(self):
        cache = EmbeddingCache()
        cache.put("test", np.zeros(3))
        cache.get("test")
        cache.get("other")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_lru_eviction(self):
        cfg = EmbeddingCacheConfig(max_entries=2, max_memory_mb=1000.0)
        cache = EmbeddingCache(cfg)
        cache.put("a", np.zeros(3))
        cache.put("b", np.zeros(3))
        cache.put("c", np.zeros(3))  # Evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None
        assert cache.get_stats()["evictions"] >= 1

    def test_ttl_expiration(self):
        cfg = EmbeddingCacheConfig(ttl_seconds=0.01)  # Very short TTL
        cache = EmbeddingCache(cfg)
        cache.put("test", np.zeros(3))
        time.sleep(0.02)
        assert cache.get("test") is None

    def test_get_or_compute(self):
        cache = EmbeddingCache()
        cache.set_compute_function(lambda text: np.array([len(text)]))
        result = cache.get_or_compute("hello")
        assert result is not None
        # Second call should hit cache
        result2 = cache.get_or_compute("hello")
        assert np.array_equal(result, result2)

    def test_get_or_compute_no_fn(self):
        cache = EmbeddingCache()
        with pytest.raises(ValueError, match="No compute function"):
            cache.get_or_compute("hello")

    def test_get_batch(self):
        cache = EmbeddingCache()
        cache.set_compute_function(lambda text: np.array([len(text)]))
        # Pre-populate one
        cache.put("hello", np.array([5]))
        results = cache.get_batch(["hello", "world"])
        assert len(results) == 2
        assert np.array_equal(results[0], np.array([5]))
        assert np.array_equal(results[1], np.array([5]))  # "world" also length 5

    def test_clear(self):
        cache = EmbeddingCache()
        cache.put("test", np.zeros(3))
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0

    def test_get_stats(self):
        cache = EmbeddingCache()
        stats = cache.get_stats()
        assert "entries_count" in stats
        assert "estimated_memory_mb" in stats

    def test_size_property(self):
        cache = EmbeddingCache()
        assert cache.size == 0
        cache.put("a", np.zeros(3))
        assert cache.size == 1


@pytest.mark.unit
class TestCreateEmbeddingCache:
    def test_factory_defaults(self):
        cache = create_embedding_cache()
        assert isinstance(cache, EmbeddingCache)

    def test_factory_with_compute_fn(self):
        def fn(text):
            return np.zeros(10)

        cache = create_embedding_cache(compute_fn=fn)
        assert cache._compute_fn is fn

    def test_factory_custom_params(self):
        cache = create_embedding_cache(
            max_entries=100,
            embedding_dim=256,
            ttl_seconds=60.0,
        )
        assert cache.config.max_entries == 100
        assert cache.config.embedding_dim == 256
