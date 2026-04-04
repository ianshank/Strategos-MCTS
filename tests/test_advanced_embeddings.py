"""
Tests for Advanced Embedding System
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from training.advanced_embeddings import (
    BGEEmbedder,
    CohereEmbedder,
    EmbedderFactory,
    EnsembleEmbedder,
    OpenAIEmbedder,
    RandomEmbedder,
    SentenceTransformerEmbedder,
    VoyageEmbedder,
)


class TestBaseEmbedder:
    """Test base embedder functionality."""

    def test_cache_key_generation(self):
        """Test cache key generation is consistent."""
        config = {"model": "test-model", "dimension": 1024}
        embedder = RandomEmbedder(config)

        key1 = embedder._get_cache_key("test text")
        key2 = embedder._get_cache_key("test text")
        key3 = embedder._get_cache_key("different text")

        assert key1 == key2, "Cache keys should be consistent for same text"
        assert key1 != key3, "Cache keys should differ for different texts"

    def test_caching_enabled(self):
        """Test caching saves and loads embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model": "test-model", "dimension": 128, "cache_dir": tmpdir, "cache_enabled": True}
            embedder = RandomEmbedder(config)

            texts = ["test text 1", "test text 2"]

            # First call - should generate new embeddings
            result1 = embedder.embed_with_cache(texts)
            assert not result1.cached, "First call should not be cached"
            assert result1.metadata["cache_hit_rate"] == 0.0

            # Second call - should load from cache
            result2 = embedder.embed_with_cache(texts)
            assert result2.cached, "Second call should be fully cached"
            assert result2.metadata["cache_hit_rate"] == 1.0

            # Verify embeddings are the same
            np.testing.assert_array_equal(result1.embeddings, result2.embeddings)

    def test_caching_disabled(self):
        """Test that caching can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model": "test-model", "dimension": 128, "cache_dir": tmpdir, "cache_enabled": False}
            embedder = RandomEmbedder(config)

            texts = ["test text 1"]
            embedder.embed_with_cache(texts)

            # Check that cache directory is empty
            cache_files = list(Path(tmpdir).glob("*.npy"))
            assert len(cache_files) == 0, "No cache files should be created when caching is disabled"


class TestVoyageEmbedder:
    """Test Voyage AI embedder."""

    @pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY not set")
    def test_voyage_embedder_initialization(self):
        """Test Voyage embedder initialization."""
        config = {"model": "voyage-large-2-instruct", "api_key": os.environ.get("VOYAGE_API_KEY")}
        embedder = VoyageEmbedder(config)

        assert embedder.is_available(), "Voyage embedder should be available with valid API key"
        assert embedder.dimension == 1024, "voyage-large-2-instruct should have dimension 1024"

    @pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY not set")
    def test_voyage_embedding(self):
        """Test Voyage embedding generation."""
        config = {"model": "voyage-large-2-instruct", "api_key": os.environ.get("VOYAGE_API_KEY")}
        embedder = VoyageEmbedder(config)

        texts = ["This is a test.", "Another test sentence."]
        embeddings = embedder.embed(texts)

        assert embeddings.shape == (2, 1024), "Should return correct shape"
        assert embeddings.dtype == np.float32, "Should return float32 arrays"

    def test_voyage_unavailable_without_key(self):
        """Test Voyage embedder is unavailable without API key."""
        config = {"model": "voyage-large-2-instruct"}
        embedder = VoyageEmbedder(config)

        if not os.environ.get("VOYAGE_API_KEY"):
            assert not embedder.is_available(), "Should not be available without API key"


class TestCohereEmbedder:
    """Test Cohere embedder."""

    @pytest.mark.skipif(not os.environ.get("COHERE_API_KEY"), reason="COHERE_API_KEY not set")
    def test_cohere_embedder_initialization(self):
        """Test Cohere embedder initialization."""
        config = {"model": "embed-english-v3.0", "dimension": 1024, "api_key": os.environ.get("COHERE_API_KEY")}
        embedder = CohereEmbedder(config)

        assert embedder.is_available(), "Cohere embedder should be available with valid API key"
        assert embedder.dimension == 1024, "Should use specified dimension"

    @pytest.mark.skipif(not os.environ.get("COHERE_API_KEY"), reason="COHERE_API_KEY not set")
    def test_cohere_matryoshka_compression(self):
        """Test Cohere Matryoshka dimension compression."""
        config = {"model": "embed-english-v3.0", "dimension": 512, "api_key": os.environ.get("COHERE_API_KEY")}
        embedder = CohereEmbedder(config)

        texts = ["This is a test."]
        embeddings = embedder.embed(texts)

        assert embeddings.shape == (1, 512), "Should return compressed dimension"

    def test_cohere_dimension_validation(self):
        """Test Cohere dimension validation."""
        config = {"model": "embed-english-v3.0", "dimension": 999}  # Invalid dimension
        embedder = CohereEmbedder(config)

        # Should default to 1024
        assert embedder.dimension == 1024, "Should default to 1024 for invalid dimension"


class TestOpenAIEmbedder:
    """Test OpenAI embedder."""

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_openai_embedder_initialization(self):
        """Test OpenAI embedder initialization."""
        config = {"model": "text-embedding-3-large", "api_key": os.environ.get("OPENAI_API_KEY")}
        embedder = OpenAIEmbedder(config)

        assert embedder.is_available(), "OpenAI embedder should be available with valid API key"
        assert embedder.dimension == 3072, "text-embedding-3-large should have dimension 3072"

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_openai_dimension_reduction(self):
        """Test OpenAI dimension reduction."""
        config = {"model": "text-embedding-3-large", "dimension": 1024, "api_key": os.environ.get("OPENAI_API_KEY")}
        embedder = OpenAIEmbedder(config)

        texts = ["This is a test."]
        embeddings = embedder.embed(texts)

        assert embeddings.shape == (1, 1024), "Should return reduced dimension"

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_openai_embedding(self):
        """Test OpenAI embedding generation."""
        config = {"model": "text-embedding-3-small", "api_key": os.environ.get("OPENAI_API_KEY")}
        embedder = OpenAIEmbedder(config)

        texts = ["This is a test.", "Another test sentence."]
        embeddings = embedder.embed(texts)

        assert embeddings.shape[0] == 2, "Should embed all texts"
        assert embeddings.dtype == np.float32, "Should return float32 arrays"


class TestBGEEmbedder:
    """Test BGE embedder."""

    def test_bge_embedder_initialization(self):
        """Test BGE embedder initialization."""
        config = {"model": "BAAI/bge-small-en-v1.5"}  # Use small model for faster testing
        embedder = BGEEmbedder(config)

        if embedder.is_available():
            assert embedder.dimension > 0, "Should have positive dimension"

    def test_bge_embedding(self):
        """Test BGE embedding generation."""
        config = {"model": "BAAI/bge-small-en-v1.5"}
        embedder = BGEEmbedder(config)

        if not embedder.is_available():
            pytest.skip("BGE model not available")

        texts = ["This is a test.", "Another test sentence."]
        embeddings = embedder.embed(texts)

        assert embeddings.shape[0] == 2, "Should embed all texts"
        assert embeddings.dtype == np.float32, "Should return float32 arrays"


class TestSentenceTransformerEmbedder:
    """Test Sentence-Transformer embedder."""

    def test_sentence_transformer_initialization(self):
        """Test Sentence-Transformer embedder initialization."""
        config = {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        embedder = SentenceTransformerEmbedder(config)

        if embedder.is_available():
            assert embedder.dimension == 384, "all-MiniLM-L6-v2 should have dimension 384"

    def test_sentence_transformer_embedding(self):
        """Test Sentence-Transformer embedding generation."""
        config = {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        embedder = SentenceTransformerEmbedder(config)

        if not embedder.is_available():
            pytest.skip("Sentence-Transformers not available")

        texts = ["This is a test.", "Another test sentence."]
        embeddings = embedder.embed(texts)

        assert embeddings.shape == (2, 384), "Should return correct shape"
        assert embeddings.dtype == np.float32, "Should return float32 arrays"


class TestEnsembleEmbedder:
    """Test ensemble embedder."""

    def test_ensemble_concat(self):
        """Test ensemble with concatenation."""
        embedder1 = RandomEmbedder({"dimension": 100})
        embedder2 = RandomEmbedder({"dimension": 200})

        config = {"combination_method": "concat"}
        ensemble = EnsembleEmbedder(config, [embedder1, embedder2])

        assert ensemble.dimension == 300, "Concatenated dimension should be sum of embedders"

        texts = ["test"]
        embeddings = ensemble.embed(texts)
        assert embeddings.shape == (1, 300), "Should return concatenated embeddings"

    def test_ensemble_mean(self):
        """Test ensemble with mean aggregation."""
        embedder1 = RandomEmbedder({"dimension": 100})
        embedder2 = RandomEmbedder({"dimension": 100})

        config = {"combination_method": "mean"}
        ensemble = EnsembleEmbedder(config, [embedder1, embedder2])

        assert ensemble.dimension == 100, "Mean dimension should match embedders"

        texts = ["test"]
        embeddings = ensemble.embed(texts)
        assert embeddings.shape == (1, 100), "Should return mean embeddings"

    def test_ensemble_weighted(self):
        """Test ensemble with weighted aggregation."""
        embedder1 = RandomEmbedder({"dimension": 100})
        embedder2 = RandomEmbedder({"dimension": 100})

        config = {"combination_method": "weighted", "weights": [0.7, 0.3]}
        ensemble = EnsembleEmbedder(config, [embedder1, embedder2])

        texts = ["test"]
        embeddings = ensemble.embed(texts)
        assert embeddings.shape == (1, 100), "Should return weighted embeddings"

    def test_ensemble_add_embedder(self):
        """Test adding embedder to ensemble."""
        config = {"combination_method": "concat"}
        ensemble = EnsembleEmbedder(config, [])

        assert ensemble.dimension == 1024, "Default dimension before adding embedders"

        embedder1 = RandomEmbedder({"dimension": 100})
        ensemble.add_embedder(embedder1)

        assert ensemble.dimension == 100, "Should update dimension after adding embedder"


class TestEmbedderFactory:
    """Test embedder factory."""

    def test_create_voyage_embedder(self):
        """Test factory creates Voyage embedder."""
        config = {"model": "voyage-large-2-instruct"}
        embedder = EmbedderFactory.create_embedder(config)
        assert isinstance(embedder, VoyageEmbedder), "Should create VoyageEmbedder"

    def test_create_cohere_embedder(self):
        """Test factory creates Cohere embedder."""
        config = {"model": "embed-english-v3.0", "provider": "cohere"}
        embedder = EmbedderFactory.create_embedder(config)
        assert isinstance(embedder, CohereEmbedder), "Should create CohereEmbedder"

    def test_create_openai_embedder(self):
        """Test factory creates OpenAI embedder."""
        config = {"model": "text-embedding-3-large"}
        embedder = EmbedderFactory.create_embedder(config)
        assert isinstance(embedder, OpenAIEmbedder), "Should create OpenAIEmbedder"

    def test_create_bge_embedder(self):
        """Test factory creates BGE embedder."""
        config = {"model": "BAAI/bge-large-en-v1.5"}
        embedder = EmbedderFactory.create_embedder(config)
        assert isinstance(embedder, BGEEmbedder), "Should create BGEEmbedder"

    def test_create_sentence_transformer_embedder(self):
        """Test factory creates Sentence-Transformer embedder."""
        config = {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        embedder = EmbedderFactory.create_embedder(config)
        assert isinstance(embedder, SentenceTransformerEmbedder), "Should create SentenceTransformerEmbedder"

    def test_create_with_fallback(self):
        """Test factory with fallback."""
        configs = [
            {"model": "nonexistent-model"},
            {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        ]
        embedder = EmbedderFactory.create_with_fallback(configs)

        # Should successfully create an embedder (either first or fallback)
        assert embedder is not None, "Should create an embedder"
        assert embedder.is_available() or isinstance(
            embedder, RandomEmbedder
        ), "Should have available embedder or random fallback"


class TestRandomEmbedder:
    """Test random embedder (for testing/fallback)."""

    def test_random_embedder(self):
        """Test random embedder generates consistent shapes."""
        config = {"dimension": 128}
        embedder = RandomEmbedder(config)

        assert embedder.is_available(), "Random embedder should always be available"
        assert embedder.dimension == 128, "Should use specified dimension"

        texts = ["test1", "test2", "test3"]
        embeddings = embedder.embed(texts)

        assert embeddings.shape == (3, 128), "Should return correct shape"
        assert embeddings.dtype == np.float32, "Should return float32 arrays"


class TestAsyncEmbedding:
    """Test async embedding functionality."""

    @pytest.mark.asyncio
    async def test_async_embed(self):
        """Test async embedding."""
        from training.advanced_embeddings import AsyncEmbedder

        config = {"dimension": 128}
        embedder = RandomEmbedder(config)
        async_embedder = AsyncEmbedder(embedder)

        texts = ["test1", "test2"]
        result = await async_embedder.embed_async(texts)

        assert result.embeddings.shape == (2, 128), "Should return correct shape"
        assert result.model == "random", "Should track model name"

    @pytest.mark.asyncio
    async def test_async_batch_embed(self):
        """Test async batch embedding."""
        from training.advanced_embeddings import AsyncEmbedder

        config = {"dimension": 128}
        embedder = RandomEmbedder(config)
        async_embedder = AsyncEmbedder(embedder)

        batches = [["test1", "test2"], ["test3", "test4"], ["test5"]]
        results = await async_embedder.embed_batch_async(batches)

        assert len(results) == 3, "Should return results for all batches"
        assert results[0].embeddings.shape == (2, 128), "First batch should have 2 embeddings"
        assert results[2].embeddings.shape == (1, 128), "Last batch should have 1 embedding"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
