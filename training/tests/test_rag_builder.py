"""Unit tests for Pinecone RAG builder."""

from dataclasses import dataclass, field
from pathlib import Path
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock the torch import for data_pipeline before importing
sys.modules["torch"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()


# Define DocumentChunk locally to avoid importing from data_pipeline
@dataclass
class DocumentChunk:
    """A chunk of a document."""

    doc_id: str
    chunk_id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Patch data_pipeline to use our local DocumentChunk
import training.data_pipeline  # noqa: E402

training.data_pipeline.DocumentChunk = DocumentChunk

from training.rag_builder import (  # noqa: E402
    ChunkingStrategy,
    IndexStats,
    RAGIndexManager,
    RetrievalOptimizer,
    SearchResult,
    VectorIndexBuilder,
)


@pytest.fixture
def rag_config():
    """Create RAG configuration."""
    return {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "chunk_strategy": "fixed",
        "index_name": "test-index",
        "namespace": "test",
        "num_neighbors": 10,
        "index_path": "./test_cache/rag_index",
        "pinecone": {
            "api_key": "test-api-key",
            "environment": "us-east-1",
            "cloud": "aws",
            "index_name": "test-index",
            "namespace": "test",
            "batch_size": 10,
        },
        "hybrid_search": {"enabled": True, "bm25_weight": 0.3, "dense_weight": 0.7},
    }


@pytest.fixture
def sample_chunks():
    """Create sample document chunks."""
    return [
        DocumentChunk(
            doc_id="doc1",
            chunk_id=0,
            text="MITRE ATT&CK framework provides a comprehensive knowledge base of adversary tactics and techniques.",
            metadata={"category": "mitre", "source": "cybersecurity_docs"},
        ),
        DocumentChunk(
            doc_id="doc1",
            chunk_id=1,
            text="Critical vulnerability CVE-2023-1234 allows remote code execution in vulnerable systems.",
            metadata={"category": "cyber_companies", "source": "vulnerability_report"},
        ),
        DocumentChunk(
            doc_id="doc2",
            chunk_id=0,
            text="Anomaly detection systems monitor network traffic for suspicious patterns.",
            metadata={"category": "wikipedia", "source": "general_knowledge"},
        ),
        DocumentChunk(
            doc_id="doc3",
            chunk_id=0,
            text="Low priority information about system administration best practices.",
            metadata={"category": "general", "source": "documentation"},
        ),
    ]


class TestChunkingStrategy:
    """Tests for chunking strategy."""

    def test_fixed_chunking(self):
        """Test fixed-size chunking."""
        config = {"chunk_size": 50, "chunk_overlap": 10, "chunk_strategy": "fixed"}
        strategy = ChunkingStrategy(config)

        text = "A" * 120  # 120 characters
        chunks = strategy.chunk_document(text, "test_doc")

        assert len(chunks) > 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert chunks[0].doc_id == "test_doc"
        assert chunks[0].chunk_id == 0

    def test_semantic_chunking(self):
        """Test semantic chunking."""
        config = {"chunk_size": 100, "chunk_overlap": 20, "chunk_strategy": "semantic"}
        strategy = ChunkingStrategy(config)

        text = "First sentence here. Second sentence follows. Third sentence ends."
        chunks = strategy.chunk_document(text, "semantic_doc")

        assert len(chunks) >= 1
        assert chunks[0].metadata.get("strategy") == "semantic"

    def test_recursive_chunking(self):
        """Test recursive chunking."""
        config = {"chunk_size": 100, "chunk_overlap": 10, "chunk_strategy": "recursive"}
        strategy = ChunkingStrategy(config)

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = strategy.chunk_document(text, "recursive_doc")

        assert len(chunks) >= 1
        assert chunks[0].metadata.get("strategy") == "recursive"

    def test_empty_text(self):
        """Test chunking with empty text."""
        config = {"chunk_size": 50, "chunk_overlap": 10, "chunk_strategy": "fixed"}
        strategy = ChunkingStrategy(config)

        chunks = strategy.chunk_document("", "empty_doc")
        assert len(chunks) == 0

    def test_small_text(self):
        """Test chunking with text smaller than chunk size."""
        config = {"chunk_size": 100, "chunk_overlap": 10, "chunk_strategy": "fixed"}
        strategy = ChunkingStrategy(config)

        text = "Small text"
        chunks = strategy.chunk_document(text, "small_doc")

        assert len(chunks) == 1
        assert chunks[0].text == text


class TestVectorIndexBuilder:
    """Tests for Pinecone vector index builder."""

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_initialization_without_transformers(self, rag_config):
        """Test builder initializes without sentence transformers."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index

            builder = VectorIndexBuilder(rag_config)

            assert builder.embedding_model is None
            assert builder.embedding_dim == 384

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_embed_text_fallback(self, rag_config):
        """Test embedding fallback to random when no model available."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = MagicMock()

            builder = VectorIndexBuilder(rag_config)
            embedding = builder._embed_text("test text")

            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_estimate_threat_level(self, rag_config):
        """Test threat level estimation."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = MagicMock()

            builder = VectorIndexBuilder(rag_config)

            assert builder._estimate_threat_level("critical vulnerability exploit") == "high"
            assert builder._estimate_threat_level("warning about suspicious activity") == "medium"
            assert builder._estimate_threat_level("normal system operation") == "low"

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    @patch("training.rag_builder.HAS_BM25", True)
    def test_build_index(self, rag_config, sample_chunks):
        """Test building index with sample chunks."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index
            mock_index.describe_index_stats.return_value = {"total_vector_count": 4}

            builder = VectorIndexBuilder(rag_config)
            stats = builder.build_index(iter(sample_chunks), batch_size=2)

            assert isinstance(stats, IndexStats)
            assert stats.total_chunks == 4
            assert stats.total_documents == 3  # doc1, doc2, doc3
            assert "mitre" in stats.domains
            assert builder.bm25_index is not None

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_upsert_batch(self, rag_config):
        """Test batch upsert to Pinecone."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index

            builder = VectorIndexBuilder(rag_config)
            vectors = [
                {"id": "v1", "values": [0.1] * 384, "metadata": {"text": "test"}},
                {"id": "v2", "values": [0.2] * 384, "metadata": {"text": "test2"}},
            ]
            builder._upsert_batch(vectors)

            mock_index.upsert.assert_called_once()
            call_args = mock_index.upsert.call_args
            assert call_args.kwargs["namespace"] == "test"

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_search(self, rag_config):
        """Test search functionality."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index

            mock_index.query.return_value = {
                "matches": [
                    {
                        "id": "doc1_0_abc123",
                        "score": 0.95,
                        "metadata": {
                            "doc_id": "doc1",
                            "chunk_id": 0,
                            "text": "MITRE ATT&CK framework",
                            "category": "mitre",
                        },
                    }
                ]
            }

            builder = VectorIndexBuilder(rag_config)
            builder.config["hybrid_search"]["enabled"] = False
            results = builder.search("ATT&CK framework", k=5)

            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].doc_id == "doc1"
            assert results[0].score == 0.95

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_search_by_category(self, rag_config):
        """Test search with category filter."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index
            mock_index.query.return_value = {"matches": []}

            builder = VectorIndexBuilder(rag_config)
            builder.config["hybrid_search"]["enabled"] = False
            builder.search_by_category("test query", "mitre", k=5)

            call_args = mock_index.query.call_args
            assert call_args.kwargs["filter"] == {"category": {"$eq": "mitre"}}

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_search_by_threat_level(self, rag_config):
        """Test search with threat level filter."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index
            mock_index.query.return_value = {"matches": []}

            builder = VectorIndexBuilder(rag_config)
            builder.config["hybrid_search"]["enabled"] = False
            builder.search_by_threat_level("vulnerability", "high", k=5)

            call_args = mock_index.query.call_args
            assert call_args.kwargs["filter"] == {"threat_level": {"$eq": "high"}}

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_save_and_load_index(self, rag_config):
        """Test saving and loading index metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir, patch("training.rag_builder.Pinecone") as mock_pc:
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = MagicMock()

            builder = VectorIndexBuilder(rag_config)
            builder.chunk_store = [{"id": "v1", "doc_id": "doc1", "chunk_id": 0, "text": "test text", "metadata": {}}]

            save_path = Path(tmp_dir) / "index"
            builder.save_index(save_path)

            assert (save_path / "chunks.json").exists()
            assert (save_path / "index_config.json").exists()

            # Load into new builder
            builder2 = VectorIndexBuilder(rag_config)
            builder2.load_index(save_path)

            assert len(builder2.chunk_store) == 1
            assert builder2.chunk_store[0]["doc_id"] == "doc1"

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_add_documents(self, rag_config, sample_chunks):
        """Test adding documents to existing index."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index

            builder = VectorIndexBuilder(rag_config)
            added = builder.add_documents(iter(sample_chunks[:2]), batch_size=10)

            assert added == 2
            assert len(builder.chunk_store) == 2

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_delete_namespace(self, rag_config):
        """Test deleting namespace."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index

            builder = VectorIndexBuilder(rag_config)
            builder.chunk_store = [{"id": "v1", "text": "test"}]

            result = builder.delete_namespace()

            assert result is True
            assert len(builder.chunk_store) == 0
            mock_index.delete.assert_called_once_with(delete_all=True, namespace="test")

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_get_index_stats(self, rag_config):
        """Test getting index statistics."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index
            mock_index.describe_index_stats.return_value = {
                "total_vector_count": 100,
                "dimension": 384,
                "namespaces": {"test": {"vector_count": 100}},
            }

            builder = VectorIndexBuilder(rag_config)
            builder.chunk_store = [{"id": "v1"}] * 5

            stats = builder.get_index_stats()

            assert stats["available"] is True
            assert stats["total_vector_count"] == 100
            assert stats["local_chunks"] == 5

    @patch("training.rag_builder.HAS_PINECONE", False)
    def test_pinecone_not_available(self, rag_config):
        """Test behavior when Pinecone is not available."""
        builder = VectorIndexBuilder(rag_config)

        assert builder.is_available is False
        results = builder.search("test query")
        assert results == []

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_is_available_property(self, rag_config):
        """Test is_available property."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = MagicMock()

            builder = VectorIndexBuilder(rag_config)
            assert builder.is_available is True


class TestRetrievalOptimizer:
    """Tests for retrieval optimizer."""

    def test_initialization(self, rag_config):
        """Test optimizer initialization."""
        optimizer = RetrievalOptimizer(rag_config)
        assert optimizer is not None

    def test_query_expansion(self, rag_config):
        """Test query expansion with synonyms."""
        optimizer = RetrievalOptimizer(rag_config)

        expansions = optimizer.query_expansion("attack defense")

        assert len(expansions) >= 1
        assert "attack defense" in expansions
        # Check that synonyms are included
        assert any("threat" in exp or "breach" in exp for exp in expansions) or len(expansions) > 1

    def test_query_expansion_no_synonyms(self, rag_config):
        """Test query expansion with no matching synonyms."""
        optimizer = RetrievalOptimizer(rag_config)

        expansions = optimizer.query_expansion("random query")

        assert len(expansions) == 1
        assert expansions[0] == "random query"


class TestRAGIndexManager:
    """Tests for RAG index manager."""

    @pytest.fixture
    def temp_config_file(self, rag_config, tmp_path):
        """Create temporary config file."""
        full_config = {"rag": rag_config, "data": {"primus_seed": {"categories": ["mitre", "cyber_companies"]}}}
        config_path = tmp_path / "config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(full_config, f)
        return str(config_path)

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_manager_initialization(self, temp_config_file):
        """Test manager initialization."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = MagicMock()

            manager = RAGIndexManager(temp_config_file)

            assert manager is not None
            assert manager.chunking_strategy is not None

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_save_all_indices(self, temp_config_file, tmp_path):
        """Test saving all indices."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = MagicMock()

            manager = RAGIndexManager(temp_config_file)

            # Create index with tmp_path as index_path
            index_config = manager.rag_config.copy()
            index_config["index_path"] = str(tmp_path / "test_index")
            test_index = VectorIndexBuilder(index_config)
            test_index.chunk_store = [{"id": "v1", "doc_id": "doc1", "chunk_id": 0, "text": "test", "metadata": {}}]
            manager.indices = {"test": test_index}

            # Save indices (method doesn't take arguments)
            manager.save_all_indices()

            # Check that files were saved
            assert (tmp_path / "test_index" / "chunks.json").exists()


class TestIntegration:
    """Integration tests for RAG builder components."""

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    @patch("training.rag_builder.HAS_BM25", True)
    def test_full_indexing_and_search_pipeline(self, rag_config, sample_chunks):
        """Test complete indexing and search pipeline."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index
            mock_index.describe_index_stats.return_value = {"total_vector_count": 4}
            mock_index.query.return_value = {
                "matches": [
                    {
                        "id": "doc1_0_abc123",
                        "score": 0.95,
                        "metadata": {
                            "doc_id": "doc1",
                            "chunk_id": 0,
                            "text": "MITRE ATT&CK framework provides a comprehensive knowledge base",
                            "category": "mitre",
                            "threat_level": "high",
                        },
                    },
                    {
                        "id": "doc1_1_def456",
                        "score": 0.85,
                        "metadata": {
                            "doc_id": "doc1",
                            "chunk_id": 1,
                            "text": "Critical vulnerability CVE-2023-1234",
                            "category": "cyber_companies",
                            "threat_level": "high",
                        },
                    },
                ]
            }

            builder = VectorIndexBuilder(rag_config)

            # Build index
            stats = builder.build_index(iter(sample_chunks), batch_size=2)
            assert stats.total_chunks == 4

            # Perform search
            results = builder.search("cybersecurity framework", k=2)
            assert len(results) == 2
            assert results[0].score > results[1].score

    @patch("training.rag_builder.HAS_PINECONE", True)
    @patch("training.rag_builder.HAS_SENTENCE_TRANSFORMERS", False)
    def test_metadata_filtering_integration(self, rag_config):
        """Test metadata filtering works correctly."""
        with patch("training.rag_builder.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.list_indexes.return_value = []
            mock_pc.return_value.Index.return_value = mock_index

            # Mock different results based on filter
            def query_side_effect(**kwargs):
                filter_val = kwargs.get("filter", {})
                if filter_val.get("category", {}).get("$eq") == "mitre":
                    return {
                        "matches": [
                            {
                                "id": "mitre_doc",
                                "score": 0.9,
                                "metadata": {
                                    "doc_id": "mitre1",
                                    "chunk_id": 0,
                                    "text": "MITRE specific content",
                                    "category": "mitre",
                                },
                            }
                        ]
                    }
                return {"matches": []}

            mock_index.query.side_effect = query_side_effect

            builder = VectorIndexBuilder(rag_config)
            builder.config["hybrid_search"]["enabled"] = False

            mitre_results = builder.search_by_category("framework", "mitre")
            assert len(mitre_results) == 1
            assert mitre_results[0].metadata["category"] == "mitre"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
