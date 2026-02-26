"""
Unit tests for FAISS vector store.

Tests initialization, document management, search, and persistence.

Based on: NEXT_STEPS_PLAN.md Phase 3.4
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("aioboto3", reason="aioboto3 required for storage tests")

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Check FAISS availability
# =============================================================================

try:
    import faiss  # noqa: F401 - availability check
    import numpy as np  # noqa: F401 - availability check

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

skip_if_no_faiss = pytest.mark.skipif(
    not FAISS_AVAILABLE,
    reason="FAISS not installed",
)


# =============================================================================
# Mock FAISS for tests without the actual library
# =============================================================================


@pytest.fixture
def mock_faiss():
    """Create mock FAISS module for testing without FAISS installed."""
    mock_index = MagicMock()
    mock_index.is_trained = True
    mock_index.add = MagicMock()
    mock_index.search = MagicMock(
        return_value=(
            [[0.9, 0.8, 0.7]],  # distances
            [[0, 1, 2]],  # indices
        )
    )

    mock_faiss_module = MagicMock()
    mock_faiss_module.IndexFlatIP.return_value = mock_index
    mock_faiss_module.IndexFlatL2.return_value = mock_index
    mock_faiss_module.IndexIVFFlat.return_value = mock_index
    mock_faiss_module.IndexHNSWFlat.return_value = mock_index
    mock_faiss_module.get_num_gpus.return_value = 0
    mock_faiss_module.write_index = MagicMock()
    mock_faiss_module.read_index = MagicMock(return_value=mock_index)

    return mock_faiss_module, mock_index


@pytest.fixture
def mock_sentence_transformer():
    """Create mock SentenceTransformer."""
    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        return_value=MagicMock(
            shape=(384,),
            astype=MagicMock(
                return_value=MagicMock(
                    reshape=MagicMock(return_value=[[0.1] * 384]),
                )
            ),
        )
    )
    return mock_model


# =============================================================================
# Module Import Tests
# =============================================================================


class TestFAISSModuleImport:
    """Tests for FAISS module availability detection."""

    def test_faiss_availability_flag_exists(self):
        """Test module has FAISS availability flag."""
        from src.storage import faiss_store

        assert hasattr(faiss_store, "_HAS_FAISS")
        assert isinstance(faiss_store._HAS_FAISS, bool)

    def test_faiss_gpu_flag_exists(self):
        """Test module has GPU availability flag."""
        from src.storage import faiss_store

        assert hasattr(faiss_store, "_HAS_FAISS_GPU")
        assert isinstance(faiss_store._HAS_FAISS_GPU, bool)


# =============================================================================
# FAISSStoreDefaults Tests
# =============================================================================


class TestFAISSStoreDefaults:
    """Tests for FAISS store default configuration."""

    def test_defaults_have_expected_values(self):
        """Test default configuration values are set."""
        from src.storage.faiss_store import FAISSStoreDefaults

        assert FAISSStoreDefaults.MODEL_NAME == "all-MiniLM-L6-v2"
        assert FAISSStoreDefaults.EMBEDDING_DIM == 384
        assert FAISSStoreDefaults.DEFAULT_TOP_K == 5
        assert FAISSStoreDefaults.MAX_TOP_K == 10000


class TestIndexType:
    """Tests for index type constants."""

    def test_index_types_defined(self):
        """Test all index types are defined."""
        from src.storage.faiss_store import IndexType

        assert IndexType.FLAT_L2 == "flat_l2"
        assert IndexType.FLAT_IP == "flat_ip"
        assert IndexType.IVF_FLAT == "ivf_flat"
        assert IndexType.IVF_PQ == "ivf_pq"
        assert IndexType.HNSW == "hnsw"


# =============================================================================
# FAISSDocument Tests
# =============================================================================


class TestFAISSDocument:
    """Tests for FAISSDocument dataclass."""

    def test_document_creation(self):
        """Test document can be created."""
        from src.storage.faiss_store import FAISSDocument

        doc = FAISSDocument(
            id="doc-001",
            content="Test content",
            embedding_id=0,
            metadata={"source": "test"},
        )

        assert doc.id == "doc-001"
        assert doc.content == "Test content"
        assert doc.embedding_id == 0
        assert doc.metadata["source"] == "test"

    def test_document_hash_and_equality(self):
        """Test document hashing and equality."""
        from src.storage.faiss_store import FAISSDocument

        doc1 = FAISSDocument(id="doc-001", content="Content 1", embedding_id=0)
        doc2 = FAISSDocument(id="doc-001", content="Content 2", embedding_id=1)
        doc3 = FAISSDocument(id="doc-002", content="Content 1", embedding_id=0)

        # Same ID means equal
        assert doc1 == doc2
        assert hash(doc1) == hash(doc2)

        # Different ID means not equal
        assert doc1 != doc3


# =============================================================================
# FAISSVectorStore Initialization Tests
# =============================================================================


class TestFAISSVectorStoreInitialization:
    """Tests for FAISS store initialization."""

    def test_store_creation_with_defaults(self):
        """Test store can be created with default settings."""
        from src.storage.faiss_store import FAISSVectorStore

        store = FAISSVectorStore()

        assert store is not None
        assert store._is_initialized is False

    def test_store_creation_with_custom_settings(self):
        """Test store accepts custom settings."""
        from src.storage.faiss_store import FAISSVectorStore, IndexType

        store = FAISSVectorStore(
            model_name="custom-model",
            embedding_dim=512,
            index_type=IndexType.HNSW,
            use_gpu=False,
        )

        assert store._model_name == "custom-model"
        assert store._embedding_dim == 512
        assert store._index_type == IndexType.HNSW
        assert store._use_gpu is False

    @skip_if_no_faiss
    def test_initialization_requires_dependencies(self):
        """Test initialization checks dependencies."""
        from src.storage.faiss_store import FAISSVectorStore

        store = FAISSVectorStore()
        # This will fail or succeed based on dependencies
        result = store.initialize()

        # Result depends on sentence-transformers being installed
        assert isinstance(result, bool)


# =============================================================================
# FAISSVectorStore Document Operations Tests (with mocks)
# =============================================================================


class TestFAISSVectorStoreDocumentOps:
    """Tests for document operations with mocked dependencies."""

    def test_add_documents_returns_count(self, mock_faiss, mock_sentence_transformer):
        """Test add_documents returns count of added documents."""
        from src.storage.faiss_store import FAISSVectorStore

        with patch.dict("sys.modules", {"faiss": mock_faiss[0]}):
            with patch("src.storage.faiss_store.faiss", mock_faiss[0]):
                with patch("src.storage.faiss_store._HAS_FAISS", True):
                    with patch("src.storage.faiss_store._HAS_NUMPY", True):
                        with patch("src.storage.faiss_store._HAS_SENTENCE_TRANSFORMERS", True):
                            store = FAISSVectorStore()
                            store._model = mock_sentence_transformer
                            store._index = mock_faiss[1]
                            store._is_initialized = True

                            # Mock numpy operations
                            import numpy as np

                            mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384]).astype("float32")

                            added = store.add_documents(
                                [
                                    {"content": "Test document 1"},
                                    {"content": "Test document 2"},
                                ]
                            )

                            assert added == 2

    def test_add_empty_documents_returns_zero(self):
        """Test adding empty documents returns 0."""
        from src.storage.faiss_store import FAISSVectorStore

        store = FAISSVectorStore()
        # Not initialized
        added = store.add_documents([])

        assert added == 0

    def test_add_documents_skips_empty_content(self, mock_faiss, mock_sentence_transformer):
        """Test documents with empty content are skipped."""
        from src.storage.faiss_store import FAISSVectorStore

        with patch("src.storage.faiss_store.faiss", mock_faiss[0]):
            with patch("src.storage.faiss_store._HAS_FAISS", True):
                with patch("src.storage.faiss_store._HAS_NUMPY", True):
                    with patch("src.storage.faiss_store._HAS_SENTENCE_TRANSFORMERS", True):
                        store = FAISSVectorStore()
                        store._model = mock_sentence_transformer
                        store._index = mock_faiss[1]
                        store._is_initialized = True

                        import numpy as np

                        mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384]).astype("float32")

                        added = store.add_documents(
                            [
                                {"content": "Valid content"},
                                {"content": ""},  # Empty
                                {"content": None},  # None
                            ]
                        )

                        assert added == 1


# =============================================================================
# FAISSVectorStore Search Tests
# =============================================================================


class TestFAISSVectorStoreSearch:
    """Tests for search operations."""

    def test_search_returns_empty_when_not_initialized(self):
        """Test search returns empty list when not initialized."""
        from src.storage.faiss_store import FAISSVectorStore

        store = FAISSVectorStore()
        results = store.search("query")

        assert results == []

    def test_search_returns_empty_for_empty_query(self, mock_faiss):
        """Test search returns empty for empty query."""
        from src.storage.faiss_store import FAISSVectorStore

        with patch("src.storage.faiss_store.faiss", mock_faiss[0]):
            store = FAISSVectorStore()
            store._is_initialized = True
            store._index = mock_faiss[1]
            store._documents = {0: MagicMock()}

            results = store.search("")

            assert results == []


# =============================================================================
# FAISSVectorStore Properties Tests
# =============================================================================


class TestFAISSVectorStoreProperties:
    """Tests for store properties."""

    def test_is_available_property(self):
        """Test is_available property."""
        from src.storage.faiss_store import FAISSVectorStore

        store = FAISSVectorStore()
        assert store.is_available is False

        store._is_initialized = True
        assert store.is_available is True

    def test_document_count_property(self):
        """Test document_count property."""
        from src.storage.faiss_store import FAISSDocument, FAISSVectorStore

        store = FAISSVectorStore()
        assert store.document_count == 0

        store._documents = {
            0: FAISSDocument(id="1", content="a", embedding_id=0),
            1: FAISSDocument(id="2", content="b", embedding_id=1),
        }
        assert store.document_count == 2

    def test_index_type_property(self):
        """Test index_type property."""
        from src.storage.faiss_store import FAISSVectorStore, IndexType

        store = FAISSVectorStore(index_type=IndexType.HNSW)
        assert store.index_type == IndexType.HNSW

    def test_gpu_enabled_property(self):
        """Test gpu_enabled property."""
        from src.storage.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(use_gpu=True)
        # Will be False unless GPU is actually available
        assert isinstance(store.gpu_enabled, bool)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateFaissStore:
    """Tests for factory function."""

    def test_factory_returns_none_without_faiss(self):
        """Test factory returns None when FAISS not installed."""
        from src.storage.faiss_store import create_faiss_store

        with patch("src.storage.faiss_store._HAS_FAISS", False):
            store = create_faiss_store(auto_initialize=False)

            assert store is None

    def test_factory_accepts_parameters(self):
        """Test factory accepts configuration parameters."""
        from src.storage.faiss_store import IndexType, create_faiss_store

        with patch("src.storage.faiss_store._HAS_FAISS", True):
            store = create_faiss_store(
                model_name="test-model",
                index_type=IndexType.FLAT_L2,
                use_gpu=False,
                auto_initialize=False,
            )

            if store is not None:
                assert store._model_name == "test-model"
                assert store._index_type == IndexType.FLAT_L2


# =============================================================================
# Persistence Tests
# =============================================================================


class TestFAISSPersistence:
    """Tests for index persistence."""

    def test_persist_dir_stored(self):
        """Test persist directory is stored."""
        from src.storage.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(persist_dir=Path("/tmp/faiss"))

        assert store._persist_dir == Path("/tmp/faiss")

    @skip_if_no_faiss
    def test_load_if_exists_returns_false_for_missing_files(self):
        """Test _load_if_exists returns False when files don't exist."""
        from src.storage.faiss_store import FAISSVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSVectorStore(persist_dir=Path(tmpdir))
            result = store._load_if_exists()

            assert result is False


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestFAISSThreadSafety:
    """Tests for thread safety."""

    def test_lock_exists(self):
        """Test store has a lock for thread safety."""
        from src.storage.faiss_store import FAISSVectorStore

        store = FAISSVectorStore()

        assert hasattr(store, "_lock")
        assert store._lock is not None


# =============================================================================
# Remove and Rebuild Tests
# =============================================================================


class TestFAISSRemoveRebuild:
    """Tests for remove and rebuild operations."""

    def test_remove_document_not_found(self):
        """Test removing non-existent document returns False."""
        from src.storage.faiss_store import FAISSVectorStore

        store = FAISSVectorStore()
        result = store.remove_document("non-existent")

        assert result is False

    def test_remove_document_found(self):
        """Test removing existing document returns True."""
        from src.storage.faiss_store import FAISSDocument, FAISSVectorStore

        store = FAISSVectorStore()
        store._document_id_map = {"doc-001": 0}
        store._documents = {0: FAISSDocument(id="doc-001", content="test", embedding_id=0)}

        result = store.remove_document("doc-001")

        assert result is True
        assert "doc-001" not in store._document_id_map
        assert 0 not in store._documents

    def test_clear_resets_state(self, mock_faiss):
        """Test clear resets all state."""
        from src.storage.faiss_store import FAISSDocument, FAISSVectorStore

        with patch("src.storage.faiss_store.faiss", mock_faiss[0]):
            store = FAISSVectorStore()
            store._documents = {0: FAISSDocument(id="1", content="a", embedding_id=0)}
            store._document_id_map = {"1": 0}
            store._next_id = 5
            store._index = mock_faiss[1]

            store.clear()

            assert len(store._documents) == 0
            assert len(store._document_id_map) == 0
            assert store._next_id == 0
