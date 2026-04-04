"""Unit tests for src/storage/faiss_store.py."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from src.storage.faiss_store import (
    FAISSDocument,
    FAISSStoreDefaults,
    FAISSVectorStore,
    IndexType,
    create_faiss_store,
)


@pytest.mark.unit
class TestFAISSStoreDefaults:
    """Tests for FAISSStoreDefaults constants."""

    def test_model_name(self):
        assert FAISSStoreDefaults.MODEL_NAME == "all-MiniLM-L6-v2"

    def test_embedding_dim(self):
        assert FAISSStoreDefaults.EMBEDDING_DIM == 384

    def test_default_top_k(self):
        assert FAISSStoreDefaults.DEFAULT_TOP_K == 5

    def test_max_top_k(self):
        assert FAISSStoreDefaults.MAX_TOP_K == 10000

    def test_min_score(self):
        assert FAISSStoreDefaults.MIN_SCORE == 0.0

    def test_max_score(self):
        assert FAISSStoreDefaults.MAX_SCORE == 1.0

    def test_batch_size(self):
        assert FAISSStoreDefaults.BATCH_SIZE == 32

    def test_flat_max_vectors(self):
        assert FAISSStoreDefaults.FLAT_MAX_VECTORS == 10000

    def test_ivf_min_vectors(self):
        assert FAISSStoreDefaults.IVF_MIN_VECTORS == 10000

    def test_ivf_nlist(self):
        assert FAISSStoreDefaults.IVF_NLIST == 100


@pytest.mark.unit
class TestIndexType:
    """Tests for IndexType constants."""

    def test_flat_l2(self):
        assert IndexType.FLAT_L2 == "flat_l2"

    def test_flat_ip(self):
        assert IndexType.FLAT_IP == "flat_ip"

    def test_ivf_flat(self):
        assert IndexType.IVF_FLAT == "ivf_flat"

    def test_ivf_pq(self):
        assert IndexType.IVF_PQ == "ivf_pq"

    def test_hnsw(self):
        assert IndexType.HNSW == "hnsw"


@pytest.mark.unit
class TestFAISSDocument:
    """Tests for FAISSDocument dataclass."""

    def test_creation(self):
        doc = FAISSDocument(id="abc", content="hello world", embedding_id=0)
        assert doc.id == "abc"
        assert doc.content == "hello world"
        assert doc.embedding_id == 0
        assert doc.metadata == {}

    def test_creation_with_metadata(self):
        doc = FAISSDocument(id="abc", content="hello", embedding_id=1, metadata={"source": "test"})
        assert doc.metadata == {"source": "test"}

    def test_hash(self):
        doc1 = FAISSDocument(id="abc", content="hello", embedding_id=0)
        doc2 = FAISSDocument(id="abc", content="different", embedding_id=1)
        assert hash(doc1) == hash(doc2)

    def test_equality(self):
        doc1 = FAISSDocument(id="abc", content="hello", embedding_id=0)
        doc2 = FAISSDocument(id="abc", content="different", embedding_id=1)
        assert doc1 == doc2

    def test_inequality(self):
        doc1 = FAISSDocument(id="abc", content="hello", embedding_id=0)
        doc2 = FAISSDocument(id="xyz", content="hello", embedding_id=0)
        assert doc1 != doc2

    def test_equality_with_non_document(self):
        doc = FAISSDocument(id="abc", content="hello", embedding_id=0)
        assert doc != "not a document"
        assert doc != 42

    def test_can_be_used_in_set(self):
        doc1 = FAISSDocument(id="abc", content="hello", embedding_id=0)
        doc2 = FAISSDocument(id="abc", content="world", embedding_id=1)
        doc3 = FAISSDocument(id="xyz", content="hello", embedding_id=2)
        s = {doc1, doc2, doc3}
        assert len(s) == 2  # doc1 and doc2 have same id


@pytest.mark.unit
class TestFAISSVectorStoreInit:
    """Tests for FAISSVectorStore initialization."""

    def test_default_init(self):
        store = FAISSVectorStore()
        assert store._model_name == FAISSStoreDefaults.MODEL_NAME
        assert store._embedding_dim == FAISSStoreDefaults.EMBEDDING_DIM
        assert store._index_type == IndexType.FLAT_IP
        assert store._persist_dir is None
        assert store._is_initialized is False
        assert store._next_id == 0
        assert store.document_count == 0

    def test_custom_init(self):
        from pathlib import Path

        store = FAISSVectorStore(
            model_name="custom-model",
            embedding_dim=768,
            index_type=IndexType.FLAT_L2,
            use_gpu=False,
            persist_dir=Path("/tmp/test"),
        )
        assert store._model_name == "custom-model"
        assert store._embedding_dim == 768
        assert store._index_type == IndexType.FLAT_L2
        assert store._persist_dir == Path("/tmp/test")

    def test_is_available_false_before_init(self):
        store = FAISSVectorStore()
        assert store.is_available is False

    def test_index_type_property(self):
        store = FAISSVectorStore(index_type=IndexType.HNSW)
        assert store.index_type == IndexType.HNSW


@pytest.mark.unit
class TestFAISSVectorStoreHelpers:
    """Tests for FAISSVectorStore helper methods."""

    def test_generate_doc_id(self):
        store = FAISSVectorStore()
        doc_id = store._generate_doc_id("hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()[:16]
        assert doc_id == expected

    def test_generate_doc_id_deterministic(self):
        store = FAISSVectorStore()
        id1 = store._generate_doc_id("same content")
        id2 = store._generate_doc_id("same content")
        assert id1 == id2

    def test_generate_doc_id_different_for_different_content(self):
        store = FAISSVectorStore()
        id1 = store._generate_doc_id("content A")
        id2 = store._generate_doc_id("content B")
        assert id1 != id2

    def test_matches_filter_all_match(self):
        store = FAISSVectorStore()
        metadata = {"source": "test", "type": "doc"}
        filter_metadata = {"source": "test"}
        assert store._matches_filter(metadata, filter_metadata) is True

    def test_matches_filter_no_match(self):
        store = FAISSVectorStore()
        metadata = {"source": "test"}
        filter_metadata = {"source": "other"}
        assert store._matches_filter(metadata, filter_metadata) is False

    def test_matches_filter_missing_key(self):
        store = FAISSVectorStore()
        metadata = {"source": "test"}
        filter_metadata = {"missing_key": "value"}
        assert store._matches_filter(metadata, filter_metadata) is False

    def test_matches_filter_empty_filter(self):
        store = FAISSVectorStore()
        metadata = {"source": "test"}
        assert store._matches_filter(metadata, {}) is True

    def test_matches_filter_multiple_criteria(self):
        store = FAISSVectorStore()
        metadata = {"source": "test", "type": "doc", "lang": "en"}
        filter_metadata = {"source": "test", "lang": "en"}
        assert store._matches_filter(metadata, filter_metadata) is True

    def test_matches_filter_partial_match_fails(self):
        store = FAISSVectorStore()
        metadata = {"source": "test", "type": "doc"}
        filter_metadata = {"source": "test", "type": "other"}
        assert store._matches_filter(metadata, filter_metadata) is False


@pytest.mark.unit
class TestFAISSVectorStoreOperations:
    """Tests for FAISSVectorStore operations using mocked dependencies."""

    def test_search_returns_empty_when_not_initialized(self):
        store = FAISSVectorStore()
        results = store.search("hello")
        assert results == []

    def test_search_returns_empty_for_empty_query(self):
        store = FAISSVectorStore()
        store._is_initialized = True
        store._index = MagicMock()
        results = store.search("")
        assert results == []

    def test_search_returns_empty_for_whitespace_query(self):
        store = FAISSVectorStore()
        store._is_initialized = True
        store._index = MagicMock()
        results = store.search("   ")
        assert results == []

    def test_search_returns_empty_when_no_documents(self):
        store = FAISSVectorStore()
        store._is_initialized = True
        store._index = MagicMock()
        results = store.search("hello")
        assert results == []

    def test_add_documents_returns_zero_when_not_initialized(self):
        """Test add_documents returns 0 when store cannot initialize."""
        store = FAISSVectorStore()
        # Patch initialize to return False
        store.initialize = MagicMock(return_value=False)
        result = store.add_documents([{"content": "hello"}])
        assert result == 0

    def test_remove_document_nonexistent(self):
        store = FAISSVectorStore()
        assert store.remove_document("nonexistent") is False

    def test_remove_document_existing(self):
        store = FAISSVectorStore()
        # Manually add a document to internal state
        store._document_id_map["doc1"] = 0
        store._documents[0] = FAISSDocument(id="doc1", content="hello", embedding_id=0)

        assert store.remove_document("doc1") is True
        assert "doc1" not in store._document_id_map
        assert 0 not in store._documents

    def test_clear_resets_state(self):
        """Test that clear resets documents and ID mappings."""
        store = FAISSVectorStore()
        # Mock _create_index to avoid needing faiss
        mock_index = MagicMock()
        store._create_index = MagicMock(return_value=mock_index)

        store._documents = {0: FAISSDocument(id="a", content="x", embedding_id=0)}
        store._document_id_map = {"a": 0}
        store._next_id = 1

        store.clear()

        assert store.document_count == 0
        assert store._next_id == 0
        assert len(store._document_id_map) == 0

    def test_document_count_property(self):
        store = FAISSVectorStore()
        assert store.document_count == 0
        store._documents[0] = FAISSDocument(id="a", content="x", embedding_id=0)
        assert store.document_count == 1

    def test_gpu_enabled_property_no_gpu(self):
        store = FAISSVectorStore(use_gpu=False)
        assert store.gpu_enabled is False


@pytest.mark.unit
class TestFAISSVectorStoreInitialize:
    """Tests for FAISSVectorStore.initialize method."""

    @patch("src.storage.faiss_store._HAS_NUMPY", False)
    def test_initialize_fails_without_numpy(self):
        store = FAISSVectorStore()
        assert store.initialize() is False

    @patch("src.storage.faiss_store._HAS_NUMPY", True)
    @patch("src.storage.faiss_store._HAS_FAISS", False)
    def test_initialize_fails_without_faiss(self):
        store = FAISSVectorStore()
        assert store.initialize() is False

    @patch("src.storage.faiss_store._HAS_NUMPY", True)
    @patch("src.storage.faiss_store._HAS_FAISS", True)
    @patch("src.storage.faiss_store._HAS_SENTENCE_TRANSFORMERS", False)
    def test_initialize_fails_without_sentence_transformers(self):
        store = FAISSVectorStore()
        assert store.initialize() is False

    def test_initialize_returns_true_if_already_initialized(self):
        store = FAISSVectorStore()
        store._is_initialized = True
        assert store.initialize() is True


@pytest.mark.unit
class TestCreateFaissStoreFactory:
    """Tests for create_faiss_store factory function."""

    @patch("src.storage.faiss_store._HAS_FAISS", False)
    def test_returns_none_without_faiss(self):
        result = create_faiss_store()
        assert result is None

    @patch("src.storage.faiss_store._HAS_FAISS", True)
    def test_returns_none_when_init_fails(self):
        """Test that create_faiss_store returns None when initialization fails."""
        with patch.object(FAISSVectorStore, "initialize", return_value=False):
            result = create_faiss_store(auto_initialize=True)
            assert result is None

    @patch("src.storage.faiss_store._HAS_FAISS", True)
    def test_returns_store_without_auto_init(self):
        """Test that create_faiss_store returns store when auto_initialize=False."""
        result = create_faiss_store(auto_initialize=False)
        assert isinstance(result, FAISSVectorStore)
        assert result.is_available is False


@pytest.mark.unit
class TestFAISSVectorStoreLogging:
    """Tests for logging helper methods."""

    def test_log_debug(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger)
        store._log_debug("test message", key="value")
        mock_logger.debug.assert_called_once()

    def test_log_info(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger)
        store._log_info("test message")
        mock_logger.info.assert_called_once()

    def test_log_warning(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger)
        store._log_warning("test warning")
        mock_logger.warning.assert_called_once()

    def test_log_error(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger)
        store._log_error("test error", detail="something")
        mock_logger.error.assert_called_once()

    def test_log_debug_no_kwargs(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger)
        store._log_debug("simple message")
        mock_logger.debug.assert_called_once_with("simple message")

    def test_log_info_no_kwargs(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger)
        store._log_info("simple message")
        mock_logger.info.assert_called_once_with("simple message")
