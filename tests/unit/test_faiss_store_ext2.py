"""Extended unit tests for src/storage/faiss_store.py - targeting uncovered lines.

Covers: initialize with mocked deps, _create_index variants, add_documents,
search, rebuild_index, _save, _load_if_exists, GPU paths, logging with
structured logging, and create_faiss_store factory.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.storage.faiss_store import (
    FAISSDocument,
    FAISSStoreDefaults,
    FAISSVectorStore,
    IndexType,
    create_faiss_store,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_initialized_store(
    index_type: str = IndexType.FLAT_IP,
    persist_dir: Path | None = None,
    use_gpu: bool = False,
) -> FAISSVectorStore:
    """Return a FAISSVectorStore with mocked internals marked as initialized."""
    store = FAISSVectorStore(
        index_type=index_type,
        use_gpu=use_gpu,
        persist_dir=persist_dir,
        embedding_dim=4,
    )
    store._is_initialized = True
    store._model = MagicMock()
    store._index = MagicMock()
    store._index.is_trained = True
    return store


def _fake_encode(texts, **kwargs):
    """Return a (N, 4) float32 numpy array mimicking SentenceTransformer.encode."""
    if isinstance(texts, str):
        return np.random.rand(4).astype("float32")
    return np.random.rand(len(texts), 4).astype("float32")


# ---------------------------------------------------------------------------
# initialize() – full success path with mocked FAISS / SentenceTransformer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitializeFull:
    """Test the full initialize() path (lines 193-222)."""

    @patch("src.storage.faiss_store._HAS_SENTENCE_TRANSFORMERS", True)
    @patch("src.storage.faiss_store._HAS_FAISS", True)
    @patch("src.storage.faiss_store._HAS_NUMPY", True)
    @patch("src.storage.faiss_store.SentenceTransformer")
    @patch("src.storage.faiss_store.faiss")
    def test_initialize_success(self, mock_faiss, mock_st_cls):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(4, dtype="float32")
        mock_st_cls.return_value = mock_model

        mock_faiss.IndexFlatIP.return_value = MagicMock()

        store = FAISSVectorStore(embedding_dim=4, use_gpu=False)
        assert store.initialize() is True
        assert store.is_available is True
        mock_st_cls.assert_called_once()

    @patch("src.storage.faiss_store._HAS_SENTENCE_TRANSFORMERS", True)
    @patch("src.storage.faiss_store._HAS_FAISS", True)
    @patch("src.storage.faiss_store._HAS_NUMPY", True)
    @patch("src.storage.faiss_store.SentenceTransformer")
    @patch("src.storage.faiss_store.faiss")
    def test_initialize_double_check_lock(self, mock_faiss, mock_st_cls):
        """Second call inside the lock should return True without re-init."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(4, dtype="float32")
        mock_st_cls.return_value = mock_model
        mock_faiss.IndexFlatIP.return_value = MagicMock()

        store = FAISSVectorStore(embedding_dim=4, use_gpu=False)
        store.initialize()
        # Second call returns True immediately from outer check
        assert store.initialize() is True

    @patch("src.storage.faiss_store._HAS_SENTENCE_TRANSFORMERS", True)
    @patch("src.storage.faiss_store._HAS_FAISS", True)
    @patch("src.storage.faiss_store._HAS_NUMPY", True)
    @patch("src.storage.faiss_store.SentenceTransformer")
    def test_initialize_runtime_error(self, mock_st_cls):
        """RuntimeError during init returns False (line 220-222)."""
        mock_st_cls.side_effect = RuntimeError("model load failed")
        store = FAISSVectorStore(use_gpu=False)
        assert store.initialize() is False
        assert store.is_available is False

    @patch("src.storage.faiss_store._HAS_SENTENCE_TRANSFORMERS", True)
    @patch("src.storage.faiss_store._HAS_FAISS", True)
    @patch("src.storage.faiss_store._HAS_NUMPY", True)
    @patch("src.storage.faiss_store.SentenceTransformer")
    def test_initialize_os_error(self, mock_st_cls):
        """OSError during init returns False."""
        mock_st_cls.side_effect = OSError("disk error")
        store = FAISSVectorStore(use_gpu=False)
        assert store.initialize() is False

    @patch("src.storage.faiss_store._HAS_SENTENCE_TRANSFORMERS", True)
    @patch("src.storage.faiss_store._HAS_FAISS", True)
    @patch("src.storage.faiss_store._HAS_NUMPY", True)
    @patch("src.storage.faiss_store.SentenceTransformer")
    @patch("src.storage.faiss_store.faiss")
    def test_initialize_with_persist_dir_calls_load(self, mock_faiss, mock_st_cls, tmp_path):
        """When persist_dir is set, _load_if_exists is called (line 207-208)."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(4, dtype="float32")
        mock_st_cls.return_value = mock_model
        mock_faiss.IndexFlatIP.return_value = MagicMock()

        store = FAISSVectorStore(embedding_dim=4, use_gpu=False, persist_dir=tmp_path)
        with patch.object(store, "_load_if_exists") as mock_load:
            store.initialize()
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# _create_index – all index type branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateIndex:
    """Test _create_index for each IndexType (lines 228-248)."""

    @patch("src.storage.faiss_store.faiss")
    def test_flat_l2(self, mock_faiss):
        store = FAISSVectorStore(index_type=IndexType.FLAT_L2, embedding_dim=4, use_gpu=False)
        store._create_index()
        mock_faiss.IndexFlatL2.assert_called_once_with(4)

    @patch("src.storage.faiss_store.faiss")
    def test_flat_ip(self, mock_faiss):
        store = FAISSVectorStore(index_type=IndexType.FLAT_IP, embedding_dim=4, use_gpu=False)
        store._create_index()
        mock_faiss.IndexFlatIP.assert_called_once_with(4)

    @patch("src.storage.faiss_store.faiss")
    def test_ivf_flat(self, mock_faiss):
        mock_quantizer = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_quantizer
        store = FAISSVectorStore(index_type=IndexType.IVF_FLAT, embedding_dim=4, use_gpu=False)
        store._create_index()
        mock_faiss.IndexIVFFlat.assert_called_once_with(
            mock_quantizer, 4, FAISSStoreDefaults.IVF_NLIST
        )

    @patch("src.storage.faiss_store.faiss")
    def test_hnsw(self, mock_faiss):
        store = FAISSVectorStore(index_type=IndexType.HNSW, embedding_dim=4, use_gpu=False)
        store._create_index()
        mock_faiss.IndexHNSWFlat.assert_called_once_with(4, 32)

    @patch("src.storage.faiss_store.faiss")
    def test_unknown_defaults_to_flat_ip(self, mock_faiss):
        """Unknown index type falls through to default (lines 238-239)."""
        store = FAISSVectorStore(index_type="unknown_type", embedding_dim=4, use_gpu=False)
        store._create_index()
        mock_faiss.IndexFlatIP.assert_called_with(4)

    @patch("src.storage.faiss_store._HAS_FAISS_GPU", True)
    @patch("src.storage.faiss_store.faiss")
    def test_gpu_move_success(self, mock_faiss):
        """GPU path succeeds (lines 243-246)."""
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_gpu_res = MagicMock()
        mock_faiss.StandardGpuResources.return_value = mock_gpu_res
        mock_faiss.index_cpu_to_gpu.return_value = MagicMock()

        store = FAISSVectorStore(index_type=IndexType.FLAT_IP, embedding_dim=4, use_gpu=True)
        store._use_gpu = True  # Force GPU usage
        store._create_index()
        mock_faiss.index_cpu_to_gpu.assert_called_once_with(mock_gpu_res, 0, mock_index)

    @patch("src.storage.faiss_store._HAS_FAISS_GPU", True)
    @patch("src.storage.faiss_store.faiss")
    def test_gpu_move_failure_falls_back(self, mock_faiss):
        """GPU failure falls back to CPU (lines 247-248)."""
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.StandardGpuResources.side_effect = RuntimeError("GPU fail")

        store = FAISSVectorStore(index_type=IndexType.FLAT_IP, embedding_dim=4, use_gpu=True)
        store._use_gpu = True
        result = store._create_index()
        # Should return the CPU index despite GPU failure
        assert result == mock_index


# ---------------------------------------------------------------------------
# add_documents – full paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddDocuments:
    """Test add_documents covering lines 274, 288, 293, 316, 341, 345-347."""

    def test_returns_zero_when_model_none(self):
        """Line 274: _model is None."""
        store = _make_initialized_store()
        store._model = None
        assert store.add_documents([{"content": "hello"}]) == 0

    def test_returns_zero_when_index_none(self):
        """Line 274: _index is None."""
        store = _make_initialized_store()
        store._index = None
        assert store.add_documents([{"content": "hello"}]) == 0

    def test_skips_empty_content(self):
        """Documents with empty content are skipped."""
        store = _make_initialized_store()
        store._model.encode.side_effect = _fake_encode
        result = store.add_documents([{"content": ""}, {"content": "   valid   "}])
        # Only the valid doc should be added
        assert result == 1

    def test_skips_duplicate_documents(self):
        """Line 288: duplicate doc_id is skipped."""
        store = _make_initialized_store()
        store._model.encode.side_effect = _fake_encode
        store._document_id_map["existing_id"] = 0

        result = store.add_documents([{"id": "existing_id", "content": "hello"}])
        assert result == 0

    def test_returns_zero_for_all_duplicates(self):
        """Line 293: new_docs is empty after filtering."""
        store = _make_initialized_store()
        store._document_id_map["id1"] = 0
        result = store.add_documents([{"id": "id1", "content": "hello"}])
        assert result == 0

    def test_adds_single_document(self):
        store = _make_initialized_store()
        store._model.encode.side_effect = _fake_encode

        result = store.add_documents([{"content": "hello world", "metadata": {"src": "test"}}])
        assert result == 1
        assert store.document_count == 1
        assert store._next_id == 1

    def test_adds_multiple_documents_with_batching(self):
        store = _make_initialized_store()
        store._model.encode.side_effect = _fake_encode

        docs = [{"content": f"doc {i}"} for i in range(5)]
        result = store.add_documents(docs, batch_size=2)
        assert result == 5
        assert store._next_id == 5

    def test_auto_generates_doc_id(self):
        store = _make_initialized_store()
        store._model.encode.side_effect = _fake_encode
        store.add_documents([{"content": "auto id test"}])
        # Check that a generated ID (not user-supplied) was used
        assert len(store._document_id_map) == 1
        doc_id = list(store._document_id_map.keys())[0]
        assert len(doc_id) == 16  # sha256[:16]

    def test_uses_provided_doc_id(self):
        store = _make_initialized_store()
        store._model.encode.side_effect = _fake_encode
        store.add_documents([{"id": "my-id", "content": "test"}])
        assert "my-id" in store._document_id_map

    def test_ivf_training_triggered(self):
        """Line 316: IVF index trains when enough data."""
        store = _make_initialized_store(index_type=IndexType.IVF_FLAT)
        store._index.is_trained = False
        store._model.encode.side_effect = _fake_encode

        # Need >= IVF_NLIST (100) docs to trigger training
        docs = [{"content": f"doc {i}"} for i in range(101)]
        store.add_documents(docs)
        store._index.train.assert_called_once()

    def test_ivf_no_training_when_insufficient_data(self):
        """IVF index does NOT train with too few documents."""
        store = _make_initialized_store(index_type=IndexType.IVF_FLAT)
        store._index.is_trained = False
        store._model.encode.side_effect = _fake_encode

        docs = [{"content": f"doc {i}"} for i in range(5)]
        store.add_documents(docs)
        store._index.train.assert_not_called()

    def test_auto_saves_with_persist_dir(self, tmp_path):
        """Line 341: auto-save when persist_dir is set."""
        store = _make_initialized_store(persist_dir=tmp_path)
        store._model.encode.side_effect = _fake_encode

        with patch.object(store, "_save") as mock_save:
            store.add_documents([{"content": "hello"}])
            mock_save.assert_called_once()

    def test_exception_returns_zero(self):
        """Lines 345-347: exception during add returns 0."""
        store = _make_initialized_store()
        store._model.encode.side_effect = ValueError("encoding failed")
        result = store.add_documents([{"content": "hello"}])
        assert result == 0

    def test_runtime_error_returns_zero(self):
        store = _make_initialized_store()
        store._model.encode.side_effect = RuntimeError("runtime fail")
        result = store.add_documents([{"content": "hello"}])
        assert result == 0

    def test_type_error_returns_zero(self):
        store = _make_initialized_store()
        store._model.encode.side_effect = TypeError("type fail")
        result = store.add_documents([{"content": "hello"}])
        assert result == 0

    def test_add_documents_triggers_initialize_if_needed(self):
        """Line 270: not initialized triggers initialize()."""
        store = FAISSVectorStore(use_gpu=False)
        store.initialize = MagicMock(return_value=False)
        result = store.add_documents([{"content": "hello"}])
        assert result == 0
        store.initialize.assert_called_once()


# ---------------------------------------------------------------------------
# search – full path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearch:
    """Test search covering lines 380-441."""

    def test_search_model_none_returns_empty(self):
        """Line 381-382: model is None inside lock."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(id="d1", content="hi", embedding_id=0)
        store._model = None
        assert store.search("test") == []

    def test_search_success(self):
        """Full search success path."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(
            id="d1", content="hello world", embedding_id=0, metadata={"src": "test"}
        )

        query_emb = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")
        store._model.encode.return_value = query_emb

        distances = np.array([[0.95]], dtype="float32")
        indices = np.array([[0]], dtype="int64")
        store._index.search.return_value = (distances, indices)

        results = store.search("hello", top_k=1)
        assert len(results) == 1
        assert results[0]["content"] == "hello world"
        assert results[0]["score"] == pytest.approx(0.95)
        assert results[0]["id"] == "d1"

    def test_search_skips_negative_indices(self):
        """Line 403-404: FAISS returns -1 for missing results."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(id="d1", content="hi", embedding_id=0)

        store._model.encode.return_value = np.zeros(4, dtype="float32")
        store._index.search.return_value = (
            np.array([[0.9, -1.0]], dtype="float32"),
            np.array([[0, -1]], dtype="int64"),
        )

        results = store.search("test", top_k=5)
        assert len(results) == 1

    def test_search_skips_missing_doc(self):
        """Line 407-408: idx not found in _documents."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(id="d1", content="hi", embedding_id=0)

        store._model.encode.return_value = np.zeros(4, dtype="float32")
        store._index.search.return_value = (
            np.array([[0.9, 0.8]], dtype="float32"),
            np.array([[0, 99]], dtype="int64"),  # 99 doesn't exist
        )

        results = store.search("test", top_k=5)
        assert len(results) == 1

    def test_search_flat_l2_score_conversion(self):
        """Lines 412-414: L2 distance converted to similarity."""
        store = _make_initialized_store(index_type=IndexType.FLAT_L2)
        store._documents[0] = FAISSDocument(id="d1", content="hi", embedding_id=0)

        store._model.encode.return_value = np.zeros(4, dtype="float32")
        # L2 distance of 0.0 -> similarity of 1.0 / (1.0 + 0.0) = 1.0
        store._index.search.return_value = (
            np.array([[0.0]], dtype="float32"),
            np.array([[0]], dtype="int64"),
        )

        results = store.search("test", top_k=1)
        assert results[0]["score"] == pytest.approx(1.0)

    def test_search_flat_l2_large_distance(self):
        """L2 distance of 4.0 -> similarity of 1/(1+4) = 0.2."""
        store = _make_initialized_store(index_type=IndexType.FLAT_L2)
        store._documents[0] = FAISSDocument(id="d1", content="hi", embedding_id=0)

        store._model.encode.return_value = np.zeros(4, dtype="float32")
        store._index.search.return_value = (
            np.array([[4.0]], dtype="float32"),
            np.array([[0]], dtype="int64"),
        )

        results = store.search("test", top_k=1)
        assert results[0]["score"] == pytest.approx(0.2)

    def test_search_min_score_filter(self):
        """Line 418-419: results below min_score are excluded."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(id="d1", content="hi", embedding_id=0)
        store._documents[1] = FAISSDocument(id="d2", content="lo", embedding_id=1)

        store._model.encode.return_value = np.zeros(4, dtype="float32")
        store._index.search.return_value = (
            np.array([[0.9, 0.1]], dtype="float32"),
            np.array([[0, 1]], dtype="int64"),
        )

        results = store.search("test", top_k=5, min_score=0.5)
        assert len(results) == 1
        assert results[0]["id"] == "d1"

    def test_search_metadata_filter(self):
        """Lines 422-423: metadata filter applied."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(
            id="d1", content="hi", embedding_id=0, metadata={"type": "a"}
        )
        store._documents[1] = FAISSDocument(
            id="d2", content="lo", embedding_id=1, metadata={"type": "b"}
        )

        store._model.encode.return_value = np.zeros(4, dtype="float32")
        store._index.search.return_value = (
            np.array([[0.9, 0.8]], dtype="float32"),
            np.array([[0, 1]], dtype="int64"),
        )

        results = store.search("test", top_k=5, filter_metadata={"type": "b"})
        assert len(results) == 1
        assert results[0]["id"] == "d2"

    def test_search_top_k_limit(self):
        """Line 434-435: results capped at top_k."""
        store = _make_initialized_store()
        for i in range(5):
            store._documents[i] = FAISSDocument(id=f"d{i}", content=f"doc {i}", embedding_id=i)

        store._model.encode.return_value = np.zeros(4, dtype="float32")
        store._index.search.return_value = (
            np.array([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype="float32"),
            np.array([[0, 1, 2, 3, 4]], dtype="int64"),
        )

        results = store.search("test", top_k=2)
        assert len(results) == 2

    def test_search_with_filter_uses_expanded_k(self):
        """Line 398: search_k = top_k * 3 when filter_metadata is set."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(
            id="d1", content="hi", embedding_id=0, metadata={"type": "a"}
        )

        store._model.encode.return_value = np.zeros(4, dtype="float32")
        store._index.search.return_value = (
            np.array([[0.9]], dtype="float32"),
            np.array([[0]], dtype="int64"),
        )

        store.search("test", top_k=2, filter_metadata={"type": "a"})
        # search_k = 2 * 3 = 6, but capped at len(documents)=1
        store._index.search.assert_called_once()
        actual_k = store._index.search.call_args[0][1]
        assert actual_k == 1  # min(6, 1)

    def test_search_exception_returns_empty(self):
        """Lines 439-441: exception during search returns []."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(id="d1", content="hi", embedding_id=0)
        store._model.encode.side_effect = ValueError("encode error")
        assert store.search("test") == []

    def test_search_runtime_error(self):
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(id="d1", content="hi", embedding_id=0)
        store._model.encode.side_effect = RuntimeError("search fail")
        assert store.search("test") == []


# ---------------------------------------------------------------------------
# rebuild_index
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRebuildIndex:
    """Test rebuild_index covering lines 476-533."""

    def test_rebuild_not_initialized(self):
        """Line 477: not initialized returns False."""
        store = FAISSVectorStore(use_gpu=False)
        assert store.rebuild_index() is False

    def test_rebuild_model_none(self):
        """Line 477: model is None returns False."""
        store = _make_initialized_store()
        store._model = None
        assert store.rebuild_index() is False

    def test_rebuild_empty_docs(self):
        """Lines 483-487: empty docs resets index."""
        store = _make_initialized_store()
        mock_new_index = MagicMock()
        store._create_index = MagicMock(return_value=mock_new_index)

        result = store.rebuild_index()
        assert result is True
        assert store._next_id == 0
        assert store._index == mock_new_index

    def test_rebuild_with_documents(self):
        """Lines 490-526: rebuild with existing documents."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(id="d1", content="hello", embedding_id=0)
        store._documents[1] = FAISSDocument(id="d2", content="world", embedding_id=1)
        store._document_id_map = {"d1": 0, "d2": 1}

        store._model.encode.return_value = np.random.rand(2, 4).astype("float32")
        mock_new_index = MagicMock()
        mock_new_index.is_trained = True
        store._create_index = MagicMock(return_value=mock_new_index)

        result = store.rebuild_index()
        assert result is True
        assert store._next_id == 2
        assert store.document_count == 2
        assert "d1" in store._document_id_map
        assert "d2" in store._document_id_map
        mock_new_index.add.assert_called_once()

    def test_rebuild_ivf_trains(self):
        """Lines 501-506: IVF training during rebuild."""
        store = _make_initialized_store(index_type=IndexType.IVF_FLAT)
        # Add enough docs
        for i in range(101):
            store._documents[i] = FAISSDocument(id=f"d{i}", content=f"doc {i}", embedding_id=i)
            store._document_id_map[f"d{i}"] = i

        store._model.encode.return_value = np.random.rand(101, 4).astype("float32")
        mock_new_index = MagicMock()
        mock_new_index.is_trained = False
        store._create_index = MagicMock(return_value=mock_new_index)

        result = store.rebuild_index()
        assert result is True
        mock_new_index.train.assert_called_once()

    def test_rebuild_exception_returns_false(self):
        """Lines 531-533: exception returns False."""
        store = _make_initialized_store()
        store._documents[0] = FAISSDocument(id="d1", content="hello", embedding_id=0)
        store._model.encode.side_effect = RuntimeError("fail")

        assert store.rebuild_index() is False


# ---------------------------------------------------------------------------
# _save
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSave:
    """Test _save covering lines 546-582."""

    def test_save_no_persist_dir(self):
        """Line 546: no persist_dir does nothing."""
        store = _make_initialized_store()
        store._persist_dir = None
        store._save()  # Should not raise

    def test_save_no_index(self):
        """Line 546: no index does nothing."""
        store = _make_initialized_store(persist_dir=Path("/tmp/fake"))
        store._index = None
        store._save()

    @patch("src.storage.faiss_store.faiss")
    def test_save_creates_files(self, mock_faiss, tmp_path):
        """Lines 549-577: saves index and metadata."""
        store = _make_initialized_store(persist_dir=tmp_path)
        store._documents[0] = FAISSDocument(
            id="d1", content="hello", embedding_id=0, metadata={"src": "test"}
        )
        store._document_id_map = {"d1": 0}
        store._next_id = 1

        store._save()

        mock_faiss.write_index.assert_called_once()
        metadata_path = tmp_path / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["next_id"] == 1
        assert "0" in metadata["documents"]
        assert metadata["documents"]["0"]["id"] == "d1"

    @patch("src.storage.faiss_store._HAS_FAISS_GPU", True)
    @patch("src.storage.faiss_store.faiss")
    def test_save_gpu_converts_to_cpu(self, mock_faiss, tmp_path):
        """Lines 555-557: GPU index converted to CPU before saving."""
        store = _make_initialized_store(persist_dir=tmp_path, use_gpu=True)
        store._use_gpu = True

        mock_cpu_index = MagicMock()
        mock_faiss.index_gpu_to_cpu.return_value = mock_cpu_index

        store._save()

        mock_faiss.index_gpu_to_cpu.assert_called_once_with(store._index)
        mock_faiss.write_index.assert_called_once_with(mock_cpu_index, str(tmp_path / "faiss.index"))

    @patch("src.storage.faiss_store.faiss")
    def test_save_os_error(self, mock_faiss, tmp_path):
        """Lines 581-582: OSError is caught."""
        store = _make_initialized_store(persist_dir=tmp_path)
        mock_faiss.write_index.side_effect = OSError("disk full")
        store._save()  # Should not raise


# ---------------------------------------------------------------------------
# _load_if_exists
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadIfExists:
    """Test _load_if_exists covering lines 586-629."""

    def test_load_no_persist_dir(self):
        """Line 586-587: no persist_dir returns False."""
        store = _make_initialized_store()
        store._persist_dir = None
        assert store._load_if_exists() is False

    def test_load_missing_files(self, tmp_path):
        """Lines 592-593: missing files returns False."""
        store = _make_initialized_store(persist_dir=tmp_path)
        assert store._load_if_exists() is False

    def test_load_missing_index_file(self, tmp_path):
        """Only metadata exists, no index file."""
        (tmp_path / "metadata.json").write_text("{}")
        store = _make_initialized_store(persist_dir=tmp_path)
        assert store._load_if_exists() is False

    def test_load_missing_metadata_file(self, tmp_path):
        """Only index file exists, no metadata."""
        (tmp_path / "faiss.index").write_text("fake")
        store = _make_initialized_store(persist_dir=tmp_path)
        assert store._load_if_exists() is False

    @patch("src.storage.faiss_store.faiss")
    def test_load_success(self, mock_faiss, tmp_path):
        """Lines 596-625: successful load."""
        # Create metadata file
        metadata = {
            "next_id": 2,
            "documents": {
                "0": {"id": "d1", "content": "hello", "embedding_id": 0, "metadata": {"src": "test"}},
                "1": {"id": "d2", "content": "world", "embedding_id": 1, "metadata": {}},
            },
            "document_id_map": {"d1": 0, "d2": 1},
        }
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "faiss.index").write_text("fake_index")

        mock_faiss.read_index.return_value = MagicMock()

        store = _make_initialized_store(persist_dir=tmp_path)
        result = store._load_if_exists()

        assert result is True
        assert store._next_id == 2
        assert store.document_count == 2
        assert store._document_id_map == {"d1": 0, "d2": 1}
        assert store._documents[0].id == "d1"
        assert store._documents[1].content == "world"

    @patch("src.storage.faiss_store._HAS_FAISS_GPU", True)
    @patch("src.storage.faiss_store.faiss")
    def test_load_with_gpu(self, mock_faiss, tmp_path):
        """Lines 600-602: GPU load path."""
        metadata = {
            "next_id": 0,
            "documents": {},
            "document_id_map": {},
        }
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "faiss.index").write_text("fake")

        mock_faiss.read_index.return_value = MagicMock()
        mock_gpu_res = MagicMock()
        mock_faiss.StandardGpuResources.return_value = mock_gpu_res
        mock_faiss.index_cpu_to_gpu.return_value = MagicMock()

        store = _make_initialized_store(persist_dir=tmp_path, use_gpu=True)
        store._use_gpu = True
        result = store._load_if_exists()

        assert result is True
        mock_faiss.index_cpu_to_gpu.assert_called_once()

    @patch("src.storage.faiss_store.faiss")
    def test_load_os_error(self, mock_faiss, tmp_path):
        """Lines 627-629: OSError returns False."""
        (tmp_path / "metadata.json").write_text("{}")
        (tmp_path / "faiss.index").write_text("fake")

        mock_faiss.read_index.side_effect = OSError("read fail")
        store = _make_initialized_store(persist_dir=tmp_path)
        assert store._load_if_exists() is False

    @patch("src.storage.faiss_store.faiss")
    def test_load_value_error(self, mock_faiss, tmp_path):
        """ValueError returns False."""
        (tmp_path / "metadata.json").write_text("invalid json{{{")
        (tmp_path / "faiss.index").write_text("fake")

        mock_faiss.read_index.return_value = MagicMock()
        store = _make_initialized_store(persist_dir=tmp_path)
        assert store._load_if_exists() is False


# ---------------------------------------------------------------------------
# Logging helpers with structured logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoggingWithStructuredLogging:
    """Test log helpers when _HAS_STRUCTURED_LOGGING is True (lines 648, 654, 660, 666)."""

    @patch("src.storage.faiss_store._HAS_STRUCTURED_LOGGING", True)
    def test_log_debug_structured(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger, use_gpu=False)
        store._log_debug("msg", key="val")
        mock_logger.debug.assert_called_once_with("msg", key="val")

    @patch("src.storage.faiss_store._HAS_STRUCTURED_LOGGING", True)
    def test_log_info_structured(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger, use_gpu=False)
        store._log_info("msg", key="val")
        mock_logger.info.assert_called_once_with("msg", key="val")

    @patch("src.storage.faiss_store._HAS_STRUCTURED_LOGGING", True)
    def test_log_warning_structured(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger, use_gpu=False)
        store._log_warning("msg", key="val")
        mock_logger.warning.assert_called_once_with("msg", key="val")

    @patch("src.storage.faiss_store._HAS_STRUCTURED_LOGGING", True)
    def test_log_error_structured(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger, use_gpu=False)
        store._log_error("msg", key="val")
        mock_logger.error.assert_called_once_with("msg", key="val")

    @patch("src.storage.faiss_store._HAS_STRUCTURED_LOGGING", False)
    def test_log_warning_no_kwargs_unstructured(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger, use_gpu=False)
        store._log_warning("simple")
        mock_logger.warning.assert_called_once_with("simple")

    @patch("src.storage.faiss_store._HAS_STRUCTURED_LOGGING", False)
    def test_log_error_no_kwargs_unstructured(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger, use_gpu=False)
        store._log_error("simple")
        mock_logger.error.assert_called_once_with("simple")

    @patch("src.storage.faiss_store._HAS_STRUCTURED_LOGGING", False)
    def test_log_debug_with_kwargs_unstructured(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger, use_gpu=False)
        store._log_debug("msg", k="v")
        mock_logger.debug.assert_called_once_with("msg {'k': 'v'}")

    @patch("src.storage.faiss_store._HAS_STRUCTURED_LOGGING", False)
    def test_log_info_with_kwargs_unstructured(self):
        mock_logger = MagicMock()
        store = FAISSVectorStore(logger_instance=mock_logger, use_gpu=False)
        store._log_info("msg", k="v")
        mock_logger.info.assert_called_once_with("msg {'k': 'v'}")


# ---------------------------------------------------------------------------
# remove_document edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRemoveDocumentEdge:
    """Extra tests for remove_document (line 461->464)."""

    def test_remove_document_id_not_in_documents_dict(self):
        """Line 461: embedding_id exists in map but not in documents dict."""
        store = FAISSVectorStore(use_gpu=False)
        store._document_id_map["orphan"] = 99
        # 99 is not in store._documents
        result = store.remove_document("orphan")
        assert result is True
        assert "orphan" not in store._document_id_map


# ---------------------------------------------------------------------------
# create_faiss_store factory with persist_dir
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateFaissStoreExtended:
    @patch("src.storage.faiss_store._HAS_FAISS", True)
    def test_with_persist_dir(self):
        with patch.object(FAISSVectorStore, "initialize", return_value=True):
            result = create_faiss_store(persist_dir="/tmp/test_store")
            assert isinstance(result, FAISSVectorStore)
            assert result._persist_dir == Path("/tmp/test_store")

    @patch("src.storage.faiss_store._HAS_FAISS", True)
    def test_with_custom_model_and_index(self):
        result = create_faiss_store(
            model_name="custom-model",
            index_type=IndexType.HNSW,
            auto_initialize=False,
        )
        assert isinstance(result, FAISSVectorStore)
        assert result._model_name == "custom-model"
        assert result._index_type == IndexType.HNSW
