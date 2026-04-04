"""
Unit tests for src/api/local_embedding_store.py.
"""

import hashlib
import logging
from unittest.mock import MagicMock, patch

import pytest

# We need to mock heavy dependencies before importing the module under test.
# The module conditionally imports numpy and sentence_transformers at module level.
# We import it directly and then manipulate the module-level flags for testing.


@pytest.mark.unit
class TestLocalDocument:
    """Tests for LocalDocument dataclass."""

    def test_creation(self):
        from src.api.local_embedding_store import LocalDocument

        doc = LocalDocument(id="d1", content="hello")
        assert doc.id == "d1"
        assert doc.content == "hello"
        assert doc.embedding is None
        assert doc.metadata == {}

    def test_hash_and_equality(self):
        from src.api.local_embedding_store import LocalDocument

        d1 = LocalDocument(id="same", content="a")
        d2 = LocalDocument(id="same", content="b")
        d3 = LocalDocument(id="diff", content="a")
        assert hash(d1) == hash(d2)
        assert d1 == d2
        assert d1 != d3

    def test_equality_non_document(self):
        from src.api.local_embedding_store import LocalDocument

        doc = LocalDocument(id="d1", content="x")
        assert doc != "not a document"


@pytest.mark.unit
class TestLocalEmbeddingDefaults:
    """Tests for LocalEmbeddingDefaults."""

    def test_constants(self):
        from src.api.local_embedding_store import LocalEmbeddingDefaults

        assert LocalEmbeddingDefaults.MODEL_NAME == "all-MiniLM-L6-v2"
        assert LocalEmbeddingDefaults.EMBEDDING_DIM == 384
        assert LocalEmbeddingDefaults.BATCH_SIZE == 32
        assert LocalEmbeddingDefaults.MIN_SCORE == 0.0
        assert LocalEmbeddingDefaults.MAX_SCORE == 1.0
        assert LocalEmbeddingDefaults.DEFAULT_TOP_K == 5
        assert LocalEmbeddingDefaults.MAX_TOP_K == 1000


@pytest.mark.unit
class TestLocalEmbeddingStore:
    """Tests for LocalEmbeddingStore with mocked dependencies."""

    def _make_store(self, **kwargs):
        from src.api.local_embedding_store import LocalEmbeddingStore

        return LocalEmbeddingStore(**kwargs)

    # -- __init__ / properties --

    def test_init_defaults(self):
        store = self._make_store()
        assert store.model_name == "all-MiniLM-L6-v2"
        assert store.embedding_dim == 384
        assert store.is_available is False
        assert store.has_documents is False
        assert store.document_count == 0

    def test_init_custom(self):
        store = self._make_store(model_name="custom-model", embedding_dim=768)
        assert store.model_name == "custom-model"
        assert store.embedding_dim == 768

    # -- initialize --

    def test_initialize_no_numpy(self):
        import src.api.local_embedding_store as mod

        original = mod._HAS_NUMPY
        try:
            mod._HAS_NUMPY = False
            store = self._make_store()
            assert store.initialize() is False
        finally:
            mod._HAS_NUMPY = original

    def test_initialize_no_sentence_transformers(self):
        import src.api.local_embedding_store as mod

        orig_np = mod._HAS_NUMPY
        orig_st = mod._HAS_SENTENCE_TRANSFORMERS
        try:
            mod._HAS_NUMPY = True
            mod._HAS_SENTENCE_TRANSFORMERS = False
            store = self._make_store()
            assert store.initialize() is False
        finally:
            mod._HAS_NUMPY = orig_np
            mod._HAS_SENTENCE_TRANSFORMERS = orig_st

    def test_initialize_already_initialized(self):
        store = self._make_store()
        store._is_initialized = True
        assert store.initialize() is True

    def test_initialize_success(self):
        import numpy as np

        import src.api.local_embedding_store as mod

        orig_np = mod._HAS_NUMPY
        orig_st = mod._HAS_SENTENCE_TRANSFORMERS
        try:
            mod._HAS_NUMPY = True
            mod._HAS_SENTENCE_TRANSFORMERS = True

            mock_model = MagicMock()
            mock_model.encode.return_value = np.zeros(384)

            with patch.object(mod, "SentenceTransformer", return_value=mock_model):
                store = self._make_store()
                result = store.initialize()
                assert result is True
                assert store.is_available is True
        finally:
            mod._HAS_NUMPY = orig_np
            mod._HAS_SENTENCE_TRANSFORMERS = orig_st

    def test_initialize_exception(self):
        import src.api.local_embedding_store as mod

        orig_np = mod._HAS_NUMPY
        orig_st = mod._HAS_SENTENCE_TRANSFORMERS
        try:
            mod._HAS_NUMPY = True
            mod._HAS_SENTENCE_TRANSFORMERS = True

            with patch.object(mod, "SentenceTransformer", side_effect=RuntimeError("fail")):
                store = self._make_store()
                assert store.initialize() is False
        finally:
            mod._HAS_NUMPY = orig_np
            mod._HAS_SENTENCE_TRANSFORMERS = orig_st

    # -- _validate_top_k --

    def test_validate_top_k_none(self):
        store = self._make_store()
        assert store._validate_top_k(None) == 5

    def test_validate_top_k_valid(self):
        store = self._make_store()
        assert store._validate_top_k(10) == 10

    def test_validate_top_k_too_small(self):
        store = self._make_store()
        assert store._validate_top_k(0) == 5
        assert store._validate_top_k(-1) == 5

    def test_validate_top_k_too_large(self):
        store = self._make_store()
        assert store._validate_top_k(2000) == 1000

    # -- _validate_min_score --

    def test_validate_min_score_none(self):
        store = self._make_store()
        assert store._validate_min_score(None) == 0.0

    def test_validate_min_score_valid(self):
        store = self._make_store()
        assert store._validate_min_score(0.5) == 0.5

    def test_validate_min_score_too_low(self):
        store = self._make_store()
        assert store._validate_min_score(-0.1) == 0.0

    def test_validate_min_score_too_high(self):
        store = self._make_store()
        assert store._validate_min_score(1.5) == 1.0

    # -- _generate_doc_id --

    def test_generate_doc_id(self):
        store = self._make_store()
        doc_id = store._generate_doc_id("hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()[:16]
        assert doc_id == expected

    def test_generate_doc_id_deterministic(self):
        store = self._make_store()
        assert store._generate_doc_id("test") == store._generate_doc_id("test")

    def test_generate_doc_id_different_content(self):
        store = self._make_store()
        assert store._generate_doc_id("a") != store._generate_doc_id("b")

    # -- _matches_filter --

    def test_matches_filter_match(self):
        store = self._make_store()
        assert store._matches_filter({"source": "web", "lang": "en"}, {"source": "web"}) is True

    def test_matches_filter_no_match(self):
        store = self._make_store()
        assert store._matches_filter({"source": "web"}, {"source": "api"}) is False

    def test_matches_filter_missing_key(self):
        store = self._make_store()
        assert store._matches_filter({}, {"source": "web"}) is False

    def test_matches_filter_empty_filter(self):
        store = self._make_store()
        assert store._matches_filter({"any": "thing"}, {}) is True

    # -- add_documents --

    def test_add_documents_not_initialized(self):
        import src.api.local_embedding_store as mod

        orig_st = mod._HAS_SENTENCE_TRANSFORMERS
        try:
            mod._HAS_SENTENCE_TRANSFORMERS = False
            store = self._make_store()
            count = store.add_documents([{"content": "hello"}])
            assert count == 0
        finally:
            mod._HAS_SENTENCE_TRANSFORMERS = orig_st

    def test_add_documents_empty_content_skipped(self):
        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        store._model = mock_model
        count = store.add_documents([{"content": ""}, {"content": None}])
        assert count == 0

    def test_add_documents_success(self):
        import numpy as np

        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        store._model = mock_model

        count = store.add_documents([{"content": "hello", "metadata": {"src": "test"}}])
        assert count == 1
        assert store.document_count == 1
        assert store.has_documents is True

    def test_add_documents_skips_duplicates(self):
        import numpy as np

        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        store._model = mock_model

        store.add_documents([{"content": "hello", "id": "d1"}])
        count = store.add_documents([{"content": "world", "id": "d1"}])
        assert count == 0
        assert store.document_count == 1

    def test_add_documents_appends_embeddings(self):
        import numpy as np

        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        store._model = mock_model

        store.add_documents([{"content": "first"}])
        assert store._embeddings is not None
        assert store._embeddings.shape == (1, 3)

        mock_model.encode.return_value = np.array([[0.4, 0.5, 0.6]])
        store.add_documents([{"content": "second"}])
        assert store._embeddings.shape == (2, 3)

    def test_add_documents_invalid_batch_size(self):
        import numpy as np

        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        store._model = mock_model

        count = store.add_documents([{"content": "hello"}], batch_size=0)
        assert count == 1  # Should use default batch size and succeed

    def test_add_documents_encode_exception(self):
        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("encode failed")
        store._model = mock_model

        count = store.add_documents([{"content": "hello"}])
        assert count == 0

    def test_add_documents_model_none_after_init(self):
        store = self._make_store()
        store._is_initialized = True
        store._model = None
        count = store.add_documents([{"content": "hello"}])
        assert count == 0

    # -- search --

    def test_search_not_initialized(self):
        store = self._make_store()
        results = store.search("query")
        assert results == []

    def test_search_no_documents(self):
        store = self._make_store()
        store._is_initialized = True
        results = store.search("query")
        assert results == []

    def test_search_empty_query(self):
        import numpy as np

        store = self._make_store()
        store._is_initialized = True
        store._embeddings = np.array([[0.1]])
        store._documents = [MagicMock()]
        results = store.search("")
        assert results == []
        results2 = store.search("   ")
        assert results2 == []

    def test_search_success(self):
        import numpy as np

        from src.api.local_embedding_store import LocalDocument

        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        query_emb = np.array([1.0, 0.0, 0.0])
        mock_model.encode.return_value = query_emb
        store._model = mock_model

        store._documents = [
            LocalDocument(id="d1", content="hello", metadata={"src": "a"}),
            LocalDocument(id="d2", content="world", metadata={"src": "b"}),
        ]
        store._embeddings = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.0],
            ]
        )

        results = store.search("test", top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "d1"  # higher dot product with [1,0,0]
        assert "content" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]

    def test_search_with_min_score(self):
        import numpy as np

        from src.api.local_embedding_store import LocalDocument

        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0, 0.0])
        store._model = mock_model

        store._documents = [
            LocalDocument(id="d1", content="good"),
            LocalDocument(id="d2", content="bad"),
        ]
        store._embeddings = np.array(
            [
                [0.95, 0.05],
                [0.1, 0.9],
            ]
        )

        results = store.search("q", top_k=10, min_score=0.5)
        assert all(r["score"] >= 0.5 for r in results)

    def test_search_with_metadata_filter(self):
        import numpy as np

        from src.api.local_embedding_store import LocalDocument

        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0, 0.0])
        store._model = mock_model

        store._documents = [
            LocalDocument(id="d1", content="hello", metadata={"lang": "en"}),
            LocalDocument(id="d2", content="bonjour", metadata={"lang": "fr"}),
        ]
        store._embeddings = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
            ]
        )

        results = store.search("test", filter_metadata={"lang": "fr"})
        assert len(results) == 1
        assert results[0]["id"] == "d2"

    def test_search_model_none(self):
        import numpy as np

        from src.api.local_embedding_store import LocalDocument

        store = self._make_store()
        store._is_initialized = True
        store._model = None
        store._documents = [LocalDocument(id="d1", content="x")]
        store._embeddings = np.array([[0.1]])
        results = store.search("q")
        assert results == []

    def test_search_exception(self):
        import numpy as np

        from src.api.local_embedding_store import LocalDocument

        store = self._make_store()
        store._is_initialized = True
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("fail")
        store._model = mock_model
        store._documents = [LocalDocument(id="d1", content="x")]
        store._embeddings = np.array([[0.1]])
        results = store.search("q")
        assert results == []

    # -- remove_document --

    def test_remove_document_success(self):
        import numpy as np

        from src.api.local_embedding_store import LocalDocument

        store = self._make_store()
        store._documents = [
            LocalDocument(id="d1", content="a"),
            LocalDocument(id="d2", content="b"),
        ]
        store._document_ids = {"d1", "d2"}
        store._embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

        assert store.remove_document("d1") is True
        assert store.document_count == 1
        assert "d1" not in store._document_ids
        assert store._embeddings.shape == (1, 2)

    def test_remove_document_not_found(self):
        store = self._make_store()
        assert store.remove_document("missing") is False

    def test_remove_last_document(self):
        import numpy as np

        from src.api.local_embedding_store import LocalDocument

        store = self._make_store()
        store._documents = [LocalDocument(id="d1", content="a")]
        store._document_ids = {"d1"}
        store._embeddings = np.array([[1.0, 0.0]])

        assert store.remove_document("d1") is True
        assert store.document_count == 0
        assert store._embeddings is None

    def test_remove_document_inconsistent_state(self):
        store = self._make_store()
        store._document_ids = {"ghost"}
        store._documents = []
        result = store.remove_document("ghost")
        assert result is False
        assert "ghost" not in store._document_ids

    # -- clear --

    def test_clear(self):
        import numpy as np

        from src.api.local_embedding_store import LocalDocument

        store = self._make_store()
        store._documents = [LocalDocument(id="d1", content="a")]
        store._document_ids = {"d1"}
        store._embeddings = np.array([[1.0]])

        store.clear()
        assert store.document_count == 0
        assert store._embeddings is None
        assert len(store._document_ids) == 0

    # -- logging helpers --

    def test_log_methods_no_structured(self):
        import src.api.local_embedding_store as mod

        orig = mod._HAS_STRUCTURED_LOGGING
        try:
            mod._HAS_STRUCTURED_LOGGING = False
            test_logger = MagicMock(spec=logging.Logger)
            store = self._make_store(logger_instance=test_logger)
            store._log_debug("debug msg", key="val")
            store._log_info("info msg")
            store._log_warning("warn msg", x=1)
            store._log_error("error msg")
            assert test_logger.debug.called
            assert test_logger.info.called
            assert test_logger.warning.called
            assert test_logger.error.called
        finally:
            mod._HAS_STRUCTURED_LOGGING = orig

    def test_log_methods_with_structured(self):
        import src.api.local_embedding_store as mod

        orig = mod._HAS_STRUCTURED_LOGGING
        try:
            mod._HAS_STRUCTURED_LOGGING = True
            test_logger = MagicMock()
            store = self._make_store(logger_instance=test_logger)
            store._log_debug("debug msg", key="val")
            test_logger.debug.assert_called_with("debug msg", key="val")
        finally:
            mod._HAS_STRUCTURED_LOGGING = orig


@pytest.mark.unit
class TestCreateLocalEmbeddingStore:
    """Tests for the factory function."""

    def test_no_numpy(self):
        import src.api.local_embedding_store as mod

        orig_np = mod._HAS_NUMPY
        try:
            mod._HAS_NUMPY = False
            result = mod.create_local_embedding_store(auto_initialize=False)
            assert result is None
        finally:
            mod._HAS_NUMPY = orig_np

    def test_no_sentence_transformers(self):
        import src.api.local_embedding_store as mod

        orig_st = mod._HAS_SENTENCE_TRANSFORMERS
        try:
            mod._HAS_SENTENCE_TRANSFORMERS = False
            result = mod.create_local_embedding_store(auto_initialize=False)
            assert result is None
        finally:
            mod._HAS_SENTENCE_TRANSFORMERS = orig_st

    def test_create_no_auto_init(self):
        import src.api.local_embedding_store as mod

        orig_np = mod._HAS_NUMPY
        orig_st = mod._HAS_SENTENCE_TRANSFORMERS
        try:
            mod._HAS_NUMPY = True
            mod._HAS_SENTENCE_TRANSFORMERS = True
            store = mod.create_local_embedding_store(auto_initialize=False)
            assert store is not None
            assert store.is_available is False
        finally:
            mod._HAS_NUMPY = orig_np
            mod._HAS_SENTENCE_TRANSFORMERS = orig_st

    def test_create_auto_init_fails(self):
        import src.api.local_embedding_store as mod

        orig_np = mod._HAS_NUMPY
        orig_st = mod._HAS_SENTENCE_TRANSFORMERS
        try:
            mod._HAS_NUMPY = True
            mod._HAS_SENTENCE_TRANSFORMERS = True
            with patch.object(mod, "SentenceTransformer", side_effect=RuntimeError("fail")):
                result = mod.create_local_embedding_store(auto_initialize=True)
                assert result is None
        finally:
            mod._HAS_NUMPY = orig_np
            mod._HAS_SENTENCE_TRANSFORMERS = orig_st

    def test_create_with_cache_dir(self):
        import src.api.local_embedding_store as mod

        orig_np = mod._HAS_NUMPY
        orig_st = mod._HAS_SENTENCE_TRANSFORMERS
        try:
            mod._HAS_NUMPY = True
            mod._HAS_SENTENCE_TRANSFORMERS = True
            store = mod.create_local_embedding_store(
                model_name="test-model",
                cache_dir="/tmp/cache",
                auto_initialize=False,
            )
            assert store is not None
            assert store.model_name == "test-model"
        finally:
            mod._HAS_NUMPY = orig_np
            mod._HAS_SENTENCE_TRANSFORMERS = orig_st
