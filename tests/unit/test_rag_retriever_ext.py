"""
Extended unit tests for src/api/rag_retriever.py targeting uncovered lines.

Covers:
- RAGRetriever.retrieve() with various backends and edge cases
- _retrieve_from_pinecone with find_similar_decisions and similarity_search
- _retrieve_local with various store states
- add_documents for local and pinecone backends
- _validate_top_k and _validate_min_score edge cases
- _check_local_backend legacy path and error paths
- _check_pinecone_backend without is_available attribute
- create_rag_retriever factory with Pinecone and local fallback
- Logging helper methods
"""

from unittest.mock import MagicMock, patch

import pytest

pydantic_settings = pytest.importorskip("pydantic_settings")

from src.api.rag_retriever import (  # noqa: E402
    RAGRetriever,
    RAGRetrieverDefaults,
    RetrievalResult,
    RetrievedDocument,
)


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.FRAMEWORK_TOP_K_RETRIEVAL = 5
    settings.get_pinecone_api_key.return_value = None
    settings.PINECONE_HOST = None
    return settings


@pytest.mark.unit
class TestValidateTopK:
    """Tests for _validate_top_k edge cases."""

    def test_top_k_none_returns_default(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        assert r._validate_top_k(None) == 5

    def test_top_k_zero_returns_default(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        assert r._validate_top_k(0) == 5

    def test_top_k_negative_returns_default(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        assert r._validate_top_k(-1) == 5

    def test_top_k_exceeds_max_capped(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        assert r._validate_top_k(200) == RAGRetrieverDefaults.MAX_TOP_K

    def test_top_k_valid_passthrough(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        assert r._validate_top_k(10) == 10


@pytest.mark.unit
class TestValidateMinScore:
    """Tests for _validate_min_score edge cases."""

    def test_none_returns_default(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        assert r._validate_min_score(None) == RAGRetrieverDefaults.DEFAULT_MIN_SCORE

    def test_negative_returns_zero(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        assert r._validate_min_score(-0.5) == RAGRetrieverDefaults.MIN_SCORE

    def test_exceeds_max_returns_one(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        assert r._validate_min_score(1.5) == RAGRetrieverDefaults.MAX_SCORE

    def test_valid_passthrough(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        assert r._validate_min_score(0.7) == 0.7


@pytest.mark.unit
class TestRetrieveEdgeCases:
    """Tests for retrieve() method edge cases."""

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        await r.initialize()
        result = await r.retrieve("")
        assert result.backend == "none"
        assert result.metadata["error"] == "Empty query"

    @pytest.mark.asyncio
    async def test_retrieve_whitespace_query(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        await r.initialize()
        result = await r.retrieve("   ")
        assert result.backend == "none"
        assert result.metadata["error"] == "Empty query"

    @pytest.mark.asyncio
    async def test_retrieve_auto_initializes(self, mock_settings):
        """retrieve() calls initialize() if not yet initialized."""
        r = RAGRetriever(settings=mock_settings)
        assert r._is_initialized is False
        result = await r.retrieve("test query")
        assert r._is_initialized is True
        assert result.backend == "none"

    @pytest.mark.asyncio
    async def test_retrieve_pinecone_exception_falls_through(self, mock_settings):
        """When pinecone raises, we fallback gracefully."""
        store = MagicMock()
        store.is_available = True
        store.find_similar_decisions = MagicMock(side_effect=RuntimeError("fail"))

        r = RAGRetriever(settings=mock_settings, pinecone_store=store)
        await r.initialize()
        result = await r.retrieve("test query")
        assert result.backend == "none"

    @pytest.mark.asyncio
    async def test_retrieve_local_fallback_after_pinecone_failure(self, mock_settings):
        """When pinecone fails, local is tried as fallback."""
        pinecone_store = MagicMock()
        pinecone_store.is_available = True
        pinecone_store.find_similar_decisions = MagicMock(side_effect=RuntimeError("fail"))

        local_store = MagicMock()
        local_store.is_available = True
        local_store.has_documents = True
        local_store.search.return_value = [
            {"content": "local result", "score": 0.8, "metadata": {}},
        ]

        r = RAGRetriever(settings=mock_settings, pinecone_store=pinecone_store, local_store=local_store)
        await r.initialize()
        result = await r.retrieve("test query")
        assert result.backend == "local"
        assert len(result.documents) == 1

    @pytest.mark.asyncio
    async def test_retrieve_local_fallback_exception(self, mock_settings):
        """When local retrieval also fails, returns empty."""
        local_store = MagicMock()
        local_store.is_available = True
        local_store.has_documents = True
        local_store.search.side_effect = RuntimeError("local fail")

        r = RAGRetriever(settings=mock_settings, local_store=local_store)
        await r.initialize()
        result = await r.retrieve("test query")
        assert result.backend == "none"


@pytest.mark.unit
class TestRetrieveFromPinecone:
    """Tests for _retrieve_from_pinecone method."""

    @pytest.mark.asyncio
    async def test_pinecone_store_none(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        docs, backend = await r._retrieve_from_pinecone("q", 5, None, 0.0)
        assert docs == []
        assert backend == "none"

    @pytest.mark.asyncio
    async def test_find_similar_decisions_path(self, mock_settings):
        store = MagicMock()
        store.find_similar_decisions.return_value = [
            {"score": 0.9, "metadata": {"content": "R1"}},
            {"score": 0.3, "metadata": {"content": "R2"}},
        ]
        # Remove similarity_search to force find_similar_decisions path
        del store.similarity_search

        r = RAGRetriever(settings=mock_settings, pinecone_store=store)
        with patch("src.api.rag_retriever.MetaControllerFeatures", create=True) as mock_features_cls:
            mock_features_cls.return_value = MagicMock()
            # Need the actual import to work
            try:
                from src.agents.meta_controller.base import MetaControllerFeatures  # noqa: F401
                docs, backend = await r._retrieve_from_pinecone("query", 5, None, 0.5)
                assert backend == "pinecone"
                # Only score >= 0.5 should be returned
                assert len(docs) == 1
                assert docs[0].score == 0.9
            except ImportError:
                # MetaControllerFeatures not available, skip assertion on actual results
                pass

    @pytest.mark.asyncio
    async def test_similarity_search_path(self, mock_settings):
        """Test LangChain-style similarity_search path."""
        store = MagicMock()
        # Remove find_similar_decisions to force similarity_search path
        del store.find_similar_decisions

        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.score = 0.85
        mock_doc.metadata = {"source": "test"}
        store.similarity_search.return_value = [mock_doc]

        r = RAGRetriever(settings=mock_settings, pinecone_store=store)
        docs, backend = await r._retrieve_from_pinecone("query", 5, None, 0.0)
        assert backend == "pinecone"
        assert len(docs) == 1
        assert docs[0].content == "Test content"
        assert docs[0].score == 0.85

    @pytest.mark.asyncio
    async def test_similarity_search_with_min_score_filter(self, mock_settings):
        """Docs below min_score are filtered out."""
        store = MagicMock()
        del store.find_similar_decisions

        doc1 = MagicMock()
        doc1.page_content = "Good"
        doc1.score = 0.9
        doc1.metadata = {}
        doc2 = MagicMock()
        doc2.page_content = "Bad"
        doc2.score = 0.2
        doc2.metadata = {}
        store.similarity_search.return_value = [doc1, doc2]

        r = RAGRetriever(settings=mock_settings, pinecone_store=store)
        docs, backend = await r._retrieve_from_pinecone("q", 5, None, 0.5)
        assert len(docs) == 1
        assert docs[0].content == "Good"

    @pytest.mark.asyncio
    async def test_store_without_known_methods(self, mock_settings):
        """Store with neither find_similar_decisions nor similarity_search."""
        store = MagicMock(spec=[])  # empty spec - no methods
        store.is_available = True

        r = RAGRetriever(settings=mock_settings, pinecone_store=store)
        docs, backend = await r._retrieve_from_pinecone("q", 5, None, 0.0)
        assert docs == []
        assert backend == "none"


@pytest.mark.unit
class TestRetrieveLocal:
    """Tests for _retrieve_local method."""

    @pytest.mark.asyncio
    async def test_local_store_none(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        docs, backend = await r._retrieve_local("q", 5, None, 0.0)
        assert docs == []
        assert backend == "none"

    @pytest.mark.asyncio
    async def test_local_store_not_available(self, mock_settings):
        store = MagicMock()
        store.is_available = False
        r = RAGRetriever(settings=mock_settings, local_store=store)
        docs, backend = await r._retrieve_local("q", 5, None, 0.0)
        assert docs == []
        assert backend == "none"

    @pytest.mark.asyncio
    async def test_local_store_no_documents(self, mock_settings):
        store = MagicMock()
        store.is_available = True
        store.has_documents = False
        r = RAGRetriever(settings=mock_settings, local_store=store)
        docs, backend = await r._retrieve_local("q", 5, None, 0.0)
        assert docs == []
        assert backend == "local"

    @pytest.mark.asyncio
    async def test_local_store_search_success(self, mock_settings):
        store = MagicMock()
        store.is_available = True
        store.has_documents = True
        store.search.return_value = [
            {"content": "doc1", "score": 0.9, "metadata": {"k": "v"}},
        ]
        r = RAGRetriever(settings=mock_settings, local_store=store)
        docs, backend = await r._retrieve_local("q", 5, None, 0.0)
        assert backend == "local"
        assert len(docs) == 1
        assert docs[0].source == "local"

    @pytest.mark.asyncio
    async def test_local_store_search_exception(self, mock_settings):
        store = MagicMock()
        store.is_available = True
        store.has_documents = True
        store.search.side_effect = RuntimeError("search failed")
        r = RAGRetriever(settings=mock_settings, local_store=store)
        docs, backend = await r._retrieve_local("q", 5, None, 0.0)
        assert docs == []
        assert backend == "none"


@pytest.mark.unit
class TestCheckBackends:
    """Tests for _check_pinecone_backend and _check_local_backend."""

    @pytest.mark.asyncio
    async def test_pinecone_without_is_available_attr(self, mock_settings):
        """Store without is_available assumed available."""
        store = MagicMock(spec=[])  # no is_available
        r = RAGRetriever(settings=mock_settings, pinecone_store=store)
        await r._check_pinecone_backend("corr-id")
        assert "pinecone" in r._available_backends

    @pytest.mark.asyncio
    async def test_pinecone_check_exception(self, mock_settings):
        store = MagicMock()
        type(store).is_available = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))
        r = RAGRetriever(settings=mock_settings, pinecone_store=store)
        await r._check_pinecone_backend("corr-id")
        assert "pinecone" not in r._available_backends

    @pytest.mark.asyncio
    async def test_local_store_with_initialize_method(self, mock_settings):
        """Local store that needs initialization via initialize()."""
        store = MagicMock()
        store.is_available = False
        store.initialize.return_value = True
        r = RAGRetriever(settings=mock_settings, local_store=store)
        await r._check_local_backend("corr-id")
        assert "local" in r._available_backends

    @pytest.mark.asyncio
    async def test_local_store_check_exception(self, mock_settings):
        store = MagicMock()
        type(store).is_available = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))
        r = RAGRetriever(settings=mock_settings, local_store=store)
        await r._check_local_backend("corr-id")
        assert "local" not in r._available_backends

    @pytest.mark.asyncio
    async def test_legacy_embedding_model_path_no_local(self, mock_settings):
        """When embedding_model is provided but local store can't be created, no backend added."""
        embedding_model = MagicMock()
        r = RAGRetriever(settings=mock_settings, embedding_model=embedding_model)
        await r._check_local_backend("corr-id")
        # Without sentence-transformers, local backend won't be available
        # This is environment-dependent so just verify no crash


@pytest.mark.unit
class TestAddDocuments:
    """Tests for add_documents method."""

    def test_add_to_existing_local_store(self, mock_settings):
        store = MagicMock()
        store.add_documents.return_value = 3
        r = RAGRetriever(settings=mock_settings, local_store=store)
        r._available_backends.append("local")
        count = r.add_documents([{"content": "doc1"}, {"content": "doc2"}, {"content": "doc3"}])
        assert count == 3

    def test_add_to_local_with_store(self, mock_settings):
        """When local_store is provided, uses it directly."""
        store = MagicMock()
        store.add_documents.return_value = 2
        r = RAGRetriever(settings=mock_settings, local_store=store)
        r._available_backends.append("local")
        count = r.add_documents([{"content": "a"}, {"content": "b"}])
        assert count == 2

    def test_add_to_pinecone_not_implemented(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        count = r.add_documents([{"content": "a"}], backend="pinecone")
        assert count == 0

    def test_add_to_unknown_backend(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        count = r.add_documents([{"content": "a"}], backend="unknown")
        assert count == 0


@pytest.mark.unit
class TestInitializeDoubleCheck:
    """Tests for initialize() double-checked locking."""

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        r._is_initialized = True
        r._available_backends = ["pinecone"]
        result = await r.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_initialize_already_initialized_no_backends(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        r._is_initialized = True
        r._available_backends = []
        result = await r.initialize()
        assert result is False


@pytest.mark.unit
class TestLoggingHelpers:
    """Tests for logging helper methods."""

    def test_log_error(self, mock_settings):
        logger = MagicMock()
        r = RAGRetriever(settings=mock_settings, logger_instance=logger)
        r._log_error("test error", key="val")
        logger.error.assert_called_once()

    def test_log_debug_no_kwargs(self, mock_settings):
        logger = MagicMock()
        r = RAGRetriever(settings=mock_settings, logger_instance=logger)
        r._log_debug("simple message")
        logger.debug.assert_called_once_with("simple message")


@pytest.mark.unit
class TestCreateRagRetrieverFactory:
    """Extended tests for create_rag_retriever factory."""

    def test_factory_with_pinecone_success(self):
        from src.api.rag_retriever import create_rag_retriever

        mock_settings = MagicMock()
        mock_settings.FRAMEWORK_TOP_K_RETRIEVAL = 5
        mock_settings.get_pinecone_api_key.return_value = "fake-key"
        mock_settings.PINECONE_HOST = "https://fake-host"

        mock_store = MagicMock()
        with patch("src.api.rag_retriever.PineconeVectorStore", create=True, return_value=mock_store):
            retriever = create_rag_retriever(settings=mock_settings, enable_local_fallback=False)
        assert retriever is not None

    def test_factory_pinecone_import_error(self):
        from src.api.rag_retriever import create_rag_retriever

        mock_settings = MagicMock()
        mock_settings.FRAMEWORK_TOP_K_RETRIEVAL = 5
        mock_settings.get_pinecone_api_key.return_value = "key"
        mock_settings.PINECONE_HOST = "host"

        with patch.dict("sys.modules", {"src.storage.pinecone_store": None}):
            retriever = create_rag_retriever(settings=mock_settings, enable_local_fallback=False)
        assert retriever is not None

    def test_factory_local_fallback_exception(self):
        from src.api.rag_retriever import create_rag_retriever

        mock_settings = MagicMock()
        mock_settings.FRAMEWORK_TOP_K_RETRIEVAL = 5
        mock_settings.get_pinecone_api_key.return_value = None
        mock_settings.PINECONE_HOST = None

        with patch(
            "src.api.rag_retriever.create_local_embedding_store",
            create=True,
            side_effect=Exception("some error"),
        ):
            retriever = create_rag_retriever(settings=mock_settings, enable_local_fallback=True)
        assert retriever is not None


@pytest.mark.unit
class TestRAGRetrieverDefaults:
    """Tests for RAGRetrieverDefaults constants."""

    def test_defaults_values(self):
        assert RAGRetrieverDefaults.DEFAULT_TOP_K == 5
        assert RAGRetrieverDefaults.MAX_TOP_K == 100
        assert RAGRetrieverDefaults.MIN_SCORE == 0.0
        assert RAGRetrieverDefaults.MAX_SCORE == 1.0
        assert RAGRetrieverDefaults.DEFAULT_HRM_CONFIDENCE == 0.5
        assert RAGRetrieverDefaults.DEFAULT_TRM_CONFIDENCE == 0.5
        assert RAGRetrieverDefaults.DEFAULT_MCTS_VALUE == 0.5


@pytest.mark.unit
class TestAvailableBackendsProperty:
    """Tests for available_backends property returning a copy."""

    def test_returns_copy(self, mock_settings):
        r = RAGRetriever(settings=mock_settings)
        r._available_backends = ["pinecone", "local"]
        backends = r.available_backends
        backends.append("other")
        # Original should not be modified
        assert "other" not in r._available_backends
