"""
Unit tests for RAG Retriever module.

Tests:
- RAGRetriever initialization
- Document retrieval
- Context formatting
- Backend fallback
- Error handling
"""

from unittest.mock import MagicMock, patch

import pytest

# Skip if pydantic_settings not available
pydantic_settings = pytest.importorskip("pydantic_settings")


class TestRetrievedDocument:
    """Tests for RetrievedDocument dataclass."""

    def test_create_document(self):
        """Test creating a retrieved document."""
        from src.api.rag_retriever import RetrievedDocument

        doc = RetrievedDocument(
            content="Test content",
            score=0.95,
            metadata={"source": "test"},
            source="pinecone",
        )

        assert doc.content == "Test content"
        assert doc.score == 0.95
        assert doc.metadata == {"source": "test"}
        assert doc.source == "pinecone"

    def test_to_dict(self):
        """Test converting document to dictionary."""
        from src.api.rag_retriever import RetrievedDocument

        doc = RetrievedDocument(
            content="Test content",
            score=0.85,
            metadata={"key": "value"},
            source="local",
        )

        result = doc.to_dict()

        assert result["content"] == "Test content"
        assert result["score"] == 0.85
        assert result["metadata"] == {"key": "value"}
        assert result["source"] == "local"

    def test_default_values(self):
        """Test default values for document."""
        from src.api.rag_retriever import RetrievedDocument

        doc = RetrievedDocument(content="Content", score=0.5)

        assert doc.metadata == {}
        assert doc.source == "unknown"


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_context_property_empty(self):
        """Test context property with no documents."""
        from src.api.rag_retriever import RetrievalResult

        result = RetrievalResult(
            documents=[],
            query="test query",
            retrieval_time_ms=10.5,
            backend="none",
        )

        assert result.context == ""
        assert result.has_results is False

    def test_context_property_with_documents(self):
        """Test context property with documents."""
        from src.api.rag_retriever import RetrievalResult, RetrievedDocument

        docs = [
            RetrievedDocument(content="First document", score=0.9),
            RetrievedDocument(content="Second document", score=0.8),
        ]

        result = RetrievalResult(
            documents=docs,
            query="test query",
            retrieval_time_ms=25.0,
            backend="pinecone",
        )

        context = result.context

        assert "First document" in context
        assert "Second document" in context
        assert "0.900" in context  # Score included
        assert result.has_results is True

    def test_has_results(self):
        """Test has_results property."""
        from src.api.rag_retriever import RetrievalResult, RetrievedDocument

        empty_result = RetrievalResult(
            documents=[],
            query="test",
            retrieval_time_ms=5.0,
            backend="none",
        )

        assert empty_result.has_results is False

        non_empty = RetrievalResult(
            documents=[RetrievedDocument(content="doc", score=0.5)],
            query="test",
            retrieval_time_ms=5.0,
            backend="local",
        )

        assert non_empty.has_results is True


class TestRAGRetriever:
    """Tests for RAGRetriever class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.FRAMEWORK_TOP_K_RETRIEVAL = 5
        settings.get_pinecone_api_key.return_value = None
        settings.PINECONE_HOST = None
        return settings

    @pytest.fixture
    def mock_pinecone_store(self):
        """Create mock Pinecone store."""
        store = MagicMock()
        store.is_available = True
        store.find_similar_decisions = MagicMock(
            return_value=[
                {"score": 0.95, "metadata": {"content": "Result 1"}},
                {"score": 0.85, "metadata": {"content": "Result 2"}},
            ]
        )
        return store

    def test_init_without_backends(self, mock_settings):
        """Test initialization without any backends."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever(settings=mock_settings)

        assert retriever._settings == mock_settings
        assert retriever._pinecone_store is None
        assert retriever._is_initialized is False

    def test_init_with_pinecone(self, mock_settings, mock_pinecone_store):
        """Test initialization with Pinecone store."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever(
            settings=mock_settings,
            pinecone_store=mock_pinecone_store,
        )

        assert retriever._pinecone_store == mock_pinecone_store

    @pytest.mark.asyncio
    async def test_initialize_with_pinecone(self, mock_settings, mock_pinecone_store):
        """Test initialization marks Pinecone as available."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever(
            settings=mock_settings,
            pinecone_store=mock_pinecone_store,
        )

        result = await retriever.initialize()

        assert result is True
        assert retriever._is_initialized is True
        assert "pinecone" in retriever.available_backends

    @pytest.mark.asyncio
    async def test_initialize_without_backends(self, mock_settings):
        """Test initialization without backends."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever(settings=mock_settings)

        result = await retriever.initialize()

        assert result is False
        assert retriever._is_initialized is True
        assert len(retriever.available_backends) == 0

    @pytest.mark.asyncio
    async def test_retrieve_empty_when_no_backends(self, mock_settings):
        """Test retrieval returns empty when no backends available."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever(settings=mock_settings)
        await retriever.initialize()

        result = await retriever.retrieve("test query")

        assert result.has_results is False
        assert result.backend == "none"

    @pytest.mark.asyncio
    async def test_retrieve_with_pinecone(self, mock_settings, mock_pinecone_store):
        """Test retrieval from Pinecone."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever(
            settings=mock_settings,
            pinecone_store=mock_pinecone_store,
        )
        await retriever.initialize()

        result = await retriever.retrieve("test query", top_k=2)

        assert result.query == "test query"
        assert result.retrieval_time_ms > 0
        # Note: actual document count depends on mock implementation

    def test_format_context_empty(self, mock_settings):
        """Test formatting empty result."""
        from src.api.rag_retriever import RAGRetriever, RetrievalResult

        retriever = RAGRetriever(settings=mock_settings)
        result = RetrievalResult(
            documents=[],
            query="test",
            retrieval_time_ms=5.0,
            backend="none",
        )

        formatted = retriever.format_context(result)

        assert formatted == ""

    def test_format_context_with_documents(self, mock_settings):
        """Test formatting result with documents."""
        from src.api.rag_retriever import RAGRetriever, RetrievalResult, RetrievedDocument

        retriever = RAGRetriever(settings=mock_settings)
        docs = [
            RetrievedDocument(content="Document one", score=0.9),
            RetrievedDocument(content="Document two", score=0.8),
        ]
        result = RetrievalResult(
            documents=docs,
            query="test",
            retrieval_time_ms=10.0,
            backend="pinecone",
        )

        formatted = retriever.format_context(result)

        assert "Document one" in formatted
        assert "Document two" in formatted
        assert "[Source 1]" in formatted
        assert "[Source 2]" in formatted

    def test_format_context_truncation(self, mock_settings):
        """Test context truncation."""
        from src.api.rag_retriever import RAGRetriever, RetrievalResult, RetrievedDocument

        retriever = RAGRetriever(settings=mock_settings)
        docs = [
            RetrievedDocument(content="A" * 1000, score=0.9),
        ]
        result = RetrievalResult(
            documents=docs,
            query="test",
            retrieval_time_ms=10.0,
            backend="local",
        )

        formatted = retriever.format_context(result, max_length=100)

        assert len(formatted) <= 100
        assert formatted.endswith("...")

    def test_format_context_without_scores(self, mock_settings):
        """Test formatting without scores."""
        from src.api.rag_retriever import RAGRetriever, RetrievalResult, RetrievedDocument

        retriever = RAGRetriever(settings=mock_settings)
        docs = [
            RetrievedDocument(content="Test doc", score=0.9),
        ]
        result = RetrievalResult(
            documents=docs,
            query="test",
            retrieval_time_ms=5.0,
            backend="local",
        )

        formatted = retriever.format_context(result, include_scores=False)

        assert "[Source 1]" in formatted
        assert "relevance" not in formatted

    def test_is_available_property(self, mock_settings, mock_pinecone_store):
        """Test is_available property."""
        from src.api.rag_retriever import RAGRetriever

        # Without backends
        retriever_empty = RAGRetriever(settings=mock_settings)
        assert retriever_empty.is_available is False

        # With backend
        retriever_with_backend = RAGRetriever(
            settings=mock_settings,
            pinecone_store=mock_pinecone_store,
        )
        retriever_with_backend._available_backends = ["pinecone"]
        assert retriever_with_backend.is_available is True


class TestCreateRagRetriever:
    """Tests for create_rag_retriever factory function."""

    def test_creates_retriever(self):
        """Test factory creates retriever."""
        with patch("src.api.rag_retriever.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.FRAMEWORK_TOP_K_RETRIEVAL = 5
            mock_settings.get_pinecone_api_key.return_value = None
            mock_settings.PINECONE_HOST = None
            mock_get_settings.return_value = mock_settings

            from src.api.rag_retriever import create_rag_retriever

            retriever = create_rag_retriever(settings=mock_settings)

            assert retriever is not None
            assert retriever._settings == mock_settings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
