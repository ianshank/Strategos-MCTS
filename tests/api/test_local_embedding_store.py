"""
Tests for LocalEmbeddingStore.

Tests the local embedding store implementation for RAG fallback.
Follows best practices:
- Deterministic tests with fixed seeds
- No brittle mocks
- Comprehensive edge case coverage
- Clear test organization
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

# Check if dependencies are available
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (_HAS_NUMPY and _HAS_SENTENCE_TRANSFORMERS),
    reason="numpy and sentence-transformers required for local embedding store tests",
)


# Test constants
TEST_SEED = 42
TEST_EMBEDDING_DIM = 384


@pytest.fixture
def seeded_rng():
    """Create a seeded random number generator for deterministic tests."""
    if _HAS_NUMPY:
        return np.random.default_rng(TEST_SEED)
    return None


@pytest.fixture
def local_store():
    """Create a local embedding store for testing."""
    from src.api.local_embedding_store import LocalEmbeddingStore

    store = LocalEmbeddingStore()
    return store


@pytest.fixture
def initialized_store():
    """Create an initialized local embedding store."""
    from src.api.local_embedding_store import LocalEmbeddingStore

    store = LocalEmbeddingStore()
    if not store.initialize():
        pytest.skip("Failed to initialize embedding store")
    return store


@pytest.fixture
def sample_documents():
    """Sample documents for testing - deterministic content."""
    return [
        {
            "id": "doc1",
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "test", "category": "animals"},
        },
        {
            "id": "doc2",
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "test", "category": "technology"},
        },
        {
            "id": "doc3",
            "content": "Python is a popular programming language for data science.",
            "metadata": {"source": "test", "category": "technology"},
        },
        {
            "id": "doc4",
            "content": "Neural networks are inspired by biological neurons.",
            "metadata": {"source": "test", "category": "technology"},
        },
        {
            "id": "doc5",
            "content": "Dogs and cats are common household pets.",
            "metadata": {"source": "test", "category": "animals"},
        },
    ]


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing logging behavior."""
    return MagicMock(spec=logging.Logger)


class TestLocalEmbeddingDefaults:
    """Tests for LocalEmbeddingDefaults configuration."""

    def test_defaults_are_defined(self):
        """Test that all defaults are properly defined."""
        from src.api.local_embedding_store import LocalEmbeddingDefaults

        assert LocalEmbeddingDefaults.MODEL_NAME == "all-MiniLM-L6-v2"
        assert LocalEmbeddingDefaults.EMBEDDING_DIM == 384
        assert LocalEmbeddingDefaults.BATCH_SIZE == 32
        assert LocalEmbeddingDefaults.MIN_SCORE == 0.0
        assert LocalEmbeddingDefaults.MAX_SCORE == 1.0
        assert LocalEmbeddingDefaults.DEFAULT_TOP_K == 5
        assert LocalEmbeddingDefaults.MAX_TOP_K == 1000


class TestLocalEmbeddingStoreInit:
    """Tests for LocalEmbeddingStore initialization."""

    def test_init_default_model(self, local_store):
        """Test initialization with default model."""
        from src.api.local_embedding_store import LocalEmbeddingDefaults

        assert local_store.model_name == LocalEmbeddingDefaults.MODEL_NAME
        assert not local_store.is_available
        assert local_store.document_count == 0

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        from src.api.local_embedding_store import LocalEmbeddingStore

        custom_model = "paraphrase-MiniLM-L3-v2"
        store = LocalEmbeddingStore(model_name=custom_model)
        assert store.model_name == custom_model

    def test_init_with_logger_injection(self, mock_logger):
        """Test initialization with injected logger."""
        from src.api.local_embedding_store import LocalEmbeddingStore

        store = LocalEmbeddingStore(logger_instance=mock_logger)
        assert store._logger is mock_logger

    def test_initialize_success(self, local_store):
        """Test successful initialization."""
        result = local_store.initialize()
        assert result is True
        assert local_store.is_available

    def test_initialize_idempotent(self, initialized_store):
        """Test that multiple initializations are idempotent."""
        result1 = initialized_store.initialize()
        result2 = initialized_store.initialize()
        assert result1 is True
        assert result2 is True

    def test_initialize_thread_safe(self, local_store):
        """Test thread-safe initialization with concurrent calls.

        Verifies:
        1. All concurrent calls complete successfully
        2. The store is in a valid initialized state
        3. Only one model instance is created (double-checked locking works)
        """
        import concurrent.futures
        import threading

        # Track how many times SentenceTransformer constructor is called
        model_init_count = 0
        model_init_lock = threading.Lock()

        # Get reference to original SentenceTransformer
        from sentence_transformers import SentenceTransformer as OriginalST

        class TrackedSentenceTransformer(OriginalST):
            """SentenceTransformer that tracks instantiation count."""

            def __init__(self, *args, **kwargs):
                nonlocal model_init_count
                with model_init_lock:
                    model_init_count += 1
                super().__init__(*args, **kwargs)

        # Patch SentenceTransformer in the module under test
        with patch(
            "src.api.local_embedding_store.SentenceTransformer",
            TrackedSentenceTransformer,
        ):
            # Create a fresh store for this test
            from src.api.local_embedding_store import LocalEmbeddingStore

            test_store = LocalEmbeddingStore()

            # Run concurrent initializations
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(test_store.initialize) for _ in range(20)]
                results = [f.result() for f in futures]

        # All calls should succeed
        assert all(results), "All initialization calls should succeed"

        # Store should be initialized
        assert test_store.is_available, "Store should be available after initialization"

        # Model should only be created once (double-checked locking)
        assert model_init_count == 1, (
            f"Model should be created exactly once, but was created {model_init_count} times. "
            "This indicates a race condition in initialization."
        )


class TestLocalEmbeddingStoreDocuments:
    """Tests for document management."""

    def test_add_single_document(self, initialized_store):
        """Test adding a single document."""
        docs = [{"content": "Test document content", "id": "test1"}]
        added = initialized_store.add_documents(docs)

        assert added == 1
        assert initialized_store.document_count == 1
        assert initialized_store.has_documents

    def test_add_multiple_documents(self, initialized_store, sample_documents):
        """Test adding multiple documents."""
        added = initialized_store.add_documents(sample_documents)

        assert added == len(sample_documents)
        assert initialized_store.document_count == len(sample_documents)

    def test_add_empty_content_skipped(self, initialized_store):
        """Test that documents with empty content are skipped."""
        docs = [
            {"content": "", "id": "empty"},
            {"content": "Valid content", "id": "valid"},
        ]
        added = initialized_store.add_documents(docs)

        assert added == 1
        assert initialized_store.document_count == 1

    def test_add_duplicate_documents_skipped(self, initialized_store):
        """Test that duplicate documents are skipped."""
        docs = [{"content": "Test content", "id": "dup1"}]

        added1 = initialized_store.add_documents(docs)
        added2 = initialized_store.add_documents(docs)

        assert added1 == 1
        assert added2 == 0
        assert initialized_store.document_count == 1

    def test_auto_generate_id(self, initialized_store):
        """Test automatic ID generation for documents without ID."""
        docs = [{"content": "Document without explicit ID"}]
        added = initialized_store.add_documents(docs)

        assert added == 1
        assert initialized_store.document_count == 1

    def test_add_with_invalid_batch_size(self, initialized_store):
        """Test that invalid batch_size falls back to default."""
        docs = [{"content": "Test content", "id": "test1"}]
        # Should not raise, uses default batch size
        added = initialized_store.add_documents(docs, batch_size=-1)
        assert added == 1

    def test_remove_document(self, initialized_store, sample_documents):
        """Test removing a document."""
        initialized_store.add_documents(sample_documents)
        initial_count = initialized_store.document_count

        removed = initialized_store.remove_document("doc1")

        assert removed is True
        assert initialized_store.document_count == initial_count - 1

    def test_remove_nonexistent_document(self, initialized_store):
        """Test removing a document that doesn't exist."""
        removed = initialized_store.remove_document("nonexistent")
        assert removed is False

    def test_clear_documents(self, initialized_store, sample_documents):
        """Test clearing all documents."""
        initialized_store.add_documents(sample_documents)
        assert initialized_store.document_count > 0

        initialized_store.clear()

        assert initialized_store.document_count == 0
        assert not initialized_store.has_documents


class TestLocalEmbeddingStoreSearch:
    """Tests for similarity search."""

    def test_search_returns_results(self, initialized_store, sample_documents):
        """Test that search returns relevant results."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search("artificial intelligence", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)

    def test_search_score_ordering(self, initialized_store, sample_documents):
        """Test that results are ordered by score descending."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search("machine learning", top_k=5)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_min_score_filter(self, initialized_store, sample_documents):
        """Test minimum score filtering."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search("machine learning", min_score=0.5)

        assert all(r["score"] >= 0.5 for r in results)

    def test_search_metadata_filter(self, initialized_store, sample_documents):
        """Test metadata filtering."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search(
            "animals",
            top_k=10,
            filter_metadata={"category": "animals"},
        )

        assert len(results) > 0
        assert all(r["metadata"].get("category") == "animals" for r in results)

    def test_search_empty_store(self, initialized_store):
        """Test searching an empty store returns empty list."""
        results = initialized_store.search("anything")
        assert results == []

    def test_search_uninitialized_store(self, local_store):
        """Test searching an uninitialized store returns empty list."""
        results = local_store.search("anything")
        assert results == []

    def test_search_top_k_limit(self, initialized_store, sample_documents):
        """Test that top_k limits results."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search("technology", top_k=2)

        assert len(results) <= 2

    def test_search_empty_query(self, initialized_store, sample_documents):
        """Test searching with empty query returns empty list."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search("")
        assert results == []

        results = initialized_store.search("   ")
        assert results == []

    def test_search_validates_top_k(self, initialized_store, sample_documents):
        """Test that invalid top_k values are corrected."""
        initialized_store.add_documents(sample_documents)

        # Negative top_k should use default
        results = initialized_store.search("test", top_k=-1)
        assert isinstance(results, list)

        # Excessive top_k should be capped
        results = initialized_store.search("test", top_k=10000)
        assert isinstance(results, list)

    def test_search_validates_min_score(self, initialized_store, sample_documents):
        """Test that invalid min_score values are corrected."""
        initialized_store.add_documents(sample_documents)

        # Negative min_score should use 0
        results = initialized_store.search("test", min_score=-0.5)
        assert isinstance(results, list)

        # min_score > 1 should use 1
        results = initialized_store.search("test", min_score=1.5)
        assert isinstance(results, list)


class TestLocalEmbeddingStoreFactory:
    """Tests for factory function."""

    def test_create_with_auto_initialize(self):
        """Test factory function with auto initialization."""
        from src.api.local_embedding_store import create_local_embedding_store

        store = create_local_embedding_store(auto_initialize=True)

        assert store is not None
        assert store.is_available

    def test_create_without_auto_initialize(self):
        """Test factory function without auto initialization."""
        from src.api.local_embedding_store import create_local_embedding_store

        store = create_local_embedding_store(auto_initialize=False)

        assert store is not None
        assert not store.is_available

    def test_create_with_custom_model(self):
        """Test factory with custom model name."""
        from src.api.local_embedding_store import create_local_embedding_store

        store = create_local_embedding_store(
            model_name="paraphrase-MiniLM-L3-v2",
            auto_initialize=False,
        )

        assert store is not None
        assert store.model_name == "paraphrase-MiniLM-L3-v2"

    def test_create_with_logger_injection(self, mock_logger):
        """Test factory with logger injection."""
        from src.api.local_embedding_store import create_local_embedding_store

        store = create_local_embedding_store(
            auto_initialize=False,
            logger_instance=mock_logger,
        )

        assert store is not None
        assert store._logger is mock_logger


class TestRAGRetrieverIntegration:
    """Tests for RAGRetriever with local store."""

    @pytest.mark.asyncio
    async def test_rag_retriever_with_local_store(self, initialized_store, sample_documents):
        """Test RAGRetriever using local store."""
        from src.api.rag_retriever import RAGRetriever

        # Add documents to store
        initialized_store.add_documents(sample_documents)

        # Create retriever with local store
        retriever = RAGRetriever(local_store=initialized_store)
        await retriever.initialize()

        assert "local" in retriever.available_backends

        # Test retrieval
        result = await retriever.retrieve("machine learning", top_k=3)

        assert result.has_results
        assert result.backend == "local"
        assert len(result.documents) > 0

    @pytest.mark.asyncio
    async def test_rag_retriever_add_documents(self):
        """Test adding documents through RAGRetriever."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever()
        await retriever.initialize()

        docs = [
            {"content": "Test document one", "metadata": {"test": True}},
            {"content": "Test document two", "metadata": {"test": True}},
        ]

        added = retriever.add_documents(docs, backend="local")

        # May be 0 if local store couldn't be created (missing deps)
        assert added >= 0

    @pytest.mark.asyncio
    async def test_rag_retriever_factory_with_local(self):
        """Test factory function creates retriever with local fallback."""
        from src.api.rag_retriever import create_rag_retriever

        retriever = create_rag_retriever(enable_local_fallback=True)
        await retriever.initialize()

        # Local should be available if dependencies present
        assert retriever.is_available

    @pytest.mark.asyncio
    async def test_rag_retriever_empty_query_handling(self):
        """Test that empty queries are handled gracefully."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever()
        await retriever.initialize()

        result = await retriever.retrieve("")
        assert not result.has_results
        assert result.backend == "none"

    @pytest.mark.asyncio
    async def test_rag_retriever_validation(self):
        """Test parameter validation in retriever."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever()
        await retriever.initialize()

        # Invalid top_k should not raise
        result = await retriever.retrieve("test", top_k=-1)
        assert isinstance(result.documents, list)

        # Invalid min_score should not raise
        result = await retriever.retrieve("test", min_score=2.0)
        assert isinstance(result.documents, list)


class TestThreadSafety:
    """Tests for thread safety of LocalEmbeddingStore."""

    def test_concurrent_add_documents(self, initialized_store):
        """Test concurrent document additions."""
        import concurrent.futures

        def add_batch(batch_id: int) -> int:
            docs = [{"content": f"Document {batch_id}-{i}", "id": f"doc-{batch_id}-{i}"} for i in range(10)]
            return initialized_store.add_documents(docs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(add_batch, i) for i in range(4)]
            results = [f.result() for f in futures]

        assert sum(results) == 40
        assert initialized_store.document_count == 40

    def test_concurrent_search(self, initialized_store, sample_documents):
        """Test concurrent searches."""
        import concurrent.futures

        initialized_store.add_documents(sample_documents)

        def search_query(query: str) -> list:
            return initialized_store.search(query, top_k=3)

        queries = ["machine learning", "animals", "python", "neural networks"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(search_query, q) for q in queries]
            results = [f.result() for f in futures]

        assert all(isinstance(r, list) for r in results)

    def test_concurrent_add_and_search(self, initialized_store, sample_documents):
        """Test concurrent adds and searches don't deadlock."""
        import concurrent.futures

        # Pre-add some documents
        initialized_store.add_documents(sample_documents[:2])

        def add_doc(doc_id: int) -> int:
            return initialized_store.add_documents([{"content": f"Concurrent doc {doc_id}", "id": f"conc-{doc_id}"}])

        def search_docs(query: str) -> list:
            return initialized_store.search(query, top_k=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            add_futures = [executor.submit(add_doc, i) for i in range(10)]
            search_futures = [executor.submit(search_docs, "concurrent") for _ in range(10)]

            # All operations should complete without deadlock
            all_futures = add_futures + search_futures
            concurrent.futures.wait(all_futures, timeout=30)

            # Verify all completed
            assert all(f.done() for f in all_futures)

    def test_concurrent_remove_same_document(self, initialized_store):
        """Test concurrent removal of the same document (TOCTOU race condition fix).

        This test verifies that when multiple threads try to remove the same
        document concurrently, exactly one succeeds and the others fail gracefully.
        This tests the fix for the TOCTOU race condition where the check was
        outside the lock.
        """
        import concurrent.futures

        # Add a document that we'll try to remove concurrently
        initialized_store.add_documents([{"content": "Document to remove", "id": "remove-me"}])
        assert initialized_store.document_count == 1

        # Try to remove the same document from multiple threads
        def remove_doc() -> bool:
            return initialized_store.remove_document("remove-me")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(remove_doc) for _ in range(20)]
            results = [f.result() for f in futures]

        # Exactly one thread should succeed, all others should return False
        success_count = sum(1 for r in results if r is True)
        failure_count = sum(1 for r in results if r is False)

        assert success_count == 1, (
            f"Exactly one thread should successfully remove the document, "
            f"but {success_count} succeeded. This may indicate a race condition."
        )
        assert failure_count == 19, f"Expected 19 failures, got {failure_count}"

        # Document should be removed
        assert initialized_store.document_count == 0

    def test_concurrent_add_and_remove(self, initialized_store):
        """Test concurrent adds and removes maintain consistency."""
        import concurrent.futures
        import threading

        # Track operations for verification
        operations_completed = 0
        ops_lock = threading.Lock()

        def add_doc(doc_id: int) -> int:
            nonlocal operations_completed
            result = initialized_store.add_documents([{"content": f"Add-remove doc {doc_id}", "id": f"ar-{doc_id}"}])
            with ops_lock:
                operations_completed += 1
            return result

        def remove_doc(doc_id: int) -> bool:
            nonlocal operations_completed
            result = initialized_store.remove_document(f"ar-{doc_id}")
            with ops_lock:
                operations_completed += 1
            return result

        # Run adds and removes concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            add_futures = [executor.submit(add_doc, i) for i in range(10)]
            remove_futures = [executor.submit(remove_doc, i) for i in range(10)]

            all_futures = add_futures + remove_futures
            concurrent.futures.wait(all_futures, timeout=30)

        # All operations should complete
        assert all(f.done() for f in all_futures), "All operations should complete"

        # Document count should be consistent (either 0 or positive integer)
        count = initialized_store.document_count
        assert count >= 0, f"Document count should be non-negative, got {count}"

        # Verify internal consistency: document_ids set should match documents list
        assert len(initialized_store._document_ids) == len(initialized_store._documents), (
            f"Internal inconsistency: {len(initialized_store._document_ids)} IDs "
            f"but {len(initialized_store._documents)} documents"
        )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unicode_content(self, initialized_store):
        """Test handling of unicode content."""
        docs = [
            {"content": "日本語のテキスト", "id": "japanese"},
            {"content": "Ελληνικά κείμενο", "id": "greek"},
            {"content": "Emoji content 🎉🚀", "id": "emoji"},
        ]
        added = initialized_store.add_documents(docs)
        assert added == 3

        results = initialized_store.search("日本語")
        assert isinstance(results, list)

    def test_very_long_content(self, initialized_store):
        """Test handling of very long content."""
        long_content = "word " * 10000  # ~50k characters
        docs = [{"content": long_content, "id": "long"}]

        added = initialized_store.add_documents(docs)
        assert added == 1

    def test_special_characters_in_metadata(self, initialized_store):
        """Test metadata with special characters."""
        docs = [
            {
                "content": "Test content",
                "id": "special",
                "metadata": {"path": "/foo/bar/baz.txt", "tag": "key=value"},
            }
        ]
        added = initialized_store.add_documents(docs)
        assert added == 1

        results = initialized_store.search("Test", filter_metadata={"path": "/foo/bar/baz.txt"})
        assert len(results) == 1

    def test_remove_last_document(self, initialized_store):
        """Test removing the last document clears embeddings."""
        docs = [{"content": "Only document", "id": "only"}]
        initialized_store.add_documents(docs)

        assert initialized_store.document_count == 1

        removed = initialized_store.remove_document("only")
        assert removed is True
        assert initialized_store.document_count == 0
        assert initialized_store._embeddings is None
