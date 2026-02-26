"""
Local Embedding Store for RAG Fallback.

Provides local document storage and similarity search using sentence-transformers.
This enables RAG functionality even when cloud vector stores (Pinecone) are unavailable.

Best Practices 2025:
- Configuration via Settings (no hardcoded values)
- Graceful degradation without dependencies
- Efficient numpy-based similarity search
- Thread-safe document management with proper locking
- Structured logging with correlation IDs
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Module-level logger (will be replaced with structured logger if available)
logger: logging.Logger = logging.getLogger(__name__)

# Try to import structured logging
_HAS_STRUCTURED_LOGGING: bool = False
try:
    from src.observability.logging import get_structured_logger

    logger = get_structured_logger(__name__)  # type: ignore[assignment]
    _HAS_STRUCTURED_LOGGING = True
except ImportError:
    pass

# Try to import numpy - required for this module
_HAS_NUMPY: bool = False
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment,unused-ignore]

# Try to import sentence-transformers
_HAS_SENTENCE_TRANSFORMERS: bool = False
try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc,unused-ignore]


# Configuration constants (can be overridden via Settings)
class LocalEmbeddingDefaults:
    """Default configuration values for local embedding store."""

    MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: Final[int] = 384
    BATCH_SIZE: Final[int] = 32
    MIN_SCORE: Final[float] = 0.0
    MAX_SCORE: Final[float] = 1.0
    DEFAULT_TOP_K: Final[int] = 5
    MAX_TOP_K: Final[int] = 1000


@dataclass
class LocalDocument:
    """Document stored in local index."""

    id: str
    content: str
    embedding: NDArray[np.floating[Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash based on document ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on document ID."""
        if not isinstance(other, LocalDocument):
            return False
        return self.id == other.id


class LocalEmbeddingStore:
    """
    Local embedding store using sentence-transformers and numpy.

    Provides:
    - Document embedding with sentence-transformers
    - Similarity search with cosine similarity
    - In-memory storage with optional persistence
    - Thread-safe operations with proper locking

    Usage:
        store = LocalEmbeddingStore()
        if store.initialize():
            store.add_documents([{"content": "Hello world", "metadata": {"source": "test"}}])
            results = store.search("Hello", top_k=5)
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: Path | None = None,
        embedding_dim: int | None = None,
        logger_instance: logging.Logger | None = None,
    ):
        """
        Initialize local embedding store.

        Args:
            model_name: Sentence transformer model name (defaults from LocalEmbeddingDefaults)
            cache_dir: Optional directory for caching models
            embedding_dim: Embedding dimension (auto-detected from model if not specified)
            logger_instance: Optional logger instance for dependency injection
        """
        self._model_name = model_name or LocalEmbeddingDefaults.MODEL_NAME
        self._cache_dir = cache_dir
        self._embedding_dim = embedding_dim or LocalEmbeddingDefaults.EMBEDDING_DIM
        self._model: SentenceTransformer | None = None
        self._documents: list[LocalDocument] = []
        self._embeddings: NDArray[np.floating[Any]] | None = None
        self._document_ids: set[str] = set()
        self._is_initialized = False
        self._lock = threading.RLock()
        self._logger = logger_instance or logger

    def initialize(self) -> bool:
        """
        Initialize the embedding model.

        Uses double-checked locking pattern for thread safety.

        Returns:
            True if initialization successful, False otherwise
        """
        # Fast path: already initialized
        if self._is_initialized:
            return True

        # Check dependencies before acquiring lock
        if not _HAS_NUMPY:
            self._log_warning("numpy not installed, local embedding store unavailable")
            return False

        if not _HAS_SENTENCE_TRANSFORMERS:
            self._log_warning("sentence-transformers not installed, local embedding store unavailable")
            return False

        # Thread-safe initialization with double-checked locking
        with self._lock:
            # Re-check after acquiring lock (another thread may have initialized)
            if self._is_initialized:
                return True

            try:
                self._model = SentenceTransformer(
                    self._model_name,
                    cache_folder=str(self._cache_dir) if self._cache_dir else None,
                )

                # Auto-detect embedding dimension
                test_embedding = self._model.encode("test", convert_to_numpy=True)
                self._embedding_dim = int(test_embedding.shape[0])

                self._is_initialized = True
                self._log_info(
                    "Initialized local embedding model",
                    model_name=self._model_name,
                    embedding_dim=self._embedding_dim,
                )
                return True
            except Exception as e:
                self._log_error("Failed to initialize embedding model", error=str(e))
                return False

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int | None = None,
    ) -> int:
        """
        Add documents to the store.

        Args:
            documents: List of dicts with 'content' and optional 'metadata', 'id'
            batch_size: Batch size for embedding generation (defaults from config)

        Returns:
            Number of documents added
        """
        batch_size = batch_size or LocalEmbeddingDefaults.BATCH_SIZE

        # Validate batch_size
        if batch_size < 1:
            self._log_warning("Invalid batch_size, using default", batch_size=batch_size)
            batch_size = LocalEmbeddingDefaults.BATCH_SIZE

        # Thread-safe lazy initialization
        with self._lock:
            if not self._is_initialized and not self.initialize():
                self._log_warning("Cannot add documents - store not initialized")
                return 0

            # Ensure model is available after initialization
            if self._model is None:
                self._log_error("Model not available after initialization")
                return 0

            added = 0
            new_docs: list[LocalDocument] = []
            new_contents: list[str] = []

            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue

                # Generate document ID if not provided
                doc_id = doc.get("id") or self._generate_doc_id(content)

                # Skip duplicates
                if doc_id in self._document_ids:
                    self._log_debug("Skipping duplicate document", doc_id=doc_id)
                    continue

                local_doc = LocalDocument(
                    id=doc_id,
                    content=content,
                    metadata=doc.get("metadata", {}),
                )
                new_docs.append(local_doc)
                new_contents.append(content)
                added += 1

            if not new_contents:
                return 0

            # Generate embeddings in batches
            try:
                all_embeddings: list[NDArray[np.floating[Any]]] = []
                for i in range(0, len(new_contents), batch_size):
                    batch = new_contents[i : i + batch_size]
                    batch_embeddings = self._model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True,  # Pre-normalize for cosine similarity
                    )
                    all_embeddings.append(batch_embeddings)

                new_embeddings = np.vstack(all_embeddings)

                # Set embeddings on documents
                for local_doc, emb in zip(new_docs, new_embeddings, strict=True):
                    local_doc.embedding = emb

                # Add to store
                self._documents.extend(new_docs)
                self._document_ids.update(d.id for d in new_docs)

                # Update embedding matrix
                if self._embeddings is None:
                    self._embeddings = new_embeddings
                else:
                    self._embeddings = np.vstack([self._embeddings, new_embeddings])

                self._log_debug(
                    "Added documents to local store",
                    added=added,
                    total=len(self._documents),
                )
                return added

            except Exception as e:
                self._log_error("Failed to add documents", error=str(e))
                return 0

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query string
            top_k: Number of results to return (default: 5, max: 1000)
            min_score: Minimum similarity score (0.0-1.0)
            filter_metadata: Optional metadata filters (exact match)

        Returns:
            List of results with content, score, metadata, and id
        """
        # Apply defaults and validation
        top_k = self._validate_top_k(top_k)
        min_score = self._validate_min_score(min_score)

        if not self._is_initialized or self._embeddings is None:
            return []

        if len(self._documents) == 0:
            return []

        if not query or not query.strip():
            self._log_warning("Empty query provided to search")
            return []

        with self._lock:
            # Ensure model is available
            if self._model is None:
                self._log_error("Model not available for search")
                return []

            try:
                # Encode query (normalized for cosine similarity)
                query_embedding = self._model.encode(
                    query,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

                # Compute cosine similarities (dot product since vectors are normalized)
                similarities = np.dot(self._embeddings, query_embedding)

                # Apply metadata filter if provided
                valid_indices = list(range(len(self._documents)))
                if filter_metadata:
                    valid_indices = [
                        i for i in valid_indices if self._matches_filter(self._documents[i].metadata, filter_metadata)
                    ]

                # Get top-k indices from valid documents
                if filter_metadata:
                    filtered_similarities = [(i, similarities[i]) for i in valid_indices]
                    filtered_similarities.sort(key=lambda x: x[1], reverse=True)
                    top_indices = [i for i, _ in filtered_similarities[:top_k]]
                else:
                    top_indices = np.argsort(similarities)[::-1][:top_k].tolist()

                results: list[dict[str, Any]] = []
                for idx in top_indices:
                    score = float(similarities[idx])
                    if score < min_score:
                        continue

                    doc = self._documents[idx]
                    results.append(
                        {
                            "content": doc.content,
                            "score": score,
                            "metadata": doc.metadata,
                            "id": doc.id,
                        }
                    )

                self._log_debug(
                    "Search completed",
                    query_length=len(query),
                    results_count=len(results),
                    top_k=top_k,
                )
                return results

            except Exception as e:
                self._log_error("Search failed", error=str(e))
                return []

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document by ID.

        Thread-safe operation with atomic check-and-remove pattern.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if document was removed, False otherwise
        """
        with self._lock:
            # Check inside lock to avoid TOCTOU race condition
            if doc_id not in self._document_ids:
                self._log_debug("Document not found for removal", doc_id=doc_id)
                return False

            # Find document index
            idx: int | None = None
            for i, doc in enumerate(self._documents):
                if doc.id == doc_id:
                    idx = i
                    break

            if idx is None:
                # This should not happen if _document_ids and _documents are in sync
                self._log_warning(
                    "Document ID in set but not in list - inconsistent state",
                    doc_id=doc_id,
                )
                self._document_ids.discard(doc_id)
                return False

            # Remove from documents list
            self._documents.pop(idx)
            self._document_ids.discard(doc_id)

            # Remove from embeddings matrix
            if self._embeddings is not None and len(self._documents) > 0:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)
            elif len(self._documents) == 0:
                self._embeddings = None

            self._log_debug("Removed document", doc_id=doc_id)
            return True

    def clear(self) -> None:
        """Clear all documents from the store."""
        with self._lock:
            self._documents.clear()
            self._document_ids.clear()
            self._embeddings = None
            self._log_debug("Cleared local embedding store")

    def _validate_top_k(self, top_k: int | None) -> int:
        """Validate and return top_k parameter."""
        if top_k is None:
            return LocalEmbeddingDefaults.DEFAULT_TOP_K
        if top_k < 1:
            self._log_warning("top_k must be >= 1, using default", provided=top_k)
            return LocalEmbeddingDefaults.DEFAULT_TOP_K
        if top_k > LocalEmbeddingDefaults.MAX_TOP_K:
            self._log_warning(
                "top_k exceeds maximum, capping",
                provided=top_k,
                max_value=LocalEmbeddingDefaults.MAX_TOP_K,
            )
            return LocalEmbeddingDefaults.MAX_TOP_K
        return top_k

    def _validate_min_score(self, min_score: float | None) -> float:
        """Validate and return min_score parameter."""
        if min_score is None:
            return LocalEmbeddingDefaults.MIN_SCORE
        if min_score < LocalEmbeddingDefaults.MIN_SCORE:
            self._log_warning("min_score must be >= 0, using 0", provided=min_score)
            return LocalEmbeddingDefaults.MIN_SCORE
        if min_score > LocalEmbeddingDefaults.MAX_SCORE:
            self._log_warning("min_score must be <= 1, using 1", provided=min_score)
            return LocalEmbeddingDefaults.MAX_SCORE
        return min_score

    def _generate_doc_id(self, content: str) -> str:
        """Generate a unique document ID from content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _matches_filter(
        self,
        metadata: dict[str, Any],
        filter_metadata: dict[str, Any],
    ) -> bool:
        """Check if document metadata matches filter criteria."""
        return all(metadata.get(key) == value for key, value in filter_metadata.items())

    # Logging helper methods for consistent structured logging
    def _log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with optional structured data."""
        if _HAS_STRUCTURED_LOGGING:
            self._logger.debug(message, **kwargs)
        else:
            self._logger.debug(f"{message} {kwargs}" if kwargs else message)

    def _log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with optional structured data."""
        if _HAS_STRUCTURED_LOGGING:
            self._logger.info(message, **kwargs)
        else:
            self._logger.info(f"{message} {kwargs}" if kwargs else message)

    def _log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with optional structured data."""
        if _HAS_STRUCTURED_LOGGING:
            self._logger.warning(message, **kwargs)
        else:
            self._logger.warning(f"{message} {kwargs}" if kwargs else message)

    def _log_error(self, message: str, **kwargs: Any) -> None:
        """Log error message with optional structured data."""
        if _HAS_STRUCTURED_LOGGING:
            self._logger.error(message, **kwargs)
        else:
            self._logger.error(f"{message} {kwargs}" if kwargs else message)

    @property
    def is_available(self) -> bool:
        """Check if store is initialized."""
        return self._is_initialized

    @property
    def has_documents(self) -> bool:
        """Check if store has any documents."""
        return len(self._documents) > 0

    @property
    def document_count(self) -> int:
        """Get number of documents in store."""
        return len(self._documents)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


def create_local_embedding_store(
    model_name: str | None = None,
    cache_dir: str | None = None,
    auto_initialize: bool = True,
    logger_instance: logging.Logger | None = None,
) -> LocalEmbeddingStore | None:
    """
    Factory function to create a local embedding store.

    Args:
        model_name: Sentence transformer model name
        cache_dir: Directory for caching models
        auto_initialize: Whether to initialize on creation
        logger_instance: Optional logger for dependency injection

    Returns:
        LocalEmbeddingStore instance or None if initialization fails
    """
    if not _HAS_NUMPY or not _HAS_SENTENCE_TRANSFORMERS:
        logger.warning(
            "Cannot create local embedding store: "
            f"numpy={'yes' if _HAS_NUMPY else 'no'}, "
            f"sentence-transformers={'yes' if _HAS_SENTENCE_TRANSFORMERS else 'no'}"
        )
        return None

    store = LocalEmbeddingStore(
        model_name=model_name,
        cache_dir=Path(cache_dir) if cache_dir else None,
        logger_instance=logger_instance,
    )

    if auto_initialize and not store.initialize():
        return None

    return store


__all__ = [
    "LocalDocument",
    "LocalEmbeddingDefaults",
    "LocalEmbeddingStore",
    "create_local_embedding_store",
]
