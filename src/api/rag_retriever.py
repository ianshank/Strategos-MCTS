"""
RAG Retriever Service for Framework Integration.

Provides document retrieval and context formatting for the query pipeline.
Supports multiple backends (Pinecone, local embeddings) with graceful fallback.

Best Practices 2025:
- Configuration-driven (no hardcoded values)
- Async-first design with proper locking
- Comprehensive error handling
- Type-safe interfaces
- Structured logging with correlation IDs
- Thread-safe initialization
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from src.config.settings import Settings, get_settings

if TYPE_CHECKING:
    from .local_embedding_store import LocalEmbeddingStore

# Module-level logger
_logger: logging.Logger = logging.getLogger(__name__)

# Try to import structured logging (optional dependency)
_HAS_STRUCTURED_LOGGING: bool = False
try:
    from src.observability.logging import get_correlation_id, get_structured_logger

    _logger = get_structured_logger(__name__)  # type: ignore[assignment]
    _HAS_STRUCTURED_LOGGING = True
except ImportError:

    def get_correlation_id() -> str:
        """Fallback correlation ID function."""
        return "unknown"


# Configuration defaults (can be overridden via Settings)
class RAGRetrieverDefaults:
    """Default configuration values for RAG retriever."""

    DEFAULT_TOP_K: Final[int] = 5
    MAX_TOP_K: Final[int] = 100
    MIN_SCORE: Final[float] = 0.0
    MAX_SCORE: Final[float] = 1.0
    DEFAULT_MIN_SCORE: Final[float] = 0.0
    # Default feature values for meta-controller queries (configurable)
    DEFAULT_HRM_CONFIDENCE: Final[float] = 0.5
    DEFAULT_TRM_CONFIDENCE: Final[float] = 0.5
    DEFAULT_MCTS_VALUE: Final[float] = 0.5


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with content and metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
        }


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    documents: list[RetrievedDocument]
    query: str
    retrieval_time_ms: float
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context(self) -> str:
        """Format documents as context string."""
        if not self.documents:
            return ""

        context_parts = []
        for i, doc in enumerate(self.documents, 1):
            context_parts.append(f"[Document {i}] (score: {doc.score:.3f})\n{doc.content}")

        return "\n\n".join(context_parts)

    @property
    def has_results(self) -> bool:
        """Check if any documents were retrieved."""
        return len(self.documents) > 0


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector store backends."""

    def is_available(self) -> bool: ...

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class RAGRetriever:
    """
    RAG Retriever service with multi-backend support.

    Supports:
    - Pinecone vector store
    - Local embedding fallback (sentence-transformers + numpy)
    - Graceful degradation when backends unavailable
    - Thread-safe async initialization
    """

    def __init__(
        self,
        settings: Settings | None = None,
        pinecone_store: Any | None = None,
        embedding_model: Any | None = None,
        local_store: LocalEmbeddingStore | None = None,
        logger_instance: logging.Logger | None = None,
    ):
        """
        Initialize RAG retriever.

        Args:
            settings: Application settings (uses global if None)
            pinecone_store: Optional Pinecone store instance
            embedding_model: Optional embedding model for queries
            local_store: Optional local embedding store for fallback
            logger_instance: Optional logger for dependency injection
        """
        self._settings = settings or get_settings()
        self._pinecone_store = pinecone_store
        self._embedding_model = embedding_model
        self._local_store = local_store
        self._logger = logger_instance or _logger
        self._is_initialized = False
        self._available_backends: list[str] = []

        # Async lock for thread-safe async initialization
        self._init_lock = asyncio.Lock()

        # Threading lock for thread-safe synchronous operations
        self._sync_lock = threading.RLock()

        # Configuration from settings with defaults
        self._top_k_default = getattr(
            self._settings,
            "FRAMEWORK_TOP_K_RETRIEVAL",
            RAGRetrieverDefaults.DEFAULT_TOP_K,
        )

    async def initialize(self) -> bool:
        """
        Initialize retriever and check available backends.

        Uses double-checked locking pattern for thread-safe async initialization.

        Returns:
            True if at least one backend is available
        """
        # Fast path: already initialized
        if self._is_initialized:
            return len(self._available_backends) > 0

        # Thread-safe async initialization
        async with self._init_lock:
            # Re-check after acquiring lock
            if self._is_initialized:
                return len(self._available_backends) > 0

            init_start = time.perf_counter()
            correlation_id = get_correlation_id()

            self._log_info(
                "Initializing RAG retriever",
                correlation_id=correlation_id,
                pinecone_configured=self._pinecone_store is not None,
                local_store_configured=self._local_store is not None,
            )

            # Check Pinecone availability
            await self._check_pinecone_backend(correlation_id)

            # Check local embedding store
            await self._check_local_backend(correlation_id)

            self._is_initialized = True
            init_time = (time.perf_counter() - init_start) * 1000

            self._log_info(
                "RAG retriever initialized",
                correlation_id=correlation_id,
                available_backends=self._available_backends,
                init_time_ms=round(init_time, 2),
            )

            return len(self._available_backends) > 0

    async def _check_pinecone_backend(self, correlation_id: str) -> None:
        """Check and register Pinecone backend if available."""
        if self._pinecone_store is None:
            return

        try:
            if hasattr(self._pinecone_store, "is_available"):
                if self._pinecone_store.is_available:
                    self._available_backends.append("pinecone")
                    self._log_info(
                        "Pinecone backend available",
                        correlation_id=correlation_id,
                    )
            else:
                # Assume available if no check method
                self._available_backends.append("pinecone")
                self._log_debug(
                    "Pinecone backend assumed available (no check method)",
                    correlation_id=correlation_id,
                )
        except Exception as e:
            self._log_warning(
                "Pinecone backend check failed",
                correlation_id=correlation_id,
                error=str(e),
            )

    async def _check_local_backend(self, correlation_id: str) -> None:
        """Check and register local embedding backend if available."""
        if self._local_store is not None:
            try:
                if hasattr(self._local_store, "is_available") and self._local_store.is_available:
                    self._available_backends.append("local")
                    self._log_info(
                        "Local embedding store available",
                        correlation_id=correlation_id,
                        document_count=getattr(self._local_store, "document_count", 0),
                    )
                elif hasattr(self._local_store, "initialize") and self._local_store.initialize():
                    self._available_backends.append("local")
                    self._log_info(
                        "Local embedding store initialized",
                        correlation_id=correlation_id,
                    )
            except Exception as e:
                self._log_warning(
                    "Local embedding store check failed",
                    correlation_id=correlation_id,
                    error=str(e),
                )
        elif self._embedding_model is not None:
            # Legacy path: try to create local store from embedding model
            try:
                from .local_embedding_store import create_local_embedding_store

                self._local_store = create_local_embedding_store(auto_initialize=True)
                if self._local_store is not None:
                    self._available_backends.append("local")
                    self._log_info(
                        "Local embedding backend created from model",
                        correlation_id=correlation_id,
                    )
            except ImportError:
                self._log_debug(
                    "Local embedding store module not available",
                    correlation_id=correlation_id,
                )

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string to search for
            top_k: Number of documents to retrieve (uses config default if None)
            filter_metadata: Optional metadata filters
            min_score: Minimum similarity score threshold (0.0-1.0)

        Returns:
            RetrievalResult with documents and metadata
        """
        if not self._is_initialized:
            await self.initialize()

        retrieval_start = time.perf_counter()
        correlation_id = get_correlation_id()

        # Validate and apply defaults
        top_k = self._validate_top_k(top_k)
        min_score = self._validate_min_score(min_score)

        # Validate query
        if not query or not query.strip():
            self._log_warning("Empty query provided to retrieve", correlation_id=correlation_id)
            return RetrievalResult(
                documents=[],
                query=query,
                retrieval_time_ms=0.0,
                backend="none",
                metadata={"error": "Empty query"},
            )

        self._log_debug(
            "Starting document retrieval",
            correlation_id=correlation_id,
            query_length=len(query),
            top_k=top_k,
            min_score=min_score,
            available_backends=self._available_backends,
        )

        # Try backends in priority order
        documents: list[RetrievedDocument] = []
        backend_used = "none"

        # Try Pinecone first
        if "pinecone" in self._available_backends:
            try:
                documents, backend_used = await self._retrieve_from_pinecone(
                    query=query,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    min_score=min_score,
                )
            except Exception as e:
                self._log_warning(
                    "Pinecone retrieval failed, trying fallback",
                    correlation_id=correlation_id,
                    error=str(e),
                )

        # Fallback to local if Pinecone failed or returned no results
        if not documents and "local" in self._available_backends:
            try:
                documents, backend_used = await self._retrieve_local(
                    query=query,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    min_score=min_score,
                )
            except Exception as e:
                self._log_warning(
                    "Local retrieval failed",
                    correlation_id=correlation_id,
                    error=str(e),
                )

        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        result = RetrievalResult(
            documents=documents,
            query=query,
            retrieval_time_ms=retrieval_time,
            backend=backend_used,
            metadata={
                "top_k_requested": top_k,
                "documents_returned": len(documents),
                "min_score": min_score,
                "filter_applied": filter_metadata is not None,
            },
        )

        avg_score = sum(d.score for d in documents) / len(documents) if documents else 0.0
        self._log_info(
            "Document retrieval completed",
            correlation_id=correlation_id,
            documents_retrieved=len(documents),
            backend=backend_used,
            retrieval_time_ms=round(retrieval_time, 2),
            avg_score=round(avg_score, 3),
        )

        return result

    async def _retrieve_from_pinecone(
        self,
        query: str,
        top_k: int,
        filter_metadata: dict[str, Any] | None,
        min_score: float,
    ) -> tuple[list[RetrievedDocument], str]:
        """Retrieve from Pinecone vector store."""
        if self._pinecone_store is None:
            return [], "none"

        # Use the store's search method
        if hasattr(self._pinecone_store, "find_similar_decisions"):
            # Meta-controller style store
            from src.agents.meta_controller.base import MetaControllerFeatures

            # Use configurable default feature values
            features = MetaControllerFeatures(
                hrm_confidence=RAGRetrieverDefaults.DEFAULT_HRM_CONFIDENCE,
                trm_confidence=RAGRetrieverDefaults.DEFAULT_TRM_CONFIDENCE,
                mcts_value=RAGRetrieverDefaults.DEFAULT_MCTS_VALUE,
                consensus_score=0.0,
                last_agent="none",
                iteration=0,
                query_length=len(query),
                has_rag_context=False,
            )

            results = self._pinecone_store.find_similar_decisions(
                features=features,
                top_k=top_k,
            )

            documents = []
            for result in results:
                score = result.get("score", 0.0)
                if score >= min_score:
                    metadata = result.get("metadata", {})
                    documents.append(
                        RetrievedDocument(
                            content=str(metadata),
                            score=score,
                            metadata=metadata,
                            source="pinecone",
                        )
                    )

            return documents, "pinecone"

        elif hasattr(self._pinecone_store, "similarity_search"):
            # LangChain-style store
            results = self._pinecone_store.similarity_search(
                query=query,
                k=top_k,
            )

            documents = []
            for doc in results:
                score = getattr(doc, "score", 0.7)  # Default score if not provided
                if score >= min_score:
                    documents.append(
                        RetrievedDocument(
                            content=doc.page_content,
                            score=score,
                            metadata=getattr(doc, "metadata", {}),
                            source="pinecone",
                        )
                    )

            return documents, "pinecone"

        return [], "none"

    async def _retrieve_local(
        self,
        query: str,
        top_k: int,
        filter_metadata: dict[str, Any] | None,
        min_score: float,
    ) -> tuple[list[RetrievedDocument], str]:
        """Retrieve using local embeddings."""
        if self._local_store is None:
            self._log_debug("Local store not available")
            return [], "none"

        if not getattr(self._local_store, "is_available", False):
            self._log_debug("Local store not initialized")
            return [], "none"

        if not getattr(self._local_store, "has_documents", False):
            self._log_debug("Local store has no documents")
            return [], "local"

        try:
            # Search local store
            results = self._local_store.search(
                query=query,
                top_k=top_k,
                min_score=min_score,
                filter_metadata=filter_metadata,
            )

            documents = [
                RetrievedDocument(
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {}),
                    source="local",
                )
                for result in results
            ]

            self._log_debug(
                "Local retrieval completed",
                documents_found=len(documents),
                query_length=len(query),
            )

            return documents, "local"

        except Exception as e:
            self._log_warning("Local retrieval failed", error=str(e))
            return [], "none"

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        backend: str = "local",
    ) -> int:
        """
        Add documents to a backend store.

        Thread-safe operation with proper locking for backend list modifications.

        Args:
            documents: List of documents with 'content' and optional 'metadata'
            backend: Backend to add to ('local' or 'pinecone')

        Returns:
            Number of documents added
        """
        if backend == "local":
            # Thread-safe creation of local store if needed
            with self._sync_lock:
                if self._local_store is None:
                    # Try to create local store
                    try:
                        from .local_embedding_store import create_local_embedding_store

                        self._local_store = create_local_embedding_store(auto_initialize=True)
                        if self._local_store is not None and "local" not in self._available_backends:
                            self._available_backends.append("local")
                            self._log_debug(
                                "Created local store and added to backends",
                                backends=self._available_backends,
                            )
                    except ImportError:
                        self._log_warning("Cannot create local store - module not available")
                        return 0

                local_store = self._local_store

            # Add documents outside the lock (LocalEmbeddingStore has its own locking)
            if local_store is not None:
                return local_store.add_documents(documents)

        elif backend == "pinecone":
            self._log_warning("Adding documents to Pinecone not implemented")
            return 0

        return 0

    def format_context(
        self,
        result: RetrievalResult,
        max_length: int | None = None,
        include_scores: bool = True,
    ) -> str:
        """
        Format retrieval result as context string.

        Args:
            result: Retrieval result to format
            max_length: Maximum context length (truncates if exceeded)
            include_scores: Whether to include similarity scores

        Returns:
            Formatted context string
        """
        if not result.has_results:
            return ""

        context_parts = []
        for i, doc in enumerate(result.documents, 1):
            header = f"[Source {i}] (relevance: {doc.score:.2f})" if include_scores else f"[Source {i}]"
            context_parts.append(f"{header}\n{doc.content}")

        context = "\n\n---\n\n".join(context_parts)

        # Truncate if needed
        if max_length and len(context) > max_length:
            original_length = len(context)
            context = context[: max_length - 3] + "..."
            self._log_debug(
                "Context truncated",
                original_length=original_length,
                truncated_length=max_length,
            )

        return context

    def _validate_top_k(self, top_k: int | None) -> int:
        """Validate and return top_k parameter."""
        if top_k is None:
            return self._top_k_default
        if top_k < 1:
            self._log_warning("top_k must be >= 1, using default", provided=top_k)
            return self._top_k_default
        if top_k > RAGRetrieverDefaults.MAX_TOP_K:
            self._log_warning(
                "top_k exceeds maximum, capping",
                provided=top_k,
                max_value=RAGRetrieverDefaults.MAX_TOP_K,
            )
            return RAGRetrieverDefaults.MAX_TOP_K
        return top_k

    def _validate_min_score(self, min_score: float | None) -> float:
        """Validate and return min_score parameter."""
        if min_score is None:
            return RAGRetrieverDefaults.DEFAULT_MIN_SCORE
        if min_score < RAGRetrieverDefaults.MIN_SCORE:
            self._log_warning("min_score must be >= 0, using 0", provided=min_score)
            return RAGRetrieverDefaults.MIN_SCORE
        if min_score > RAGRetrieverDefaults.MAX_SCORE:
            self._log_warning("min_score must be <= 1, using 1", provided=min_score)
            return RAGRetrieverDefaults.MAX_SCORE
        return min_score

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
        """Check if retriever has any available backends."""
        return len(self._available_backends) > 0

    @property
    def available_backends(self) -> list[str]:
        """Get list of available backends."""
        return self._available_backends.copy()


# Factory function for creating retrievers
def create_rag_retriever(
    settings: Settings | None = None,
    pinecone_api_key: str | None = None,
    pinecone_host: str | None = None,
    enable_local_fallback: bool = True,
    local_model_name: str | None = None,
    logger_instance: logging.Logger | None = None,
) -> RAGRetriever:
    """
    Factory function to create a RAG retriever with proper configuration.

    Args:
        settings: Application settings
        pinecone_api_key: Override Pinecone API key
        pinecone_host: Override Pinecone host
        enable_local_fallback: Whether to create local embedding store as fallback
        local_model_name: Model name for local embeddings
        logger_instance: Optional logger for dependency injection

    Returns:
        Configured RAGRetriever instance
    """
    settings = settings or get_settings()

    # Try to create Pinecone store
    pinecone_store = None
    try:
        from src.storage.pinecone_store import PineconeVectorStore

        api_key = pinecone_api_key or settings.get_pinecone_api_key()
        host = pinecone_host or settings.PINECONE_HOST

        if api_key and host:
            pinecone_store = PineconeVectorStore(
                api_key=api_key,
                host=host,
                namespace="rag_documents",
                auto_init=True,
            )
            _logger.info("Pinecone store created for RAG retriever")
    except ImportError:
        _logger.debug("Pinecone not available for RAG retriever")
    except Exception as e:
        _logger.warning(f"Failed to create Pinecone store: {e}")

    # Try to create local embedding store as fallback
    local_store = None
    if enable_local_fallback:
        try:
            from .local_embedding_store import create_local_embedding_store

            local_store = create_local_embedding_store(
                model_name=local_model_name,
                auto_initialize=True,
            )
            if local_store is not None:
                _logger.info("Local embedding store created for RAG retriever")
        except ImportError:
            _logger.debug("Local embedding store not available (missing dependencies)")
        except Exception as e:
            _logger.warning(f"Failed to create local embedding store: {e}")

    return RAGRetriever(
        settings=settings,
        pinecone_store=pinecone_store,
        local_store=local_store,
        logger_instance=logger_instance,
    )


__all__ = [
    "RAGRetriever",
    "RAGRetrieverDefaults",
    "RetrievedDocument",
    "RetrievalResult",
    "VectorStoreProtocol",
    "create_rag_retriever",
]
