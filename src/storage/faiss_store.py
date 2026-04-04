"""
FAISS Vector Store for RAG Pipeline.

Provides local, high-performance vector search using FAISS (Facebook AI Similarity Search).
Supports both CPU and GPU acceleration with various index types.

Best Practices 2025:
- Configuration via Settings (no hardcoded values)
- Graceful degradation without dependencies
- Index type selection based on data size
- Persistence with automatic backup
- Thread-safe operations with proper locking
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

# Module-level logger
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

# Try to import FAISS
_HAS_FAISS: bool = False
_HAS_FAISS_GPU: bool = False
try:
    import faiss

    _HAS_FAISS = True
    # Check for GPU support
    if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        _HAS_FAISS_GPU = True
except ImportError:
    faiss = None  # type: ignore[assignment,unused-ignore]

# Try to import sentence-transformers for embedding
_HAS_SENTENCE_TRANSFORMERS: bool = False
try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc,unused-ignore]


class FAISSStoreDefaults:
    """Default configuration values for FAISS store."""

    MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: Final[int] = 384
    DEFAULT_TOP_K: Final[int] = 5
    MAX_TOP_K: Final[int] = 10000
    MIN_SCORE: Final[float] = 0.0
    MAX_SCORE: Final[float] = 1.0
    BATCH_SIZE: Final[int] = 32
    # Index type thresholds
    FLAT_MAX_VECTORS: Final[int] = 10000
    IVF_MIN_VECTORS: Final[int] = 10000
    IVF_NLIST: Final[int] = 100  # Number of clusters for IVF


class IndexType:
    """FAISS index types."""

    FLAT_L2 = "flat_l2"  # Exact search, L2 distance
    FLAT_IP = "flat_ip"  # Exact search, inner product (cosine with normalized vectors)
    IVF_FLAT = "ivf_flat"  # Approximate search with inverted file
    IVF_PQ = "ivf_pq"  # Approximate search with product quantization
    HNSW = "hnsw"  # Hierarchical Navigable Small World


@dataclass
class FAISSDocument:
    """Document stored in FAISS index."""

    id: str
    content: str
    embedding_id: int  # FAISS internal ID
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash based on document ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on document ID."""
        if not isinstance(other, FAISSDocument):
            return False
        return self.id == other.id


class FAISSVectorStore:
    """
    FAISS-based vector store for high-performance similarity search.

    Features:
    - Multiple index types (Flat, IVF, HNSW) for different use cases
    - GPU acceleration when available
    - Persistence with automatic backup
    - Thread-safe operations

    Usage:
        store = FAISSVectorStore()
        if store.initialize():
            store.add_documents([{"content": "Hello world", "metadata": {"source": "test"}}])
            results = store.search("Hello", top_k=5)
    """

    def __init__(
        self,
        model_name: str | None = None,
        embedding_dim: int | None = None,
        index_type: str = IndexType.FLAT_IP,
        use_gpu: bool = True,
        persist_dir: Path | None = None,
        logger_instance: logging.Logger | None = None,
    ):
        """
        Initialize FAISS vector store.

        Args:
            model_name: Sentence transformer model for embeddings
            embedding_dim: Embedding dimension (auto-detected if not specified)
            index_type: FAISS index type (flat_ip, ivf_flat, hnsw)
            use_gpu: Use GPU if available
            persist_dir: Directory for persisting index
            logger_instance: Optional logger for dependency injection
        """
        self._model_name = model_name or FAISSStoreDefaults.MODEL_NAME
        self._embedding_dim = embedding_dim or FAISSStoreDefaults.EMBEDDING_DIM
        self._index_type = index_type
        self._use_gpu = use_gpu and _HAS_FAISS_GPU
        self._persist_dir = persist_dir
        self._logger = logger_instance or logger

        self._model: SentenceTransformer | None = None
        self._index: Any = None  # faiss.Index
        self._documents: dict[int, FAISSDocument] = {}
        self._document_id_map: dict[str, int] = {}  # doc_id -> embedding_id
        self._next_id = 0
        self._is_initialized = False
        self._lock = threading.RLock()

    def initialize(self) -> bool:
        """
        Initialize the FAISS index and embedding model.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._is_initialized:
            return True

        if not _HAS_NUMPY:
            self._log_warning("numpy not installed, FAISS store unavailable")
            return False

        if not _HAS_FAISS:
            self._log_warning("FAISS not installed, FAISS store unavailable")
            return False

        if not _HAS_SENTENCE_TRANSFORMERS:
            self._log_warning("sentence-transformers not installed, FAISS store unavailable")
            return False

        with self._lock:
            if self._is_initialized:
                return True

            try:
                # Initialize embedding model
                self._model = SentenceTransformer(self._model_name)
                test_embedding = self._model.encode("test", convert_to_numpy=True)
                self._embedding_dim = int(test_embedding.shape[0])

                # Create FAISS index
                self._index = self._create_index()

                # Try to load persisted index if available
                if self._persist_dir:
                    self._load_if_exists()

                self._is_initialized = True
                self._log_info(
                    "Initialized FAISS store",
                    model_name=self._model_name,
                    embedding_dim=self._embedding_dim,
                    index_type=self._index_type,
                    gpu_enabled=self._use_gpu,
                )
                return True

            except (RuntimeError, OSError) as e:
                self._log_error("Failed to initialize FAISS store", error=str(e))
                return False

    def _create_index(self) -> Any:
        """Create FAISS index based on configuration."""
        dim = self._embedding_dim

        if self._index_type == IndexType.FLAT_L2:
            index = faiss.IndexFlatL2(dim)
        elif self._index_type == IndexType.FLAT_IP:
            index = faiss.IndexFlatIP(dim)
        elif self._index_type == IndexType.IVF_FLAT:
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, FAISSStoreDefaults.IVF_NLIST)
        elif self._index_type == IndexType.HNSW:
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per node
        else:
            # Default to flat inner product
            index = faiss.IndexFlatIP(dim)

        # Move to GPU if available and requested
        if self._use_gpu and _HAS_FAISS_GPU:
            try:
                gpu_resource = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
                self._log_info("FAISS index moved to GPU")
            except (RuntimeError, OSError) as e:
                self._log_warning("Failed to move index to GPU, using CPU", error=str(e))

        return index

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int | None = None,
    ) -> int:
        """
        Add documents to the FAISS index.

        Args:
            documents: List of dicts with 'content' and optional 'metadata', 'id'
            batch_size: Batch size for embedding generation

        Returns:
            Number of documents added.
        """
        batch_size = batch_size or FAISSStoreDefaults.BATCH_SIZE

        with self._lock:
            if not self._is_initialized and not self.initialize():
                return 0

            if self._model is None or self._index is None:
                return 0

            added = 0
            new_docs: list[tuple[str, str, dict[str, Any]]] = []  # (id, content, metadata)

            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue

                doc_id = doc.get("id") or self._generate_doc_id(content)

                # Skip duplicates
                if doc_id in self._document_id_map:
                    continue

                new_docs.append((doc_id, content, doc.get("metadata", {})))

            if not new_docs:
                return 0

            try:
                # Generate embeddings in batches
                all_embeddings = []
                for i in range(0, len(new_docs), batch_size):
                    batch = [d[1] for d in new_docs[i : i + batch_size]]
                    batch_embeddings = self._model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    all_embeddings.append(batch_embeddings)

                embeddings = np.vstack(all_embeddings).astype("float32")

                # Train IVF index if needed (first time with enough data)
                if (
                    self._index_type == IndexType.IVF_FLAT
                    and not self._index.is_trained
                    and len(embeddings) >= FAISSStoreDefaults.IVF_NLIST
                ):
                    self._index.train(embeddings)

                # Add embeddings to index
                start_id = self._next_id
                self._index.add(embeddings)

                # Store document metadata
                for i, (doc_id, content, metadata) in enumerate(new_docs):
                    embedding_id = start_id + i
                    faiss_doc = FAISSDocument(
                        id=doc_id,
                        content=content,
                        embedding_id=embedding_id,
                        metadata=metadata,
                    )
                    self._documents[embedding_id] = faiss_doc
                    self._document_id_map[doc_id] = embedding_id
                    added += 1

                self._next_id += len(new_docs)

                self._log_debug("Added documents", added=added, total=len(self._documents))

                # Auto-save if persist_dir is set
                if self._persist_dir:
                    self._save()

                return added

            except (ValueError, TypeError, RuntimeError) as e:
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
            top_k: Number of results to return
            min_score: Minimum similarity score (0.0-1.0)
            filter_metadata: Optional metadata filters

        Returns:
            List of results with content, score, metadata, and id.
        """
        top_k = min(top_k or FAISSStoreDefaults.DEFAULT_TOP_K, FAISSStoreDefaults.MAX_TOP_K)
        min_score = min_score if min_score is not None else FAISSStoreDefaults.MIN_SCORE

        if not self._is_initialized or self._index is None:
            return []

        if len(self._documents) == 0:
            return []

        if not query or not query.strip():
            return []

        with self._lock:
            if self._model is None:
                return []

            try:
                # Encode query
                query_embedding = (
                    self._model.encode(
                        query,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    .astype("float32")
                    .reshape(1, -1)
                )

                # Search FAISS index
                # For IVF, we might need to search more to apply filters
                search_k = top_k * 3 if filter_metadata else top_k
                distances, indices = self._index.search(query_embedding, min(search_k, len(self._documents)))

                results = []
                for score, idx in zip(distances[0], indices[0], strict=True):
                    if idx == -1:  # FAISS returns -1 for missing results
                        continue

                    doc = self._documents.get(idx)
                    if doc is None:
                        continue

                    # Convert L2 distance to similarity score for FLAT_L2
                    # For inner product (FLAT_IP), score is already similarity
                    if self._index_type == IndexType.FLAT_L2:
                        # Convert L2 distance to cosine-like similarity
                        similarity = 1.0 / (1.0 + float(score))
                    else:
                        similarity = float(score)

                    if similarity < min_score:
                        continue

                    # Apply metadata filter
                    if filter_metadata and not self._matches_filter(doc.metadata, filter_metadata):
                        continue

                    results.append(
                        {
                            "content": doc.content,
                            "score": similarity,
                            "metadata": doc.metadata,
                            "id": doc.id,
                        }
                    )

                    if len(results) >= top_k:
                        break

                return results

            except (ValueError, TypeError, RuntimeError) as e:
                self._log_error("Search failed", error=str(e))
                return []

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document by ID.

        Note: FAISS doesn't support true deletion. This marks the document as deleted
        and removes it from the ID map. The index entry remains until rebuild.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if document was marked as removed.
        """
        with self._lock:
            if doc_id not in self._document_id_map:
                return False

            embedding_id = self._document_id_map.pop(doc_id)
            if embedding_id in self._documents:
                del self._documents[embedding_id]

            self._log_debug("Removed document", doc_id=doc_id)
            return True

    def rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index from scratch.

        This is needed to reclaim space from deleted documents.

        Returns:
            True if rebuild successful.
        """
        with self._lock:
            if not self._is_initialized or self._model is None:
                return False

            try:
                # Get all current documents
                docs = list(self._documents.values())
                if not docs:
                    # Reset to empty index
                    self._index = self._create_index()
                    self._next_id = 0
                    return True

                # Re-embed all documents
                contents = [doc.content for doc in docs]
                embeddings = self._model.encode(
                    contents,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype("float32")

                # Create new index
                new_index = self._create_index()

                # Train if needed
                if (
                    self._index_type == IndexType.IVF_FLAT
                    and not new_index.is_trained
                    and len(embeddings) >= FAISSStoreDefaults.IVF_NLIST
                ):
                    new_index.train(embeddings)

                # Add all embeddings
                new_index.add(embeddings)

                # Update document mappings
                self._index = new_index
                self._documents.clear()
                self._document_id_map.clear()

                for i, doc in enumerate(docs):
                    new_doc = FAISSDocument(
                        id=doc.id,
                        content=doc.content,
                        embedding_id=i,
                        metadata=doc.metadata,
                    )
                    self._documents[i] = new_doc
                    self._document_id_map[doc.id] = i

                self._next_id = len(docs)

                self._log_info("Rebuilt FAISS index", document_count=len(docs))
                return True

            except (ValueError, TypeError, RuntimeError) as e:
                self._log_error("Failed to rebuild index", error=str(e))
                return False

    def clear(self) -> None:
        """Clear all documents and reset index."""
        with self._lock:
            self._index = self._create_index()
            self._documents.clear()
            self._document_id_map.clear()
            self._next_id = 0
            self._log_debug("Cleared FAISS store")

    def _save(self) -> None:
        """Save index and metadata to disk."""
        if not self._persist_dir or not self._index:
            return

        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_path = self._persist_dir / "faiss.index"
            # Move to CPU if on GPU before saving
            if self._use_gpu and _HAS_FAISS_GPU:
                cpu_index = faiss.index_gpu_to_cpu(self._index)
                faiss.write_index(cpu_index, str(index_path))
            else:
                faiss.write_index(self._index, str(index_path))

            # Save metadata
            metadata = {
                "next_id": self._next_id,
                "documents": {
                    str(k): {
                        "id": v.id,
                        "content": v.content,
                        "embedding_id": v.embedding_id,
                        "metadata": v.metadata,
                    }
                    for k, v in self._documents.items()
                },
                "document_id_map": self._document_id_map,
            }
            metadata_path = self._persist_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            self._log_debug("Saved FAISS store", path=str(self._persist_dir))

        except (OSError, ValueError) as e:
            self._log_error("Failed to save FAISS store", error=str(e))

    def _load_if_exists(self) -> bool:
        """Load index and metadata from disk if exists."""
        if not self._persist_dir:
            return False

        index_path = self._persist_dir / "faiss.index"
        metadata_path = self._persist_dir / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            return False

        try:
            # Load FAISS index
            self._index = faiss.read_index(str(index_path))

            # Move to GPU if requested
            if self._use_gpu and _HAS_FAISS_GPU:
                gpu_resource = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(gpu_resource, 0, self._index)

            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)

            self._next_id = metadata["next_id"]
            self._document_id_map = metadata["document_id_map"]

            # Reconstruct documents
            for k, v in metadata["documents"].items():
                self._documents[int(k)] = FAISSDocument(
                    id=v["id"],
                    content=v["content"],
                    embedding_id=v["embedding_id"],
                    metadata=v["metadata"],
                )

            self._log_info(
                "Loaded FAISS store",
                document_count=len(self._documents),
                path=str(self._persist_dir),
            )
            return True

        except (OSError, ValueError) as e:
            self._log_error("Failed to load FAISS store", error=str(e))
            return False

    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID from content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _matches_filter(
        self,
        metadata: dict[str, Any],
        filter_metadata: dict[str, Any],
    ) -> bool:
        """Check if document metadata matches filter criteria."""
        return all(metadata.get(key) == value for key, value in filter_metadata.items())

    # Logging helpers
    def _log_debug(self, message: str, **kwargs: Any) -> None:
        if _HAS_STRUCTURED_LOGGING:
            self._logger.debug(message, **kwargs)
        else:
            self._logger.debug(f"{message} {kwargs}" if kwargs else message)

    def _log_info(self, message: str, **kwargs: Any) -> None:
        if _HAS_STRUCTURED_LOGGING:
            self._logger.info(message, **kwargs)
        else:
            self._logger.info(f"{message} {kwargs}" if kwargs else message)

    def _log_warning(self, message: str, **kwargs: Any) -> None:
        if _HAS_STRUCTURED_LOGGING:
            self._logger.warning(message, **kwargs)
        else:
            self._logger.warning(f"{message} {kwargs}" if kwargs else message)

    def _log_error(self, message: str, **kwargs: Any) -> None:
        if _HAS_STRUCTURED_LOGGING:
            self._logger.error(message, **kwargs)
        else:
            self._logger.error(f"{message} {kwargs}" if kwargs else message)

    @property
    def is_available(self) -> bool:
        """Check if store is initialized and ready."""
        return self._is_initialized

    @property
    def document_count(self) -> int:
        """Get number of documents in store."""
        return len(self._documents)

    @property
    def index_type(self) -> str:
        """Get the FAISS index type."""
        return self._index_type

    @property
    def gpu_enabled(self) -> bool:
        """Check if GPU is enabled."""
        return self._use_gpu and _HAS_FAISS_GPU


def create_faiss_store(
    model_name: str | None = None,
    index_type: str = IndexType.FLAT_IP,
    use_gpu: bool = True,
    persist_dir: str | None = None,
    auto_initialize: bool = True,
) -> FAISSVectorStore | None:
    """
    Factory function to create a FAISS vector store.

    Args:
        model_name: Sentence transformer model name
        index_type: FAISS index type
        use_gpu: Use GPU if available
        persist_dir: Directory for persistence
        auto_initialize: Initialize on creation

    Returns:
        FAISSVectorStore instance or None if initialization fails.
    """
    if not _HAS_FAISS:
        logger.warning("FAISS not installed. Install with: pip install faiss-cpu")
        return None

    store = FAISSVectorStore(
        model_name=model_name,
        index_type=index_type,
        use_gpu=use_gpu,
        persist_dir=Path(persist_dir) if persist_dir else None,
    )

    if auto_initialize and not store.initialize():
        return None

    return store


__all__ = [
    "FAISSDocument",
    "FAISSStoreDefaults",
    "FAISSVectorStore",
    "IndexType",
    "create_faiss_store",
]
