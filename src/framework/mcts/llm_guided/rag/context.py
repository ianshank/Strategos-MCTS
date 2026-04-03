"""
RAG Context Provider for Code Generation.

Retrieves relevant code examples, documentation, and patterns
to augment LLM prompts during MCTS search.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.config.settings import get_settings
from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


@dataclass
class RAGContext:
    """Retrieved context for code generation."""

    # Code examples
    similar_solutions: list[dict[str, Any]] = field(default_factory=list)
    """Similar code solutions with metadata."""

    code_patterns: list[dict[str, Any]] = field(default_factory=list)
    """Relevant code patterns and idioms."""

    # Documentation
    api_docs: list[dict[str, Any]] = field(default_factory=list)
    """Relevant API documentation."""

    # Problem context
    similar_problems: list[dict[str, Any]] = field(default_factory=list)
    """Similar problems from training data."""

    # Metadata
    query: str = ""
    """Original query used for retrieval."""

    num_results: int = 0
    """Total number of results retrieved."""

    retrieval_time_ms: float = 0.0
    """Time taken for retrieval."""

    def is_empty(self) -> bool:
        """Check if context is empty."""
        return (
            len(self.similar_solutions) == 0
            and len(self.code_patterns) == 0
            and len(self.api_docs) == 0
            and len(self.similar_problems) == 0
        )

    def to_text(self, max_length: int = 4000) -> str:
        """Convert context to text for prompt injection."""
        sections = []

        if self.similar_solutions:
            sections.append("## Similar Solutions")
            for i, sol in enumerate(self.similar_solutions[:3], 1):
                code = sol.get("code", "")
                desc = sol.get("description", "")
                sections.append(f"### Example {i}")
                if desc:
                    sections.append(f"Description: {desc}")
                sections.append(f"```python\n{code}\n```")

        if self.code_patterns:
            sections.append("## Relevant Patterns")
            for pattern in self.code_patterns[:2]:
                name = pattern.get("name", "Pattern")
                code = pattern.get("code", "")
                sections.append(f"### {name}")
                sections.append(f"```python\n{code}\n```")

        if self.api_docs:
            sections.append("## API Reference")
            for doc in self.api_docs[:2]:
                name = doc.get("name", "API")
                content = doc.get("content", "")
                sections.append(f"### {name}")
                sections.append(content)

        text = "\n\n".join(sections)

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[Context truncated...]"

        return text

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "similar_solutions": self.similar_solutions,
            "code_patterns": self.code_patterns,
            "api_docs": self.api_docs,
            "similar_problems": self.similar_problems,
            "query": self.query,
            "num_results": self.num_results,
            "retrieval_time_ms": self.retrieval_time_ms,
        }


@dataclass
class RAGContextProviderConfig:
    """Configuration for RAG context provider."""

    # Retrieval settings
    top_k: int = 5
    """Number of results to retrieve."""

    similarity_threshold: float = 0.7
    """Minimum similarity score for inclusion."""

    # Source selection
    include_solutions: bool = True
    """Include similar solutions."""

    include_patterns: bool = True
    """Include code patterns."""

    include_docs: bool = True
    """Include API documentation."""

    include_problems: bool = True
    """Include similar problems."""

    # Vector store settings
    collection_name: str = "code_examples"
    """Vector store collection name."""

    namespace: str | None = None
    """Vector store namespace."""

    # Caching
    cache_results: bool = True
    """Cache retrieval results."""

    cache_ttl_seconds: int = 300
    """Cache TTL in seconds."""

    cache_max_size: int = 1000
    """Maximum number of entries in cache to prevent unbounded growth."""

    def validate(self) -> None:
        """Validate configuration."""
        errors = []

        if self.top_k < 1:
            errors.append("top_k must be >= 1")
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            errors.append("similarity_threshold must be in [0, 1]")

        if errors:
            raise ValueError("Invalid RAGContextProviderConfig:\n" + "\n".join(f"  - {e}" for e in errors))


class VectorStoreProtocol(Protocol):
    """Protocol for vector store backends."""

    async def query(
        self,
        query_text: str,
        top_k: int,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector store."""
        ...


class RAGContextProvider:
    """
    Provides retrieved context for code generation.

    Integrates with vector stores (Pinecone, etc.) to retrieve
    relevant code examples, patterns, and documentation.
    """

    def __init__(
        self,
        config: RAGContextProviderConfig | None = None,
        vector_store: VectorStoreProtocol | None = None,
    ):
        """
        Initialize RAG context provider.

        Args:
            config: Provider configuration
            vector_store: Vector store backend (optional)
        """
        self._config = config or RAGContextProviderConfig()
        self._config.validate()

        self._vector_store = vector_store
        self._cache: dict[str, tuple[RAGContext, float]] = {}
        self._cache_lock = asyncio.Lock()  # Thread-safe cache access

        # Try to initialize default vector store if not provided
        if self._vector_store is None:
            self._vector_store = self._init_default_vector_store()

        logger.info(
            "Initialized RAGContextProvider",
            top_k=self._config.top_k,
            has_vector_store=self._vector_store is not None,
        )

    def _init_default_vector_store(self) -> VectorStoreProtocol | None:
        """Try to initialize default vector store from settings."""
        try:
            settings = get_settings()

            # Check for Pinecone
            if settings.PINECONE_API_KEY:
                from src.storage.pinecone_store import PineconeVectorStore

                store = PineconeVectorStore()
                # Cast to protocol (PineconeVectorStore should implement query method)
                return store  # type: ignore[return-value]

            logger.debug("No vector store configured, RAG will use local fallback")
            return None

        except Exception as e:
            logger.warning(
                "Failed to initialize vector store",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    async def get_context(
        self,
        problem: str,
        current_code: str | None = None,
        tags: list[str] | None = None,
    ) -> RAGContext:
        """
        Retrieve relevant context for a code generation problem.

        Args:
            problem: Problem description
            current_code: Current code state (for refinement)
            tags: Optional tags to filter results

        Returns:
            RAGContext with retrieved information
        """
        start_time = time.perf_counter()

        # Check cache with async lock
        cache_key = self._cache_key(problem, current_code)
        async with self._cache_lock:
            if self._config.cache_results and cache_key in self._cache:
                cached_context, cached_time = self._cache[cache_key]
                # Use monotonic time for TTL comparison (handles clock adjustments)
                if time.monotonic() - cached_time < self._config.cache_ttl_seconds:
                    logger.debug("Returning cached RAG context")
                    # Return a deep copy to prevent mutation of cached data
                    return copy.deepcopy(cached_context)
                else:
                    # Remove expired entry
                    del self._cache[cache_key]

        # Build query
        query = self._build_query(problem, current_code)

        context = RAGContext(query=query)

        if self._vector_store is not None:
            # Retrieve from vector store
            try:
                results = await self._retrieve_from_vector_store(query, tags)
                context = self._process_results(results, query)
            except (TimeoutError, ConnectionError, OSError) as e:
                # Transient network errors - log as warning
                logger.warning(
                    f"Vector store retrieval failed (transient), using fallback: {e}",
                    error_type=type(e).__name__,
                    query_preview=query[:100],
                )
                context = self._get_fallback_context(problem)
            except (ValueError, TypeError, KeyError) as e:
                # Data/validation errors - likely a bug
                logger.error(
                    f"Vector store data error, using fallback: {e}",
                    error_type=type(e).__name__,
                    query_preview=query[:100],
                )
                context = self._get_fallback_context(problem)
            except Exception as e:
                # Unexpected error - log full details
                logger.error(
                    f"Unexpected vector store error, using fallback: {e}",
                    error_type=type(e).__name__,
                    query_preview=query[:100],
                )
                context = self._get_fallback_context(problem)
        else:
            # Use local fallback when no vector store is configured
            logger.debug("No vector store configured, using fallback context")
            context = self._get_fallback_context(problem)

        context.retrieval_time_ms = (time.perf_counter() - start_time) * 1000

        # Cache result with async lock
        if self._config.cache_results:
            async with self._cache_lock:
                # Evict oldest entries if cache is full
                if len(self._cache) >= self._config.cache_max_size:
                    self._evict_expired_entries()
                    # If still full after eviction, remove oldest
                    if len(self._cache) >= self._config.cache_max_size:
                        oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                        del self._cache[oldest_key]
                self._cache[cache_key] = (context, time.monotonic())

        logger.debug(
            "Retrieved RAG context",
            num_results=context.num_results,
            time_ms=context.retrieval_time_ms,
        )

        return context

    def _cache_key(self, problem: str, current_code: str | None) -> str:
        """Generate cache key."""
        content = problem + (current_code or "")
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def _evict_expired_entries(self) -> int:
        """Remove expired entries from cache. Returns count of removed entries."""
        current_time = time.monotonic()
        expired_keys = [
            key
            for key, (_, cached_time) in self._cache.items()
            if current_time - cached_time >= self._config.cache_ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def _build_query(self, problem: str, current_code: str | None) -> str:
        """Build query for vector search."""
        parts = [problem]

        if current_code:
            # Extract key elements from current code
            lines = current_code.strip().split("\n")
            # Include function signature and first few lines
            parts.append("\n".join(lines[:5]))

        return "\n\n".join(parts)

    async def _retrieve_from_vector_store(
        self,
        query: str,
        tags: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Retrieve results from vector store."""
        if self._vector_store is None:
            raise RuntimeError("Vector store must be initialized before retrieval")

        filter_dict = None
        if tags:
            filter_dict = {"tags": {"$in": tags}}

        results = await self._vector_store.query(
            query_text=query,
            top_k=self._config.top_k,
            namespace=self._config.namespace,
            filter=filter_dict,
        )

        return results

    def _process_results(self, results: list[dict[str, Any]], query: str) -> RAGContext:
        """Process vector store results into RAGContext."""
        context = RAGContext(query=query)

        for result in results:
            score = result.get("score", 0.0)
            if score < self._config.similarity_threshold:
                continue

            result_type = result.get("type", "solution")
            metadata = result.get("metadata", {})

            if result_type == "solution" and self._config.include_solutions:
                context.similar_solutions.append(
                    {
                        "code": result.get("text", ""),
                        "description": metadata.get("description", ""),
                        "score": score,
                    }
                )

            elif result_type == "pattern" and self._config.include_patterns:
                context.code_patterns.append(
                    {
                        "name": metadata.get("name", "Pattern"),
                        "code": result.get("text", ""),
                        "description": metadata.get("description", ""),
                        "score": score,
                    }
                )

            elif result_type == "doc" and self._config.include_docs:
                context.api_docs.append(
                    {
                        "name": metadata.get("name", "API"),
                        "content": result.get("text", ""),
                        "score": score,
                    }
                )

            elif result_type == "problem" and self._config.include_problems:
                context.similar_problems.append(
                    {
                        "description": result.get("text", ""),
                        "solution": metadata.get("solution", ""),
                        "score": score,
                    }
                )

        context.num_results = (
            len(context.similar_solutions)
            + len(context.code_patterns)
            + len(context.api_docs)
            + len(context.similar_problems)
        )

        return context

    def _get_fallback_context(self, problem: str) -> RAGContext:
        """Get fallback context when vector store is unavailable."""
        # Provide generic Python patterns
        patterns = [
            {
                "name": "List Comprehension",
                "code": "result = [f(x) for x in items if condition(x)]",
                "description": "Efficient filtering and transformation",
            },
            {
                "name": "Dictionary Comprehension",
                "code": "result = {k: v for k, v in items.items() if condition(k, v)}",
                "description": "Create dictionaries concisely",
            },
            {
                "name": "Generator Expression",
                "code": "result = sum(x**2 for x in numbers)",
                "description": "Memory-efficient iteration",
            },
        ]

        return RAGContext(
            query=problem,
            code_patterns=patterns,
            num_results=len(patterns),
        )

    async def clear_cache(self) -> None:
        """Clear the context cache (thread-safe)."""
        async with self._cache_lock:
            self._cache.clear()
            logger.debug("RAG context cache cleared")


def create_rag_provider(
    top_k: int = 5,
    similarity_threshold: float = 0.7,
    cache_results: bool = True,
) -> RAGContextProvider:
    """
    Create a RAG context provider with specified settings.

    Args:
        top_k: Number of results to retrieve
        similarity_threshold: Minimum similarity score
        cache_results: Whether to cache results

    Returns:
        Configured RAGContextProvider
    """
    config = RAGContextProviderConfig(
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        cache_results=cache_results,
    )

    return RAGContextProvider(config=config)
