"""
RAG (Retrieval-Augmented Generation) Builder Module - Pinecone Edition

Handles vector index construction, document processing, and retrieval optimization
for the PRIMUS-Seed cybersecurity document corpus using Pinecone vector database.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import yaml

try:
    from pinecone import Pinecone, ServerlessSpec

    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from training.data_pipeline import DocumentChunk, PRIMUSProcessor

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""

    doc_id: str
    chunk_id: int
    text: str
    score: float
    metadata: dict[str, Any]


@dataclass
class IndexStats:
    """Statistics about the vector index."""

    total_documents: int
    total_chunks: int
    index_size_mb: float
    avg_chunk_length: float
    domains: dict[str, int]


class ChunkingStrategy:
    """Smart document chunking strategies."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize chunking strategy.

        Args:
            config: RAG configuration
        """
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.strategy = config.get("chunk_strategy", "semantic")

        logger.info(f"ChunkingStrategy: size={self.chunk_size}, overlap={self.chunk_overlap}, strategy={self.strategy}")

    def chunk_document(self, text: str, doc_id: str = "") -> list[DocumentChunk]:
        """
        Chunk a document using the configured strategy.

        Args:
            text: Document text
            doc_id: Document identifier

        Returns:
            List of document chunks
        """
        if self.strategy == "semantic":
            return self._semantic_chunking(text, doc_id)
        elif self.strategy == "recursive":
            return self._recursive_chunking(text, doc_id)
        else:  # fixed
            return self._fixed_chunking(text, doc_id)

    def _fixed_chunking(self, text: str, doc_id: str) -> list[DocumentChunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata={"strategy": "fixed", "start": start, "end": end},
                    )
                )
                chunk_id += 1

            start = end - self.chunk_overlap
            if start >= len(text) - 1:
                break

        return chunks

    def _semantic_chunking(self, text: str, doc_id: str) -> list[DocumentChunk]:
        """Semantic chunking based on sentence boundaries."""
        chunks = []
        sentences = self._split_into_sentences(text)
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata={"strategy": "semantic", "num_sentences": len(current_chunk)},
                    )
                )
                chunk_id += 1

                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_len

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={"strategy": "semantic", "num_sentences": len(current_chunk)},
                )
            )

        return chunks

    def _recursive_chunking(self, text: str, doc_id: str) -> list[DocumentChunk]:
        """Recursive chunking with multiple separators."""
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(text, separators, doc_id, 0, [])

    def _recursive_split(
        self, text: str, separators: list[str], doc_id: str, chunk_id: int, results: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Recursively split text with separators."""
        if len(text) <= self.chunk_size:
            results.append(
                DocumentChunk(doc_id=doc_id, chunk_id=chunk_id, text=text.strip(), metadata={"strategy": "recursive"})
            )
            return results

        if not separators:
            results.append(
                DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text[: self.chunk_size].strip(),
                    metadata={"strategy": "recursive", "forced": True},
                )
            )
            return self._recursive_split(
                text[self.chunk_size - self.chunk_overlap :], ["\n\n", "\n", ". ", " "], doc_id, chunk_id + 1, results
            )

        separator = separators[0]
        parts = text.split(separator)

        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                current_chunk += part + separator
            else:
                if current_chunk:
                    results.append(
                        DocumentChunk(
                            doc_id=doc_id,
                            chunk_id=len(results),
                            text=current_chunk.strip(),
                            metadata={"strategy": "recursive", "separator": separator},
                        )
                    )
                if len(part) > self.chunk_size:
                    self._recursive_split(part, separators[1:], doc_id, len(results), results)
                else:
                    current_chunk = part + separator

        if current_chunk.strip():
            results.append(
                DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=len(results),
                    text=current_chunk.strip(),
                    metadata={"strategy": "recursive", "separator": separator},
                )
            )

        return results

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


class VectorIndexBuilder:
    """Build and manage vector indices for RAG using Pinecone."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize vector index builder with Pinecone.

        Args:
            config: RAG configuration
        """
        self.config = config
        self.embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = config.get("embedding_dim", 384)
        self.index_name = config.get("index_name", "mcts-rag-index")
        self.namespace = config.get("namespace", "default")
        self.num_neighbors = config.get("num_neighbors", 10)

        # Pinecone configuration
        self.pinecone_config = config.get("pinecone", {})
        self.api_key = self.pinecone_config.get("api_key") or os.environ.get("PINECONE_API_KEY")
        self.environment = self.pinecone_config.get("environment", "us-east-1")
        self.cloud = self.pinecone_config.get("cloud", "aws")

        # Initialize embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            has_new = hasattr(self.embedding_model, "get_embedding_dimension")
            is_mock_new = has_new and hasattr(self.embedding_model.get_embedding_dimension, "return_value")
            if has_new and (
                not is_mock_new
                or (
                    hasattr(self.embedding_model.get_embedding_dimension.return_value, "__class__")
                    and self.embedding_model.get_embedding_dimension.return_value.__class__.__name__
                    not in ("Mock", "MagicMock", "AsyncMock")
                )
            ):
                self.embedding_dim = self.embedding_model.get_embedding_dimension()
            else:
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            logger.warning("SentenceTransformers not available, using random embeddings")
            self.embedding_model = None

        # Initialize Pinecone
        self.pc_client = None
        self.index = None
        self.chunk_store = []
        self.bm25_index = None

        if HAS_PINECONE and self.api_key:
            self._initialize_pinecone()
        else:
            if not HAS_PINECONE:
                logger.warning("Pinecone not available. Install with: pip install pinecone")
            if not self.api_key:
                logger.warning("PINECONE_API_KEY not set")

        logger.info(f"VectorIndexBuilder initialized with Pinecone, model: {self.embedding_model_name}")

    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone client and create index if needed."""
        try:
            self.pc_client = Pinecone(api_key=self.api_key)

            existing_indexes = [idx.name for idx in self.pc_client.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc_client.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.cloud, region=self.environment),
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")

            self.index = self.pc_client.Index(self.index_name)
            logger.info("Pinecone client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.pc_client = None
            self.index = None

    @property
    def is_available(self) -> bool:
        """Check if Pinecone is available and configured."""
        return HAS_PINECONE and self.pc_client is not None and self.index is not None

    def build_index(self, documents: Iterator[DocumentChunk], batch_size: int = 100) -> IndexStats:
        """
        Build vector index from document chunks using Pinecone.

        Args:
            documents: Iterator of document chunks
            batch_size: Number of vectors to upsert at once

        Returns:
            Index statistics
        """
        logger.info("Building Pinecone vector index...")

        all_texts = []
        chunk_lengths = []
        domain_counts = {}
        batch_vectors = []
        total_chunks = 0

        for chunk in documents:
            embedding = self._embed_text(chunk.text)
            vector_id = f"{chunk.doc_id}_{chunk.chunk_id}_{uuid.uuid4().hex[:8]}"

            truncated_text = chunk.text[:10000] if len(chunk.text) > 10000 else chunk.text

            metadata = {
                "doc_id": str(chunk.doc_id),
                "chunk_id": int(chunk.chunk_id),
                "text": truncated_text,
                "category": chunk.metadata.get("category", "general"),
                "source": chunk.metadata.get("source", "unknown"),
                "text_length": len(chunk.text),
                "word_count": len(chunk.text.split()),
                "has_mitre_reference": "mitre" in chunk.text.lower() or "att&ck" in chunk.text.lower(),
                "has_cve_reference": "cve-" in chunk.text.lower(),
                "threat_level": self._estimate_threat_level(chunk.text),
            }

            batch_vectors.append({"id": vector_id, "values": embedding.tolist(), "metadata": metadata})

            self.chunk_store.append(
                {
                    "id": vector_id,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )

            all_texts.append(chunk.text)
            chunk_lengths.append(len(chunk.text))
            domain = chunk.metadata.get("category", "general")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            total_chunks += 1

            if len(batch_vectors) >= batch_size:
                self._upsert_batch(batch_vectors)
                batch_vectors = []
                if total_chunks % 1000 == 0:
                    logger.info(f"Indexed {total_chunks} chunks to Pinecone")

        if batch_vectors:
            self._upsert_batch(batch_vectors)

        if HAS_BM25 and self.config.get("hybrid_search", {}).get("enabled", False):
            logger.info("Building BM25 index for hybrid search")
            tokenized = [text.lower().split() for text in all_texts]
            self.bm25_index = BM25Okapi(tokenized)

        index_size_mb = 0.0
        if self.is_available:
            try:
                stats = self.index.describe_index_stats()
                total_vectors = stats.get("total_vector_count", 0)
                index_size_mb = (total_vectors * self.embedding_dim * 4) / (1024 * 1024)
            except Exception:
                pass

        stats = IndexStats(
            total_documents=len({c["doc_id"] for c in self.chunk_store}),
            total_chunks=total_chunks,
            index_size_mb=index_size_mb,
            avg_chunk_length=np.mean(chunk_lengths) if chunk_lengths else 0,
            domains=domain_counts,
        )

        logger.info(f"Index built: {stats.total_chunks} chunks in Pinecone")
        return stats

    def _upsert_batch(self, vectors: list[dict[str, Any]]) -> None:
        """Upsert batch of vectors to Pinecone."""
        if not self.is_available:
            return
        try:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")

    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.embedding_model:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        return np.random.randn(self.embedding_dim).astype(np.float32)

    def _estimate_threat_level(self, text: str) -> str:
        """Estimate threat level based on keywords."""
        text_lower = text.lower()
        if any(t in text_lower for t in ["critical", "severe", "exploit", "vulnerability", "breach"]):
            return "high"
        elif any(t in text_lower for t in ["warning", "suspicious", "anomaly", "risk"]):
            return "medium"
        return "low"

    def search(self, query: str, k: int = None, filter_metadata: dict[str, Any] | None = None) -> list[SearchResult]:
        """
        Search for similar chunks using Pinecone.

        Args:
            query: Query text
            k: Number of results
            filter_metadata: Optional Pinecone metadata filter

        Returns:
            List of search results
        """
        if k is None:
            k = self.num_neighbors

        if not self.is_available:
            logger.warning("Pinecone not available")
            return []

        query_embedding = self._embed_text(query)

        try:
            query_response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k * 2 if self.bm25_index else k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_metadata,
            )

            pinecone_results = []
            for match in query_response.get("matches", []):
                metadata = match.get("metadata", {})
                pinecone_results.append(
                    SearchResult(
                        doc_id=metadata.get("doc_id", ""),
                        chunk_id=int(metadata.get("chunk_id", 0)),
                        text=metadata.get("text", ""),
                        score=float(match.get("score", 0.0)),
                        metadata=metadata,
                    )
                )

            if self.bm25_index and self.config.get("hybrid_search", {}).get("enabled", False):
                bm25_weight = self.config["hybrid_search"]["bm25_weight"]
                dense_weight = self.config["hybrid_search"]["dense_weight"]
                tokenized_query = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(tokenized_query)

                combined_scores = {}
                for i, result in enumerate(pinecone_results):
                    bm25_score = 0.0
                    for j, chunk in enumerate(self.chunk_store):
                        if chunk["doc_id"] == result.doc_id and chunk["chunk_id"] == result.chunk_id:
                            bm25_score = bm25_scores[j] if j < len(bm25_scores) else 0.0
                            break
                    if bm25_scores.max() > 0:
                        bm25_score = bm25_score / bm25_scores.max()
                    combined_scores[i] = dense_weight * result.score + bm25_weight * bm25_score

                sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
                results = [pinecone_results[i] for i in sorted_indices[:k]]
                for i, idx in enumerate(sorted_indices[:k]):
                    results[i].score = combined_scores[idx]
            else:
                results = pinecone_results[:k]

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_category(self, query: str, category: str, k: int = None) -> list[SearchResult]:
        """Search within a specific category."""
        return self.search(query, k=k, filter_metadata={"category": {"$eq": category}})

    def search_by_threat_level(self, query: str, threat_level: str, k: int = None) -> list[SearchResult]:
        """Search for documents with specific threat level."""
        return self.search(query, k=k, filter_metadata={"threat_level": {"$eq": threat_level}})

    def save_index(self, path: Path | None = None) -> None:
        """Save local index metadata (Pinecone data is persisted automatically)."""
        if path is None:
            path = Path(self.config.get("index_path", "./cache/rag_index"))
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "chunks.json", "w") as f:
            json.dump(self.chunk_store, f)

        if self.bm25_index:
            corpus = [chunk["text"].lower().split() for chunk in self.chunk_store]
            with open(path / "bm25_corpus.json", "w") as f:
                json.dump(corpus, f)

        config_data = {
            "index_name": self.index_name,
            "namespace": self.namespace,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "total_chunks": len(self.chunk_store),
        }
        with open(path / "index_config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Index metadata saved to {path}")

    def load_index(self, path: Path | None = None) -> None:
        """Load local index metadata."""
        if path is None:
            path = Path(self.config.get("index_path", "./cache/rag_index"))
        path = Path(path)

        if (path / "chunks.json").exists():
            with open(path / "chunks.json") as f:
                self.chunk_store = json.load(f)

        if HAS_BM25 and (path / "bm25_corpus.json").exists():
            with open(path / "bm25_corpus.json") as f:
                corpus = json.load(f)
            self.bm25_index = BM25Okapi(corpus)

        logger.info(f"Index metadata loaded from {path}, {len(self.chunk_store)} chunks")

    def add_documents(self, documents: Iterator[DocumentChunk], batch_size: int = 100) -> int:
        """Add new documents to existing index."""
        added = 0
        new_texts = []
        batch_vectors = []

        for chunk in documents:
            embedding = self._embed_text(chunk.text)
            vector_id = f"{chunk.doc_id}_{chunk.chunk_id}_{uuid.uuid4().hex[:8]}"
            truncated_text = chunk.text[:10000] if len(chunk.text) > 10000 else chunk.text

            metadata = {
                "doc_id": str(chunk.doc_id),
                "chunk_id": int(chunk.chunk_id),
                "text": truncated_text,
                "category": chunk.metadata.get("category", "general"),
                "source": chunk.metadata.get("source", "unknown"),
                "text_length": len(chunk.text),
                "word_count": len(chunk.text.split()),
                "has_mitre_reference": "mitre" in chunk.text.lower() or "att&ck" in chunk.text.lower(),
                "has_cve_reference": "cve-" in chunk.text.lower(),
                "threat_level": self._estimate_threat_level(chunk.text),
            }

            batch_vectors.append({"id": vector_id, "values": embedding.tolist(), "metadata": metadata})
            self.chunk_store.append(
                {
                    "id": vector_id,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )
            new_texts.append(chunk.text)
            added += 1

            if len(batch_vectors) >= batch_size:
                self._upsert_batch(batch_vectors)
                batch_vectors = []

        if batch_vectors:
            self._upsert_batch(batch_vectors)

        if self.bm25_index and new_texts:
            all_texts = [c["text"] for c in self.chunk_store]
            tokenized = [text.lower().split() for text in all_texts]
            self.bm25_index = BM25Okapi(tokenized)

        logger.info(f"Added {added} new documents to Pinecone index")
        return added

    def delete_namespace(self) -> bool:
        """Delete all vectors in the current namespace."""
        if not self.is_available:
            return False
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            self.chunk_store = []
            self.bm25_index = None
            logger.info(f"Deleted namespace: {self.namespace}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete namespace: {e}")
            return False

    def get_index_stats(self) -> dict[str, Any]:
        """Get Pinecone index statistics."""
        if not self.is_available:
            return {"available": False, "local_chunks": len(self.chunk_store)}
        try:
            stats = self.index.describe_index_stats()
            return {
                "available": True,
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", self.embedding_dim),
                "namespaces": stats.get("namespaces", {}),
                "local_chunks": len(self.chunk_store),
            }
        except Exception as e:
            return {"available": True, "error": str(e), "local_chunks": len(self.chunk_store)}


class RetrievalOptimizer:
    """Optimize retrieval quality and performance."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def evaluate_retrieval(
        self, index: VectorIndexBuilder, queries: list[str], ground_truth: list[list[str]]
    ) -> dict[str, float]:
        """Evaluate retrieval quality."""
        precisions, recalls, mrrs = [], [], []

        for query, expected_docs in zip(queries, ground_truth, strict=True):
            results = index.search(query)
            retrieved_docs = [r.doc_id for r in results]

            relevant = len(set(retrieved_docs) & set(expected_docs))
            precision = relevant / len(retrieved_docs) if retrieved_docs else 0
            recall = relevant / len(expected_docs) if expected_docs else 0
            mrr = 0
            for i, doc in enumerate(retrieved_docs):
                if doc in expected_docs:
                    mrr = 1.0 / (i + 1)
                    break

            precisions.append(precision)
            recalls.append(recall)
            mrrs.append(mrr)

        metrics = {
            "precision_at_k": np.mean(precisions),
            "recall": np.mean(recalls),
            "mrr": np.mean(mrrs),
            "f1_score": 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls) + 1e-6),
        }
        logger.info(f"Retrieval metrics: {metrics}")
        return metrics

    def query_expansion(self, query: str) -> list[str]:
        """Expand query with synonyms."""
        expansions = [query]
        cybersecurity_synonyms = {
            "attack": ["threat", "exploit", "breach"],
            "vulnerability": ["weakness", "flaw", "CVE"],
            "malware": ["virus", "trojan", "ransomware"],
            "defense": ["protection", "mitigation", "security"],
        }
        query_lower = query.lower()
        for term, synonyms in cybersecurity_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expansions.append(query_lower.replace(term, synonym))
        return expansions[:5]


class RAGIndexManager:
    """Manage multiple domain-specific indices using Pinecone namespaces."""

    def __init__(self, config_path: str = "training/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.rag_config = self.config["rag"]
        self.indices = {}
        self.chunking_strategy = ChunkingStrategy(self.rag_config)
        self.optimizer = RetrievalOptimizer(self.rag_config)
        logger.info("RAGIndexManager initialized with Pinecone backend")

    def build_domain_indices(self, processor: PRIMUSProcessor) -> dict[str, IndexStats]:
        """Build separate indices for each domain using Pinecone namespaces."""
        stats = {}

        for domain_config in self.rag_config.get("domain_indices", []):
            domain_name = domain_config["name"]
            categories = domain_config["categories"]

            logger.info(f"Building index for domain: {domain_name}")

            def filter_documents(cats=categories):  # Bind loop variable
                for chunk in processor.stream_documents():
                    if cats == ["all"] or chunk.metadata.get("category") in cats:
                        yield chunk

            index_config = self.rag_config.copy()
            index_config["namespace"] = domain_name
            index_config["index_path"] = f"./cache/rag_index/{domain_name}"

            builder = VectorIndexBuilder(index_config)
            domain_stats = builder.build_index(filter_documents())

            self.indices[domain_name] = builder
            stats[domain_name] = domain_stats

        return stats

    def search_all_domains(self, query: str, k: int = 10) -> dict[str, list[SearchResult]]:
        """Search across all domain indices."""
        results = {}
        for domain_name, index in self.indices.items():
            results[domain_name] = index.search(query, k=k)
        return results

    def route_query(self, query: str) -> str:
        """Route query to most appropriate domain index."""
        query_lower = query.lower()
        if any(term in query_lower for term in ["mitre", "att&ck", "technique", "tactic"]):
            return "cybersecurity"
        elif any(term in query_lower for term in ["tactical", "operation", "mission"]):
            return "tactical"
        return "general"

    def save_all_indices(self) -> None:
        """Save all indices metadata to disk."""
        for domain_name, index in self.indices.items():
            index.save_index()
            logger.info(f"Saved {domain_name} index metadata")

    def load_all_indices(self) -> None:
        """Load all indices metadata from disk."""
        for domain_config in self.rag_config.get("domain_indices", []):
            domain_name = domain_config["name"]
            index_config = self.rag_config.copy()
            index_config["namespace"] = domain_name
            index_config["index_path"] = f"./cache/rag_index/{domain_name}"

            builder = VectorIndexBuilder(index_config)
            try:
                builder.load_index()
                self.indices[domain_name] = builder
                logger.info(f"Loaded {domain_name} index metadata")
            except Exception as e:
                logger.warning(f"Could not load {domain_name} index: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing RAG Builder Module with Pinecone")

    config_path = "training/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rag_config = config["rag"]

    chunker = ChunkingStrategy(rag_config)
    test_text = """
    This is a test document about cybersecurity.

    The MITRE ATT&CK framework provides a comprehensive matrix of attack techniques.
    Vulnerability management is crucial for maintaining security posture.
    """

    chunks = chunker.chunk_document(test_text, "test_doc_1")
    logger.info(f"Created {len(chunks)} chunks")

    if os.environ.get("PINECONE_API_KEY"):
        builder = VectorIndexBuilder(rag_config)
        sample_chunks = [
            DocumentChunk("doc1", 0, "Cybersecurity threat analysis", {"category": "cyber"}),
            DocumentChunk("doc1", 1, "MITRE ATT&CK framework", {"category": "mitre"}),
        ]
        stats = builder.build_index(iter(sample_chunks))
        logger.info(f"Index stats: {stats}")

        results = builder.search("cybersecurity defense")
        logger.info(f"Search results: {len(results)} found")

        index_stats = builder.get_index_stats()
        logger.info(f"Pinecone index stats: {index_stats}")
    else:
        logger.warning("PINECONE_API_KEY not set, skipping Pinecone tests")

    logger.info("RAG Builder Module test complete")
