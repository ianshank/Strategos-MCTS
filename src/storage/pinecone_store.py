"""
Pinecone vector storage integration for Meta-Controller features and predictions.

Provides semantic search and retrieval of agent selection history using vector embeddings.
"""

import os
import uuid
from datetime import datetime
from typing import Any

# Check if pinecone is available
try:
    from pinecone import Pinecone

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    Pinecone = None

from src.agents.meta_controller.base import MetaControllerFeatures, MetaControllerPrediction
from src.agents.meta_controller.utils import normalize_features
from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


class PineconeVectorStore:
    """
    Vector storage for Meta-Controller features and predictions using Pinecone.

    Stores agent selection decisions as vectors for:
    - Finding similar past routing decisions
    - Analyzing patterns in agent selection
    - Building retrieval-augmented routing strategies
    """

    # Dimension of normalized feature vectors
    VECTOR_DIMENSION = 10

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        namespace: str = "meta_controller",
        auto_init: bool = True,
    ):
        """
        Initialize Pinecone vector store.

        Args:
            api_key: Pinecone API key (if None, reads from PINECONE_API_KEY env var)
            host: Pinecone host URL (if None, reads from PINECONE_HOST env var)
            namespace: Namespace for storing vectors (default: "meta_controller")
            auto_init: Whether to initialize Pinecone client immediately
        """
        self._api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self._host = host or os.environ.get("PINECONE_HOST")
        self.namespace = namespace
        self._client: Any = None
        self._index: Any = None
        self._is_initialized = False
        self._operation_buffer: list[dict[str, Any]] = []

        if not PINECONE_AVAILABLE:
            logger.warning("pinecone package not installed. Install with: pip install pinecone")
            return

        if auto_init and self._api_key and self._host:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize Pinecone client and index connection."""
        if not PINECONE_AVAILABLE:
            return

        if self._api_key and self._host:
            try:
                self._client = Pinecone(api_key=self._api_key)
                self._index = self._client.Index(host=self._host)
                self._is_initialized = True
            except (ImportError, ValueError, ConnectionError) as e:
                logger.warning(f"Failed to initialize Pinecone: {e}", error_type=type(e).__name__)
                self._is_initialized = False

    @property
    def is_available(self) -> bool:
        """Check if Pinecone is available and configured."""
        return PINECONE_AVAILABLE and self._is_initialized and self._api_key is not None and self._host is not None

    def store_prediction(
        self,
        features: MetaControllerFeatures,
        prediction: MetaControllerPrediction,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Store a prediction along with its input features.

        Args:
            features: Input features used for the prediction
            prediction: The prediction result
            metadata: Optional additional metadata

        Returns:
            Vector ID if successful, None otherwise
        """
        if not self.is_available:
            # Buffer the operation for when Pinecone becomes available
            self._operation_buffer.append(
                {
                    "type": "store_prediction",
                    "features": features,
                    "prediction": prediction,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return None

        try:
            # Normalize features to create the vector
            vector = normalize_features(features)

            # Generate unique ID
            vector_id = str(uuid.uuid4())

            # Build metadata
            vector_metadata = {
                "selected_agent": prediction.agent,
                "confidence": prediction.confidence,
                "hrm_prob": prediction.probabilities.get("hrm", 0.0),
                "trm_prob": prediction.probabilities.get("trm", 0.0),
                "mcts_prob": prediction.probabilities.get("mcts", 0.0),
                "timestamp": datetime.now().isoformat(),
                "iteration": features.iteration,
                "query_length": features.query_length,
                "last_agent": features.last_agent,
                "has_rag_context": features.has_rag_context,
            }

            if metadata:
                vector_metadata.update(metadata)

            # Upsert to Pinecone
            self._index.upsert(
                vectors=[
                    {
                        "id": vector_id,
                        "values": vector,
                        "metadata": vector_metadata,
                    }
                ],
                namespace=self.namespace,
            )

            return vector_id

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning(f"Failed to store prediction in Pinecone: {e}", error_type=type(e).__name__)
            return None

    def find_similar_decisions(
        self,
        features: MetaControllerFeatures,
        top_k: int = 5,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Find similar past routing decisions based on current features.

        Args:
            features: Current features to find similar decisions for
            top_k: Number of similar decisions to return
            include_metadata: Whether to include metadata in results

        Returns:
            List of similar decisions with scores and metadata
        """
        if not self.is_available:
            return []

        try:
            # Normalize features to create query vector
            query_vector = normalize_features(features)

            # Query Pinecone
            results = self._index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                namespace=self.namespace,
            )

            # Format results
            similar_decisions = []
            for match in results.get("matches", []):
                decision = {
                    "id": match.get("id"),
                    "score": match.get("score"),
                }
                if include_metadata and "metadata" in match:
                    decision["metadata"] = match["metadata"]
                similar_decisions.append(decision)

            return similar_decisions

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning(f"Failed to query Pinecone: {e}", error_type=type(e).__name__)
            return []

    def get_agent_distribution(
        self,
        features: MetaControllerFeatures,
        top_k: int = 10,
    ) -> dict[str, float]:
        """
        Get the distribution of agent selections for similar past decisions.

        Useful for rule-based fallback that considers historical patterns.

        Args:
            features: Current features
            top_k: Number of similar decisions to consider

        Returns:
            Dictionary mapping agent names to selection frequency
        """
        similar = self.find_similar_decisions(features, top_k=top_k, include_metadata=True)

        if not similar:
            return {"hrm": 0.0, "trm": 0.0, "mcts": 0.0}

        # Count agent selections
        counts = {"hrm": 0, "trm": 0, "mcts": 0}
        total = 0

        for decision in similar:
            if "metadata" in decision:
                agent = decision["metadata"].get("selected_agent")
                if agent in counts:
                    counts[agent] += 1
                    total += 1

        # Convert to distribution
        if total > 0:
            return {agent: count / total for agent, count in counts.items()}
        else:
            return {"hrm": 0.0, "trm": 0.0, "mcts": 0.0}

    def store_batch(
        self,
        features_list: list[MetaControllerFeatures],
        predictions_list: list[MetaControllerPrediction],
        batch_metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Store multiple predictions in a batch.

        Args:
            features_list: List of input features
            predictions_list: List of corresponding predictions
            batch_metadata: Optional metadata to apply to all vectors

        Returns:
            Number of vectors successfully stored
        """
        if not self.is_available:
            # Buffer for later
            self._operation_buffer.append(
                {
                    "type": "store_batch",
                    "features_list": features_list,
                    "predictions_list": predictions_list,
                    "batch_metadata": batch_metadata,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return 0

        if len(features_list) != len(predictions_list):
            raise ValueError("Features and predictions lists must have same length")

        try:
            vectors = []
            for features, prediction in zip(features_list, predictions_list, strict=False):
                vector_id = str(uuid.uuid4())
                vector_values = normalize_features(features)

                metadata = {
                    "selected_agent": prediction.agent,
                    "confidence": prediction.confidence,
                    "hrm_prob": prediction.probabilities.get("hrm", 0.0),
                    "trm_prob": prediction.probabilities.get("trm", 0.0),
                    "mcts_prob": prediction.probabilities.get("mcts", 0.0),
                    "timestamp": datetime.now().isoformat(),
                    "iteration": features.iteration,
                    "query_length": features.query_length,
                    "last_agent": features.last_agent,
                    "has_rag_context": features.has_rag_context,
                }

                if batch_metadata:
                    metadata.update(batch_metadata)

                vectors.append(
                    {
                        "id": vector_id,
                        "values": vector_values,
                        "metadata": metadata,
                    }
                )

            # Upsert batch to Pinecone
            self._index.upsert(vectors=vectors, namespace=self.namespace)

            return len(vectors)

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning(f"Failed to store batch in Pinecone: {e}", error_type=type(e).__name__)
            return 0

    def delete_namespace(self) -> bool:
        """
        Delete all vectors in the current namespace.

        Use with caution! This permanently deletes all stored data.

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available:
            return False

        try:
            self._index.delete(delete_all=True, namespace=self.namespace)
            return True
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning(f"Failed to delete namespace: {e}", error_type=type(e).__name__)
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing index statistics
        """
        if not self.is_available:
            return {
                "available": False,
                "buffered_operations": len(self._operation_buffer),
            }

        try:
            stats = self._index.describe_index_stats()
            return {
                "available": True,
                "total_vectors": stats.get("total_vector_count", 0),
                "namespace_stats": stats.get("namespaces", {}),
                "dimension": stats.get("dimension", self.VECTOR_DIMENSION),
                "buffered_operations": len(self._operation_buffer),
            }
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            return {
                "available": True,
                "error": str(e),
                "buffered_operations": len(self._operation_buffer),
            }

    def get_buffered_operations(self) -> list[dict[str, Any]]:
        """
        Get all buffered operations (useful when Pinecone is not available).

        Returns:
            List of buffered operation dictionaries
        """
        return self._operation_buffer.copy()

    def clear_buffer(self) -> None:
        """Clear the operations buffer."""
        self._operation_buffer.clear()

    def flush_buffer(self) -> int:
        """
        Attempt to flush buffered operations to Pinecone.

        Returns:
            Number of operations successfully flushed
        """
        if not self.is_available or not self._operation_buffer:
            return 0

        flushed = 0
        remaining_buffer = []

        for operation in self._operation_buffer:
            try:
                if operation["type"] == "store_prediction":
                    result = self.store_prediction(
                        features=operation["features"],
                        prediction=operation["prediction"],
                        metadata=operation.get("metadata"),
                    )
                    if result:
                        flushed += 1
                    else:
                        remaining_buffer.append(operation)
                elif operation["type"] == "store_batch":
                    count = self.store_batch(
                        features_list=operation["features_list"],
                        predictions_list=operation["predictions_list"],
                        batch_metadata=operation.get("batch_metadata"),
                    )
                    if count > 0:
                        flushed += 1
                    else:
                        remaining_buffer.append(operation)
            except (ConnectionError, TimeoutError, RuntimeError):
                remaining_buffer.append(operation)

        self._operation_buffer = remaining_buffer
        return flushed


__all__ = [
    "PineconeVectorStore",
    "PINECONE_AVAILABLE",
]
