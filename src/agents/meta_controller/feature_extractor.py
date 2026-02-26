"""
Feature Extractor for Meta-Controller.

Replaces simple heuristic-based feature engineering with semantic embeddings.
Uses sentence-transformers for local embedding generation or OpenAI if configured.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.agents.meta_controller.base import MetaControllerFeatures

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractorConfig:
    """Configuration for FeatureExtractor."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "FeatureExtractorConfig":
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            device=os.getenv("DEVICE", "cpu"),
        )


class FeatureExtractor:
    """
    Extracts semantic features from queries using embeddings.

    Uses a pre-trained embedding model to map queries to a vector space,
    then calculates similarity scores against agent prototypes to estimate
    routing confidence.
    """

    # Agent prototypes - descriptions of what each agent is good at
    AGENT_PROTOTYPES = {
        "hrm": [
            "complex problem decomposition",
            "hierarchical reasoning",
            "breaking down multiple questions",
            "multi-step planning",
            "structured analysis",
        ],
        "trm": [
            "iterative refinement",
            "improving an answer",
            "comparison and contrast",
            "fixing code or text",
            "polishing content",
        ],
        "mcts": [
            "optimization problem",
            "strategic search",
            "finding the best path",
            "exploring alternatives",
            "decision making under uncertainty",
        ],
    }

    def __init__(self, config: FeatureExtractorConfig | None = None):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration object
        """
        if config is None:
            config = FeatureExtractorConfig()

        self.config = config

        try:
            logger.info(f"Loading embedding model: {config.model_name}")
            self.model = SentenceTransformer(config.model_name, device=config.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            # Pre-compute prototype embeddings
            self.prototype_embeddings = {}
            for agent, descriptions in self.AGENT_PROTOTYPES.items():
                self.prototype_embeddings[agent] = self.model.encode(descriptions)

            logger.info("FeatureExtractor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FeatureExtractor: {e}")
            # Fallback to simpler logic or raise
            self.model = None

    def extract_features(self, query: str, iteration: int = 0, last_agent: str = "none") -> MetaControllerFeatures:
        """
        Extract features from a query using semantic analysis.

        Args:
            query: The input query text
            iteration: Current iteration number
            last_agent: Name of the last agent used

        Returns:
            MetaControllerFeatures object populated with semantic scores
        """
        query_length = len(query)

        if self.model is None:
            # Fallback to heuristics if model failed to load
            return self._heuristic_fallback(query, iteration, last_agent)

        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)

            # Calculate similarity to each agent's prototypes
            scores = {}
            for agent, proto_embeddings in self.prototype_embeddings.items():
                # Calculate cosine similarity between query and all prototypes for this agent
                similarities = util.cos_sim(query_embedding, proto_embeddings)[0]
                # Take the maximum similarity as the score for this agent
                scores[agent] = float(similarities.max())

            # Normalize scores to sum to 1 (roughly) or just scale them
            # Here we map [-1, 1] similarity to [0, 1] confidence roughly

            hrm_conf = max(0.0, scores.get("hrm", 0.0))
            trm_conf = max(0.0, scores.get("trm", 0.0))
            mcts_conf = max(0.0, scores.get("mcts", 0.0))

            # Apply softmax-like normalization for clearer distinction
            confs = np.array([hrm_conf, trm_conf, mcts_conf])
            # Simple normalization
            if confs.sum() > 0:
                confs = confs / confs.sum()
            else:
                confs = np.array([0.33, 0.33, 0.33])

            hrm_confidence = float(confs[0])
            trm_confidence = float(confs[1])
            mcts_value = float(confs[2])

            # Calculate consensus
            max_conf = max(hrm_confidence, trm_confidence, mcts_value)
            min_conf = min(hrm_confidence, trm_confidence, mcts_value)
            consensus_score = min_conf / max_conf if max_conf > 0 else 0.0

            # Additional features
            has_technical = any(w in query.lower() for w in ["code", "function", "api", "error", "bug"])

            return MetaControllerFeatures(
                hrm_confidence=hrm_confidence,
                trm_confidence=trm_confidence,
                mcts_value=mcts_value,
                consensus_score=consensus_score,
                last_agent=last_agent,
                iteration=iteration,
                query_length=query_length,
                has_rag_context=query_length > 50,  # Simple proxy
                rag_relevance_score=0.0,  # Placeholder
                is_technical_query=has_technical,
            )

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return self._heuristic_fallback(query, iteration, last_agent)

    def _heuristic_fallback(self, query: str, iteration: int, last_agent: str) -> MetaControllerFeatures:
        """Fallback to simple string heuristics if embedding fails."""
        # Simple heuristics (copied/adapted from original app.py)
        has_multiple_questions = "?" in query and query.count("?") > 1
        has_comparison = any(word in query.lower() for word in ["vs", "versus", "compare", "difference"])
        has_optimization = any(word in query.lower() for word in ["optimize", "best", "improve", "maximize"])
        has_technical = any(word in query.lower() for word in ["algorithm", "code", "implement", "technical"])

        hrm_confidence = 0.5 + (0.3 if has_multiple_questions else 0) + (0.1 if has_technical else 0)
        trm_confidence = 0.5 + (0.3 if has_comparison else 0) + (0.1 if len(query) > 100 else 0)
        mcts_confidence = 0.5 + (0.3 if has_optimization else 0) + (0.1 if has_technical else 0)

        total = hrm_confidence + trm_confidence + mcts_confidence
        if total == 0:
            hrm_confidence = trm_confidence = mcts_confidence = 1.0 / 3.0
        else:
            hrm_confidence /= total
            trm_confidence /= total
            mcts_confidence /= total

        max_conf = max(hrm_confidence, trm_confidence, mcts_confidence)
        consensus_score = min(hrm_confidence, trm_confidence, mcts_confidence) / max_conf if max_conf > 0 else 0.0

        return MetaControllerFeatures(
            hrm_confidence=hrm_confidence,
            trm_confidence=trm_confidence,
            mcts_value=mcts_confidence,
            consensus_score=consensus_score,
            last_agent=last_agent,
            iteration=iteration,
            query_length=len(query),
            has_rag_context=len(query) > 50,
            rag_relevance_score=0.0,
            is_technical_query=has_technical,
        )
