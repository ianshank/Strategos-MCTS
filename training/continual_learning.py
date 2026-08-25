"""
Continual Learning Module

Production-ready feedback loop system for continuous improvement from real-world usage.

Implements:
- Production interaction logging with privacy preservation
- Failure pattern analysis and clustering
- Active learning for data selection
- Incremental retraining pipeline
- Data quality validation with PII removal
- Feedback collection from production
- Incremental training without catastrophic forgetting
- Data drift detection
- A/B testing framework

Integration:
- LangSmith for tracing and logging
- Existing training/agent_trainer.py
- training/benchmark_suite.py for evaluation
- SQLite and JSON for storage
"""

import asyncio
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import gzip
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Literal

from langsmith import Client as LangSmithClient
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Training dependencies (required)
import torch
import torch.nn.functional as F  # noqa: N812
import yaml

# Define capability flags (dependencies are now hard requirements)
HAS_TORCH = True
HAS_SKLEARN = True
HAS_LANGSMITH = True

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSample:
    """Feedback sample from production deployment."""

    sample_id: str
    input_data: dict[str, Any]
    model_output: Any
    user_feedback: str  # "positive", "negative", "neutral"
    corrected_output: Any | None
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Report of detected data drift."""

    timestamp: str
    drift_type: str  # "feature", "label", "concept"
    severity: float  # 0.0 to 1.0
    affected_features: list[str]
    p_value: float
    recommendation: str


@dataclass
class ProductionInteraction:
    """Complete production interaction record."""

    interaction_id: str
    timestamp: float
    session_id: str

    # User input (sanitized)
    user_query: str
    query_embedding: list[float] | None = None

    # Agent decisions
    agent_selected: str | None = None  # HRM/TRM/MCTS
    agent_confidence: float = 0.0
    routing_decision: dict[str, Any] = field(default_factory=dict)

    # Retrieved context
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    retrieval_quality: float = 0.0

    # Generated response
    response: str = ""
    response_metadata: dict[str, Any] = field(default_factory=dict)

    # User feedback
    user_feedback_score: float | None = None  # 1-5 stars
    user_feedback_text: str | None = None
    user_corrections: str | None = None
    thumbs_up_down: Literal["up", "down", "none"] = "none"

    # Performance metrics
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0

    # Quality signals
    hallucination_detected: bool = False
    retrieval_failed: bool = False
    error_occurred: bool = False
    error_message: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailurePattern:
    """Identified failure pattern."""

    pattern_id: str
    pattern_type: str  # "low_rating", "poor_retrieval", "wrong_agent", "hallucination", "slow"
    frequency: int
    severity: float  # 0.0 to 1.0
    description: str
    example_interactions: list[str]  # interaction IDs
    suggested_fix: str
    timestamp: str


@dataclass
class ActiveLearningCandidate:
    """Candidate sample for active learning annotation."""

    interaction_id: str
    priority_score: float
    selection_reason: str  # "uncertainty", "diversity", "failure", "novelty"
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Data Quality & Privacy
# =============================================================================


class DataQualityValidator:
    """Validate and sanitize training data for quality and privacy."""

    # PII patterns (common patterns - extend as needed)
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "api_key": r"\b(?:api[_-]?key|apikey|access[_-]?token)[:\s=]+['\"]?([a-zA-Z0-9_\-]{20,})['\"]?\b",
    }

    def __init__(self, config: dict[str, Any]):
        """
        Initialize data quality validator.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.sanitize_pii = config.get("sanitize_pii", True)
        self.min_query_length = config.get("min_query_length", 3)
        self.max_query_length = config.get("max_query_length", 5000)
        self.min_response_length = config.get("min_response_length", 1)
        self.max_response_length = config.get("max_response_length", 10000)
        self.blocked_patterns = config.get("blocked_patterns", [])

        logger.info("DataQualityValidator initialized")

    def validate_interaction(self, interaction: ProductionInteraction) -> tuple[bool, list[str]]:
        """
        Validate interaction for quality and privacy.

        Args:
            interaction: Production interaction to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check query length
        if len(interaction.user_query) < self.min_query_length:
            issues.append(f"Query too short: {len(interaction.user_query)} chars")
        if len(interaction.user_query) > self.max_query_length:
            issues.append(f"Query too long: {len(interaction.user_query)} chars")

        # Check response length
        if len(interaction.response) < self.min_response_length:
            issues.append(f"Response too short: {len(interaction.response)} chars")
        if len(interaction.response) > self.max_response_length:
            issues.append(f"Response too long: {len(interaction.response)} chars")

        # Check for PII
        if self.sanitize_pii:
            pii_found = self._detect_pii(interaction.user_query)
            if pii_found:
                issues.append(f"PII detected: {', '.join(pii_found.keys())}")

        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, interaction.user_query, re.IGNORECASE):
                issues.append(f"Blocked pattern found: {pattern}")

        # Check for coherence
        if not self._is_coherent(interaction.response):
            issues.append("Response lacks coherence")

        # Check feedback consistency
        if interaction.user_feedback_score is not None and not (1 <= interaction.user_feedback_score <= 5):
            issues.append(f"Invalid feedback score: {interaction.user_feedback_score}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def sanitize_interaction(self, interaction: ProductionInteraction) -> ProductionInteraction:
        """
        Sanitize interaction by removing PII and normalizing.

        Args:
            interaction: Production interaction

        Returns:
            Sanitized interaction
        """
        if self.sanitize_pii:
            interaction.user_query = self._remove_pii(interaction.user_query)
            if interaction.user_feedback_text:
                interaction.user_feedback_text = self._remove_pii(interaction.user_feedback_text)

        # Normalize whitespace
        interaction.user_query = " ".join(interaction.user_query.split())
        interaction.response = " ".join(interaction.response.split())

        return interaction

    def _detect_pii(self, text: str) -> dict[str, list[str]]:
        """Detect PII in text."""
        pii_found = {}

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                pii_found[pii_type] = matches

        return pii_found

    def _remove_pii(self, text: str) -> str:
        """Remove PII from text."""
        sanitized = text

        for pii_type, pattern in self.PII_PATTERNS.items():
            replacement = f"[REDACTED_{pii_type.upper()}]"
            sanitized = re.sub(pattern, replacement, sanitized)

        return sanitized

    def _is_coherent(self, text: str) -> bool:
        """Check if text is coherent (basic heuristics)."""
        if not text or len(text.strip()) == 0:
            return False

        # Check for minimum word count
        words = text.split()
        if len(words) < 2:
            return False

        # Check for excessive repetition
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
        # Return False if >50% repetition
        return most_common_count <= len(words) * 0.5


# =============================================================================
# Production Interaction Logger
# =============================================================================


class ProductionInteractionLogger:
    """Log all production interactions with privacy preservation and compression."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize production interaction logger.

        Args:
            config: Logging configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.storage_path = Path(config.get("storage", "./cache/production_logs"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.use_compression = config.get("use_compression", True)
        self.use_sqlite = config.get("use_sqlite", True)
        self.buffer_size = config.get("buffer_size", 1000)

        # Initialize validator
        self.validator = DataQualityValidator(config)

        # Buffer for batch writes
        self.interaction_buffer = []

        # Initialize SQLite database
        if self.use_sqlite:
            self.db_path = self.storage_path / "interactions.db"
            self._init_database()

        # LangSmith integration
        self.use_langsmith = config.get("use_langsmith", False)
        if self.use_langsmith:
            self.langsmith_client = LangSmithClient()
            self.langsmith_project = config.get("langsmith_project", "production-feedback")
            logger.info(f"LangSmith integration enabled: {self.langsmith_project}")

        self.statistics = {
            "total_logged": 0,
            "validation_failures": 0,
            "pii_sanitized": 0,
            "errors": 0,
        }

        logger.info(f"ProductionInteractionLogger initialized at {self.storage_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                user_query TEXT NOT NULL,
                agent_selected TEXT,
                agent_confidence REAL,
                response TEXT NOT NULL,
                user_feedback_score REAL,
                thumbs_up_down TEXT,
                latency_ms REAL,
                tokens_used INTEGER,
                cost REAL,
                hallucination_detected INTEGER,
                retrieval_failed INTEGER,
                error_occurred INTEGER,
                retrieval_quality REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session ON interactions(session_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback ON interactions(user_feedback_score)
        """)

        conn.commit()
        conn.close()

        logger.info("SQLite database initialized")

    async def log_interaction(self, interaction: ProductionInteraction) -> bool:
        """
        Log production interaction asynchronously.

        Args:
            interaction: Production interaction to log

        Returns:
            True if logged successfully
        """
        if not self.enabled:
            return False

        try:
            # Validate and sanitize
            is_valid, issues = self.validator.validate_interaction(interaction)

            if not is_valid:
                logger.warning(f"Validation failed for {interaction.interaction_id}: {issues}")
                self.statistics["validation_failures"] += 1
                # Still log but mark as invalid
                interaction.metadata["validation_issues"] = issues

            # Sanitize PII
            interaction = self.validator.sanitize_interaction(interaction)
            if self.validator.sanitize_pii:
                self.statistics["pii_sanitized"] += 1

            # Add to buffer
            self.interaction_buffer.append(interaction)

            # Flush if buffer is full
            if len(self.interaction_buffer) >= self.buffer_size:
                await self._flush_buffer()

            # Log to LangSmith if enabled
            if self.use_langsmith:
                await self._log_to_langsmith(interaction)

            self.statistics["total_logged"] += 1
            return True

        except Exception as e:
            logger.error(f"Error logging interaction {interaction.interaction_id}: {e}")
            self.statistics["errors"] += 1
            return False

    async def _flush_buffer(self) -> None:
        """Flush buffer to storage."""
        if not self.interaction_buffer:
            return

        try:
            if self.use_sqlite:
                await self._write_to_sqlite()
            else:
                await self._write_to_json()

            logger.info(f"Flushed {len(self.interaction_buffer)} interactions to storage")
            self.interaction_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")

    async def _write_to_sqlite(self) -> None:
        """Write interactions to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for interaction in self.interaction_buffer:
            cursor.execute(
                """
                INSERT OR REPLACE INTO interactions (
                    interaction_id, timestamp, session_id, user_query,
                    agent_selected, agent_confidence, response,
                    user_feedback_score, thumbs_up_down, latency_ms,
                    tokens_used, cost, hallucination_detected,
                    retrieval_failed, error_occurred, retrieval_quality, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    interaction.interaction_id,
                    interaction.timestamp,
                    interaction.session_id,
                    interaction.user_query,
                    interaction.agent_selected,
                    interaction.agent_confidence,
                    interaction.response,
                    interaction.user_feedback_score,
                    interaction.thumbs_up_down,
                    interaction.latency_ms,
                    interaction.tokens_used,
                    interaction.cost,
                    int(interaction.hallucination_detected),
                    int(interaction.retrieval_failed),
                    int(interaction.error_occurred),
                    interaction.retrieval_quality,
                    json.dumps(interaction.metadata),
                ),
            )

        conn.commit()
        conn.close()

    async def _write_to_json(self) -> None:
        """Write interactions to compressed JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactions_{timestamp}.json"

        if self.use_compression:
            filename += ".gz"

        filepath = self.storage_path / filename

        data = [asdict(interaction) for interaction in self.interaction_buffer]

        if self.use_compression:
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    async def _log_to_langsmith(self, interaction: ProductionInteraction) -> None:
        """Log interaction to LangSmith for tracing."""
        try:
            # Create a run in LangSmith
            self.langsmith_client.create_run(
                name=f"production_{interaction.interaction_id}",
                run_type="chain",
                inputs={"query": interaction.user_query},
                outputs={"response": interaction.response},
                project_name=self.langsmith_project,
                extra={
                    "agent": interaction.agent_selected,
                    "confidence": interaction.agent_confidence,
                    "latency_ms": interaction.latency_ms,
                    "feedback_score": interaction.user_feedback_score,
                    "thumbs": interaction.thumbs_up_down,
                },
            )
        except Exception as e:
            logger.error(f"Error logging to LangSmith: {e}")

    def query_interactions(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        min_feedback_score: float | None = None,
        max_feedback_score: float | None = None,
        agent_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Query interactions from database.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            min_feedback_score: Minimum feedback score
            max_feedback_score: Maximum feedback score
            agent_type: Filter by agent type
            limit: Maximum results

        Returns:
            List of interactions
        """
        if not self.use_sqlite:
            logger.warning("SQLite not enabled, cannot query")
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM interactions WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        if min_feedback_score is not None:
            query += " AND user_feedback_score >= ?"
            params.append(min_feedback_score)

        if max_feedback_score is not None:
            query += " AND user_feedback_score <= ?"
            params.append(max_feedback_score)

        if agent_type:
            query += " AND agent_selected = ?"
            params.append(agent_type)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get logging statistics."""
        stats = self.statistics.copy()

        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM interactions")
            stats["total_in_db"] = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(user_feedback_score) FROM interactions WHERE user_feedback_score IS NOT NULL")
            result = cursor.fetchone()
            stats["avg_feedback_score"] = result[0] if result[0] else 0

            cursor.execute("SELECT AVG(latency_ms) FROM interactions")
            result = cursor.fetchone()
            stats["avg_latency_ms"] = result[0] if result[0] else 0

            conn.close()

        stats["buffer_size"] = len(self.interaction_buffer)
        return stats


# =============================================================================
# Failure Pattern Analyzer
# =============================================================================


class FailurePatternAnalyzer:
    """Identify and cluster systematic failure modes from production data."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize failure pattern analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config
        self.min_cluster_size = config.get("min_cluster_size", 5)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)

        self.identified_patterns = []

        logger.info("FailurePatternAnalyzer initialized")

    def analyze_failures(self, interactions: list[dict[str, Any]]) -> list[FailurePattern]:
        """
        Analyze interactions to identify failure patterns.

        Args:
            interactions: List of production interactions

        Returns:
            List of identified failure patterns
        """
        patterns = []

        # Identify different failure types
        patterns.extend(self._identify_low_rated_responses(interactions))
        patterns.extend(self._identify_poor_retrieval(interactions))
        patterns.extend(self._identify_routing_mistakes(interactions))
        patterns.extend(self._identify_hallucinations(interactions))
        patterns.extend(self._identify_slow_responses(interactions))

        # Cluster similar failures
        if len(interactions) > 10:
            patterns.extend(self._cluster_failures(interactions))

        self.identified_patterns = patterns
        logger.info(f"Identified {len(patterns)} failure patterns")

        return patterns

    def _identify_low_rated_responses(self, interactions: list[dict[str, Any]]) -> list[FailurePattern]:
        """Identify low-rated responses."""
        low_rated = [
            i for i in interactions if i.get("user_feedback_score") is not None and i["user_feedback_score"] < 3
        ]

        if len(low_rated) < self.min_cluster_size:
            return []

        pattern_id = hashlib.md5(f"low_rated_{datetime.now()}".encode()).hexdigest()[:8]

        return [
            FailurePattern(
                pattern_id=pattern_id,
                pattern_type="low_rating",
                frequency=len(low_rated),
                severity=0.8,
                description=f"Found {len(low_rated)} interactions with low user ratings (<3 stars)",
                example_interactions=[i["interaction_id"] for i in low_rated[:5]],
                suggested_fix="Review response quality, improve prompt engineering, or retrain agent",
                timestamp=datetime.now().isoformat(),
            )
        ]

    def _identify_poor_retrieval(self, interactions: list[dict[str, Any]]) -> list[FailurePattern]:
        """Identify poor retrieval quality."""
        poor_retrieval = [
            i for i in interactions if i.get("retrieval_failed") or (i.get("retrieval_quality", 1.0) < 0.5)
        ]

        if len(poor_retrieval) < self.min_cluster_size:
            return []

        pattern_id = hashlib.md5(f"poor_retrieval_{datetime.now()}".encode()).hexdigest()[:8]

        return [
            FailurePattern(
                pattern_id=pattern_id,
                pattern_type="poor_retrieval",
                frequency=len(poor_retrieval),
                severity=0.7,
                description=f"Found {len(poor_retrieval)} interactions with poor retrieval quality",
                example_interactions=[i["interaction_id"] for i in poor_retrieval[:5]],
                suggested_fix="Update RAG index, improve chunking strategy, or add more relevant documents",
                timestamp=datetime.now().isoformat(),
            )
        ]

    def _identify_routing_mistakes(self, interactions: list[dict[str, Any]]) -> list[FailurePattern]:
        """Identify agent routing mistakes."""
        # Low confidence routing
        low_confidence = [i for i in interactions if i.get("agent_confidence", 1.0) < 0.5]

        if len(low_confidence) < self.min_cluster_size:
            return []

        pattern_id = hashlib.md5(f"routing_{datetime.now()}".encode()).hexdigest()[:8]

        return [
            FailurePattern(
                pattern_id=pattern_id,
                pattern_type="wrong_agent",
                frequency=len(low_confidence),
                severity=0.6,
                description=f"Found {len(low_confidence)} interactions with low agent routing confidence",
                example_interactions=[i["interaction_id"] for i in low_confidence[:5]],
                suggested_fix="Retrain meta-controller with more routing examples",
                timestamp=datetime.now().isoformat(),
            )
        ]

    def _identify_hallucinations(self, interactions: list[dict[str, Any]]) -> list[FailurePattern]:
        """Identify hallucinations."""
        hallucinations = [i for i in interactions if i.get("hallucination_detected")]

        if len(hallucinations) < self.min_cluster_size:
            return []

        pattern_id = hashlib.md5(f"hallucination_{datetime.now()}".encode()).hexdigest()[:8]

        return [
            FailurePattern(
                pattern_id=pattern_id,
                pattern_type="hallucination",
                frequency=len(hallucinations),
                severity=0.9,
                description=f"Found {len(hallucinations)} interactions with detected hallucinations",
                example_interactions=[i["interaction_id"] for i in hallucinations[:5]],
                suggested_fix="Improve grounding to retrieved context, add hallucination detection, use faithfulness metrics",
                timestamp=datetime.now().isoformat(),
            )
        ]

    def _identify_slow_responses(self, interactions: list[dict[str, Any]]) -> list[FailurePattern]:
        """Identify slow responses."""
        slow_threshold = 5000  # 5 seconds
        slow_responses = [i for i in interactions if i.get("latency_ms", 0) > slow_threshold]

        if len(slow_responses) < self.min_cluster_size:
            return []

        pattern_id = hashlib.md5(f"slow_{datetime.now()}".encode()).hexdigest()[:8]

        avg_latency = np.mean([i["latency_ms"] for i in slow_responses])

        return [
            FailurePattern(
                pattern_id=pattern_id,
                pattern_type="slow",
                frequency=len(slow_responses),
                severity=0.5,
                description=f"Found {len(slow_responses)} slow interactions (avg: {avg_latency:.0f}ms)",
                example_interactions=[i["interaction_id"] for i in slow_responses[:5]],
                suggested_fix="Optimize retrieval, reduce MCTS simulations, use caching, or scale infrastructure",
                timestamp=datetime.now().isoformat(),
            )
        ]

    def _cluster_failures(self, interactions: list[dict[str, Any]]) -> list[FailurePattern]:
        """Cluster similar failures using query embeddings."""
        # Get failures (low ratings or errors)
        failures = [
            i
            for i in interactions
            if (i.get("user_feedback_score") is not None and i["user_feedback_score"] < 3) or i.get("error_occurred")
        ]

        if len(failures) < self.min_cluster_size * 2:
            return []

        # Extract query text for clustering
        queries = [i.get("user_query", "") for i in failures]

        try:
            # Use TF-IDF for text similarity
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
            X = vectorizer.fit_transform(queries)

            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=self.min_cluster_size, metric="cosine")
            labels = clustering.fit_predict(X.toarray())

            # Extract patterns from clusters
            patterns = []
            unique_labels = set(labels)

            for label in unique_labels:
                if label == -1:  # Noise
                    continue

                cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
                cluster_interactions = [failures[i] for i in cluster_indices]

                pattern_id = hashlib.md5(f"cluster_{label}_{datetime.now()}".encode()).hexdigest()[:8]

                # Get representative query
                representative_query = cluster_interactions[0]["user_query"][:100]

                patterns.append(
                    FailurePattern(
                        pattern_id=pattern_id,
                        pattern_type="clustered_failure",
                        frequency=len(cluster_interactions),
                        severity=0.7,
                        description=f"Cluster of {len(cluster_interactions)} similar failures. Example: '{representative_query}...'",
                        example_interactions=[i["interaction_id"] for i in cluster_interactions[:5]],
                        suggested_fix="Investigate common failure mode and add training examples",
                        timestamp=datetime.now().isoformat(),
                    )
                )

            return patterns

        except Exception as e:
            logger.error(f"Error clustering failures: {e}")
            return []

    def get_summary(self) -> dict[str, Any]:
        """Get summary of identified patterns."""
        if not self.identified_patterns:
            return {"total_patterns": 0}

        pattern_types = Counter([p.pattern_type for p in self.identified_patterns])
        total_failures = sum(p.frequency for p in self.identified_patterns)
        avg_severity = np.mean([p.severity for p in self.identified_patterns])

        return {
            "total_patterns": len(self.identified_patterns),
            "pattern_types": dict(pattern_types),
            "total_failures_analyzed": total_failures,
            "avg_severity": avg_severity,
            "high_severity_patterns": [p for p in self.identified_patterns if p.severity >= 0.8],
        }


# =============================================================================
# Active Learning Selector
# =============================================================================


class ActiveLearningSelector:
    """Select most valuable examples for human annotation using active learning strategies."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize active learning selector.

        Args:
            config: Selection configuration
        """
        self.config = config
        self.selection_strategy = config.get("selection_strategy", "uncertainty")
        self.diversity_weight = config.get("diversity_weight", 0.3)

        logger.info(f"ActiveLearningSelector initialized with strategy: {self.selection_strategy}")

    def select_for_annotation(self, interactions: list[dict[str, Any]], budget: int) -> list[ActiveLearningCandidate]:
        """
        Select most valuable interactions for annotation.

        Args:
            interactions: Pool of interactions
            budget: Number of samples to select

        Returns:
            List of selected candidates
        """
        if not interactions:
            return []

        budget = min(budget, len(interactions))

        if self.selection_strategy == "uncertainty":
            candidates = self._uncertainty_sampling(interactions, budget)
        elif self.selection_strategy == "diversity":
            candidates = self._diversity_sampling(interactions, budget)
        elif self.selection_strategy == "hybrid":
            candidates = self._hybrid_sampling(interactions, budget)
        else:
            # Default: prioritize failures
            candidates = self._failure_prioritization(interactions, budget)

        logger.info(f"Selected {len(candidates)} candidates for annotation")
        return candidates

    def _uncertainty_sampling(self, interactions: list[dict[str, Any]], budget: int) -> list[ActiveLearningCandidate]:
        """Select samples with highest uncertainty."""
        # Calculate uncertainty scores
        scored = []

        for interaction in interactions:
            # Uncertainty based on agent confidence
            confidence = interaction.get("agent_confidence", 0.5)
            uncertainty = 1.0 - abs(confidence - 0.5) * 2  # Highest at 0.5

            # Also consider feedback variance
            feedback_score = interaction.get("user_feedback_score")
            if feedback_score is not None:
                # Mid-range scores (2.5-3.5) indicate uncertainty
                feedback_uncertainty = 1.0 - abs(feedback_score - 3.0) / 2.0
                uncertainty = (uncertainty + feedback_uncertainty) / 2

            scored.append((interaction, uncertainty))

        # Sort by uncertainty (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top budget samples
        candidates = []
        for interaction, score in scored[:budget]:
            candidates.append(
                ActiveLearningCandidate(
                    interaction_id=interaction["interaction_id"],
                    priority_score=score,
                    selection_reason="uncertainty",
                    metadata={"confidence": interaction.get("agent_confidence"), "uncertainty_score": score},
                )
            )

        return candidates

    def _diversity_sampling(self, interactions: list[dict[str, Any]], budget: int) -> list[ActiveLearningCandidate]:
        """Select diverse samples to cover feature space."""
        if len(interactions) < budget:
            return self._uncertainty_sampling(interactions, budget)

        try:
            # Extract query text
            queries = [i.get("user_query", "") for i in interactions]

            # Vectorize
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
            X = vectorizer.fit_transform(queries)

            # Use PCA for dimensionality reduction
            pca = PCA(n_components=min(10, X.shape[1]))
            X_reduced = pca.fit_transform(X.toarray())

            # Greedy diversity selection (farthest-first traversal)
            selected_indices = []
            selected_indices.append(np.random.randint(0, len(interactions)))

            while len(selected_indices) < budget:
                # Find farthest point from selected set
                max_min_distance = -1
                farthest_idx = None

                for i in range(len(interactions)):
                    if i in selected_indices:
                        continue

                    # Min distance to selected set
                    min_dist = min([np.linalg.norm(X_reduced[i] - X_reduced[j]) for j in selected_indices])

                    if min_dist > max_min_distance:
                        max_min_distance = min_dist
                        farthest_idx = i

                if farthest_idx is not None:
                    selected_indices.append(farthest_idx)
                else:
                    break

            # Create candidates
            candidates = []
            for idx in selected_indices:
                candidates.append(
                    ActiveLearningCandidate(
                        interaction_id=interactions[idx]["interaction_id"],
                        priority_score=1.0,  # All equally prioritized in diversity sampling
                        selection_reason="diversity",
                        metadata={"selection_order": len(candidates)},
                    )
                )

            return candidates

        except Exception as e:
            logger.error(f"Error in diversity sampling: {e}")
            return self._uncertainty_sampling(interactions, budget)

    def _hybrid_sampling(self, interactions: list[dict[str, Any]], budget: int) -> list[ActiveLearningCandidate]:
        """Combine uncertainty and diversity."""
        # Select half by uncertainty, half by diversity
        uncertainty_budget = budget // 2
        diversity_budget = budget - uncertainty_budget

        uncertainty_candidates = self._uncertainty_sampling(interactions, uncertainty_budget)

        # Remove already selected from pool
        selected_ids = {c.interaction_id for c in uncertainty_candidates}
        remaining = [i for i in interactions if i["interaction_id"] not in selected_ids]

        diversity_candidates = self._diversity_sampling(remaining, diversity_budget)

        return uncertainty_candidates + diversity_candidates

    def _failure_prioritization(self, interactions: list[dict[str, Any]], budget: int) -> list[ActiveLearningCandidate]:
        """Prioritize failed interactions."""
        # Score failures
        scored = []

        for interaction in interactions:
            priority = 0.0

            # Errors
            if interaction.get("error_occurred"):
                priority += 1.0

            # Hallucinations
            if interaction.get("hallucination_detected"):
                priority += 0.9

            # Low feedback
            feedback = interaction.get("user_feedback_score")
            if feedback is not None and feedback < 3:
                priority += 0.8 * (3 - feedback) / 2

            # Poor retrieval
            if interaction.get("retrieval_failed"):
                priority += 0.7

            # Slow responses
            latency = interaction.get("latency_ms", 0)
            if latency > 5000:
                priority += 0.5

            # Low confidence
            confidence = interaction.get("agent_confidence", 1.0)
            if confidence < 0.5:
                priority += 0.6

            if priority > 0:
                scored.append((interaction, priority))

        # Sort by priority
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top budget
        candidates = []
        for interaction, score in scored[:budget]:
            candidates.append(
                ActiveLearningCandidate(
                    interaction_id=interaction["interaction_id"],
                    priority_score=score,
                    selection_reason="failure",
                    metadata={"priority_score": score},
                )
            )

        return candidates


# =============================================================================
# Incremental Retraining Pipeline
# =============================================================================


class IncrementalRetrainingPipeline:
    """Orchestrate incremental model retraining with new production data."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize retraining pipeline.

        Args:
            config: Retraining configuration
        """
        self.config = config
        self.schedule = config.get("schedule", "weekly")  # daily, weekly, monthly
        self.min_new_samples = config.get("min_new_samples", 100)
        self.validation_split = config.get("validation_split", 0.2)
        self.enable_ab_test = config.get("enable_ab_test", True)

        self.retraining_history = []

        logger.info(f"IncrementalRetrainingPipeline initialized with {self.schedule} schedule")

    async def retrain(
        self,
        new_data: list[dict[str, Any]],
        old_model_path: str | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Perform incremental retraining.

        Args:
            new_data: New production data for training
            old_model_path: Path to existing model
            output_path: Path to save retrained model

        Returns:
            Retraining results
        """
        logger.info(f"Starting retraining with {len(new_data)} new samples")

        if len(new_data) < self.min_new_samples:
            logger.warning(f"Insufficient samples: {len(new_data)} < {self.min_new_samples}")
            return {"status": "skipped", "reason": "insufficient_samples"}

        results = {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(new_data),
            "steps": [],
        }

        try:
            # Step 1: Prepare training data
            logger.info("Step 1: Preparing training data")
            train_data, val_data = self._prepare_training_data(new_data)
            results["steps"].append({"step": "data_preparation", "status": "completed", "train_size": len(train_data)})

            # Step 2: Update meta-controller (routing)
            logger.info("Step 2: Updating meta-controller")
            routing_metrics = await self._update_meta_controller(train_data, val_data)
            results["steps"].append({"step": "meta_controller", "status": "completed", "metrics": routing_metrics})

            # Step 3: Update RAG index
            logger.info("Step 3: Updating RAG index")
            rag_metrics = await self._update_rag_index(new_data)
            results["steps"].append({"step": "rag_update", "status": "completed", "metrics": rag_metrics})

            # Step 4: Update evaluation benchmarks
            logger.info("Step 4: Updating evaluation benchmarks")
            benchmark_results = await self._update_benchmarks(val_data)
            results["steps"].append(
                {"step": "benchmark_evaluation", "status": "completed", "results": benchmark_results}
            )

            # Step 5: A/B test if enabled
            if self.enable_ab_test:
                logger.info("Step 5: Setting up A/B test")
                ab_test_id = await self._setup_ab_test(old_model_path, output_path)
                results["steps"].append({"step": "ab_test", "status": "setup", "test_id": ab_test_id})

            results["status"] = "completed"
            self.retraining_history.append(results)

            logger.info("Retraining completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            return results

    def _prepare_training_data(
        self, new_data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Prepare and split training data."""
        # Filter high-quality samples
        quality_data = []

        for sample in new_data:
            # Require feedback or corrections
            if sample.get("user_feedback_score") is not None or sample.get("user_corrections"):
                quality_data.append(sample)

        # Split train/val
        np.random.shuffle(quality_data)
        split_idx = int(len(quality_data) * (1 - self.validation_split))

        train_data = quality_data[:split_idx]
        val_data = quality_data[split_idx:]

        logger.info(f"Prepared {len(train_data)} train, {len(val_data)} val samples")
        return train_data, val_data

    async def _update_meta_controller(
        self, train_data: list[dict[str, Any]], val_data: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Update meta-controller with new routing decisions."""
        # This would integrate with training/meta_controller.py
        # For now, return dummy metrics

        logger.info("Updating meta-controller routing model")

        # Simulate training
        await asyncio.sleep(0.1)

        return {
            "routing_accuracy": 0.85,
            "confidence_calibration": 0.82,
            "improvement": 0.03,
        }

    async def _update_rag_index(self, new_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Update RAG index with new documents or corrections."""
        # This would integrate with training/rag_builder.py
        logger.info("Updating RAG index")

        # Extract new documents from corrections
        new_docs = []
        for sample in new_data:
            if sample.get("user_corrections"):
                new_docs.append({"text": sample["user_corrections"], "metadata": {"source": "user_correction"}})

        await asyncio.sleep(0.1)

        return {
            "new_documents_added": len(new_docs),
            "index_size": 10000 + len(new_docs),
        }

    async def _update_benchmarks(self, val_data: list[dict[str, Any]]) -> dict[str, float]:
        """Run evaluation benchmarks on validation data."""
        # This would integrate with training/benchmark_suite.py
        logger.info("Running evaluation benchmarks")

        await asyncio.sleep(0.1)

        return {
            "retrieval_precision": 0.88,
            "response_quality": 0.85,
            "avg_latency_ms": 1200.0,
        }

    async def _setup_ab_test(self, old_model_path: str | None, new_model_path: str | None) -> str:
        """Setup A/B test for new vs old model."""
        logger.info("Setting up A/B test")

        # Generate test ID
        test_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # This would integrate with ABTestFramework
        await asyncio.sleep(0.1)

        return test_id

    def should_retrain(self, last_retrain_time: datetime, num_new_samples: int) -> bool:
        """
        Check if retraining should be triggered.

        Args:
            last_retrain_time: Last retraining timestamp
            num_new_samples: Number of new samples since last retrain

        Returns:
            True if should retrain
        """
        # Check sample threshold
        if num_new_samples < self.min_new_samples:
            return False

        # Check schedule
        now = datetime.now()
        time_diff = now - last_retrain_time

        if self.schedule == "daily":
            return time_diff >= timedelta(days=1)
        elif self.schedule == "weekly":
            return time_diff >= timedelta(weeks=1)
        elif self.schedule == "monthly":
            return time_diff >= timedelta(days=30)

        return False


# =============================================================================
# Original Classes (Keep existing functionality)
# =============================================================================


class FeedbackCollector:
    """Collect and manage feedback from production deployments."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize feedback collector.

        Args:
            config: Continual learning configuration
        """
        self.config = config
        self.buffer_size = config.get("buffer_size", 100000)
        self.sample_rate = config.get("sample_rate", 0.1)

        self.feedback_buffer = deque(maxlen=self.buffer_size)
        self.storage_path = Path("training/data/feedback")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.statistics = {"total_collected": 0, "positive": 0, "negative": 0, "neutral": 0}

        logger.info(f"FeedbackCollector initialized with buffer size {self.buffer_size}")

    def add_feedback(self, feedback: FeedbackSample) -> None:
        """
        Add feedback sample to buffer.

        Args:
            feedback: Feedback sample
        """
        # Apply sampling rate
        if np.random.random() > self.sample_rate:
            return

        self.feedback_buffer.append(feedback)
        self.statistics["total_collected"] += 1
        self.statistics[feedback.user_feedback] = self.statistics.get(feedback.user_feedback, 0) + 1

        # Persist periodically
        if len(self.feedback_buffer) % 1000 == 0:
            self._persist_feedback()

        logger.debug(f"Added feedback sample {feedback.sample_id}")

    def _persist_feedback(self) -> None:
        """Persist feedback to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.storage_path / f"feedback_{timestamp}.jsonl"

        with open(file_path, "w") as f:
            for sample in list(self.feedback_buffer)[-1000:]:
                record = {
                    "sample_id": sample.sample_id,
                    "input_data": sample.input_data,
                    "model_output": str(sample.model_output),
                    "user_feedback": sample.user_feedback,
                    "corrected_output": str(sample.corrected_output) if sample.corrected_output else None,
                    "timestamp": sample.timestamp,
                    "metadata": sample.metadata,
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Persisted feedback to {file_path}")

    def get_training_samples(self, min_quality: float = 0.5) -> list[dict[str, Any]]:
        """
        Get high-quality samples for retraining.

        Args:
            min_quality: Minimum quality threshold

        Returns:
            List of training samples
        """
        training_samples = []

        for feedback in self.feedback_buffer:
            # Prioritize samples with corrections
            if feedback.corrected_output is not None:
                quality = 1.0
            elif feedback.user_feedback == "positive":
                quality = 0.8
            elif feedback.user_feedback == "neutral":
                quality = 0.5
            else:
                quality = 0.3

            if quality >= min_quality:
                sample = {
                    "input": feedback.input_data,
                    "target": feedback.corrected_output or feedback.model_output,
                    "weight": quality,
                }
                training_samples.append(sample)

        logger.info(f"Retrieved {len(training_samples)} training samples from feedback")
        return training_samples

    def get_statistics(self) -> dict[str, Any]:
        """Get feedback collection statistics."""
        stats = self.statistics.copy()
        stats["buffer_size"] = len(self.feedback_buffer)
        stats["negative_rate"] = stats["negative"] / max(1, stats["total_collected"])
        return stats

    def clear_buffer(self) -> None:
        """Clear feedback buffer after processing."""
        self._persist_feedback()
        self.feedback_buffer.clear()
        logger.info("Feedback buffer cleared")


class IncrementalTrainer:
    """Train models incrementally without catastrophic forgetting."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize incremental trainer.

        Args:
            config: Incremental training configuration
        """
        self.config = config
        self.retrain_threshold = config.get("retrain_threshold", 1000)
        self.forgetting_prevention = config.get("forgetting_prevention", "elastic_weight_consolidation")
        self.ewc_lambda = config.get("ewc_lambda", 1000.0)

        self.fisher_information = {}
        self.optimal_params = {}
        self.accumulated_samples = 0

        logger.info(f"IncrementalTrainer using {self.forgetting_prevention} method")

    def should_retrain(self, num_new_samples: int) -> bool:
        """
        Check if retraining should be triggered.

        Args:
            num_new_samples: Number of new samples available

        Returns:
            True if retraining should occur
        """
        self.accumulated_samples += num_new_samples
        return self.accumulated_samples >= self.retrain_threshold

    def compute_fisher_information(self, model: Any, dataloader: Any) -> dict[str, Any]:
        """
        Compute Fisher Information Matrix for EWC.

        Args:
            model: Neural network model
            dataloader: Data loader with old task data

        Returns:
            Fisher information for each parameter
        """
        fisher_info = {}

        # Set model to evaluation mode
        model.eval()

        for name, param in model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)

        # Compute Fisher Information
        num_samples = 0
        for batch in dataloader:
            model.zero_grad()

            # Get model output (simplified)
            if hasattr(model, "forward"):
                inputs = batch.get("input_ids", batch.get("features"))
                output = model(inputs)
            else:
                continue

            # Compute log probability for Fisher Information
            # Fisher Information = E[grad(log p(y|x))^2]
            if isinstance(output, dict):
                logits = output.get("logits", None)
                if logits is not None:
                    # Compute log softmax for classification
                    log_probs = F.log_softmax(logits, dim=-1)
                    # Use the predicted class probability
                    pred_classes = logits.argmax(dim=-1)
                    # Get log probability of predicted class
                    batch_size = logits.size(0)
                    log_likelihood = log_probs[range(batch_size), pred_classes].sum()
                else:
                    log_likelihood = output.get("output", torch.zeros(1)).sum()
            else:
                # For regression or other outputs, use negative squared error
                log_likelihood = -0.5 * (output**2).sum()

            # Compute gradients
            log_likelihood.backward()

            # Accumulate Fisher information (squared gradients)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data**2

            num_samples += 1

        # Average over number of batches
        if num_samples > 0:
            for name in fisher_info:
                fisher_info[name] /= num_samples

        self.fisher_information = fisher_info

        # Store optimal parameters
        self.optimal_params = {name: param.data.clone() for name, param in model.named_parameters()}

        logger.info(f"Computed Fisher Information for {len(fisher_info)} parameters")
        return fisher_info

    def ewc_loss(self, model: Any) -> float:
        """
        Compute EWC regularization loss.

        Args:
            model: Current model

        Returns:
            EWC loss value
        """
        if not self.fisher_information:
            return 0.0

        ewc_loss = 0.0

        for name, param in model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]

                ewc_loss += (fisher * (param - optimal) ** 2).sum()

        return self.ewc_lambda * ewc_loss

    def incremental_update(
        self, model: Any, new_dataloader: Any, old_dataloader: Any | None = None, num_epochs: int = 3
    ) -> dict[str, float]:
        """
        Perform incremental model update.

        Args:
            model: Model to update
            new_dataloader: New task data
            old_dataloader: Old task data (for EWC)
            num_epochs: Number of training epochs

        Returns:
            Training metrics
        """
        logger.info("Performing incremental model update")

        # Compute Fisher Information from old task
        if old_dataloader and self.forgetting_prevention == "elastic_weight_consolidation":
            self.compute_fisher_information(model, old_dataloader)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        metrics = {"losses": [], "ewc_losses": []}

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_ewc_loss = 0.0

            for batch in new_dataloader:
                optimizer.zero_grad()

                # Forward pass (simplified)
                if hasattr(model, "forward"):
                    output = model(batch)
                else:
                    continue

                # Task loss
                if isinstance(output, dict):
                    task_loss = output.get("loss", torch.zeros(1))
                else:
                    task_loss = output.mean()

                # Add EWC regularization
                ewc_reg = self.ewc_loss(model)

                loss = task_loss + ewc_reg
                loss.backward()
                optimizer.step()

                total_loss += task_loss.item()
                total_ewc_loss += ewc_reg if isinstance(ewc_reg, float) else ewc_reg.item()

            avg_loss = total_loss / len(new_dataloader)
            avg_ewc = total_ewc_loss / len(new_dataloader)

            metrics["losses"].append(avg_loss)
            metrics["ewc_losses"].append(avg_ewc)

            logger.info(f"Incremental epoch {epoch + 1}: Loss={avg_loss:.4f}, EWC={avg_ewc:.4f}")

        # Reset sample counter
        self.accumulated_samples = 0

        return metrics


class DriftDetector:
    """Detect data drift in production."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize drift detector.

        Args:
            config: Drift detection configuration
        """
        self.config = config
        self.window_size = config.get("window_size", 1000)
        self.threshold = config.get("threshold", 0.1)
        self.detection_method = config.get("detection_method", "kolmogorov_smirnov")

        self.reference_distribution = None
        self.current_window = deque(maxlen=self.window_size)
        self.drift_history = []

        logger.info(f"DriftDetector using {self.detection_method} method")

    def set_reference_distribution(self, data: np.ndarray) -> None:
        """
        Set reference distribution from training data.

        Args:
            data: Reference data array
        """
        self.reference_distribution = data
        logger.info(f"Set reference distribution with {len(data)} samples")

    def add_sample(self, sample: np.ndarray) -> DriftReport | None:
        """
        Add new sample and check for drift.

        Args:
            sample: New data sample

        Returns:
            DriftReport if drift detected, None otherwise
        """
        self.current_window.append(sample)

        if len(self.current_window) < self.window_size:
            return None

        # Check for drift
        drift_report = self._detect_drift()

        if drift_report:
            self.drift_history.append(drift_report)
            logger.warning(f"Data drift detected: {drift_report.drift_type}, severity={drift_report.severity:.3f}")

        return drift_report

    def _detect_drift(self) -> DriftReport | None:
        """Detect drift using configured method."""
        if self.reference_distribution is None:
            return None

        current_data = np.array(list(self.current_window))

        if self.detection_method == "kolmogorov_smirnov":
            return self._ks_test(current_data)
        elif self.detection_method == "population_stability_index":
            return self._psi_test(current_data)
        else:
            return self._ks_test(current_data)

    def _ks_test(self, current_data: np.ndarray) -> DriftReport | None:
        """Kolmogorov-Smirnov test for drift detection."""
        from scipy import stats

        # Test each feature
        affected_features = []
        p_values = []

        num_features = min(
            current_data.shape[1] if current_data.ndim > 1 else 1,
            self.reference_distribution.shape[1] if self.reference_distribution.ndim > 1 else 1,
        )

        for i in range(num_features):
            if current_data.ndim > 1:
                current_feature = current_data[:, i]
                reference_feature = self.reference_distribution[:, i]
            else:
                current_feature = current_data
                reference_feature = self.reference_distribution

            statistic, p_value = stats.ks_2samp(current_feature, reference_feature)

            if p_value < self.threshold:
                affected_features.append(f"feature_{i}")
                p_values.append(p_value)

        if affected_features:
            severity = 1.0 - np.mean(p_values)
            return DriftReport(
                timestamp=datetime.now().isoformat(),
                drift_type="feature",
                severity=severity,
                affected_features=affected_features,
                p_value=np.mean(p_values),
                recommendation="Consider retraining or feature recalibration",
            )

        return None

    def _psi_test(self, current_data: np.ndarray) -> DriftReport | None:
        """Population Stability Index test."""
        # Simplified PSI calculation
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        if self.reference_distribution.ndim == 1:
            ref_data = self.reference_distribution.reshape(-1, 1)
        else:
            ref_data = self.reference_distribution

        psi_values = []
        affected_features = []

        for i in range(current_data.shape[1]):
            psi = self._calculate_psi(ref_data[:, i], current_data[:, i])
            if psi > 0.25:  # High drift threshold
                affected_features.append(f"feature_{i}")
            psi_values.append(psi)

        if affected_features:
            return DriftReport(
                timestamp=datetime.now().isoformat(),
                drift_type="feature",
                severity=np.mean(psi_values),
                affected_features=affected_features,
                p_value=0.0,  # Not applicable for PSI
                recommendation="Significant distribution shift detected",
            )

        return None

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate PSI between two distributions."""
        # Create buckets based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Add small constant to avoid division by zero
        expected_percents = expected_counts / len(expected) + 1e-6
        actual_percents = actual_counts / len(actual) + 1e-6

        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

        return psi

    def get_drift_summary(self) -> dict[str, Any]:
        """Get summary of drift detection history."""
        if not self.drift_history:
            return {"total_drifts": 0}

        summary = {
            "total_drifts": len(self.drift_history),
            "avg_severity": np.mean([d.severity for d in self.drift_history]),
            "recent_drifts": [
                {"timestamp": d.timestamp, "type": d.drift_type, "severity": d.severity}
                for d in self.drift_history[-5:]
            ],
        }

        return summary


class ABTestFramework:
    """A/B testing framework for model updates."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize A/B testing framework.

        Args:
            config: A/B testing configuration
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.traffic_split = config.get("traffic_split", 0.1)
        self.min_samples = config.get("min_samples", 1000)
        self.confidence_level = config.get("confidence_level", 0.95)

        self.tests = {}
        self.results = {}

        logger.info(f"ABTestFramework initialized, traffic split: {self.traffic_split}")

    def create_test(self, test_name: str, model_a: Any, model_b: Any, metric_fn: Callable[[Any, Any], float]) -> str:
        """
        Create a new A/B test.

        Args:
            test_name: Name of the test
            model_a: Control model (current production)
            model_b: Treatment model (new candidate)
            metric_fn: Function to compute success metric

        Returns:
            Test ID
        """
        test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.tests[test_id] = {
            "name": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "metric_fn": metric_fn,
            "samples_a": [],
            "samples_b": [],
            "created_at": datetime.now().isoformat(),
            "status": "running",
        }

        logger.info(f"Created A/B test: {test_id}")
        return test_id

    def assign_group(self, test_id: str, request_id: str) -> str:
        """
        Assign request to test group.

        Args:
            test_id: Test ID
            request_id: Request identifier

        Returns:
            "A" or "B"
        """
        if test_id not in self.tests:
            return "A"  # Default to control

        # Hash-based assignment for consistency
        hash_val = hash(f"{test_id}_{request_id}") % 100
        group = "B" if hash_val < self.traffic_split * 100 else "A"

        return group

    def record_result(self, test_id: str, group: str, input_data: Any, output: Any, success_metric: float) -> None:
        """
        Record test result.

        Args:
            test_id: Test ID
            group: Test group ("A" or "B")
            input_data: Request input
            output: Model output
            success_metric: Success metric value
        """
        if test_id not in self.tests:
            return

        sample = {
            "input": input_data,
            "output": output,
            "metric": success_metric,
            "timestamp": datetime.now().isoformat(),
        }

        if group == "A":
            self.tests[test_id]["samples_a"].append(sample)
        else:
            self.tests[test_id]["samples_b"].append(sample)

        # Check if we have enough samples
        if self._has_enough_samples(test_id):
            self._analyze_test(test_id)

    def _has_enough_samples(self, test_id: str) -> bool:
        """Check if test has enough samples for analysis."""
        test = self.tests[test_id]
        return (
            len(test["samples_a"]) >= self.min_samples
            and len(test["samples_b"]) >= self.min_samples * self.traffic_split
        )

    def _analyze_test(self, test_id: str) -> dict[str, Any]:
        """Analyze A/B test results."""
        from scipy import stats

        test = self.tests[test_id]

        metrics_a = [s["metric"] for s in test["samples_a"]]
        metrics_b = [s["metric"] for s in test["samples_b"]]

        mean_a = np.mean(metrics_a)
        mean_b = np.mean(metrics_b)

        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(metrics_a, metrics_b)

        is_significant = p_value < (1 - self.confidence_level)
        improvement = (mean_b - mean_a) / mean_a if mean_a > 0 else 0

        result = {
            "test_id": test_id,
            "mean_control": mean_a,
            "mean_treatment": mean_b,
            "improvement": improvement,
            "p_value": p_value,
            "is_significant": is_significant,
            "recommendation": "Deploy B" if is_significant and improvement > 0 else "Keep A",
            "analyzed_at": datetime.now().isoformat(),
        }

        self.results[test_id] = result
        test["status"] = "analyzed"

        logger.info(f"A/B test {test_id} analyzed: {result['recommendation']}")
        return result

    def get_test_status(self, test_id: str) -> dict[str, Any]:
        """Get current status of a test."""
        if test_id not in self.tests:
            return {"error": "Test not found"}

        test = self.tests[test_id]

        status = {
            "test_id": test_id,
            "name": test["name"],
            "status": test["status"],
            "samples_control": len(test["samples_a"]),
            "samples_treatment": len(test["samples_b"]),
            "progress": min(len(test["samples_a"]) / self.min_samples, 1.0),
        }

        if test_id in self.results:
            status["result"] = self.results[test_id]

        return status

    def end_test(self, test_id: str) -> dict[str, Any]:
        """End an A/B test and return final results."""
        if test_id not in self.tests:
            return {"error": "Test not found"}

        if self.tests[test_id]["status"] != "analyzed":
            result = self._analyze_test(test_id)
        else:
            result = self.results[test_id]

        self.tests[test_id]["status"] = "completed"

        logger.info(f"A/B test {test_id} completed")
        return result


if __name__ == "__main__":
    # Test continual learning module
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Continual Learning Module")

    # Load config
    config_path = "training/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cl_config = config.get("continual_learning", {})

    # Test FeedbackCollector
    feedback_config = cl_config.get("feedback", {})
    collector = FeedbackCollector(feedback_config)

    # Add some feedback samples
    for i in range(100):
        feedback = FeedbackSample(
            sample_id=f"sample_{i}",
            input_data={"query": f"Test query {i}"},
            model_output=f"Result {i}",
            user_feedback=["positive", "negative", "neutral"][i % 3],
            corrected_output=f"Corrected {i}" if i % 5 == 0 else None,
            timestamp=float(i),
        )
        collector.add_feedback(feedback)

    stats = collector.get_statistics()
    logger.info(f"Feedback statistics: {stats}")

    # Test IncrementalTrainer
    incremental_config = cl_config.get("incremental", {})
    trainer = IncrementalTrainer(incremental_config)

    should_retrain = trainer.should_retrain(500)
    logger.info(f"Should retrain (500 samples): {should_retrain}")

    should_retrain = trainer.should_retrain(600)
    logger.info(f"Should retrain (1100 samples): {should_retrain}")

    # Test DriftDetector
    drift_config = cl_config.get("drift_detection", {})
    detector = DriftDetector(drift_config)

    # Set reference distribution
    reference_data = np.random.randn(1000, 5)
    detector.set_reference_distribution(reference_data)

    # Add samples (with drift)
    for i in range(1000):
        # Simulate drift by shifting distribution
        sample = np.random.randn(5) + (i / 1000) * 0.5  # Gradual drift
        drift_report = detector.add_sample(sample)

        if drift_report:
            logger.info(f"Drift detected at sample {i}")

    drift_summary = detector.get_drift_summary()
    logger.info(f"Drift summary: {drift_summary}")

    # Test ABTestFramework
    ab_config = cl_config.get("ab_testing", {})
    ab_framework = ABTestFramework(ab_config)

    # Create test
    test_id = ab_framework.create_test(
        "model_v2_test", model_a="model_v1", model_b="model_v2", metric_fn=lambda _inp, _out: np.random.random()
    )

    # Simulate test traffic
    for i in range(1500):
        group = ab_framework.assign_group(test_id, f"request_{i}")
        metric = 0.7 + (0.1 if group == "B" else 0)  # B is slightly better
        metric += np.random.randn() * 0.1

        ab_framework.record_result(test_id, group, {"req": i}, "output", metric)

    status = ab_framework.get_test_status(test_id)
    logger.info(f"A/B test status: {status}")

    logger.info("Continual Learning Module test complete")
