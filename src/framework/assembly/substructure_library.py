"""
Substructure Library for Pattern Reuse (Story 1.4).

Tracks frequently used assembly patterns and enables reuse of successful
reasoning subsequences.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickle  # nosec B403 - only used for gated, opt-in legacy migration (see _load_from_disk)
import time
from typing import Any

from src.config.constants import (
    DEFAULT_SUBSTRUCTURE_MAX_SIZE,
    DEFAULT_SUBSTRUCTURE_SIMILARITY_THRESHOLD,
    SUBSTRUCTURE_LIBRARY_FORMAT_VERSION,
)
from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


def _resolve_trust_legacy_pickle(explicit: bool | None) -> bool:
    """
    Resolve the "trust legacy pickle" flag.

    Precedence: explicit constructor arg > ``Settings.ASSEMBLY_TRUST_LEGACY_PICKLE``
    > ``False``. Settings access is defensive so this low-level library never fails to
    construct in environments where full settings validation (e.g. API keys) is unavailable.
    """
    if explicit is not None:
        return explicit
    try:
        from src.config.settings import get_settings

        return bool(get_settings().ASSEMBLY_TRUST_LEGACY_PICKLE)
    except Exception:  # pragma: no cover - defensive: settings unavailable/invalid
        return False


@dataclass
class Match:
    """
    Represents a pattern match from the library.

    Attributes:
        pattern_id: Unique pattern identifier
        sequence: The matched sequence of states/nodes
        frequency: How many times this pattern has been used
        similarity: Similarity score (0.0-1.0) between query and match
        metadata: Additional match metadata
    """

    pattern_id: str
    sequence: list[Any]
    frequency: int
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "sequence": [str(s) for s in self.sequence],  # Convert to strings
            "frequency": self.frequency,
            "similarity": self.similarity,
            "metadata": self.metadata,
        }


class SubstructureLibrary:
    """
    Library for tracking and reusing assembly patterns.

    Features:
    - Hash-based pattern storage
    - Frequency tracking (copy number)
    - Similarity-based pattern matching
    - LRU eviction for memory management
    - Persistence to disk
    """

    def __init__(
        self,
        max_size: int = DEFAULT_SUBSTRUCTURE_MAX_SIZE,
        similarity_threshold: float = DEFAULT_SUBSTRUCTURE_SIMILARITY_THRESHOLD,
        enable_persistence: bool = True,
        persistence_path: str | None = None,
        trust_legacy_pickle: bool | None = None,
    ):
        """
        Initialize substructure library.

        Args:
            max_size: Maximum number of patterns to store
            similarity_threshold: Minimum similarity for pattern matching
            enable_persistence: Enable auto-save to disk
            persistence_path: Path for persistence file
            trust_legacy_pickle: Allow a one-time read of a legacy pickled library before
                migrating to JSON. If ``None``, resolved from settings
                (``ASSEMBLY_TRUST_LEGACY_PICKLE``), defaulting to ``False`` (fail-safe).
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or "./workspace/assembly/substructure_library.pkl"
        self._trust_legacy_pickle = _resolve_trust_legacy_pickle(trust_legacy_pickle)

        # Pattern storage: pattern_id -> (sequence, frequency, last_used_timestamp, metadata)
        self._patterns: dict[str, tuple[list[Any], int, float, dict]] = {}

        # Index for fast lookup: sequence_hash -> pattern_id
        self._hash_index: dict[str, str] = {}

        # Statistics
        self._stats = {
            "total_additions": 0,
            "total_queries": 0,
            "cache_hits": 0,
            "evictions": 0,
        }

        # Load from disk if exists
        if self.enable_persistence:
            self._load_from_disk()

    def add_pattern(self, sequence: list[Any], frequency: int = 1, **metadata) -> str:
        """
        Add or increment assembly pattern.

        Args:
            sequence: Sequence of states/nodes representing the pattern
            frequency: Initial frequency (or increment amount)
            **metadata: Additional pattern metadata

        Returns:
            Pattern ID
        """
        if not sequence:
            raise ValueError("Sequence cannot be empty")

        # Generate pattern ID
        pattern_id = self._hash_sequence(sequence)

        # Update or add pattern
        current_time = time.time()

        if pattern_id in self._patterns:
            # Increment frequency
            seq, freq, _, meta = self._patterns[pattern_id]
            new_freq = freq + frequency
            meta.update(metadata)
            self._patterns[pattern_id] = (seq, new_freq, current_time, meta)
        else:
            # Add new pattern
            self._patterns[pattern_id] = (sequence, frequency, current_time, metadata)
            self._hash_index[pattern_id] = pattern_id

            # Check size limit
            if len(self._patterns) > self.max_size:
                self._evict_lru()

        self._stats["total_additions"] += 1

        # Auto-save
        if self.enable_persistence and self._stats["total_additions"] % 100 == 0:
            self._save_to_disk()

        return pattern_id

    def find_reusable_patterns(
        self,
        query_sequence: list[Any],
        max_matches: int = 10,
        min_frequency: int = 1,
    ) -> list[Match]:
        """
        Find similar patterns in library.

        Args:
            query_sequence: Query sequence to match
            max_matches: Maximum number of matches to return
            min_frequency: Minimum pattern frequency to consider

        Returns:
            List of matches, sorted by frequency and similarity
        """
        if not query_sequence:
            return []

        self._stats["total_queries"] += 1

        # Check for exact match first
        query_id = self._hash_sequence(query_sequence)
        if query_id in self._patterns:
            seq, freq, _, meta = self._patterns[query_id]
            if freq >= min_frequency:
                self._stats["cache_hits"] += 1
                return [
                    Match(
                        pattern_id=query_id,
                        sequence=seq,
                        frequency=freq,
                        similarity=1.0,
                        metadata=meta,
                    )
                ]

        # Find similar patterns
        matches = []

        for pattern_id, (seq, freq, _, meta) in self._patterns.items():
            if freq < min_frequency:
                continue

            # Calculate similarity
            similarity = self._calculate_similarity(query_sequence, seq)

            if similarity >= self.similarity_threshold:
                matches.append(
                    Match(
                        pattern_id=pattern_id,
                        sequence=seq,
                        frequency=freq,
                        similarity=similarity,
                        metadata=meta,
                    )
                )

        # Sort by frequency (descending) then similarity (descending)
        matches.sort(key=lambda m: (m.frequency, m.similarity), reverse=True)

        return matches[:max_matches]

    def get_pattern(self, pattern_id: str) -> Match | None:
        """
        Get pattern by ID.

        Args:
            pattern_id: Pattern identifier

        Returns:
            Match object or None if not found
        """
        if pattern_id not in self._patterns:
            return None

        seq, freq, _, meta = self._patterns[pattern_id]
        return Match(
            pattern_id=pattern_id,
            sequence=seq,
            frequency=freq,
            similarity=1.0,  # Exact match
            metadata=meta,
        )

    def get_most_frequent_patterns(self, n: int = 10) -> list[Match]:
        """
        Get N most frequently used patterns.

        Args:
            n: Number of patterns to return

        Returns:
            List of matches sorted by frequency
        """
        patterns = []

        for pattern_id, (seq, freq, _, meta) in self._patterns.items():
            patterns.append(
                Match(
                    pattern_id=pattern_id,
                    sequence=seq,
                    frequency=freq,
                    similarity=1.0,
                    metadata=meta,
                )
            )

        patterns.sort(key=lambda m: m.frequency, reverse=True)
        return patterns[:n]

    def calculate_reuse_rate(self) -> float:
        """
        Calculate overall reuse rate.

        Returns:
            Reuse rate (average pattern frequency)
        """
        if not self._patterns:
            return 0.0

        total_freq = sum(freq for _, freq, _, _ in self._patterns.values())
        return total_freq / len(self._patterns)

    def get_statistics(self) -> dict[str, Any]:
        """
        Get library statistics.

        Returns:
            Statistics dictionary
        """
        stats: dict[str, Any] = dict(self._stats)
        stats["num_patterns"] = len(self._patterns)
        stats["reuse_rate"] = self.calculate_reuse_rate()
        stats["max_frequency"] = max((freq for _, freq, _, _ in self._patterns.values()), default=0)
        stats["avg_sequence_length"] = (
            sum(len(seq) for seq, _, _, _ in self._patterns.values()) / len(self._patterns) if self._patterns else 0
        )

        return stats

    def clear(self) -> None:
        """Clear all patterns from library."""
        self._patterns.clear()
        self._hash_index.clear()
        self._stats["evictions"] += len(self._patterns)

    def _calculate_similarity(self, seq1: list[Any], seq2: list[Any]) -> float:
        """
        Calculate similarity between two sequences.

        Uses Longest Common Subsequence (LCS) ratio.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity score (0.0-1.0)
        """
        if not seq1 or not seq2:
            return 0.0

        # Convert to strings for comparison
        str1 = [str(s) for s in seq1]
        str2 = [str(s) for s in seq2]

        # LCS length
        lcs_len = self._lcs_length(str1, str2)

        # Normalize by average length
        avg_len = (len(seq1) + len(seq2)) / 2.0
        similarity = lcs_len / avg_len if avg_len > 0 else 0.0

        return min(1.0, similarity)

    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """
        Calculate Longest Common Subsequence length.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            LCS length
        """
        m, n = len(seq1), len(seq2)

        # Dynamic programming table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _hash_sequence(self, sequence: list[Any]) -> str:
        """
        Generate hash for sequence.

        Args:
            sequence: Sequence to hash

        Returns:
            Hash string
        """
        # Convert sequence to string representation
        seq_str = "|".join(str(s) for s in sequence)
        return hashlib.sha256(seq_str.encode()).hexdigest()

    def _evict_lru(self) -> None:
        """Evict least recently used pattern."""
        if not self._patterns:
            return

        # Find least recently used
        lru_id = min(
            self._patterns.keys(),
            key=lambda pid: self._patterns[pid][2],  # timestamp
        )

        # Remove
        del self._patterns[lru_id]
        if lru_id in self._hash_index:
            del self._hash_index[lru_id]

        self._stats["evictions"] += 1

    @staticmethod
    def _pattern_to_record(pid: str, seq: list[Any], freq: int, timestamp: float, meta: dict) -> dict[str, Any]:
        """
        Build a single JSON-serializable persistence record for a stored pattern.

        Shared by ``_save_to_disk`` and ``save_json`` so the on-disk schema (including the
        ``sequence`` string-coercion) is defined in exactly one place.
        """
        return {
            "pattern_id": pid,
            "sequence": [str(s) for s in seq],
            "frequency": freq,
            "timestamp": timestamp,
            "metadata": meta,
        }

    @staticmethod
    def _record_to_pattern(record: dict[str, Any]) -> tuple[str, tuple[list[Any], int, float, dict]]:
        """Inverse of :meth:`_pattern_to_record`: produce ``(pattern_id, (seq, freq, ts, meta))``."""
        return (
            record["pattern_id"],
            (
                list(record["sequence"]),
                int(record["frequency"]),
                float(record["timestamp"]),
                dict(record.get("metadata", {})),
            ),
        )

    def _serialize(self) -> dict[str, Any]:
        """Build the versioned, JSON-serializable representation of the library."""
        return {
            "format_version": SUBSTRUCTURE_LIBRARY_FORMAT_VERSION,
            "patterns": [
                self._pattern_to_record(pid, seq, freq, timestamp, meta)
                for pid, (seq, freq, timestamp, meta) in self._patterns.items()
            ],
            "stats": self._stats,
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold,
        }

    def _save_to_disk(self) -> None:
        """Persist the library to disk as versioned JSON (safe, human-inspectable)."""
        try:
            persistence_path = Path(self.persistence_path)
            persistence_path.parent.mkdir(parents=True, exist_ok=True)

            with open(persistence_path, "w", encoding="utf-8") as f:
                # default=str coerces any non-JSON value in pattern metadata so a single bad
                # entry can never make the whole library fail to persist.
                json.dump(self._serialize(), f, default=str)

            logger.debug(
                "Saved substructure library",
                event="substructure_library_save",
                path=str(persistence_path),
                pattern_count=len(self._patterns),
                format_version=SUBSTRUCTURE_LIBRARY_FORMAT_VERSION,
            )
        except Exception as e:
            logger.warning(
                "Failed to save substructure library",
                event="substructure_library_save_failed",
                path=str(self.persistence_path),
                error=str(e),
            )

    def _apply_loaded(self, records: list[dict[str, Any]], stats: dict[str, Any]) -> None:
        """Restore in-memory state from deserialized records."""
        self._patterns = dict(self._record_to_pattern(record) for record in records)
        self._hash_index = {pid: pid for pid in self._patterns}
        self._stats.update(stats or {})

    def _load_from_disk(self) -> None:
        """
        Load the library from disk.

        Prefers the safe versioned-JSON format. If the file is a legacy pickle, it is read
        only when ``trust_legacy_pickle`` is enabled and then immediately re-saved as JSON
        (one-time migration); otherwise the load is skipped and the library starts empty.
        """
        persistence_path = Path(self.persistence_path)
        if not persistence_path.exists():
            return

        # Preferred path: safe versioned JSON.
        try:
            with open(persistence_path, encoding="utf-8") as f:
                data = json.load(f)
            self._apply_loaded(data.get("patterns", []), data.get("stats", {}))
            logger.debug(
                "Loaded substructure library",
                event="substructure_library_load",
                path=str(persistence_path),
                pattern_count=len(self._patterns),
                format_version=data.get("format_version"),
            )
            return
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
            # Not valid JSON — possibly a legacy pickle file. Fall through to migration.
            pass
        except Exception as e:  # pragma: no cover - unexpected IO error
            logger.warning(
                "Failed to load substructure library",
                event="substructure_library_load_failed",
                path=str(persistence_path),
                error=str(e),
            )
            self._patterns = {}
            self._hash_index = {}
            return

        self._migrate_legacy_pickle(persistence_path)

    def _migrate_legacy_pickle(self, persistence_path: Path) -> None:
        """Read a legacy pickled library (opt-in) and migrate it to JSON, or skip safely."""
        if not self._trust_legacy_pickle:
            logger.warning(
                "Ignoring legacy pickled substructure library (set ASSEMBLY_TRUST_LEGACY_PICKLE "
                "to migrate it); starting empty",
                event="substructure_library_legacy_pickle_skipped",
                path=str(persistence_path),
            )
            self._patterns = {}
            self._hash_index = {}
            return

        try:
            with open(persistence_path, "rb") as f:
                data = pickle.load(f)  # nosec B301 - gated by ASSEMBLY_TRUST_LEGACY_PICKLE, opt-in migration

            # Legacy schema stored patterns as {pid: (seq, freq, ts, meta)}.
            legacy_patterns = data.get("patterns", {})
            records = [
                self._pattern_to_record(pid, seq, freq, timestamp, meta)
                for pid, (seq, freq, timestamp, meta) in legacy_patterns.items()
            ]
            self._apply_loaded(records, data.get("stats", {}))

            logger.warning(
                "Migrated legacy pickled substructure library to JSON",
                event="legacy_pickle_migration",
                path=str(persistence_path),
                pattern_count=len(self._patterns),
            )
            # Re-save immediately in the safe format.
            self._save_to_disk()
        except Exception as e:
            logger.warning(
                "Failed to migrate legacy substructure library",
                event="substructure_library_legacy_migration_failed",
                path=str(persistence_path),
                error=str(e),
            )
            self._patterns = {}
            self._hash_index = {}

    def save_json(self, path: str) -> None:
        """
        Save library to JSON (for inspection/debugging).

        Args:
            path: Output file path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "patterns": [
                self._pattern_to_record(pid, seq, freq, timestamp, meta)
                for pid, (seq, freq, timestamp, meta) in self._patterns.items()
            ],
            "statistics": self.get_statistics(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
