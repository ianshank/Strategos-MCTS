"""
Substructure Library for Pattern Reuse (Story 1.4).

Tracks frequently used assembly patterns and enables reuse of successful
reasoning subsequences.
"""

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
            'pattern_id': self.pattern_id,
            'sequence': [str(s) for s in self.sequence],  # Convert to strings
            'frequency': self.frequency,
            'similarity': self.similarity,
            'metadata': self.metadata,
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
        max_size: int = 10000,
        similarity_threshold: float = 0.7,
        enable_persistence: bool = True,
        persistence_path: str | None = None,
    ):
        """
        Initialize substructure library.

        Args:
            max_size: Maximum number of patterns to store
            similarity_threshold: Minimum similarity for pattern matching
            enable_persistence: Enable auto-save to disk
            persistence_path: Path for persistence file
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or "./workspace/assembly/substructure_library.pkl"

        # Pattern storage: pattern_id -> (sequence, frequency, last_used_timestamp, metadata)
        self._patterns: dict[str, tuple[list[Any], int, float, dict]] = {}

        # Index for fast lookup: sequence_hash -> pattern_id
        self._hash_index: dict[str, str] = {}

        # Statistics
        self._stats = {
            'total_additions': 0,
            'total_queries': 0,
            'cache_hits': 0,
            'evictions': 0,
        }

        # Load from disk if exists
        if self.enable_persistence:
            self._load_from_disk()

    def add_pattern(
        self,
        sequence: list[Any],
        frequency: int = 1,
        **metadata
    ) -> str:
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

        self._stats['total_additions'] += 1

        # Auto-save
        if self.enable_persistence and self._stats['total_additions'] % 100 == 0:
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

        self._stats['total_queries'] += 1

        # Check for exact match first
        query_id = self._hash_sequence(query_sequence)
        if query_id in self._patterns:
            seq, freq, _, meta = self._patterns[query_id]
            if freq >= min_frequency:
                self._stats['cache_hits'] += 1
                return [Match(
                    pattern_id=query_id,
                    sequence=seq,
                    frequency=freq,
                    similarity=1.0,
                    metadata=meta,
                )]

        # Find similar patterns
        matches = []

        for pattern_id, (seq, freq, _, meta) in self._patterns.items():
            if freq < min_frequency:
                continue

            # Calculate similarity
            similarity = self._calculate_similarity(query_sequence, seq)

            if similarity >= self.similarity_threshold:
                matches.append(Match(
                    pattern_id=pattern_id,
                    sequence=seq,
                    frequency=freq,
                    similarity=similarity,
                    metadata=meta,
                ))

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
            patterns.append(Match(
                pattern_id=pattern_id,
                sequence=seq,
                frequency=freq,
                similarity=1.0,
                metadata=meta,
            ))

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
        stats = dict(self._stats)
        stats['num_patterns'] = len(self._patterns)
        stats['reuse_rate'] = self.calculate_reuse_rate()
        stats['max_frequency'] = max((freq for _, freq, _, _ in self._patterns.values()), default=0)
        stats['avg_sequence_length'] = sum(len(seq) for seq, _, _, _ in self._patterns.values()) / len(self._patterns) if self._patterns else 0

        return stats

    def clear(self) -> None:
        """Clear all patterns from library."""
        self._patterns.clear()
        self._hash_index.clear()
        self._stats['evictions'] += len(self._patterns)

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
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

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
        seq_str = '|'.join(str(s) for s in sequence)
        return hashlib.sha256(seq_str.encode()).hexdigest()

    def _evict_lru(self) -> None:
        """Evict least recently used pattern."""
        if not self._patterns:
            return

        # Find least recently used
        lru_id = min(
            self._patterns.keys(),
            key=lambda pid: self._patterns[pid][2]  # timestamp
        )

        # Remove
        del self._patterns[lru_id]
        if lru_id in self._hash_index:
            del self._hash_index[lru_id]

        self._stats['evictions'] += 1

    def _save_to_disk(self) -> None:
        """Save library to disk."""
        try:
            persistence_path = Path(self.persistence_path)
            persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            data = {
                'patterns': {
                    pid: (
                        [str(s) for s in seq],  # Convert to strings
                        freq,
                        timestamp,
                        meta
                    )
                    for pid, (seq, freq, timestamp, meta) in self._patterns.items()
                },
                'stats': self._stats,
                'max_size': self.max_size,
                'similarity_threshold': self.similarity_threshold,
            }

            with open(persistence_path, 'wb') as f:
                pickle.dump(data, f)

        except Exception as e:
            logger.warning("Failed to save substructure library: %s", e)

    def _load_from_disk(self) -> None:
        """Load library from disk."""
        try:
            persistence_path = Path(self.persistence_path)

            if not persistence_path.exists():
                return

            with open(persistence_path, 'rb') as f:
                data = pickle.load(f)

            # Restore patterns
            self._patterns = {
                pid: (seq, freq, timestamp, meta)
                for pid, (seq, freq, timestamp, meta) in data['patterns'].items()
            }

            # Rebuild hash index
            self._hash_index = {pid: pid for pid in self._patterns}

            # Restore stats
            self._stats.update(data.get('stats', {}))

            # Update config (but preserve constructor values if different)
            # self.max_size = data.get('max_size', self.max_size)
            # self.similarity_threshold = data.get('similarity_threshold', self.similarity_threshold)

        except Exception as e:
            logger.warning("Failed to load substructure library: %s", e)
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
            'patterns': [
                {
                    'pattern_id': pid,
                    'sequence': [str(s) for s in seq],
                    'frequency': freq,
                    'timestamp': timestamp,
                    'metadata': meta,
                }
                for pid, (seq, freq, timestamp, meta) in self._patterns.items()
            ],
            'statistics': self.get_statistics(),
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
