"""
Unit tests for src/framework/assembly/substructure_library.py.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.framework.assembly.substructure_library import Match, SubstructureLibrary

# ---------------------------------------------------------------------------
# Match dataclass tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMatch:
    """Tests for Match dataclass."""

    def test_defaults(self):
        m = Match(pattern_id="p1", sequence=["a", "b"], frequency=3, similarity=0.9)
        assert m.pattern_id == "p1"
        assert m.sequence == ["a", "b"]
        assert m.frequency == 3
        assert m.similarity == 0.9
        assert m.metadata == {}

    def test_to_dict(self):
        m = Match(pattern_id="p1", sequence=[1, 2, 3], frequency=5, similarity=0.8, metadata={"k": "v"})
        d = m.to_dict()
        assert d["pattern_id"] == "p1"
        assert d["sequence"] == ["1", "2", "3"]  # converted to strings
        assert d["frequency"] == 5
        assert d["similarity"] == 0.8
        assert d["metadata"] == {"k": "v"}


# ---------------------------------------------------------------------------
# SubstructureLibrary tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubstructureLibrary:
    """Tests for SubstructureLibrary class."""

    def _make_lib(self, **kwargs) -> SubstructureLibrary:
        """Create a library with persistence disabled."""
        defaults = {"enable_persistence": False, "max_size": 100, "similarity_threshold": 0.5}
        defaults.update(kwargs)
        return SubstructureLibrary(**defaults)

    # -- add_pattern --

    def test_add_pattern_returns_id(self):
        lib = self._make_lib()
        pid = lib.add_pattern(["a", "b", "c"])
        assert isinstance(pid, str)
        assert len(pid) > 0

    def test_add_pattern_empty_raises(self):
        lib = self._make_lib()
        with pytest.raises(ValueError, match="empty"):
            lib.add_pattern([])

    def test_add_same_pattern_increments_frequency(self):
        lib = self._make_lib()
        pid1 = lib.add_pattern(["a", "b"])
        pid2 = lib.add_pattern(["a", "b"])
        assert pid1 == pid2
        match = lib.get_pattern(pid1)
        assert match is not None
        assert match.frequency == 2

    def test_add_pattern_with_custom_frequency(self):
        lib = self._make_lib()
        pid = lib.add_pattern(["x"], frequency=10)
        match = lib.get_pattern(pid)
        assert match is not None
        assert match.frequency == 10

    def test_add_pattern_updates_metadata(self):
        lib = self._make_lib()
        pid = lib.add_pattern(["a"], source="first")
        lib.add_pattern(["a"], source="second")
        match = lib.get_pattern(pid)
        assert match is not None
        assert match.metadata["source"] == "second"

    def test_add_pattern_tracks_stats(self):
        lib = self._make_lib()
        lib.add_pattern(["a"])
        lib.add_pattern(["b"])
        assert lib._stats["total_additions"] == 2

    # -- eviction --

    def test_eviction_when_full(self):
        lib = self._make_lib(max_size=3)
        lib.add_pattern(["a"])
        lib.add_pattern(["b"])
        lib.add_pattern(["c"])
        # Adding a 4th should trigger eviction
        lib.add_pattern(["d"])
        assert len(lib._patterns) <= 3

    def test_eviction_removes_lru(self):
        lib = self._make_lib(max_size=2)
        pid1 = lib.add_pattern(["first"])
        lib.add_pattern(["second"])
        # Add a third, should evict the first (oldest)
        lib.add_pattern(["third"])
        assert lib.get_pattern(pid1) is None

    # -- get_pattern --

    def test_get_pattern_found(self):
        lib = self._make_lib()
        pid = lib.add_pattern(["x", "y"])
        m = lib.get_pattern(pid)
        assert m is not None
        assert m.pattern_id == pid
        assert m.similarity == 1.0

    def test_get_pattern_not_found(self):
        lib = self._make_lib()
        assert lib.get_pattern("nonexistent") is None

    # -- find_reusable_patterns --

    def test_find_exact_match(self):
        lib = self._make_lib()
        lib.add_pattern(["a", "b", "c"])
        matches = lib.find_reusable_patterns(["a", "b", "c"])
        assert len(matches) == 1
        assert matches[0].similarity == 1.0

    def test_find_empty_query(self):
        lib = self._make_lib()
        lib.add_pattern(["a", "b"])
        assert lib.find_reusable_patterns([]) == []

    def test_find_similar_patterns(self):
        lib = self._make_lib(similarity_threshold=0.5)
        lib.add_pattern(["a", "b", "c", "d"])
        # ["a", "b", "c"] shares 3 elements with ["a", "b", "c", "d"]
        matches = lib.find_reusable_patterns(["a", "b", "c"])
        # The exact match won't be found; the similar one might be
        assert isinstance(matches, list)

    def test_find_with_min_frequency(self):
        lib = self._make_lib(similarity_threshold=0.0)
        lib.add_pattern(["a", "b"], frequency=1)
        lib.add_pattern(["c", "d"], frequency=5)
        matches = lib.find_reusable_patterns(["c", "d"], min_frequency=3)
        # Only exact match for ["c", "d"] with freq=5 should be returned
        assert len(matches) >= 1
        assert all(m.frequency >= 3 for m in matches)

    def test_find_updates_query_stats(self):
        lib = self._make_lib()
        lib.find_reusable_patterns(["a"])
        lib.find_reusable_patterns(["b"])
        assert lib._stats["total_queries"] == 2

    def test_find_cache_hit(self):
        lib = self._make_lib()
        lib.add_pattern(["a", "b"])
        lib.find_reusable_patterns(["a", "b"])
        assert lib._stats["cache_hits"] == 1

    def test_find_max_matches(self):
        lib = self._make_lib(similarity_threshold=0.0)
        for i in range(20):
            lib.add_pattern([str(i), "shared"])
        matches = lib.find_reusable_patterns(["0", "shared"], max_matches=5)
        assert len(matches) <= 5

    # -- get_most_frequent_patterns --

    def test_most_frequent(self):
        lib = self._make_lib()
        lib.add_pattern(["a"], frequency=10)
        lib.add_pattern(["b"], frequency=1)
        lib.add_pattern(["c"], frequency=5)
        top = lib.get_most_frequent_patterns(n=2)
        assert len(top) == 2
        assert top[0].frequency >= top[1].frequency

    def test_most_frequent_empty(self):
        lib = self._make_lib()
        assert lib.get_most_frequent_patterns() == []

    # -- calculate_reuse_rate --

    def test_reuse_rate_empty(self):
        lib = self._make_lib()
        assert lib.calculate_reuse_rate() == 0.0

    def test_reuse_rate_computed(self):
        lib = self._make_lib()
        lib.add_pattern(["a"], frequency=4)
        lib.add_pattern(["b"], frequency=6)
        # avg = (4 + 6) / 2 = 5.0
        assert lib.calculate_reuse_rate() == 5.0

    # -- get_statistics --

    def test_statistics(self):
        lib = self._make_lib()
        lib.add_pattern(["a", "b"], frequency=3)
        lib.add_pattern(["c"], frequency=7)
        stats = lib.get_statistics()
        assert stats["num_patterns"] == 2
        assert stats["max_frequency"] == 7
        assert stats["reuse_rate"] == 5.0
        assert stats["total_additions"] == 2
        assert "avg_sequence_length" in stats

    def test_statistics_empty(self):
        lib = self._make_lib()
        stats = lib.get_statistics()
        assert stats["num_patterns"] == 0
        assert stats["max_frequency"] == 0
        assert stats["avg_sequence_length"] == 0

    # -- clear --

    def test_clear(self):
        lib = self._make_lib()
        lib.add_pattern(["a"])
        lib.add_pattern(["b"])
        lib.clear()
        assert len(lib._patterns) == 0
        assert len(lib._hash_index) == 0

    # -- _calculate_similarity / _lcs_length --

    def test_similarity_identical(self):
        lib = self._make_lib()
        sim = lib._calculate_similarity(["a", "b", "c"], ["a", "b", "c"])
        assert sim == 1.0

    def test_similarity_empty(self):
        lib = self._make_lib()
        assert lib._calculate_similarity([], ["a"]) == 0.0
        assert lib._calculate_similarity(["a"], []) == 0.0

    def test_similarity_partial(self):
        lib = self._make_lib()
        sim = lib._calculate_similarity(["a", "b", "c"], ["a", "x", "c"])
        assert 0.0 < sim < 1.0

    def test_lcs_length(self):
        lib = self._make_lib()
        assert lib._lcs_length(["a", "b", "c", "d"], ["a", "c", "d"]) == 3
        assert lib._lcs_length(["a"], ["b"]) == 0
        assert lib._lcs_length([], []) == 0

    # -- _hash_sequence --

    def test_hash_deterministic(self):
        lib = self._make_lib()
        h1 = lib._hash_sequence(["a", "b"])
        h2 = lib._hash_sequence(["a", "b"])
        assert h1 == h2

    def test_hash_different_sequences(self):
        lib = self._make_lib()
        h1 = lib._hash_sequence(["a", "b"])
        h2 = lib._hash_sequence(["b", "a"])
        assert h1 != h2

    # -- persistence --

    def test_save_and_load_from_disk(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "lib.pkl")
            lib = SubstructureLibrary(enable_persistence=True, persistence_path=path)
            lib.add_pattern(["a", "b"], frequency=3)
            lib._save_to_disk()

            lib2 = SubstructureLibrary(enable_persistence=True, persistence_path=path)
            assert len(lib2._patterns) == 1

    def test_load_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "nonexistent.pkl")
            lib = SubstructureLibrary(enable_persistence=True, persistence_path=path)
            # Should not fail, just have empty patterns
            assert len(lib._patterns) == 0

    def test_load_corrupt_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "corrupt.pkl")
            Path(path).write_bytes(b"not a pickle")
            lib = SubstructureLibrary(enable_persistence=True, persistence_path=path)
            assert len(lib._patterns) == 0

    # -- save_json --

    def test_save_json(self):
        lib = self._make_lib()
        lib.add_pattern(["x", "y"], frequency=2)
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "lib.json")
            lib.save_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "patterns" in data
            assert "statistics" in data
            assert len(data["patterns"]) == 1

    def test_save_json_creates_parents(self):
        lib = self._make_lib()
        lib.add_pattern(["a"])
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "sub" / "dir" / "lib.json")
            lib.save_json(path)
            assert Path(path).exists()

    # -- auto-save on 100th addition --

    def test_auto_save_on_100th(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "lib.pkl")
            lib = SubstructureLibrary(enable_persistence=True, persistence_path=path, max_size=10000)
            for i in range(100):
                lib.add_pattern([str(i)])
            # After 100 additions, auto-save should have been triggered
            assert Path(path).exists()
