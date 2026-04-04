"""
Tests for MCTS experiment tracking module.

Tests ExperimentResult, ExperimentTracker, and utility functions
including JSON/CSV export, statistical analysis, and config comparison.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.framework.mcts.config import MCTSConfig
from src.framework.mcts.experiments import (
    ExperimentResult,
    ExperimentTracker,
    run_determinism_test,
)


def _make_result(exp_id="exp-1", best_action="a1", value=0.8, visits=50, seed=42, config=None):
    return ExperimentResult(
        experiment_id=exp_id,
        seed=seed,
        best_action=best_action,
        best_action_value=value,
        best_action_visits=visits,
        root_visits=100,
        total_iterations=100,
        execution_time_ms=50.0,
        cache_hit_rate=0.6,
        tree_depth=5,
        tree_node_count=30,
        config=config or {"name": "default"},
    )


@pytest.mark.unit
class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_defaults(self):
        r = ExperimentResult(experiment_id="test")
        assert r.experiment_id == "test"
        assert r.seed == 42
        assert r.best_action is None
        assert r.best_action_value == 0.0
        assert r.metadata == {}

    def test_to_dict(self):
        r = _make_result()
        d = r.to_dict()
        assert d["experiment_id"] == "exp-1"
        assert d["best_action"] == "a1"
        assert isinstance(d, dict)

    def test_to_json(self):
        r = _make_result()
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["experiment_id"] == "exp-1"

    def test_from_dict(self):
        original = _make_result()
        d = original.to_dict()
        restored = ExperimentResult.from_dict(d)
        assert restored.experiment_id == original.experiment_id
        assert restored.best_action_value == original.best_action_value

    def test_from_json(self):
        original = _make_result()
        j = original.to_json()
        restored = ExperimentResult.from_json(j)
        assert restored.experiment_id == original.experiment_id

    def test_roundtrip(self):
        original = _make_result(config={"name": "test", "iterations": 100})
        j = original.to_json()
        restored = ExperimentResult.from_json(j)
        assert restored.to_dict() == original.to_dict()


@pytest.mark.unit
class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    def test_init(self):
        tracker = ExperimentTracker(name="test")
        assert tracker.name == "test"
        assert len(tracker) == 0

    def test_add_result(self):
        tracker = ExperimentTracker()
        tracker.add_result(_make_result())
        assert len(tracker) == 1

    def test_create_result(self):
        tracker = ExperimentTracker()
        config = MCTSConfig()
        stats = {
            "best_action": "move_a",
            "best_action_value": 0.9,
            "best_action_visits": 40,
            "root_visits": 100,
            "iterations": 100,
            "total_simulations": 200,
            "cache_hits": 30,
            "cache_misses": 70,
            "cache_hit_rate": 0.3,
            "action_stats": {"move_a": {"visits": 40}},
        }
        result = tracker.create_result(
            experiment_id="exp-1",
            config=config,
            mcts_stats=stats,
            execution_time_ms=100.0,
            tree_depth=4,
            tree_node_count=20,
            metadata={"note": "test"},
        )
        assert result.best_action == "move_a"
        assert result.best_action_value == 0.9
        assert result.branching_factor == pytest.approx(19.0 / 4)
        assert len(tracker) == 1

    def test_create_result_no_tree(self):
        """Branching factor should be 0 when tree_depth=0."""
        tracker = ExperimentTracker()
        result = tracker.create_result(
            experiment_id="exp-2",
            config=MCTSConfig(),
            mcts_stats={},
            tree_depth=0,
            tree_node_count=0,
        )
        assert result.branching_factor == 0.0

    def test_get_summary_statistics_empty(self):
        tracker = ExperimentTracker()
        stats = tracker.get_summary_statistics()
        assert "error" in stats

    def test_get_summary_statistics(self):
        tracker = ExperimentTracker()
        tracker.add_result(_make_result("e1", "a1", 0.8, 50))
        tracker.add_result(_make_result("e2", "a1", 0.9, 60))
        tracker.add_result(_make_result("e3", "a2", 0.7, 40))

        stats = tracker.get_summary_statistics()
        assert stats["num_experiments"] == 3
        assert "best_action_value_stats" in stats
        assert stats["best_action_value_stats"]["mean"] == pytest.approx(0.8)
        assert stats["action_consistency"]["most_common_action"] == "a1"
        assert stats["action_consistency"]["consistency_rate"] == pytest.approx(2 / 3)

    def test_compare_configs(self):
        tracker = ExperimentTracker()
        tracker.add_result(_make_result("e1", "a1", 0.8, config={"name": "fast"}))
        tracker.add_result(_make_result("e2", "a1", 0.9, config={"name": "fast"}))
        tracker.add_result(_make_result("e3", "a2", 0.7, config={"name": "slow"}))

        comparison = tracker.compare_configs()
        assert "fast" in comparison
        assert "slow" in comparison
        assert comparison["fast"]["num_runs"] == 2
        assert comparison["slow"]["num_runs"] == 1

    def test_compare_configs_filter(self):
        tracker = ExperimentTracker()
        tracker.add_result(_make_result("e1", config={"name": "fast"}))
        tracker.add_result(_make_result("e2", config={"name": "slow"}))

        comparison = tracker.compare_configs(config_names=["fast"])
        assert "fast" in comparison
        assert "slow" not in comparison

    def test_compare_configs_no_config(self):
        tracker = ExperimentTracker()
        r = _make_result()
        r.config = None
        tracker.add_result(r)
        comparison = tracker.compare_configs()
        assert comparison == {}

    def test_analyze_seed_consistency(self):
        tracker = ExperimentTracker()
        tracker.add_result(_make_result("e1", "a1", 0.8, seed=42))
        tracker.add_result(_make_result("e2", "a1", 0.8, seed=42))

        analysis = tracker.analyze_seed_consistency(42)
        assert analysis["seed"] == 42
        assert analysis["num_runs"] == 2
        assert analysis["is_deterministic"] is True

    def test_analyze_seed_no_results(self):
        tracker = ExperimentTracker()
        analysis = tracker.analyze_seed_consistency(99)
        assert "error" in analysis

    def test_analyze_seed_non_deterministic(self):
        tracker = ExperimentTracker()
        tracker.add_result(_make_result("e1", "a1", 0.8, seed=42))
        tracker.add_result(_make_result("e2", "a2", 0.7, seed=42))

        analysis = tracker.analyze_seed_consistency(42)
        assert analysis["is_deterministic"] is False

    def test_export_to_json(self, tmp_path):
        tracker = ExperimentTracker(name="export_test")
        tracker.add_result(_make_result("e1", "a1", 0.8))

        path = str(tmp_path / "results.json")
        tracker.export_to_json(path)

        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "export_test"
        assert len(data["results"]) == 1

    def test_export_to_csv(self, tmp_path):
        tracker = ExperimentTracker()
        tracker.add_result(
            _make_result("e1", "a1", 0.8, config={"name": "test", "num_iterations": 100, "exploration_weight": 1.4})
        )

        path = str(tmp_path / "results.csv")
        tracker.export_to_csv(path)

        with open(path) as f:
            content = f.read()
        assert "experiment_id" in content
        assert "e1" in content

    def test_export_to_csv_empty(self, tmp_path):
        tracker = ExperimentTracker()
        path = str(tmp_path / "empty.csv")
        tracker.export_to_csv(path)
        # File should not be created when no results
        import os

        assert not os.path.exists(path)

    def test_load_from_json(self, tmp_path):
        tracker = ExperimentTracker(name="load_test")
        tracker.add_result(_make_result("e1", "a1", 0.8))
        path = str(tmp_path / "load.json")
        tracker.export_to_json(path)

        loaded = ExperimentTracker.load_from_json(path)
        assert loaded.name == "load_test"
        assert len(loaded) == 1
        assert loaded.results[0].experiment_id == "e1"

    def test_clear(self):
        tracker = ExperimentTracker()
        tracker.add_result(_make_result())
        tracker.clear()
        assert len(tracker) == 0

    def test_repr(self):
        tracker = ExperimentTracker(name="test")
        tracker.add_result(_make_result())
        r = repr(tracker)
        assert "test" in r
        assert "1" in r


@pytest.mark.unit
class TestRunDeterminismTest:
    """Tests for run_determinism_test utility."""

    def test_returns_tuple(self):
        factory = MagicMock()
        config = MCTSConfig()
        is_det, analysis = run_determinism_test(factory, config, num_runs=3)
        assert is_det is True
        assert analysis["num_runs"] == 3
        assert "is_deterministic" in analysis
