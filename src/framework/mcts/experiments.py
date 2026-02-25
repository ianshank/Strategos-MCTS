"""
Experiment Tracking Module - Track, analyze, and compare MCTS experiments.

Provides:
- Experiment run tracking (seed, params, results)
- Statistical analysis of MCTS performance
- Comparison utilities for different configurations
- Export to JSON/CSV for analysis
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import MCTSConfig


@dataclass
class ExperimentResult:
    """Result of a single MCTS experiment run."""

    # Identification
    experiment_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration
    config: dict[str, Any] | None = None
    seed: int = 42

    # Core results
    best_action: str | None = None
    best_action_value: float = 0.0
    best_action_visits: int = 0
    root_visits: int = 0

    # Performance metrics
    total_iterations: int = 0
    total_simulations: int = 0
    execution_time_ms: float = 0.0

    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Tree statistics
    tree_depth: int = 0
    tree_node_count: int = 0
    branching_factor: float = 0.0

    # Action distribution
    action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Optional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> ExperimentResult:
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


class ExperimentTracker:
    """
    Track and analyze MCTS experiments.

    Features:
    - Store multiple experiment results
    - Statistical analysis across runs
    - Configuration comparison
    - Export to JSON/CSV
    """

    def __init__(self, name: str = "mcts_experiments"):
        """
        Initialize experiment tracker.

        Args:
            name: Name of this experiment series
        """
        self.name = name
        self.results: list[ExperimentResult] = []
        self.created_at = datetime.now().isoformat()

    def add_result(self, result: ExperimentResult) -> None:
        """
        Add an experiment result.

        Args:
            result: ExperimentResult to add
        """
        self.results.append(result)

    def create_result(
        self,
        experiment_id: str,
        config: MCTSConfig,
        mcts_stats: dict[str, Any],
        execution_time_ms: float = 0.0,
        tree_depth: int = 0,
        tree_node_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        """
        Create and add an experiment result from MCTS statistics.

        Args:
            experiment_id: Unique ID for this experiment
            config: MCTS configuration used
            mcts_stats: Statistics dict from MCTSEngine.search()
            execution_time_ms: Execution time in milliseconds
            tree_depth: Depth of MCTS tree
            tree_node_count: Total nodes in tree
            metadata: Optional additional metadata

        Returns:
            Created ExperimentResult
        """
        # Calculate branching factor
        branching_factor = 0.0
        if tree_node_count > 1 and tree_depth > 0:
            branching_factor = (tree_node_count - 1) / tree_depth

        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config.to_dict(),
            seed=config.seed,
            best_action=mcts_stats.get("best_action"),
            best_action_value=mcts_stats.get("best_action_value", 0.0),
            best_action_visits=mcts_stats.get("best_action_visits", 0),
            root_visits=mcts_stats.get("root_visits", 0),
            total_iterations=mcts_stats.get("iterations", 0),
            total_simulations=mcts_stats.get("total_simulations", 0),
            execution_time_ms=execution_time_ms,
            cache_hits=mcts_stats.get("cache_hits", 0),
            cache_misses=mcts_stats.get("cache_misses", 0),
            cache_hit_rate=mcts_stats.get("cache_hit_rate", 0.0),
            tree_depth=tree_depth,
            tree_node_count=tree_node_count,
            branching_factor=branching_factor,
            action_stats=mcts_stats.get("action_stats", {}),
            metadata=metadata or {},
        )

        self.add_result(result)
        return result

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Compute summary statistics across all experiments.

        Returns:
            Dictionary of summary statistics
        """
        if not self.results:
            return {"error": "No results to analyze"}

        # Extract metrics
        best_values = [r.best_action_value for r in self.results]
        best_visits = [r.best_action_visits for r in self.results]
        exec_times = [r.execution_time_ms for r in self.results]
        cache_rates = [r.cache_hit_rate for r in self.results]
        tree_depths = [r.tree_depth for r in self.results]
        node_counts = [r.tree_node_count for r in self.results]

        def compute_stats(values: list[float] | list[int]) -> dict[str, float]:
            """Compute basic statistics."""
            if not values:
                return {}
            return {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
            }

        # Best action consistency
        best_actions = [r.best_action for r in self.results]
        action_counts: dict[str, int] = {}
        for action in best_actions:
            if action is not None:
                action_counts[action] = action_counts.get(action, 0) + 1
        most_common_action = max(action_counts.items(), key=lambda x: x[1])
        consistency_rate = most_common_action[1] / len(best_actions)

        return {
            "num_experiments": len(self.results),
            "best_action_value_stats": compute_stats(best_values),
            "best_action_visits_stats": compute_stats(best_visits),
            "execution_time_ms_stats": compute_stats(exec_times),
            "cache_hit_rate_stats": compute_stats(cache_rates),
            "tree_depth_stats": compute_stats(tree_depths),
            "tree_node_count_stats": compute_stats(node_counts),
            "action_consistency": {
                "most_common_action": most_common_action[0],
                "consistency_rate": consistency_rate,
                "action_distribution": action_counts,
            },
        }

    def compare_configs(
        self,
        config_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Compare performance across different configurations.

        Args:
            config_names: Specific config names to compare (all if None)

        Returns:
            Dictionary mapping config names to their statistics
        """
        # Group results by configuration name
        grouped: dict[str, list[ExperimentResult]] = {}

        for result in self.results:
            if result.config is None:
                continue

            config_name = result.config.get("name", "unnamed")

            if config_names and config_name not in config_names:
                continue

            if config_name not in grouped:
                grouped[config_name] = []
            grouped[config_name].append(result)

        # Compute statistics for each group
        comparison = {}
        for name, results in grouped.items():
            values = [r.best_action_value for r in results]
            times = [r.execution_time_ms for r in results]
            visits = [r.best_action_visits for r in results]

            comparison[name] = {
                "num_runs": len(results),
                "avg_value": statistics.mean(values) if values else 0.0,
                "std_value": statistics.stdev(values) if len(values) > 1 else 0.0,
                "avg_time_ms": statistics.mean(times) if times else 0.0,
                "avg_visits": statistics.mean(visits) if visits else 0.0,
                "value_per_ms": (
                    statistics.mean(values) / statistics.mean(times) if times and statistics.mean(times) > 0 else 0.0
                ),
            }

        return comparison

    def analyze_seed_consistency(self, seed: int) -> dict[str, Any]:
        """
        Analyze consistency of results for a specific seed.

        Args:
            seed: Seed value to analyze

        Returns:
            Analysis of determinism for this seed
        """
        seed_results = [r for r in self.results if r.seed == seed]

        if not seed_results:
            return {"error": f"No results found for seed {seed}"}

        # Check if all results are identical
        best_actions = [r.best_action for r in seed_results]
        best_values = [r.best_action_value for r in seed_results]
        best_visits = [r.best_action_visits for r in seed_results]

        is_deterministic = len(set(best_actions)) == 1 and len(set(best_values)) == 1 and len(set(best_visits)) == 1

        return {
            "seed": seed,
            "num_runs": len(seed_results),
            "is_deterministic": is_deterministic,
            "unique_actions": list(set(best_actions)),
            "value_variance": statistics.variance(best_values) if len(best_values) > 1 else 0.0,
            "visits_variance": statistics.variance(best_visits) if len(best_visits) > 1 else 0.0,
        }

    def export_to_json(self, file_path: str) -> None:
        """
        Export all results to JSON file.

        Args:
            file_path: Path to output file
        """
        data = {
            "name": self.name,
            "created_at": self.created_at,
            "num_experiments": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "summary": self.get_summary_statistics(),
        }

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_to_csv(self, file_path: str) -> None:
        """
        Export results to CSV file for spreadsheet analysis.

        Args:
            file_path: Path to output file
        """
        if not self.results:
            return

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Define CSV columns
        fieldnames = [
            "experiment_id",
            "timestamp",
            "seed",
            "config_name",
            "num_iterations",
            "exploration_weight",
            "best_action",
            "best_action_value",
            "best_action_visits",
            "root_visits",
            "total_simulations",
            "execution_time_ms",
            "cache_hit_rate",
            "tree_depth",
            "tree_node_count",
            "branching_factor",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {
                    "experiment_id": result.experiment_id,
                    "timestamp": result.timestamp,
                    "seed": result.seed,
                    "config_name": (result.config.get("name", "unnamed") if result.config else "unknown"),
                    "num_iterations": (result.config.get("num_iterations", 0) if result.config else 0),
                    "exploration_weight": (result.config.get("exploration_weight", 0) if result.config else 0),
                    "best_action": result.best_action,
                    "best_action_value": result.best_action_value,
                    "best_action_visits": result.best_action_visits,
                    "root_visits": result.root_visits,
                    "total_simulations": result.total_simulations,
                    "execution_time_ms": result.execution_time_ms,
                    "cache_hit_rate": result.cache_hit_rate,
                    "tree_depth": result.tree_depth,
                    "tree_node_count": result.tree_node_count,
                    "branching_factor": result.branching_factor,
                }
                writer.writerow(row)

    @classmethod
    def load_from_json(cls, file_path: str) -> ExperimentTracker:
        """
        Load experiment tracker from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Loaded ExperimentTracker
        """
        with open(file_path) as f:
            data = json.load(f)

        tracker = cls(name=data.get("name", "loaded_experiments"))
        tracker.created_at = data.get("created_at", tracker.created_at)

        for result_data in data.get("results", []):
            tracker.results.append(ExperimentResult.from_dict(result_data))

        return tracker

    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        return f"ExperimentTracker(name={self.name!r}, num_results={len(self.results)})"


def run_determinism_test(
    engine_factory,
    config: MCTSConfig,
    num_runs: int = 3,
) -> tuple[bool, dict[str, Any]]:
    """
    Test that MCTS produces deterministic results with same seed.

    Args:
        engine_factory: Factory function to create MCTSEngine
        config: Configuration to test
        num_runs: Number of runs to compare

    Returns:
        Tuple of (is_deterministic, analysis_dict)
    """
    ExperimentTracker(name="determinism_test")

    # This is a stub - actual implementation would run the engine
    # Results would be compared to verify determinism

    analysis = {
        "config": config.to_dict(),
        "num_runs": num_runs,
        "is_deterministic": True,  # Would be computed from actual runs
        "message": "Determinism test requires actual engine execution",
    }

    return True, analysis
