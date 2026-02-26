"""
Metrics collection infrastructure for multi-agent MCTS framework.

Provides:
- MCTS iteration counters
- UCB score distributions
- Agent confidence tracking
- Timing metrics for each graph node
- Memory usage monitoring
- Export to Prometheus format (optional)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class MCTSMetrics:
    """Container for MCTS-specific metrics."""

    iterations: int = 0
    total_simulations: int = 0
    tree_depth: int = 0
    total_nodes: int = 0
    ucb_scores: list[float] = field(default_factory=list)
    selection_times_ms: list[float] = field(default_factory=list)
    expansion_times_ms: list[float] = field(default_factory=list)
    simulation_times_ms: list[float] = field(default_factory=list)
    backprop_times_ms: list[float] = field(default_factory=list)
    best_action_visits: int = 0
    best_action_value: float = 0.0


@dataclass
class AgentMetrics:
    """Container for agent-specific metrics."""

    name: str
    executions: int = 0
    total_time_ms: float = 0.0
    avg_confidence: float = 0.0
    confidence_scores: list[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    memory_usage_mb: list[float] = field(default_factory=list)


class MetricsCollector:
    """
    Central metrics collection and reporting for the MCTS framework.

    Collects:
    - MCTS iteration counters and UCB scores
    - Agent confidence and execution times
    - Graph node timing metrics
    - Memory usage monitoring
    - Request/response latencies

    Supports optional Prometheus export.
    """

    _instance: Optional["MetricsCollector"] = None

    def __init__(self):
        self._mcts_metrics: dict[str, MCTSMetrics] = defaultdict(MCTSMetrics)
        self._agent_metrics: dict[str, AgentMetrics] = {}
        self._node_timings: dict[str, list[float]] = defaultdict(list)
        self._request_latencies: list[float] = []
        self._memory_samples: list[dict[str, float]] = []
        self._start_time = datetime.utcnow()
        self._process = psutil.Process()

        # Prometheus metrics (if available)
        self._prometheus_initialized = False
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()

        logger.debug("MetricsCollector initialized")

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Get singleton instance of MetricsCollector."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE or self._prometheus_initialized:
            return

        # MCTS counters
        self._prom_mcts_iterations = Counter(
            "mcts_iterations_total",
            "Total number of MCTS iterations",
            ["session_id"],
        )
        self._prom_mcts_simulations = Counter(
            "mcts_simulations_total",
            "Total number of MCTS simulations",
            ["session_id"],
        )

        # MCTS gauges
        self._prom_mcts_tree_depth = Gauge(
            "mcts_tree_depth",
            "Current MCTS tree depth",
            ["session_id"],
        )
        self._prom_mcts_total_nodes = Gauge(
            "mcts_total_nodes",
            "Total nodes in MCTS tree",
            ["session_id"],
        )

        # UCB score histogram
        self._prom_ucb_scores = Histogram(
            "mcts_ucb_score",
            "UCB score distribution",
            ["session_id"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")],
        )

        # Agent metrics
        self._prom_agent_executions = Counter(
            "agent_executions_total",
            "Total agent executions",
            ["agent_name"],
        )
        self._prom_agent_confidence = Summary(
            "agent_confidence",
            "Agent confidence scores",
            ["agent_name"],
        )
        self._prom_agent_execution_time = Histogram(
            "agent_execution_time_ms",
            "Agent execution time in milliseconds",
            ["agent_name"],
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
        )

        # System metrics
        self._prom_memory_usage = Gauge(
            "framework_memory_usage_mb",
            "Memory usage in MB",
        )
        self._prom_cpu_percent = Gauge(
            "framework_cpu_percent",
            "CPU usage percentage",
        )

        # Request latency
        self._prom_request_latency = Histogram(
            "request_latency_ms",
            "Request latency in milliseconds",
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
        )

        self._prometheus_initialized = True

    def start_prometheus_server(self, port: int = 8000) -> None:
        """Start Prometheus metrics HTTP server."""
        if PROMETHEUS_AVAILABLE:
            logger.info("Starting Prometheus metrics server on port %d", port)
            start_http_server(port)
        else:
            logger.warning("Prometheus client not available, cannot start metrics server")

    def record_mcts_iteration(
        self,
        session_id: str,
        ucb_score: float,
        selection_time_ms: float = 0.0,
        expansion_time_ms: float = 0.0,
        simulation_time_ms: float = 0.0,
        backprop_time_ms: float = 0.0,
    ) -> None:
        """Record metrics for a single MCTS iteration."""
        metrics = self._mcts_metrics[session_id]
        metrics.iterations += 1
        metrics.ucb_scores.append(ucb_score)

        if selection_time_ms > 0:
            metrics.selection_times_ms.append(selection_time_ms)
        if expansion_time_ms > 0:
            metrics.expansion_times_ms.append(expansion_time_ms)
        if simulation_time_ms > 0:
            metrics.simulation_times_ms.append(simulation_time_ms)
        if backprop_time_ms > 0:
            metrics.backprop_times_ms.append(backprop_time_ms)

        # Prometheus
        if self._prometheus_initialized:
            self._prom_mcts_iterations.labels(session_id=session_id).inc()
            self._prom_ucb_scores.labels(session_id=session_id).observe(ucb_score)

    def record_mcts_simulation(self, session_id: str) -> None:
        """Record an MCTS simulation."""
        self._mcts_metrics[session_id].total_simulations += 1
        if self._prometheus_initialized:
            self._prom_mcts_simulations.labels(session_id=session_id).inc()

    def update_mcts_tree_stats(
        self,
        session_id: str,
        tree_depth: int,
        total_nodes: int,
        best_action_visits: int = 0,
        best_action_value: float = 0.0,
    ) -> None:
        """Update MCTS tree statistics."""
        metrics = self._mcts_metrics[session_id]
        metrics.tree_depth = tree_depth
        metrics.total_nodes = total_nodes
        metrics.best_action_visits = best_action_visits
        metrics.best_action_value = best_action_value

        if self._prometheus_initialized:
            self._prom_mcts_tree_depth.labels(session_id=session_id).set(tree_depth)
            self._prom_mcts_total_nodes.labels(session_id=session_id).set(total_nodes)

    def record_agent_execution(
        self,
        agent_name: str,
        execution_time_ms: float,
        confidence: float,
        success: bool = True,
    ) -> None:
        """Record agent execution metrics."""
        logger.debug(
            "Recording agent execution: agent=%s, time_ms=%.2f, confidence=%.4f, success=%s",
            agent_name,
            execution_time_ms,
            confidence,
            success,
        )
        if agent_name not in self._agent_metrics:
            self._agent_metrics[agent_name] = AgentMetrics(name=agent_name)

        metrics = self._agent_metrics[agent_name]
        metrics.executions += 1
        metrics.total_time_ms += execution_time_ms
        metrics.confidence_scores.append(confidence)
        metrics.avg_confidence = sum(metrics.confidence_scores) / len(metrics.confidence_scores)

        if success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1

        # Memory sample
        memory_mb = self._process.memory_info().rss / (1024 * 1024)
        metrics.memory_usage_mb.append(memory_mb)

        # Prometheus
        if self._prometheus_initialized:
            self._prom_agent_executions.labels(agent_name=agent_name).inc()
            self._prom_agent_confidence.labels(agent_name=agent_name).observe(confidence)
            self._prom_agent_execution_time.labels(agent_name=agent_name).observe(execution_time_ms)

    def record_node_timing(self, node_name: str, execution_time_ms: float) -> None:
        """Record execution time for a graph node."""
        self._node_timings[node_name].append(execution_time_ms)

    def record_request_latency(self, latency_ms: float) -> None:
        """Record end-to-end request latency."""
        self._request_latencies.append(latency_ms)
        if self._prometheus_initialized:
            self._prom_request_latency.observe(latency_ms)

    def sample_system_metrics(self) -> dict[str, float]:
        """Sample current system metrics."""
        memory_info = self._process.memory_info()
        cpu_percent = self._process.cpu_percent()

        sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_rss_mb": memory_info.rss / (1024 * 1024),
            "memory_vms_mb": memory_info.vms / (1024 * 1024),
            "cpu_percent": cpu_percent,
            "thread_count": self._process.num_threads(),
            "open_files": len(self._process.open_files()),
        }

        self._memory_samples.append(sample)

        if self._prometheus_initialized:
            self._prom_memory_usage.set(sample["memory_rss_mb"])
            self._prom_cpu_percent.set(cpu_percent)

        return sample

    def get_mcts_summary(self, session_id: str) -> dict[str, Any]:
        """Get summary statistics for MCTS session."""
        metrics = self._mcts_metrics.get(session_id)
        if not metrics:
            return {}

        def safe_avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        def safe_percentile(lst: list[float], p: float) -> float:
            if not lst:
                return 0.0
            sorted_lst = sorted(lst)
            idx = int(len(sorted_lst) * p)
            return sorted_lst[min(idx, len(sorted_lst) - 1)]

        return {
            "session_id": session_id,
            "total_iterations": metrics.iterations,
            "total_simulations": metrics.total_simulations,
            "tree_depth": metrics.tree_depth,
            "total_nodes": metrics.total_nodes,
            "best_action_visits": metrics.best_action_visits,
            "best_action_value": round(metrics.best_action_value, 4),
            "ucb_scores": {
                "count": len(metrics.ucb_scores),
                "mean": round(safe_avg(metrics.ucb_scores), 4),
                "min": round(min(metrics.ucb_scores), 4) if metrics.ucb_scores else 0.0,
                "max": round(max(metrics.ucb_scores), 4) if metrics.ucb_scores else 0.0,
                "p50": round(safe_percentile(metrics.ucb_scores, 0.5), 4),
                "p95": round(safe_percentile(metrics.ucb_scores, 0.95), 4),
            },
            "timing_ms": {
                "selection_avg": round(safe_avg(metrics.selection_times_ms), 2),
                "expansion_avg": round(safe_avg(metrics.expansion_times_ms), 2),
                "simulation_avg": round(safe_avg(metrics.simulation_times_ms), 2),
                "backprop_avg": round(safe_avg(metrics.backprop_times_ms), 2),
            },
        }

    def get_agent_summary(self, agent_name: str) -> dict[str, Any]:
        """Get summary statistics for an agent."""
        metrics = self._agent_metrics.get(agent_name)
        if not metrics:
            return {}

        def safe_avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "agent_name": agent_name,
            "total_executions": metrics.executions,
            "success_count": metrics.success_count,
            "error_count": metrics.error_count,
            "success_rate": round(metrics.success_count / max(metrics.executions, 1), 4),
            "avg_execution_time_ms": round(metrics.total_time_ms / max(metrics.executions, 1), 2),
            "total_time_ms": round(metrics.total_time_ms, 2),
            "confidence": {
                "mean": round(safe_avg(metrics.confidence_scores), 4),
                "min": round(min(metrics.confidence_scores), 4) if metrics.confidence_scores else 0.0,
                "max": round(max(metrics.confidence_scores), 4) if metrics.confidence_scores else 0.0,
            },
            "avg_memory_mb": round(safe_avg(metrics.memory_usage_mb), 2),
        }

    def get_node_timing_summary(self) -> dict[str, dict[str, float]]:
        """Get timing summary for all graph nodes."""
        summary = {}
        for node_name, timings in self._node_timings.items():
            if timings:
                summary[node_name] = {
                    "count": len(timings),
                    "mean_ms": round(sum(timings) / len(timings), 2),
                    "min_ms": round(min(timings), 2),
                    "max_ms": round(max(timings), 2),
                    "total_ms": round(sum(timings), 2),
                }
        return summary

    def get_full_report(self) -> dict[str, Any]:
        """Generate a comprehensive metrics report."""
        # Sample current system state
        current_system = self.sample_system_metrics()

        report = {
            "report_time": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "system_metrics": current_system,
            "mcts_sessions": {session_id: self.get_mcts_summary(session_id) for session_id in self._mcts_metrics},
            "agents": {agent_name: self.get_agent_summary(agent_name) for agent_name in self._agent_metrics},
            "node_timings": self.get_node_timing_summary(),
            "request_latencies": {
                "count": len(self._request_latencies),
                "mean_ms": round(sum(self._request_latencies) / max(len(self._request_latencies), 1), 2),
                "min_ms": round(min(self._request_latencies), 2) if self._request_latencies else 0.0,
                "max_ms": round(max(self._request_latencies), 2) if self._request_latencies else 0.0,
            },
        }

        return report

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        if PROMETHEUS_AVAILABLE:
            return str(generate_latest(REGISTRY).decode("utf-8"))
        else:
            return "# Prometheus client not available\n"

    def reset(self) -> None:
        """Reset all collected metrics."""
        logger.info("Resetting all collected metrics")
        self._mcts_metrics.clear()
        self._agent_metrics.clear()
        self._node_timings.clear()
        self._request_latencies.clear()
        self._memory_samples.clear()
        self._start_time = datetime.utcnow()


# Convenience singleton accessors
def mcts_metrics() -> MetricsCollector:
    """Get the singleton MetricsCollector instance."""
    return MetricsCollector.get_instance()


def agent_metrics() -> MetricsCollector:
    """Alias for mcts_metrics() - same singleton."""
    return MetricsCollector.get_instance()


class MetricsTimer:
    """Context manager for timing operations and recording metrics."""

    def __init__(
        self,
        collector: MetricsCollector | None = None,
        node_name: str | None = None,
        mcts_session_id: str | None = None,
        agent_name: str | None = None,
    ):
        self.collector = collector or MetricsCollector.get_instance()
        self.node_name = node_name
        self.mcts_session_id = mcts_session_id
        self.agent_name = agent_name
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if self.node_name:
            self.collector.record_node_timing(self.node_name, self.elapsed_ms)

        return False

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if self.node_name:
            self.collector.record_node_timing(self.node_name, self.elapsed_ms)

        return False
