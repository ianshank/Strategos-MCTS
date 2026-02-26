"""
Unit tests for metrics collection system.

Tests MCTS metrics, agent metrics, system metrics, and Prometheus integration.

Based on: NEXT_STEPS_PLAN.md Phase 2.4
"""

from __future__ import annotations

import pytest

pytest.importorskip("psutil", reason="psutil required for observability tests")

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def metrics_collector():
    """Create a fresh metrics collector for testing."""
    from collections import defaultdict

    from src.observability.metrics import MCTSMetrics, MetricsCollector

    # Create new instance (don't use singleton for tests)
    collector = MetricsCollector.__new__(MetricsCollector)
    # Use defaultdict like the real implementation
    collector._mcts_metrics = defaultdict(MCTSMetrics)
    collector._agent_metrics = {}
    collector._node_timings = defaultdict(list)
    collector._request_latencies = []
    collector._memory_samples = []
    collector._prometheus_initialized = False
    # Initialize process handle for system metrics
    import psutil

    collector._process = psutil.Process()
    from datetime import datetime

    collector._start_time = datetime.utcnow()

    return collector


@pytest.fixture
def sample_session_id():
    """Provide a sample session ID."""
    return "test-session-001"


# =============================================================================
# MCTS Metrics Tests
# =============================================================================


class TestMCTSMetrics:
    """Tests for MCTS metrics recording."""

    def test_record_mcts_iteration(self, metrics_collector, sample_session_id):
        """Test recording MCTS iteration metrics."""
        metrics_collector.record_mcts_iteration(
            session_id=sample_session_id,
            ucb_score=0.75,
            selection_time_ms=5.0,
            expansion_time_ms=10.0,
            simulation_time_ms=50.0,
            backprop_time_ms=2.0,
        )

        metrics = metrics_collector._mcts_metrics.get(sample_session_id)
        assert metrics is not None
        assert metrics.iterations == 1
        assert 0.75 in metrics.ucb_scores

    def test_record_multiple_iterations(self, metrics_collector, sample_session_id):
        """Test recording multiple MCTS iterations."""
        for i in range(5):
            metrics_collector.record_mcts_iteration(
                session_id=sample_session_id,
                ucb_score=0.5 + i * 0.1,
            )

        metrics = metrics_collector._mcts_metrics.get(sample_session_id)
        assert metrics.iterations == 5
        assert len(metrics.ucb_scores) == 5

    def test_record_mcts_simulation(self, metrics_collector, sample_session_id):
        """Test recording MCTS simulations."""
        metrics_collector.record_mcts_simulation(sample_session_id)
        metrics_collector.record_mcts_simulation(sample_session_id)

        metrics = metrics_collector._mcts_metrics.get(sample_session_id)
        assert metrics.total_simulations == 2

    def test_update_mcts_tree_stats(self, metrics_collector, sample_session_id):
        """Test updating MCTS tree statistics."""
        metrics_collector.update_mcts_tree_stats(
            session_id=sample_session_id,
            tree_depth=15,
            total_nodes=500,
            best_action_visits=100,
            best_action_value=0.85,
        )

        metrics = metrics_collector._mcts_metrics.get(sample_session_id)
        assert metrics.tree_depth == 15
        assert metrics.total_nodes == 500
        assert metrics.best_action_visits == 100
        assert metrics.best_action_value == 0.85

    def test_get_mcts_summary(self, metrics_collector, sample_session_id):
        """Test getting MCTS summary."""
        # Record some data
        metrics_collector.record_mcts_iteration(
            session_id=sample_session_id,
            ucb_score=0.8,
            selection_time_ms=5.0,
        )

        summary = metrics_collector.get_mcts_summary(sample_session_id)

        # Implementation uses "total_iterations" not "iterations"
        assert "total_iterations" in summary
        assert "total_simulations" in summary
        assert summary["total_iterations"] == 1


# =============================================================================
# Agent Metrics Tests
# =============================================================================


class TestAgentMetrics:
    """Tests for agent metrics recording."""

    def test_record_agent_execution(self, metrics_collector):
        """Test recording agent execution metrics."""
        metrics_collector.record_agent_execution(
            agent_name="hrm",
            execution_time_ms=250.0,
            confidence=0.85,
            success=True,
        )

        metrics = metrics_collector._agent_metrics.get("hrm")
        assert metrics is not None
        assert metrics.executions == 1
        assert metrics.success_count == 1
        assert 0.85 in metrics.confidence_scores

    def test_record_failed_agent_execution(self, metrics_collector):
        """Test recording failed agent execution."""
        metrics_collector.record_agent_execution(
            agent_name="trm",
            execution_time_ms=100.0,
            confidence=0.3,
            success=False,
        )

        metrics = metrics_collector._agent_metrics.get("trm")
        assert metrics.error_count == 1
        assert metrics.success_count == 0

    def test_get_agent_summary(self, metrics_collector):
        """Test getting agent summary."""
        metrics_collector.record_agent_execution(
            agent_name="hrm",
            execution_time_ms=200.0,
            confidence=0.9,
            success=True,
        )

        summary = metrics_collector.get_agent_summary("hrm")

        # Implementation uses "total_executions" and "confidence" dict
        assert "total_executions" in summary
        assert "confidence" in summary
        assert "success_rate" in summary


# =============================================================================
# Node Timing Tests
# =============================================================================


class TestNodeTimingMetrics:
    """Tests for node timing metrics."""

    def test_record_node_timing(self, metrics_collector):
        """Test recording node timing."""
        metrics_collector.record_node_timing(
            node_name="entry",
            execution_time_ms=15.0,
        )

        assert "entry" in metrics_collector._node_timings
        assert 15.0 in metrics_collector._node_timings["entry"]

    def test_record_multiple_node_timings(self, metrics_collector):
        """Test recording multiple timings for same node."""
        for time_ms in [10.0, 15.0, 20.0]:
            metrics_collector.record_node_timing(
                node_name="synthesize",
                execution_time_ms=time_ms,
            )

        timings = metrics_collector._node_timings.get("synthesize")
        assert len(timings) == 3

    def test_get_node_timing_summary(self, metrics_collector):
        """Test getting node timing summary."""
        for time_ms in [10.0, 20.0, 30.0]:
            metrics_collector.record_node_timing(
                node_name="route",
                execution_time_ms=time_ms,
            )

        summary = metrics_collector.get_node_timing_summary()

        assert "route" in summary
        route_summary = summary["route"]
        assert route_summary["count"] == 3
        # Implementation uses "mean_ms" not "avg_ms"
        assert route_summary["mean_ms"] == 20.0
        assert route_summary["min_ms"] == 10.0
        assert route_summary["max_ms"] == 30.0


# =============================================================================
# Request Latency Tests
# =============================================================================


class TestRequestLatencyMetrics:
    """Tests for request latency metrics."""

    def test_record_request_latency(self, metrics_collector):
        """Test recording request latency."""
        metrics_collector.record_request_latency(latency_ms=150.0)

        assert 150.0 in metrics_collector._request_latencies

    def test_record_multiple_latencies(self, metrics_collector):
        """Test recording multiple request latencies."""
        for latency in [100.0, 150.0, 200.0]:
            metrics_collector.record_request_latency(latency)

        assert len(metrics_collector._request_latencies) == 3


# =============================================================================
# System Metrics Tests
# =============================================================================


class TestSystemMetrics:
    """Tests for system metrics sampling."""

    def test_sample_system_metrics_returns_dict(self, metrics_collector):
        """Test system metrics sampling returns dict."""
        system_metrics = metrics_collector.sample_system_metrics()

        assert isinstance(system_metrics, dict)
        # Implementation uses memory_rss_mb, memory_vms_mb
        assert "memory_rss_mb" in system_metrics
        assert "cpu_percent" in system_metrics

    def test_sample_system_metrics_has_valid_values(self, metrics_collector):
        """Test system metrics have valid values."""
        system_metrics = metrics_collector.sample_system_metrics()

        assert system_metrics["memory_rss_mb"] >= 0
        # CPU percent can vary; just check it's a number
        assert isinstance(system_metrics["cpu_percent"], (int, float))


# =============================================================================
# Full Report Tests
# =============================================================================


class TestFullReport:
    """Tests for full metrics report generation."""

    def test_get_full_report_structure(self, metrics_collector):
        """Test full report has expected structure."""
        # Add some data
        metrics_collector.record_mcts_iteration(
            session_id="test",
            ucb_score=0.5,
        )
        metrics_collector.record_agent_execution(
            agent_name="hrm",
            execution_time_ms=100.0,
            confidence=0.8,
            success=True,
        )

        report = metrics_collector.get_full_report()

        # Implementation uses different key names
        assert "mcts_sessions" in report
        assert "agents" in report
        assert "node_timings" in report
        assert "system_metrics" in report


# =============================================================================
# Reset Tests
# =============================================================================


class TestMetricsReset:
    """Tests for metrics reset functionality."""

    def test_reset_clears_all_metrics(self, metrics_collector):
        """Test reset clears all collected metrics."""
        # Add data
        metrics_collector.record_mcts_iteration(
            session_id="test",
            ucb_score=0.5,
        )
        metrics_collector.record_agent_execution(
            agent_name="hrm",
            execution_time_ms=100.0,
            confidence=0.8,
            success=True,
        )

        metrics_collector.reset()

        assert len(metrics_collector._mcts_metrics) == 0
        assert len(metrics_collector._agent_metrics) == 0
        assert len(metrics_collector._node_timings) == 0


# =============================================================================
# Metrics Timer Tests
# =============================================================================


class TestMetricsTimer:
    """Tests for MetricsTimer context manager."""

    def test_metrics_timer_context_manager(self, metrics_collector):
        """Test MetricsTimer as context manager."""
        from src.observability.metrics import MetricsTimer

        with MetricsTimer(collector=metrics_collector, node_name="test_node"):
            pass

        # Timer should have recorded the timing
        assert "test_node" in metrics_collector._node_timings

    @pytest.mark.asyncio
    async def test_metrics_timer_async_context_manager(self, metrics_collector):
        """Test MetricsTimer as async context manager."""
        from src.observability.metrics import MetricsTimer

        async with MetricsTimer(collector=metrics_collector, node_name="async_node"):
            pass

        assert "async_node" in metrics_collector._node_timings


# =============================================================================
# Singleton Tests
# =============================================================================


class TestMetricsCollectorSingleton:
    """Tests for MetricsCollector singleton pattern."""

    def test_get_instance_returns_singleton(self):
        """Test get_instance returns same instance."""
        from src.observability.metrics import MetricsCollector

        instance1 = MetricsCollector.get_instance()
        instance2 = MetricsCollector.get_instance()

        assert instance1 is instance2

    def test_convenience_functions_use_singleton(self):
        """Test convenience functions use singleton."""
        from src.observability.metrics import agent_metrics, mcts_metrics

        collector1 = mcts_metrics()
        collector2 = agent_metrics()

        assert collector1 is collector2
