"""
Unit tests for src/observability/metrics.py - extended coverage.

Covers: MetricsCollector methods (record_mcts_iteration with prometheus,
update_mcts_tree_stats, record_agent_execution, sample_system_metrics,
get_mcts_summary, get_agent_summary, get_node_timing_summary,
get_full_report, export_prometheus_format, reset, MetricsTimer, singletons).
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_metrics_singleton():
    """Reset the MetricsCollector singleton between tests."""
    from src.observability.metrics import MetricsCollector

    MetricsCollector._instance = None
    yield
    MetricsCollector._instance = None


class TestMetricsCollectorInit:
    """Tests for MetricsCollector initialization."""

    @patch("src.observability.metrics.psutil.Process")
    def test_init_creates_defaults(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        assert mc._mcts_metrics is not None
        assert mc._agent_metrics == {}
        assert mc._request_latencies == []

    @patch("src.observability.metrics.psutil.Process")
    def test_singleton(self, mock_process):
        from src.observability.metrics import MetricsCollector

        a = MetricsCollector.get_instance()
        b = MetricsCollector.get_instance()
        assert a is b


class TestMCTSIteration:
    """Tests for MCTS iteration recording."""

    @patch("src.observability.metrics.psutil.Process")
    def test_record_mcts_iteration(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        mc.record_mcts_iteration(
            "s1",
            ucb_score=0.75,
            selection_time_ms=10.0,
            expansion_time_ms=5.0,
            simulation_time_ms=20.0,
            backprop_time_ms=3.0,
        )

        m = mc._mcts_metrics["s1"]
        assert m.iterations == 1
        assert m.ucb_scores == [0.75]
        assert m.selection_times_ms == [10.0]
        assert m.expansion_times_ms == [5.0]
        assert m.simulation_times_ms == [20.0]
        assert m.backprop_times_ms == [3.0]

    @patch("src.observability.metrics.psutil.Process")
    def test_record_mcts_iteration_zero_times_skipped(self, mock_process):
        """Zero-value times should not be appended."""
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        mc.record_mcts_iteration("s1", ucb_score=0.5)

        m = mc._mcts_metrics["s1"]
        assert m.selection_times_ms == []
        assert m.expansion_times_ms == []

    @patch("src.observability.metrics.psutil.Process")
    def test_record_mcts_iteration_with_prometheus(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = True
        mc._prom_mcts_iterations = MagicMock()
        mc._prom_ucb_scores = MagicMock()

        mc.record_mcts_iteration("s1", ucb_score=0.8)

        mc._prom_mcts_iterations.labels.assert_called_once_with(session_id="s1")
        mc._prom_ucb_scores.labels.assert_called_once_with(session_id="s1")


class TestMCTSSimulation:
    """Tests for MCTS simulation recording."""

    @patch("src.observability.metrics.psutil.Process")
    def test_record_simulation(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        mc.record_mcts_simulation("s1")
        assert mc._mcts_metrics["s1"].total_simulations == 1

    @patch("src.observability.metrics.psutil.Process")
    def test_record_simulation_with_prometheus(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = True
        mc._prom_mcts_simulations = MagicMock()

        mc.record_mcts_simulation("s1")

        mc._prom_mcts_simulations.labels.assert_called_once_with(session_id="s1")


class TestUpdateMCTSTreeStats:
    """Tests for update_mcts_tree_stats."""

    @patch("src.observability.metrics.psutil.Process")
    def test_update_tree_stats(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        mc.update_mcts_tree_stats("s1", tree_depth=5, total_nodes=20, best_action_visits=10, best_action_value=0.9)

        m = mc._mcts_metrics["s1"]
        assert m.tree_depth == 5
        assert m.total_nodes == 20
        assert m.best_action_visits == 10
        assert m.best_action_value == 0.9

    @patch("src.observability.metrics.psutil.Process")
    def test_update_tree_stats_with_prometheus(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = True
        mc._prom_mcts_tree_depth = MagicMock()
        mc._prom_mcts_total_nodes = MagicMock()

        mc.update_mcts_tree_stats("s1", tree_depth=3, total_nodes=15)

        mc._prom_mcts_tree_depth.labels.assert_called_once()
        mc._prom_mcts_total_nodes.labels.assert_called_once()


class TestAgentExecution:
    """Tests for agent execution recording."""

    @patch("src.observability.metrics.psutil.Process")
    def test_record_agent_execution_success(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mock_mem = MagicMock()
        mock_mem.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_mem

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        mc.record_agent_execution("hrm", execution_time_ms=50.0, confidence=0.85, success=True)

        agent = mc._agent_metrics["hrm"]
        assert agent.executions == 1
        assert agent.success_count == 1
        assert agent.error_count == 0
        assert agent.avg_confidence == 0.85

    @patch("src.observability.metrics.psutil.Process")
    def test_record_agent_execution_failure(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mock_mem = MagicMock()
        mock_mem.rss = 50 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        mc.record_agent_execution("trm", execution_time_ms=100.0, confidence=0.3, success=False)

        agent = mc._agent_metrics["trm"]
        assert agent.error_count == 1
        assert agent.success_count == 0

    @patch("src.observability.metrics.psutil.Process")
    def test_record_agent_execution_with_prometheus(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mock_mem = MagicMock()
        mock_mem.rss = 50 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem

        mc = MetricsCollector()
        mc._prometheus_initialized = True
        mc._prom_agent_executions = MagicMock()
        mc._prom_agent_confidence = MagicMock()
        mc._prom_agent_execution_time = MagicMock()

        mc.record_agent_execution("hrm", execution_time_ms=10.0, confidence=0.9)

        mc._prom_agent_executions.labels.assert_called_once_with(agent_name="hrm")
        mc._prom_agent_confidence.labels.assert_called_once_with(agent_name="hrm")


class TestNodeTiming:
    """Tests for node timing recording."""

    @patch("src.observability.metrics.psutil.Process")
    def test_record_node_timing(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc.record_node_timing("node_a", 15.0)
        mc.record_node_timing("node_a", 25.0)

        assert mc._node_timings["node_a"] == [15.0, 25.0]


class TestRequestLatency:
    """Tests for request latency recording."""

    @patch("src.observability.metrics.psutil.Process")
    def test_record_request_latency(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        mc.record_request_latency(100.0)
        assert mc._request_latencies == [100.0]

    @patch("src.observability.metrics.psutil.Process")
    def test_record_request_latency_with_prometheus(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = True
        mc._prom_request_latency = MagicMock()

        mc.record_request_latency(200.0)
        mc._prom_request_latency.observe.assert_called_once_with(200.0)


class TestSampleSystemMetrics:
    """Tests for system metrics sampling."""

    @patch("src.observability.metrics.psutil.Process")
    def test_sample_system_metrics(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mock_inst = mock_process.return_value
        mock_mem = MagicMock()
        mock_mem.rss = 200 * 1024 * 1024
        mock_mem.vms = 400 * 1024 * 1024
        mock_inst.memory_info.return_value = mock_mem
        mock_inst.cpu_percent.return_value = 25.0
        mock_inst.num_threads.return_value = 4
        mock_inst.open_files.return_value = []

        mc = MetricsCollector()
        sample = mc.sample_system_metrics()

        assert sample["memory_rss_mb"] == pytest.approx(200.0)
        assert sample["cpu_percent"] == 25.0
        assert sample["thread_count"] == 4

    @patch("src.observability.metrics.psutil.Process")
    def test_sample_system_metrics_with_prometheus(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mock_inst = mock_process.return_value
        mock_mem = MagicMock()
        mock_mem.rss = 100 * 1024 * 1024
        mock_mem.vms = 200 * 1024 * 1024
        mock_inst.memory_info.return_value = mock_mem
        mock_inst.cpu_percent.return_value = 10.0
        mock_inst.num_threads.return_value = 2
        mock_inst.open_files.return_value = []

        mc = MetricsCollector()
        mc._prometheus_initialized = True
        mc._prom_memory_usage = MagicMock()
        mc._prom_cpu_percent = MagicMock()

        mc.sample_system_metrics()

        mc._prom_memory_usage.set.assert_called_once()
        mc._prom_cpu_percent.set.assert_called_once()


class TestGetMCTSSummary:
    """Tests for MCTS summary generation."""

    @patch("src.observability.metrics.psutil.Process")
    def test_empty_session(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        assert mc.get_mcts_summary("nonexistent") == {}

    @patch("src.observability.metrics.psutil.Process")
    def test_summary_with_data(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        mc.record_mcts_iteration("s1", ucb_score=0.5, selection_time_ms=10.0)
        mc.record_mcts_iteration("s1", ucb_score=0.8, expansion_time_ms=5.0)
        mc.update_mcts_tree_stats("s1", tree_depth=3, total_nodes=10, best_action_visits=5, best_action_value=0.7)

        summary = mc.get_mcts_summary("s1")
        assert summary["session_id"] == "s1"
        assert summary["total_iterations"] == 2
        assert summary["tree_depth"] == 3
        assert summary["ucb_scores"]["count"] == 2
        assert summary["ucb_scores"]["mean"] > 0


class TestGetAgentSummary:
    """Tests for agent summary generation."""

    @patch("src.observability.metrics.psutil.Process")
    def test_empty_agent(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        assert mc.get_agent_summary("nonexistent") == {}

    @patch("src.observability.metrics.psutil.Process")
    def test_summary_with_data(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mock_mem = MagicMock()
        mock_mem.rss = 50 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        mc.record_agent_execution("hrm", 100.0, 0.9, success=True)
        mc.record_agent_execution("hrm", 200.0, 0.7, success=False)

        summary = mc.get_agent_summary("hrm")
        assert summary["agent_name"] == "hrm"
        assert summary["total_executions"] == 2
        assert summary["success_count"] == 1
        assert summary["error_count"] == 1
        assert 0 < summary["success_rate"] < 1


class TestGetNodeTimingSummary:
    """Tests for node timing summary."""

    @patch("src.observability.metrics.psutil.Process")
    def test_node_timing_summary(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc.record_node_timing("n1", 10.0)
        mc.record_node_timing("n1", 20.0)
        mc.record_node_timing("n2", 5.0)

        summary = mc.get_node_timing_summary()
        assert "n1" in summary
        assert summary["n1"]["count"] == 2
        assert summary["n1"]["mean_ms"] == 15.0
        assert "n2" in summary


class TestGetFullReport:
    """Tests for full report generation."""

    @patch("src.observability.metrics.psutil.Process")
    def test_full_report(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mock_inst = mock_process.return_value
        mock_mem = MagicMock()
        mock_mem.rss = 100 * 1024 * 1024
        mock_mem.vms = 200 * 1024 * 1024
        mock_inst.memory_info.return_value = mock_mem
        mock_inst.cpu_percent.return_value = 5.0
        mock_inst.num_threads.return_value = 1
        mock_inst.open_files.return_value = []

        mc = MetricsCollector()
        mc._prometheus_initialized = False

        report = mc.get_full_report()
        assert "report_time" in report
        assert "uptime_seconds" in report
        assert "system_metrics" in report
        assert "mcts_sessions" in report
        assert "agents" in report
        assert "node_timings" in report
        assert "request_latencies" in report


class TestExportPrometheusFormat:
    """Tests for Prometheus export."""

    @patch("src.observability.metrics.psutil.Process")
    def test_export_with_prometheus_available(self, mock_process):
        from src.observability.metrics import MetricsCollector, PROMETHEUS_AVAILABLE

        mc = MetricsCollector()
        result = mc.export_prometheus_format()
        if PROMETHEUS_AVAILABLE:
            assert isinstance(result, str)
        else:
            assert "not available" in result

    @patch("src.observability.metrics.psutil.Process")
    @patch("src.observability.metrics.PROMETHEUS_AVAILABLE", False)
    def test_export_without_prometheus(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = False
        result = mc.export_prometheus_format()
        assert "not available" in result


class TestReset:
    """Tests for reset method."""

    @patch("src.observability.metrics.psutil.Process")
    def test_reset_clears_all(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = False
        mc.record_node_timing("n1", 10.0)
        mc.record_request_latency(50.0)

        mc.reset()

        assert len(mc._mcts_metrics) == 0
        assert len(mc._agent_metrics) == 0
        assert len(mc._node_timings) == 0
        assert len(mc._request_latencies) == 0
        assert len(mc._memory_samples) == 0


class TestStartPrometheusServer:
    """Tests for Prometheus server start."""

    @patch("src.observability.metrics.psutil.Process")
    def test_start_server(self, mock_process):
        from src.observability.metrics import MetricsCollector, PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        mc = MetricsCollector()
        with patch("src.observability.metrics.start_http_server") as mock_start:
            mc.start_prometheus_server(port=9090)
            mock_start.assert_called_once_with(9090)

    @patch("src.observability.metrics.psutil.Process")
    @patch("src.observability.metrics.PROMETHEUS_AVAILABLE", False)
    def test_start_server_no_prometheus(self, mock_process):
        from src.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc._prometheus_initialized = False
        # Should not raise
        mc.start_prometheus_server()


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("src.observability.metrics.psutil.Process")
    def test_mcts_metrics_singleton(self, mock_process):
        from src.observability.metrics import mcts_metrics

        a = mcts_metrics()
        b = mcts_metrics()
        assert a is b

    @patch("src.observability.metrics.psutil.Process")
    def test_agent_metrics_alias(self, mock_process):
        from src.observability.metrics import agent_metrics, mcts_metrics

        assert agent_metrics() is mcts_metrics()


class TestMetricsTimer:
    """Tests for MetricsTimer context manager."""

    @patch("src.observability.metrics.psutil.Process")
    def test_sync_context_manager(self, mock_process):
        from src.observability.metrics import MetricsCollector, MetricsTimer

        mc = MetricsCollector()
        timer = MetricsTimer(collector=mc, node_name="test_node")

        with timer:
            time.sleep(0.01)

        assert timer.elapsed_ms >= 10
        assert mc._node_timings["test_node"] == [timer.elapsed_ms]

    @patch("src.observability.metrics.psutil.Process")
    def test_sync_context_without_node_name(self, mock_process):
        from src.observability.metrics import MetricsCollector, MetricsTimer

        mc = MetricsCollector()
        timer = MetricsTimer(collector=mc)

        with timer:
            pass

        assert timer.elapsed_ms >= 0
        assert len(mc._node_timings) == 0

    @patch("src.observability.metrics.psutil.Process")
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_process):
        from src.observability.metrics import MetricsCollector, MetricsTimer

        mc = MetricsCollector()
        timer = MetricsTimer(collector=mc, node_name="async_node")

        async with timer:
            pass

        assert timer.elapsed_ms >= 0
        assert "async_node" in mc._node_timings

    @patch("src.observability.metrics.psutil.Process")
    @pytest.mark.asyncio
    async def test_async_context_without_node_name(self, mock_process):
        from src.observability.metrics import MetricsCollector, MetricsTimer

        mc = MetricsCollector()
        timer = MetricsTimer(collector=mc)

        async with timer:
            pass

        assert timer.elapsed_ms >= 0
        assert len(mc._node_timings) == 0

    @patch("src.observability.metrics.psutil.Process")
    def test_timer_default_collector(self, mock_process):
        """MetricsTimer should use singleton when no collector provided."""
        from src.observability.metrics import MetricsTimer

        timer = MetricsTimer(node_name="default")
        with timer:
            pass
        assert timer.elapsed_ms >= 0
