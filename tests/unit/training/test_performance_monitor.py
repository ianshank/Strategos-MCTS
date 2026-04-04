"""
Tests for performance_monitor module.

Tests PerformanceMetrics, PerformanceMonitor, and TimingContext.
"""

import time
from unittest.mock import patch

import pytest

from src.training.performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
    TimingContext,
)

# ---------------------------------------------------------------------------
# PerformanceMetrics tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_defaults(self):
        """Test all defaults are zero or sensible."""
        m = PerformanceMetrics()
        assert m.hrm_decomposition_time == 0.0
        assert m.mcts_exploration_time == 0.0
        assert m.trm_refinement_time == 0.0
        assert m.total_inference_time == 0.0
        assert m.network_forward_time == 0.0
        assert m.cpu_memory_used == 0.0
        assert m.gpu_memory_used == 0.0
        assert m.policy_loss == 0.0
        assert m.value_loss == 0.0
        assert m.total_loss == 0.0
        assert m.mcts_simulations == 0
        assert m.cache_hit_rate == 0.0
        assert m.avg_tree_depth == 0.0
        assert m.hrm_halt_step == 0
        assert m.trm_convergence_step == 0
        assert isinstance(m.timestamp, float)
        assert m.timestamp > 0

    def test_custom_values(self):
        """Test creating metrics with custom values."""
        m = PerformanceMetrics(
            total_inference_time=150.0,
            policy_loss=0.5,
            mcts_simulations=800,
        )
        assert m.total_inference_time == 150.0
        assert m.policy_loss == 0.5
        assert m.mcts_simulations == 800


# ---------------------------------------------------------------------------
# PerformanceMonitor tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a PerformanceMonitor with GPU monitoring disabled."""
        with patch("src.training.performance_monitor.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mon = PerformanceMonitor(
                window_size=50,
                enable_gpu_monitoring=False,
                alert_threshold_ms=500.0,
            )
        return mon

    def test_init(self, monitor):
        """Test monitor initializes correctly."""
        assert monitor.window_size == 50
        assert monitor.enable_gpu_monitoring is False
        assert monitor.alert_threshold_ms == 500.0
        assert monitor.total_inferences == 0
        assert monitor.slow_inference_count == 0

    def test_init_metric_queues(self, monitor):
        """Test all expected metric queues exist."""
        expected_keys = [
            "hrm_decomposition_time",
            "mcts_exploration_time",
            "trm_refinement_time",
            "total_inference_time",
            "network_forward_time",
            "cpu_memory_used",
            "gpu_memory_used",
            "policy_loss",
            "value_loss",
            "total_loss",
            "cache_hit_rate",
        ]
        for key in expected_keys:
            assert key in monitor._metric_queues

    def test_log_timing_known_stage(self, monitor):
        """Test logging timing for a known stage."""
        monitor.log_timing("hrm_decomposition", 42.0)
        assert len(monitor._metric_queues["hrm_decomposition_time"]) == 1
        assert monitor._metric_queues["hrm_decomposition_time"][0] == 42.0

    def test_log_timing_unknown_stage(self, monitor):
        """Test logging timing for unknown stage is silently ignored."""
        monitor.log_timing("unknown_stage", 10.0)
        # Should not raise; metric_name "unknown_stage_time" not in queues

    def test_log_memory(self, monitor):
        """Test logging memory usage."""
        monitor.log_memory()
        assert len(monitor._metric_queues["cpu_memory_used"]) == 1
        assert monitor._metric_queues["cpu_memory_used"][0] > 0

    def test_log_loss(self, monitor):
        """Test logging training losses."""
        monitor.log_loss(0.5, 0.3, 0.8)
        assert monitor._metric_queues["policy_loss"][0] == 0.5
        assert monitor._metric_queues["value_loss"][0] == 0.3
        assert monitor._metric_queues["total_loss"][0] == 0.8

    def test_log_mcts_stats(self, monitor):
        """Test logging MCTS statistics."""
        monitor.log_mcts_stats(cache_hit_rate=0.75, simulations=100)
        assert monitor._metric_queues["cache_hit_rate"][0] == 0.75

    def test_log_inference(self, monitor):
        """Test logging inference time."""
        monitor.log_inference(100.0)
        assert monitor.total_inferences == 1
        assert monitor._metric_queues["total_inference_time"][0] == 100.0
        assert monitor.slow_inference_count == 0

    def test_log_inference_slow(self, monitor):
        """Test slow inference is counted."""
        monitor.log_inference(600.0)  # > 500 threshold
        assert monitor.slow_inference_count == 1
        assert monitor.total_inferences == 1

    def test_log_inference_multiple(self, monitor):
        """Test multiple inferences are tracked correctly."""
        monitor.log_inference(100.0)
        monitor.log_inference(200.0)
        monitor.log_inference(600.0)
        assert monitor.total_inferences == 3
        assert monitor.slow_inference_count == 1

    def test_get_stats_single_metric(self, monitor):
        """Test getting stats for a single metric."""
        for v in [10.0, 20.0, 30.0]:
            monitor.log_timing("hrm_decomposition", v)

        stats = monitor.get_stats("hrm_decomposition_time")
        assert stats["mean"] == pytest.approx(20.0)
        assert stats["min"] == pytest.approx(10.0)
        assert stats["max"] == pytest.approx(30.0)
        assert stats["count"] == 3

    def test_get_stats_all_metrics(self, monitor):
        """Test getting stats for all metrics."""
        monitor.log_loss(0.5, 0.3, 0.8)
        monitor.log_inference(100.0)

        stats = monitor.get_stats()
        assert "system" in stats
        assert stats["system"]["total_inferences"] == 1

    def test_get_stats_empty_metric(self, monitor):
        """Test getting stats for empty metric returns empty dict."""
        stats = monitor.get_stats("hrm_decomposition_time")
        assert stats == {}

    def test_get_stats_unknown_metric(self, monitor):
        """Test getting stats for unknown metric returns empty dict."""
        stats = monitor.get_stats("nonexistent_metric")
        assert stats == {}

    def test_compute_metric_stats(self, monitor):
        """Test statistical computation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            monitor._metric_queues["policy_loss"].append(v)

        stats = monitor._compute_metric_stats("policy_loss")
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["median"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["count"] == 5
        assert "std" in stats
        assert "p95" in stats
        assert "p99" in stats

    def test_get_current_memory(self, monitor):
        """Test getting current memory snapshot."""
        memory = monitor.get_current_memory()
        assert "cpu_rss_gb" in memory
        assert "cpu_vms_gb" in memory
        assert "system_used_percent" in memory
        assert memory["cpu_rss_gb"] > 0

    def test_get_current_memory_no_gpu(self, monitor):
        """Test memory snapshot without GPU."""
        memory = monitor.get_current_memory()
        assert "gpu_allocated_gb" not in memory

    def test_alert_if_slow_no_alert(self, monitor, capsys):
        """Test no alert when times are under threshold."""
        monitor.log_inference(100.0)
        monitor.alert_if_slow()
        captured = capsys.readouterr()
        assert "Performance Alert" not in captured.out

    def test_alert_if_slow_triggers(self, monitor, capsys):
        """Test alert triggers when times exceed threshold."""
        for _ in range(10):
            monitor._metric_queues["total_inference_time"].append(600.0)
        monitor.alert_if_slow()
        captured = capsys.readouterr()
        assert "Performance Alert" in captured.out

    def test_print_summary(self, monitor, capsys):
        """Test print_summary produces output."""
        monitor.log_loss(0.5, 0.3, 0.8)
        monitor.log_inference(100.0)
        monitor.print_summary()
        captured = capsys.readouterr()
        assert "Performance Summary" in captured.out
        assert "Timing Statistics" in captured.out
        assert "System Statistics" in captured.out

    def test_export_to_dict(self, monitor):
        """Test exporting all stats to dict."""
        monitor.log_inference(100.0)
        result = monitor.export_to_dict()
        assert "stats" in result
        assert "memory" in result
        assert "window_size" in result
        assert result["window_size"] == 50

    def test_export_to_wandb(self, monitor):
        """Test exporting metrics for wandb."""
        monitor.log_loss(0.5, 0.3, 0.8)
        monitor.log_inference(100.0)

        wandb_metrics = monitor.export_to_wandb(step=1)
        assert isinstance(wandb_metrics, dict)
        # Should contain flattened metrics
        assert "system/total_inferences" in wandb_metrics
        # Should contain memory keys
        assert any(k.startswith("memory/") for k in wandb_metrics)

    def test_reset(self, monitor):
        """Test resetting all metrics."""
        monitor.log_loss(0.5, 0.3, 0.8)
        monitor.log_inference(100.0)
        monitor.log_inference(600.0)

        monitor.reset()
        assert monitor.total_inferences == 0
        assert monitor.slow_inference_count == 0
        assert len(monitor.metrics_history) == 0
        for queue in monitor._metric_queues.values():
            assert len(queue) == 0

    def test_window_size_respected(self):
        """Test rolling window size is respected."""
        with patch("src.training.performance_monitor.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mon = PerformanceMonitor(window_size=5, enable_gpu_monitoring=False)

        for i in range(10):
            mon.log_timing("hrm_decomposition", float(i))
        assert len(mon._metric_queues["hrm_decomposition_time"]) == 5

    def test_system_stats_no_inferences(self, monitor):
        """Test system stats when no inferences have been made."""
        stats = monitor.get_stats()
        assert stats["system"]["total_inferences"] == 0
        assert stats["system"]["slow_inference_count"] == 0
        assert stats["system"]["slow_inference_rate"] == 0.0


# ---------------------------------------------------------------------------
# TimingContext tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimingContext:
    """Tests for TimingContext context manager."""

    def test_timing_context_logs(self):
        """Test TimingContext logs elapsed time."""
        with patch("src.training.performance_monitor.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            monitor = PerformanceMonitor(enable_gpu_monitoring=False)

        with TimingContext(monitor, "hrm_decomposition"):
            time.sleep(0.01)  # Small sleep to ensure measurable time

        assert len(monitor._metric_queues["hrm_decomposition_time"]) == 1
        elapsed = monitor._metric_queues["hrm_decomposition_time"][0]
        assert elapsed > 0

    def test_timing_context_returns_self(self):
        """Test context manager returns self on enter."""
        with patch("src.training.performance_monitor.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            monitor = PerformanceMonitor(enable_gpu_monitoring=False)

        ctx = TimingContext(monitor, "hrm_decomposition")
        with ctx as c:
            assert c is ctx

    def test_timing_context_on_exception(self):
        """Test TimingContext still logs on exception."""
        with patch("src.training.performance_monitor.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            monitor = PerformanceMonitor(enable_gpu_monitoring=False)

        with pytest.raises(ValueError):
            with TimingContext(monitor, "hrm_decomposition"):
                raise ValueError("test error")

        # Should still log the timing
        assert len(monitor._metric_queues["hrm_decomposition_time"]) == 1
