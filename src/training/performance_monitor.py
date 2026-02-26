"""
Performance Monitoring System for LangGraph Multi-Agent MCTS.

Tracks and analyzes system performance including:
- Inference latency
- Memory usage
- Training metrics
- Cache efficiency
- Throughput statistics
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil
import torch


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Timing metrics (milliseconds)
    hrm_decomposition_time: float = 0.0
    mcts_exploration_time: float = 0.0
    trm_refinement_time: float = 0.0
    total_inference_time: float = 0.0
    network_forward_time: float = 0.0

    # Memory metrics (GB)
    cpu_memory_used: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_allocated: float = 0.0

    # Training metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    learning_rate: float = 0.0

    # MCTS metrics
    mcts_simulations: int = 0
    cache_hit_rate: float = 0.0
    avg_tree_depth: float = 0.0

    # Convergence metrics
    hrm_halt_step: int = 0
    trm_convergence_step: int = 0

    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """
    Track and analyze system performance metrics.

    Features:
    - Rolling window statistics
    - Automatic anomaly detection
    - Performance alerts
    - Export to various formats (dict, JSON, wandb)
    """

    def __init__(
        self,
        window_size: int = 100,
        enable_gpu_monitoring: bool = True,
        alert_threshold_ms: float = 1000.0,
    ):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of recent measurements to keep
            enable_gpu_monitoring: Whether to monitor GPU usage
            alert_threshold_ms: Threshold for slow inference alerts
        """
        self.window_size = window_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.alert_threshold_ms = alert_threshold_ms

        # Time series data
        self.metrics_history: deque = deque(maxlen=window_size)

        # Individual metric queues for faster access
        self._metric_queues: dict[str, deque] = {
            "hrm_decomposition_time": deque(maxlen=window_size),
            "mcts_exploration_time": deque(maxlen=window_size),
            "trm_refinement_time": deque(maxlen=window_size),
            "total_inference_time": deque(maxlen=window_size),
            "network_forward_time": deque(maxlen=window_size),
            "cpu_memory_used": deque(maxlen=window_size),
            "gpu_memory_used": deque(maxlen=window_size),
            "policy_loss": deque(maxlen=window_size),
            "value_loss": deque(maxlen=window_size),
            "total_loss": deque(maxlen=window_size),
            "cache_hit_rate": deque(maxlen=window_size),
        }

        # Counters
        self.total_inferences = 0
        self.slow_inference_count = 0

        # Process info
        self.process = psutil.Process()

    def log_timing(self, stage: str, elapsed_ms: float):
        """
        Log execution time for a processing stage.

        Args:
            stage: Stage name (e.g., "hrm_decomposition", "mcts_exploration")
            elapsed_ms: Elapsed time in milliseconds
        """
        metric_name = f"{stage}_time"
        if metric_name in self._metric_queues:
            self._metric_queues[metric_name].append(elapsed_ms)

    def log_memory(self):
        """Log current memory usage."""
        # CPU memory
        memory_info = self.process.memory_info()
        cpu_memory_gb = memory_info.rss / (1024**3)  # Bytes to GB
        self._metric_queues["cpu_memory_used"].append(cpu_memory_gb)

        # GPU memory
        if self.enable_gpu_monitoring:
            gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
            self._metric_queues["gpu_memory_used"].append(gpu_memory_gb)

    def log_loss(self, policy_loss: float, value_loss: float, total_loss: float):
        """
        Log training losses.

        Args:
            policy_loss: Policy head loss
            value_loss: Value head loss
            total_loss: Combined loss
        """
        self._metric_queues["policy_loss"].append(policy_loss)
        self._metric_queues["value_loss"].append(value_loss)
        self._metric_queues["total_loss"].append(total_loss)

    def log_mcts_stats(self, cache_hit_rate: float, simulations: int = 0):  # noqa: ARG002
        """
        Log MCTS statistics.

        Args:
            cache_hit_rate: Cache hit rate (0-1)
            simulations: Number of simulations performed
        """
        self._metric_queues["cache_hit_rate"].append(cache_hit_rate)

    def log_inference(self, total_time_ms: float):
        """
        Log complete inference.

        Args:
            total_time_ms: Total inference time in milliseconds
        """
        self.total_inferences += 1
        self._metric_queues["total_inference_time"].append(total_time_ms)

        # Check for slow inference
        if total_time_ms > self.alert_threshold_ms:
            self.slow_inference_count += 1

    def get_stats(self, metric_name: str | None = None) -> dict[str, Any]:
        """
        Get performance statistics.

        Args:
            metric_name: Specific metric to get stats for (None = all metrics)

        Returns:
            Dictionary of statistics
        """
        if metric_name:
            return self._compute_metric_stats(metric_name)

        # Compute stats for all metrics
        stats = {}
        for name, queue in self._metric_queues.items():
            if len(queue) > 0:
                stats[name] = self._compute_metric_stats(name)

        # Add system stats
        stats["system"] = {
            "total_inferences": self.total_inferences,
            "slow_inference_count": self.slow_inference_count,
            "slow_inference_rate": (
                self.slow_inference_count / self.total_inferences if self.total_inferences > 0 else 0.0
            ),
        }

        return stats

    def _compute_metric_stats(self, metric_name: str) -> dict[str, float]:
        """Compute statistics for a single metric."""
        if metric_name not in self._metric_queues:
            return {}

        values = list(self._metric_queues[metric_name])
        if not values:
            return {}

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "count": len(values),
        }

    def get_current_memory(self) -> dict[str, float]:
        """Get current memory usage snapshot."""
        memory = {}

        # CPU memory
        memory_info = self.process.memory_info()
        memory["cpu_rss_gb"] = memory_info.rss / (1024**3)
        memory["cpu_vms_gb"] = memory_info.vms / (1024**3)

        # System-wide CPU memory
        system_memory = psutil.virtual_memory()
        memory["system_used_percent"] = system_memory.percent

        # GPU memory
        if self.enable_gpu_monitoring:
            memory["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            memory["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            memory["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

        return memory

    def alert_if_slow(self):
        """Print alert if recent inferences are slow."""
        recent_times = list(self._metric_queues["total_inference_time"])[-10:]
        if recent_times and np.mean(recent_times) > self.alert_threshold_ms:
            print(
                f"⚠️  Performance Alert: Avg inference time {np.mean(recent_times):.1f}ms "
                f"(threshold: {self.alert_threshold_ms}ms)"
            )

    def print_summary(self):
        """Print formatted summary of performance statistics."""
        print("\n" + "=" * 80)
        print("Performance Summary")
        print("=" * 80)

        stats = self.get_stats()

        # Timing statistics
        print("\n[Timing Statistics (ms)]")
        timing_metrics = [
            "total_inference_time",
            "hrm_decomposition_time",
            "mcts_exploration_time",
            "trm_refinement_time",
            "network_forward_time",
        ]
        for metric in timing_metrics:
            if metric in stats:
                s = stats[metric]
                print(
                    f"  {metric:30s}: mean={s['mean']:6.1f}  "
                    f"std={s['std']:6.1f}  p95={s['p95']:6.1f}  max={s['max']:6.1f}"
                )

        # Memory statistics
        print("\n[Memory Statistics (GB)]")
        memory = self.get_current_memory()
        print(f"  CPU RSS:                {memory['cpu_rss_gb']:.2f} GB")
        print(f"  System Memory Used:     {memory['system_used_percent']:.1f}%")
        if self.enable_gpu_monitoring:
            print(f"  GPU Allocated:          {memory['gpu_allocated_gb']:.2f} GB")
            print(f"  GPU Reserved:           {memory['gpu_reserved_gb']:.2f} GB")

        # Loss statistics
        if "total_loss" in stats:
            print("\n[Training Loss]")
            for metric in ["policy_loss", "value_loss", "total_loss"]:
                if metric in stats:
                    s = stats[metric]
                    print(f"  {metric:20s}: mean={s['mean']:.4f}  std={s['std']:.4f}")

        # System statistics
        print("\n[System Statistics]")
        sys_stats = stats.get("system", {})
        print(f"  Total Inferences:       {sys_stats.get('total_inferences', 0)}")
        print(f"  Slow Inferences:        {sys_stats.get('slow_inference_count', 0)}")
        print(f"  Slow Inference Rate:    {sys_stats.get('slow_inference_rate', 0):.2%}")

        # Cache statistics
        if "cache_hit_rate" in stats:
            s = stats["cache_hit_rate"]
            print("\n[Cache Performance]")
            print(f"  Hit Rate:               {s['mean']:.2%}")

        print("=" * 80 + "\n")

    def export_to_dict(self) -> dict[str, Any]:
        """Export all statistics to dictionary."""
        return {
            "stats": self.get_stats(),
            "memory": self.get_current_memory(),
            "window_size": self.window_size,
        }

    def export_to_wandb(self, step: int) -> dict[str, float]:  # noqa: ARG002
        """
        Export metrics for Weights & Biases logging.

        Args:
            step: Training step/iteration

        Returns:
            Flattened metrics dictionary
        """
        stats = self.get_stats()
        wandb_metrics = {}

        # Flatten nested statistics
        for metric_name, metric_stats in stats.items():
            if metric_name == "system":
                for key, value in metric_stats.items():
                    wandb_metrics[f"system/{key}"] = value
            elif isinstance(metric_stats, dict):
                # Log mean and p95 for each metric
                wandb_metrics[f"{metric_name}/mean"] = metric_stats.get("mean", 0)
                wandb_metrics[f"{metric_name}/p95"] = metric_stats.get("p95", 0)

        # Add memory
        memory = self.get_current_memory()
        for key, value in memory.items():
            wandb_metrics[f"memory/{key}"] = value

        return wandb_metrics

    def reset(self):
        """Reset all metrics."""
        for queue in self._metric_queues.values():
            queue.clear()
        self.metrics_history.clear()
        self.total_inferences = 0
        self.slow_inference_count = 0


class TimingContext:
    """Context manager for timing code blocks."""

    def __init__(self, monitor: PerformanceMonitor, stage: str):
        self.monitor = monitor
        self.stage = stage
        self.start_time: float | None = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = (time.perf_counter() - self.start_time) * 1000  # ms
            self.monitor.log_timing(self.stage, elapsed)
