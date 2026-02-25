"""
Performance profiling infrastructure for multi-agent MCTS framework.

Provides:
- Context manager for timing code blocks
- Memory profiling hooks
- Async-aware profiling
- Report generation
"""

import asyncio
import functools
import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import psutil

from .logging import get_logger


@dataclass
class TimingResult:
    """Result of a timed operation."""

    name: str
    elapsed_ms: float
    start_time: float
    end_time: float
    memory_start_mb: float
    memory_end_mb: float
    memory_delta_mb: float
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfilingSession:
    """Container for profiling results within a session."""

    session_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    timings: list[TimingResult] = field(default_factory=list)
    memory_samples: list[dict[str, float]] = field(default_factory=list)
    cpu_samples: list[float] = field(default_factory=list)
    markers: list[dict[str, Any]] = field(default_factory=list)


class AsyncProfiler:
    """
    Async-aware profiler for multi-agent MCTS framework.

    Tracks:
    - Execution times for async operations
    - Memory usage patterns
    - CPU utilization
    - Custom markers and events
    """

    _instance: Optional["AsyncProfiler"] = None

    def __init__(self):
        self.logger = get_logger("observability.profiling")
        self._sessions: dict[str, ProfilingSession] = {}
        self._current_session: str | None = None
        self._process = psutil.Process()
        self._aggregate_timings: dict[str, list[float]] = defaultdict(list)

    @classmethod
    def get_instance(cls) -> "AsyncProfiler":
        """Get singleton instance of AsyncProfiler."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new profiling session."""
        if session_id is None:
            session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

        self._sessions[session_id] = ProfilingSession(session_id=session_id)
        self._current_session = session_id
        self.logger.info(f"Started profiling session: {session_id}")
        return session_id

    def end_session(self, session_id: str | None = None) -> ProfilingSession:
        """End a profiling session and return results."""
        if session_id is None:
            session_id = self._current_session

        if session_id not in self._sessions:
            raise ValueError(f"Unknown session: {session_id}")

        session = self._sessions[session_id]
        self.logger.info(f"Ended profiling session: {session_id}")

        if self._current_session == session_id:
            self._current_session = None

        return session

    def get_current_session(self) -> ProfilingSession | None:
        """Get current profiling session."""
        if self._current_session and self._current_session in self._sessions:
            return self._sessions[self._current_session]
        return None

    @contextmanager
    def time_block(
        self,
        name: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for timing synchronous code blocks.

        Args:
            name: Name of the operation being timed
            session_id: Optional session ID (uses current if not specified)
            metadata: Additional metadata to record

        Example:
            with profiler.time_block("mcts.selection"):
                # perform selection
        """
        if session_id is None:
            session_id = self._current_session

        start_time = time.perf_counter()
        memory_start = self._process.memory_info().rss / (1024 * 1024)
        success = True
        error = None

        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            memory_end = self._process.memory_info().rss / (1024 * 1024)
            elapsed_ms = (end_time - start_time) * 1000

            result = TimingResult(
                name=name,
                elapsed_ms=elapsed_ms,
                start_time=start_time,
                end_time=end_time,
                memory_start_mb=memory_start,
                memory_end_mb=memory_end,
                memory_delta_mb=memory_end - memory_start,
                success=success,
                error=error,
                metadata=metadata or {},
            )

            # Record in session if available
            if session_id and session_id in self._sessions:
                self._sessions[session_id].timings.append(result)

            # Record in aggregates
            self._aggregate_timings[name].append(elapsed_ms)

            self.logger.debug(
                f"Timed block '{name}': {elapsed_ms:.2f}ms",
                extra={
                    "profiling": {
                        "name": name,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "memory_delta_mb": round(result.memory_delta_mb, 2),
                        "success": success,
                    }
                },
            )

    @asynccontextmanager
    async def async_time_block(
        self,
        name: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Async context manager for timing asynchronous code blocks.

        Args:
            name: Name of the operation being timed
            session_id: Optional session ID
            metadata: Additional metadata

        Example:
            async with profiler.async_time_block("llm.call"):
                await model.generate(...)
        """
        if session_id is None:
            session_id = self._current_session

        start_time = time.perf_counter()
        memory_start = self._process.memory_info().rss / (1024 * 1024)
        success = True
        error = None

        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            memory_end = self._process.memory_info().rss / (1024 * 1024)
            elapsed_ms = (end_time - start_time) * 1000

            result = TimingResult(
                name=name,
                elapsed_ms=elapsed_ms,
                start_time=start_time,
                end_time=end_time,
                memory_start_mb=memory_start,
                memory_end_mb=memory_end,
                memory_delta_mb=memory_end - memory_start,
                success=success,
                error=error,
                metadata=metadata or {},
            )

            if session_id and session_id in self._sessions:
                self._sessions[session_id].timings.append(result)

            self._aggregate_timings[name].append(elapsed_ms)

            self.logger.debug(
                f"Async timed block '{name}': {elapsed_ms:.2f}ms",
                extra={
                    "profiling": {
                        "name": name,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "memory_delta_mb": round(result.memory_delta_mb, 2),
                        "success": success,
                    }
                },
            )

    def sample_memory(self, session_id: str | None = None) -> dict[str, float]:
        """Sample current memory usage."""
        memory_info = self._process.memory_info()

        sample = {
            "timestamp": time.time(),
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": self._process.memory_percent(),
        }

        if session_id is None:
            session_id = self._current_session

        if session_id and session_id in self._sessions:
            self._sessions[session_id].memory_samples.append(sample)

        return sample

    def sample_cpu(self, session_id: str | None = None) -> float:
        """Sample current CPU usage."""
        cpu_percent = self._process.cpu_percent()

        if session_id is None:
            session_id = self._current_session

        if session_id and session_id in self._sessions:
            self._sessions[session_id].cpu_samples.append(cpu_percent)

        return float(cpu_percent)

    def add_marker(
        self,
        name: str,
        data: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Add a custom marker/event to the profiling session."""
        marker = {
            "timestamp": time.time(),
            "name": name,
            "data": data or {},
        }

        if session_id is None:
            session_id = self._current_session

        if session_id and session_id in self._sessions:
            self._sessions[session_id].markers.append(marker)

        self.logger.debug(f"Added profiling marker: {name}")

    def get_timing_summary(self, name: str | None = None) -> dict[str, Any]:
        """
        Get summary statistics for timed operations.

        Args:
            name: Optional specific operation name (all if None)

        Returns:
            Summary statistics
        """
        if name:
            timings = self._aggregate_timings.get(name, [])
            if not timings:
                return {}
            return self._compute_stats(name, timings)
        else:
            return {op_name: self._compute_stats(op_name, times) for op_name, times in self._aggregate_timings.items()}

    def _compute_stats(self, name: str, timings: list[float]) -> dict[str, Any]:
        """Compute statistics for a list of timings."""
        if not timings:
            return {}

        sorted_timings = sorted(timings)
        n = len(sorted_timings)

        return {
            "name": name,
            "count": n,
            "total_ms": round(sum(timings), 2),
            "mean_ms": round(sum(timings) / n, 2),
            "min_ms": round(min(timings), 2),
            "max_ms": round(max(timings), 2),
            "p50_ms": round(sorted_timings[n // 2], 2),
            "p90_ms": round(sorted_timings[int(n * 0.9)], 2),
            "p95_ms": round(sorted_timings[int(n * 0.95)], 2),
            "p99_ms": round(sorted_timings[min(int(n * 0.99), n - 1)], 2),
        }

    def reset(self) -> None:
        """Reset all profiling data."""
        self._sessions.clear()
        self._current_session = None
        self._aggregate_timings.clear()
        self.logger.info("Profiler reset")


class MemoryProfiler:
    """
    Memory-focused profiler for tracking memory usage patterns.
    """

    def __init__(self):
        self.logger = get_logger("observability.profiling.memory")
        self._process = psutil.Process()
        self._baseline: float | None = None
        self._peak: float = 0.0
        self._samples: list[dict[str, Any]] = []

    def set_baseline(self) -> float:
        """Set current memory as baseline."""
        self._baseline = float(self._process.memory_info().rss / (1024 * 1024))
        self.logger.info(f"Memory baseline set: {self._baseline:.2f} MB")
        return self._baseline

    def get_current(self) -> float:
        """Get current memory usage in MB."""
        return float(self._process.memory_info().rss / (1024 * 1024))

    def get_delta(self) -> float:
        """Get memory change from baseline."""
        if self._baseline is None:
            self.set_baseline()
            return 0.0

        current = self.get_current()
        return current - self._baseline

    def sample(self, label: str = "") -> dict[str, Any]:
        """Take a memory sample with optional label."""
        memory_info = self._process.memory_info()
        current_mb = memory_info.rss / (1024 * 1024)

        if current_mb > self._peak:
            self._peak = current_mb

        sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "label": label,
            "rss_mb": round(current_mb, 2),
            "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
            "percent": round(self._process.memory_percent(), 2),
            "delta_from_baseline_mb": round(self.get_delta(), 2) if self._baseline else 0.0,
            "peak_mb": round(self._peak, 2),
        }

        self._samples.append(sample)
        self.logger.debug(f"Memory sample [{label}]: {current_mb:.2f} MB")
        return sample

    def check_leak(self, threshold_mb: float = 10.0) -> dict[str, Any]:
        """
        Check for potential memory leak.

        Args:
            threshold_mb: Memory increase threshold to consider as leak

        Returns:
            Leak detection result
        """
        if self._baseline is None:
            return {"status": "no_baseline", "message": "Set baseline first"}

        current = self.get_current()
        delta = current - self._baseline

        if delta > threshold_mb:
            self.logger.warning(f"Potential memory leak detected: {delta:.2f} MB increase")
            return {
                "status": "potential_leak",
                "baseline_mb": round(self._baseline, 2),
                "current_mb": round(current, 2),
                "delta_mb": round(delta, 2),
                "threshold_mb": threshold_mb,
            }

        return {
            "status": "ok",
            "baseline_mb": round(self._baseline, 2),
            "current_mb": round(current, 2),
            "delta_mb": round(delta, 2),
            "threshold_mb": threshold_mb,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get memory profiling summary."""
        if not self._samples:
            return {"message": "No samples collected"}

        rss_values = [s["rss_mb"] for s in self._samples]

        return {
            "sample_count": len(self._samples),
            "baseline_mb": round(self._baseline, 2) if self._baseline else None,
            "current_mb": round(self.get_current(), 2),
            "peak_mb": round(self._peak, 2),
            "mean_mb": round(sum(rss_values) / len(rss_values), 2),
            "min_mb": round(min(rss_values), 2),
            "max_mb": round(max(rss_values), 2),
        }


@contextmanager
def profile_block(
    name: str,
    metadata: dict[str, Any] | None = None,
):
    """
    Convenience context manager for profiling a code block.

    Uses the global AsyncProfiler singleton.

    Example:
        with profile_block("data_processing", {"batch_size": 100}):
            process_data(batch)
    """
    profiler = AsyncProfiler.get_instance()
    with profiler.time_block(name, metadata=metadata):
        yield


def generate_performance_report(session_id: str | None = None) -> dict[str, Any]:
    """
    Generate a comprehensive performance report.

    Args:
        session_id: Optional specific session (uses current if not specified)

    Returns:
        Performance report with timing summaries, memory stats, etc.
    """
    profiler = AsyncProfiler.get_instance()

    report: dict[str, Any] = {
        "report_time": datetime.utcnow().isoformat(),
        "timing_summary": profiler.get_timing_summary(),
    }

    # Add session-specific data if available
    session = profiler._sessions.get(session_id) if session_id else profiler.get_current_session()

    if session:
        report["session"] = {
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "timing_count": len(session.timings),
            "memory_samples": len(session.memory_samples),
            "cpu_samples": len(session.cpu_samples),
            "markers_count": len(session.markers),
        }

        # Compute session-specific stats
        if session.timings:
            session_times: dict[str, list[float]] = {}
            for timing in session.timings:
                if timing.name not in session_times:
                    session_times[timing.name] = []
                session_times[timing.name].append(timing.elapsed_ms)

            report["session"]["timing_breakdown"] = {
                name: profiler._compute_stats(name, times) for name, times in session_times.items()
            }

        if session.memory_samples:
            rss_values = [s["rss_mb"] for s in session.memory_samples]
            report["session"]["memory_summary"] = {
                "sample_count": len(rss_values),
                "mean_mb": round(sum(rss_values) / len(rss_values), 2),
                "min_mb": round(min(rss_values), 2),
                "max_mb": round(max(rss_values), 2),
            }

        if session.cpu_samples:
            report["session"]["cpu_summary"] = {
                "sample_count": len(session.cpu_samples),
                "mean_percent": round(sum(session.cpu_samples) / len(session.cpu_samples), 2),
                "min_percent": round(min(session.cpu_samples), 2),
                "max_percent": round(max(session.cpu_samples), 2),
            }

    # Current system state
    process = psutil.Process()
    report["current_system"] = {
        "memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "cpu_percent": process.cpu_percent(),
        "thread_count": process.num_threads(),
    }

    return report


def profile_function(name: str | None = None, metadata: dict[str, Any] | None = None):
    """
    Decorator for profiling function execution.

    Args:
        name: Optional custom name (defaults to function name)
        metadata: Additional metadata

    Example:
        @profile_function()
        def process_batch(data):
            ...

        @profile_function(name="custom_name")
        async def async_operation():
            ...
    """

    def decorator(func):
        op_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            profiler = AsyncProfiler.get_instance()
            with profiler.time_block(op_name, metadata=metadata):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler = AsyncProfiler.get_instance()
            async with profiler.async_time_block(op_name, metadata=metadata):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
