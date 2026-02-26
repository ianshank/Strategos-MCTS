"""
Comprehensive tests for observability profiling utilities.

Tests AsyncProfiler, MemoryProfiler, timing blocks, and report generation.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("psutil", reason="psutil required for observability tests")

from src.observability.profiling import (
    AsyncProfiler,
    MemoryProfiler,
    ProfilingSession,
    TimingResult,
    generate_performance_report,
    profile_block,
    profile_function,
)

# ============================================================================
# TimingResult Tests
# ============================================================================


class TestTimingResult:
    """Tests for TimingResult dataclass."""

    def test_timing_result_creation(self) -> None:
        """Test creating a timing result."""
        result = TimingResult(
            name="test_op",
            elapsed_ms=100.5,
            start_time=0.0,
            end_time=0.1,
            memory_start_mb=50.0,
            memory_end_mb=52.0,
            memory_delta_mb=2.0,
        )

        assert result.name == "test_op"
        assert result.elapsed_ms == 100.5
        assert result.memory_delta_mb == 2.0
        assert result.success is True
        assert result.error is None

    def test_timing_result_with_error(self) -> None:
        """Test timing result with error."""
        result = TimingResult(
            name="failed_op",
            elapsed_ms=50.0,
            start_time=0.0,
            end_time=0.05,
            memory_start_mb=50.0,
            memory_end_mb=50.0,
            memory_delta_mb=0.0,
            success=False,
            error="Test error",
        )

        assert result.success is False
        assert result.error == "Test error"

    def test_timing_result_with_metadata(self) -> None:
        """Test timing result with metadata."""
        result = TimingResult(
            name="op",
            elapsed_ms=10.0,
            start_time=0.0,
            end_time=0.01,
            memory_start_mb=50.0,
            memory_end_mb=50.0,
            memory_delta_mb=0.0,
            metadata={"batch_size": 100, "model": "test"},
        )

        assert result.metadata["batch_size"] == 100


class TestProfilingSession:
    """Tests for ProfilingSession dataclass."""

    def test_session_creation(self) -> None:
        """Test creating a profiling session."""
        session = ProfilingSession(session_id="test_session")

        assert session.session_id == "test_session"
        assert len(session.timings) == 0
        assert len(session.memory_samples) == 0
        assert len(session.cpu_samples) == 0
        assert len(session.markers) == 0


# ============================================================================
# AsyncProfiler Tests
# ============================================================================


class TestAsyncProfilerInit:
    """Tests for AsyncProfiler initialization."""

    def test_get_instance_singleton(self) -> None:
        """Test singleton pattern."""
        instance1 = AsyncProfiler.get_instance()
        instance2 = AsyncProfiler.get_instance()
        assert instance1 is instance2

    def test_new_instance(self) -> None:
        """Test creating a new instance directly."""
        profiler = AsyncProfiler()
        assert profiler._current_session is None
        assert len(profiler._sessions) == 0


class TestAsyncProfilerSessions:
    """Tests for AsyncProfiler session management."""

    @pytest.fixture
    def profiler(self) -> AsyncProfiler:
        """Create a fresh profiler instance."""
        profiler = AsyncProfiler()
        yield profiler
        profiler.reset()

    def test_start_session(self, profiler: AsyncProfiler) -> None:
        """Test starting a session."""
        session_id = profiler.start_session("my_session")

        assert session_id == "my_session"
        assert profiler._current_session == "my_session"
        assert "my_session" in profiler._sessions

    def test_start_session_auto_id(self, profiler: AsyncProfiler) -> None:
        """Test starting a session with auto-generated ID."""
        session_id = profiler.start_session()

        assert session_id.startswith("session_")
        assert profiler._current_session == session_id

    def test_end_session(self, profiler: AsyncProfiler) -> None:
        """Test ending a session."""
        profiler.start_session("test")
        session = profiler.end_session("test")

        assert session.session_id == "test"
        assert profiler._current_session is None

    def test_end_session_unknown(self, profiler: AsyncProfiler) -> None:
        """Test ending unknown session raises error."""
        with pytest.raises(ValueError, match="Unknown session"):
            profiler.end_session("nonexistent")

    def test_get_current_session(self, profiler: AsyncProfiler) -> None:
        """Test getting current session."""
        assert profiler.get_current_session() is None

        profiler.start_session("test")
        session = profiler.get_current_session()

        assert session is not None
        assert session.session_id == "test"


class TestAsyncProfilerTimeBlock:
    """Tests for time_block context manager."""

    @pytest.fixture
    def profiler(self) -> AsyncProfiler:
        """Create a fresh profiler with active session."""
        profiler = AsyncProfiler()
        profiler.start_session("test")
        yield profiler
        profiler.reset()

    def test_time_block_basic(self, profiler: AsyncProfiler) -> None:
        """Test basic time block."""
        with profiler.time_block("test_operation"):
            time.sleep(0.01)

        assert len(profiler._aggregate_timings["test_operation"]) == 1
        assert profiler._aggregate_timings["test_operation"][0] >= 10  # At least 10ms

    def test_time_block_records_in_session(self, profiler: AsyncProfiler) -> None:
        """Test time block records in session."""
        with profiler.time_block("op"):
            pass

        session = profiler.get_current_session()
        assert session is not None
        assert len(session.timings) == 1
        assert session.timings[0].name == "op"

    def test_time_block_with_metadata(self, profiler: AsyncProfiler) -> None:
        """Test time block with metadata."""
        with profiler.time_block("op", metadata={"key": "value"}):
            pass

        session = profiler.get_current_session()
        assert session.timings[0].metadata["key"] == "value"

    def test_time_block_captures_exception(self, profiler: AsyncProfiler) -> None:
        """Test time block captures exceptions."""
        with pytest.raises(ValueError):
            with profiler.time_block("failing_op"):
                raise ValueError("Test error")

        session = profiler.get_current_session()
        assert len(session.timings) == 1
        assert session.timings[0].success is False
        assert "Test error" in session.timings[0].error

    def test_time_block_no_session(self) -> None:
        """Test time block without session still aggregates."""
        profiler = AsyncProfiler()
        profiler.reset()

        with profiler.time_block("op"):
            pass

        assert len(profiler._aggregate_timings["op"]) == 1


class TestAsyncProfilerAsyncTimeBlock:
    """Tests for async_time_block context manager."""

    @pytest.fixture
    def profiler(self) -> AsyncProfiler:
        """Create a fresh profiler with active session."""
        profiler = AsyncProfiler()
        profiler.start_session("test")
        yield profiler
        profiler.reset()

    @pytest.mark.asyncio
    async def test_async_time_block_basic(self, profiler: AsyncProfiler) -> None:
        """Test basic async time block."""
        async with profiler.async_time_block("async_op"):
            await asyncio.sleep(0.01)

        assert len(profiler._aggregate_timings["async_op"]) == 1
        assert profiler._aggregate_timings["async_op"][0] >= 10

    @pytest.mark.asyncio
    async def test_async_time_block_captures_exception(self, profiler: AsyncProfiler) -> None:
        """Test async time block captures exceptions."""
        with pytest.raises(ValueError):
            async with profiler.async_time_block("failing"):
                raise ValueError("Async error")

        session = profiler.get_current_session()
        assert session.timings[0].success is False


class TestAsyncProfilerSampling:
    """Tests for memory and CPU sampling."""

    @pytest.fixture
    def profiler(self) -> AsyncProfiler:
        """Create a fresh profiler with active session."""
        profiler = AsyncProfiler()
        profiler.start_session("test")
        yield profiler
        profiler.reset()

    def test_sample_memory(self, profiler: AsyncProfiler) -> None:
        """Test memory sampling."""
        sample = profiler.sample_memory()

        assert "rss_mb" in sample
        assert "vms_mb" in sample
        assert "percent" in sample
        assert sample["rss_mb"] > 0

        session = profiler.get_current_session()
        assert len(session.memory_samples) == 1

    def test_sample_cpu(self, profiler: AsyncProfiler) -> None:
        """Test CPU sampling."""
        cpu = profiler.sample_cpu()

        assert isinstance(cpu, float)
        assert cpu >= 0

        session = profiler.get_current_session()
        assert len(session.cpu_samples) == 1


class TestAsyncProfilerMarkers:
    """Tests for profiling markers."""

    @pytest.fixture
    def profiler(self) -> AsyncProfiler:
        """Create a fresh profiler with active session."""
        profiler = AsyncProfiler()
        profiler.start_session("test")
        yield profiler
        profiler.reset()

    def test_add_marker(self, profiler: AsyncProfiler) -> None:
        """Test adding a marker."""
        profiler.add_marker("checkpoint_1", {"step": 100})

        session = profiler.get_current_session()
        assert len(session.markers) == 1
        assert session.markers[0]["name"] == "checkpoint_1"
        assert session.markers[0]["data"]["step"] == 100


class TestAsyncProfilerSummary:
    """Tests for timing summary."""

    @pytest.fixture
    def profiler(self) -> AsyncProfiler:
        """Create profiler with some timing data."""
        profiler = AsyncProfiler()
        profiler._aggregate_timings["op1"] = [10.0, 20.0, 30.0, 40.0, 50.0]
        profiler._aggregate_timings["op2"] = [5.0, 10.0]
        yield profiler
        profiler.reset()

    def test_get_timing_summary_all(self, profiler: AsyncProfiler) -> None:
        """Test getting all timing summaries."""
        summary = profiler.get_timing_summary()

        assert "op1" in summary
        assert "op2" in summary
        assert summary["op1"]["count"] == 5
        assert summary["op2"]["count"] == 2

    def test_get_timing_summary_specific(self, profiler: AsyncProfiler) -> None:
        """Test getting specific timing summary."""
        summary = profiler.get_timing_summary("op1")

        assert summary["name"] == "op1"
        assert summary["count"] == 5
        assert summary["mean_ms"] == 30.0
        assert summary["min_ms"] == 10.0
        assert summary["max_ms"] == 50.0

    def test_get_timing_summary_unknown(self, profiler: AsyncProfiler) -> None:
        """Test getting summary for unknown operation."""
        summary = profiler.get_timing_summary("nonexistent")
        assert summary == {}

    def test_compute_stats(self, profiler: AsyncProfiler) -> None:
        """Test computing statistics."""
        stats = profiler._compute_stats("test", [10.0, 20.0, 30.0, 40.0, 50.0])

        assert stats["count"] == 5
        assert stats["total_ms"] == 150.0
        assert stats["p50_ms"] == 30.0


class TestAsyncProfilerReset:
    """Tests for profiler reset."""

    def test_reset(self) -> None:
        """Test resetting the profiler."""
        profiler = AsyncProfiler()
        profiler.start_session("test")
        profiler._aggregate_timings["op"] = [10.0]

        profiler.reset()

        assert len(profiler._sessions) == 0
        assert profiler._current_session is None
        assert len(profiler._aggregate_timings) == 0


# ============================================================================
# MemoryProfiler Tests
# ============================================================================


class TestMemoryProfiler:
    """Tests for MemoryProfiler."""

    @pytest.fixture
    def memory_profiler(self) -> MemoryProfiler:
        """Create a fresh memory profiler."""
        return MemoryProfiler()

    def test_set_baseline(self, memory_profiler: MemoryProfiler) -> None:
        """Test setting memory baseline."""
        baseline = memory_profiler.set_baseline()

        assert baseline > 0
        assert memory_profiler._baseline == baseline

    def test_get_current(self, memory_profiler: MemoryProfiler) -> None:
        """Test getting current memory."""
        current = memory_profiler.get_current()
        assert current > 0

    def test_get_delta_no_baseline(self, memory_profiler: MemoryProfiler) -> None:
        """Test getting delta when no baseline set."""
        delta = memory_profiler.get_delta()
        assert delta == 0.0
        # Baseline should now be set
        assert memory_profiler._baseline is not None

    def test_get_delta_with_baseline(self, memory_profiler: MemoryProfiler) -> None:
        """Test getting delta with baseline."""
        memory_profiler.set_baseline()
        delta = memory_profiler.get_delta()
        # Delta should be small (close to 0)
        assert isinstance(delta, float)

    def test_sample(self, memory_profiler: MemoryProfiler) -> None:
        """Test taking a memory sample."""
        sample = memory_profiler.sample("test_label")

        assert sample["label"] == "test_label"
        assert sample["rss_mb"] > 0
        assert "vms_mb" in sample
        assert "percent" in sample
        assert len(memory_profiler._samples) == 1

    def test_sample_updates_peak(self, memory_profiler: MemoryProfiler) -> None:
        """Test sample updates peak memory."""
        sample = memory_profiler.sample()
        # Allow small floating point tolerance due to rounding
        assert memory_profiler._peak >= sample["rss_mb"] - 0.1

    def test_check_leak_no_baseline(self, memory_profiler: MemoryProfiler) -> None:
        """Test leak check without baseline."""
        result = memory_profiler.check_leak()
        assert result["status"] == "no_baseline"

    def test_check_leak_ok(self, memory_profiler: MemoryProfiler) -> None:
        """Test leak check when OK."""
        memory_profiler.set_baseline()
        result = memory_profiler.check_leak(threshold_mb=1000)  # High threshold
        assert result["status"] == "ok"

    def test_get_summary_no_samples(self, memory_profiler: MemoryProfiler) -> None:
        """Test summary with no samples."""
        summary = memory_profiler.get_summary()
        assert summary["message"] == "No samples collected"

    def test_get_summary_with_samples(self, memory_profiler: MemoryProfiler) -> None:
        """Test summary with samples."""
        memory_profiler.set_baseline()
        memory_profiler.sample()
        memory_profiler.sample()

        summary = memory_profiler.get_summary()

        assert summary["sample_count"] == 2
        assert summary["current_mb"] > 0
        assert summary["peak_mb"] > 0


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestProfileBlock:
    """Tests for profile_block context manager."""

    def test_profile_block(self) -> None:
        """Test profile_block convenience function."""
        # Reset singleton first
        AsyncProfiler._instance = None
        profiler = AsyncProfiler.get_instance()

        with profile_block("test_block"):
            time.sleep(0.01)

        assert len(profiler._aggregate_timings["test_block"]) == 1


class TestGeneratePerformanceReport:
    """Tests for generate_performance_report function."""

    def test_generate_report_no_session(self) -> None:
        """Test generating report without session."""
        # Reset singleton
        AsyncProfiler._instance = None
        profiler = AsyncProfiler.get_instance()
        profiler.reset()

        report = generate_performance_report()

        assert "report_time" in report
        assert "timing_summary" in report
        assert "current_system" in report
        assert report["current_system"]["memory_mb"] > 0

    def test_generate_report_with_session(self) -> None:
        """Test generating report with active session."""
        AsyncProfiler._instance = None
        profiler = AsyncProfiler.get_instance()
        profiler.start_session("test")

        with profiler.time_block("op1"):
            pass
        with profiler.time_block("op2"):
            pass

        profiler.sample_memory()
        profiler.sample_cpu()

        report = generate_performance_report()

        assert "session" in report
        assert report["session"]["timing_count"] == 2
        assert report["session"]["memory_samples"] == 1

        profiler.reset()


class TestProfileFunctionDecorator:
    """Tests for profile_function decorator."""

    def test_sync_function(self) -> None:
        """Test decorating sync function."""
        AsyncProfiler._instance = None
        profiler = AsyncProfiler.get_instance()
        profiler.reset()

        @profile_function(name="decorated_sync")
        def sync_func(x: int) -> int:
            return x * 2

        result = sync_func(5)

        assert result == 10
        assert len(profiler._aggregate_timings["decorated_sync"]) == 1

    @pytest.mark.asyncio
    async def test_async_function(self) -> None:
        """Test decorating async function."""
        AsyncProfiler._instance = None
        profiler = AsyncProfiler.get_instance()
        profiler.reset()

        @profile_function(name="decorated_async")
        async def async_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = await async_func(5)

        assert result == 10
        assert len(profiler._aggregate_timings["decorated_async"]) == 1

    def test_default_name(self) -> None:
        """Test decorator uses function name as default."""
        AsyncProfiler._instance = None
        profiler = AsyncProfiler.get_instance()
        profiler.reset()

        @profile_function()
        def my_function():
            pass

        my_function()

        # Name should include function name
        assert any("my_function" in name for name in profiler._aggregate_timings)
