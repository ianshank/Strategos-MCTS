"""
Unit tests for optimization utilities.

Tests batch processing, memory optimization, and benchmarking.

Based on: NEXT_STEPS_PLAN.md Phase 5.1
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("psutil", reason="psutil required for performance optimization tests")

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
def batch_processor():
    """Create test batch processor."""
    from src.performance.optimization import BatchProcessor

    return BatchProcessor(batch_size=4, max_concurrent=2)


@pytest.fixture
def memory_optimizer():
    """Create test memory optimizer."""
    from src.performance.optimization import MemoryOptimizer

    return MemoryOptimizer(target_memory_mb=1000.0)


# =============================================================================
# BatchResult Tests
# =============================================================================


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_batch_result_creation(self):
        """Test batch result creation."""
        from src.performance.optimization import BatchResult

        result = BatchResult(
            results=[1, 2, 3],
            batch_count=1,
            total_items=3,
            elapsed_time_ms=100.0,
            items_per_second=30.0,
        )

        assert len(result.results) == 3
        assert result.batch_count == 1
        assert result.elapsed_time_ms == 100.0

    def test_batch_result_to_dict(self):
        """Test batch result to dictionary."""
        from src.performance.optimization import BatchResult

        result = BatchResult(
            results=[],
            batch_count=2,
            total_items=10,
            elapsed_time_ms=50.0,
            items_per_second=200.0,
        )

        data = result.to_dict()

        assert data["batch_count"] == 2
        assert data["total_items"] == 10
        assert data["items_per_second"] == 200.0


# =============================================================================
# BenchmarkResult Tests
# =============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test benchmark result creation."""
        from src.performance.optimization import BenchmarkResult

        result = BenchmarkResult(
            name="test_benchmark",
            iterations=100,
            total_time_ms=1000.0,
            mean_time_ms=10.0,
            min_time_ms=5.0,
            max_time_ms=20.0,
            std_time_ms=3.0,
            iterations_per_second=100.0,
        )

        assert result.name == "test_benchmark"
        assert result.iterations == 100
        assert result.iterations_per_second == 100.0
        assert result.passed_threshold is True

    def test_benchmark_result_to_dict(self):
        """Test benchmark result to dictionary."""
        from src.performance.optimization import BenchmarkResult

        result = BenchmarkResult(
            name="test",
            iterations=10,
            total_time_ms=100.0,
            mean_time_ms=10.0,
            min_time_ms=5.0,
            max_time_ms=15.0,
            std_time_ms=2.0,
            iterations_per_second=100.0,
            memory_delta_mb=5.0,
        )

        data = result.to_dict()

        assert data["name"] == "test"
        assert data["memory_delta_mb"] == 5.0


# =============================================================================
# BatchProcessor Tests
# =============================================================================


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_processor_creation(self, batch_processor):
        """Test processor can be created."""
        assert batch_processor is not None
        assert batch_processor.batch_size == 4

    def test_batch_iterator(self, batch_processor):
        """Test batch iterator yields correct batches."""
        items = list(range(10))

        batches = list(batch_processor.batch_iterator(items))

        assert len(batches) == 3  # 4 + 4 + 2
        assert batches[0] == (0, [0, 1, 2, 3])
        assert batches[1] == (1, [4, 5, 6, 7])
        assert batches[2] == (2, [8, 9])

    def test_batch_iterator_single_batch(self, batch_processor):
        """Test batch iterator with fewer items than batch size."""
        items = [1, 2]

        batches = list(batch_processor.batch_iterator(items))

        assert len(batches) == 1
        assert batches[0] == (0, [1, 2])

    def test_process_all_items(self, batch_processor):
        """Test process handles all items."""
        items = list(range(10))

        def process_fn(batch):
            return [x * 2 for x in batch]

        result = batch_processor.process(items, process_fn)

        assert len(result.results) == 10
        assert result.results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        assert result.batch_count == 3

    def test_process_timing(self, batch_processor):
        """Test process tracks timing."""
        items = list(range(4))

        def process_fn(batch):
            time.sleep(0.01)  # Small delay
            return batch

        result = batch_processor.process(items, process_fn)

        assert result.elapsed_time_ms >= 10  # At least 10ms
        assert result.items_per_second > 0

    def test_process_handles_errors(self, batch_processor):
        """Test process handles batch errors gracefully."""
        items = list(range(8))

        def process_fn(batch):
            if batch[0] == 4:  # Fail on second batch
                raise ValueError("Test error")
            return batch

        result = batch_processor.process(items, process_fn)

        # First batch succeeds, second fails, returns None placeholders
        assert len(result.results) == 8
        assert result.results[:4] == [0, 1, 2, 3]
        assert result.results[4:8] == [None, None, None, None]

    def test_process_stats_accumulated(self, batch_processor):
        """Test stats are accumulated across calls."""
        items1 = list(range(4))
        items2 = list(range(4))

        batch_processor.process(items1, lambda x: x)
        batch_processor.process(items2, lambda x: x)

        stats = batch_processor.get_stats()

        assert stats["total_processed"] == 8
        assert stats["total_batches"] == 2

    @pytest.mark.asyncio
    async def test_process_async(self, batch_processor):
        """Test async batch processing."""
        items = list(range(8))

        async def async_process_fn(batch):
            return [x + 1 for x in batch]

        result = await batch_processor.process_async(items, async_process_fn)

        assert len(result.results) == 8
        assert result.results == [1, 2, 3, 4, 5, 6, 7, 8]


# =============================================================================
# MemoryOptimizer Tests
# =============================================================================


class TestMemoryOptimizer:
    """Tests for MemoryOptimizer class."""

    def test_optimizer_creation(self, memory_optimizer):
        """Test optimizer can be created."""
        assert memory_optimizer is not None
        assert memory_optimizer.target_memory_mb == 1000.0

    @pytest.mark.skipif(
        not True,  # Skip if psutil not available
        reason="psutil required",
    )
    def test_get_memory_usage(self, memory_optimizer):
        """Test getting memory usage."""
        memory_mb = memory_optimizer.get_memory_usage_mb()

        # Should return positive value
        assert memory_mb >= 0

    def test_sample_memory(self, memory_optimizer):
        """Test sampling memory."""
        sample1 = memory_optimizer.sample_memory()
        sample2 = memory_optimizer.sample_memory()

        assert sample1 >= 0
        assert sample2 >= 0
        assert len(memory_optimizer._memory_samples) == 2

    def test_sample_memory_bounded(self, memory_optimizer):
        """Test memory samples are bounded."""
        memory_optimizer._max_samples = 5

        for _ in range(10):
            memory_optimizer.sample_memory()

        assert len(memory_optimizer._memory_samples) == 5

    def test_check_and_collect_not_triggered(self, memory_optimizer):
        """Test GC not triggered when under limit."""
        memory_optimizer.target_memory_mb = 100000.0  # Very high limit

        result = memory_optimizer.check_and_collect()

        assert result is False
        assert memory_optimizer._gc_runs == 0

    def test_detect_memory_leak_insufficient_samples(self, memory_optimizer):
        """Test leak detection with insufficient samples."""
        for _ in range(5):
            memory_optimizer.sample_memory()

        result = memory_optimizer.detect_memory_leak()

        assert result is False  # Not enough samples

    def test_get_stats(self, memory_optimizer):
        """Test getting optimizer stats."""
        memory_optimizer.sample_memory()

        stats = memory_optimizer.get_stats()

        assert "current_memory_mb" in stats
        assert "gc_runs" in stats
        assert stats["target_memory_mb"] == 1000.0


# =============================================================================
# Benchmark Function Tests
# =============================================================================


class TestBenchmarkIterationsPerSecond:
    """Tests for benchmark_iterations_per_second function."""

    def test_benchmark_basic(self):
        """Test basic benchmarking."""
        from src.performance.optimization import benchmark_iterations_per_second

        counter = [0]

        def simple_fn():
            counter[0] += 1

        result = benchmark_iterations_per_second(
            simple_fn,
            iterations=20,
            warmup_iterations=5,
            name="simple_test",
        )

        assert result.iterations == 20
        assert result.name == "simple_test"
        assert result.iterations_per_second > 0
        assert counter[0] == 25  # 20 + 5 warmup

    def test_benchmark_timing_accuracy(self):
        """Test benchmark timing is reasonably accurate."""
        from src.performance.optimization import benchmark_iterations_per_second

        def slow_fn():
            time.sleep(0.01)  # 10ms

        result = benchmark_iterations_per_second(
            slow_fn,
            iterations=5,
            warmup_iterations=1,
            name="slow_test",
        )

        # Each iteration ~10ms, so total should be ~50ms
        assert result.total_time_ms >= 40  # Allow some margin
        assert result.mean_time_ms >= 8

    def test_benchmark_target_pass(self):
        """Test benchmark passes target threshold."""
        from src.performance.optimization import benchmark_iterations_per_second

        def fast_fn():
            pass  # Very fast

        result = benchmark_iterations_per_second(
            fast_fn,
            iterations=100,
            warmup_iterations=10,
            name="fast_test",
            target_ips=100.0,  # Should easily pass
        )

        assert result.passed_threshold is True

    def test_benchmark_target_fail(self):
        """Test benchmark fails target threshold."""
        from src.performance.optimization import benchmark_iterations_per_second

        def slow_fn():
            time.sleep(0.1)  # 100ms

        result = benchmark_iterations_per_second(
            slow_fn,
            iterations=5,
            warmup_iterations=1,
            name="slow_fail_test",
            target_ips=100.0,  # Will fail
        )

        assert result.passed_threshold is False

    def test_benchmark_statistics(self):
        """Test benchmark computes statistics."""
        from src.performance.optimization import benchmark_iterations_per_second

        def fn():
            pass

        result = benchmark_iterations_per_second(
            fn,
            iterations=50,
            warmup_iterations=5,
        )

        assert result.min_time_ms <= result.mean_time_ms
        assert result.mean_time_ms <= result.max_time_ms
        assert result.std_time_ms >= 0


class TestAsyncBenchmark:
    """Tests for async benchmark function."""

    @pytest.mark.asyncio
    async def test_async_benchmark_basic(self):
        """Test async benchmarking."""
        from src.performance.optimization import benchmark_async_iterations_per_second

        counter = [0]

        async def async_fn():
            counter[0] += 1

        result = await benchmark_async_iterations_per_second(
            async_fn,
            iterations=10,
            warmup_iterations=2,
            name="async_test",
        )

        assert result.iterations == 10
        assert counter[0] == 12  # 10 + 2 warmup

    @pytest.mark.asyncio
    async def test_async_benchmark_with_sync_fn(self):
        """Test async benchmark with sync function."""
        from src.performance.optimization import benchmark_async_iterations_per_second

        def sync_fn():
            return 42

        result = await benchmark_async_iterations_per_second(
            sync_fn,
            iterations=10,
            warmup_iterations=2,
        )

        assert result.iterations == 10


# =============================================================================
# PerformanceThresholds Tests
# =============================================================================


class TestPerformanceThresholds:
    """Tests for PerformanceThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        from src.performance.optimization import PerformanceThresholds

        thresholds = PerformanceThresholds()

        assert thresholds.mcts_iterations_per_second == 1000.0
        assert thresholds.embedding_cache_hit_rate == 0.8
        assert thresholds.memory_usage_mb == 1000.0

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        from src.performance.optimization import PerformanceThresholds

        thresholds = PerformanceThresholds(
            mcts_iterations_per_second=500.0,
        )

        assert thresholds.mcts_iterations_per_second == 500.0


# =============================================================================
# Regression Test Factory Tests
# =============================================================================


class TestCreatePerformanceRegressionTest:
    """Tests for create_performance_regression_test factory."""

    def test_mcts_regression_check(self):
        """Test MCTS regression check."""
        from src.performance.optimization import (
            BenchmarkResult,
            PerformanceThresholds,
            create_performance_regression_test,
        )

        thresholds = PerformanceThresholds(mcts_iterations_per_second=100.0)
        check_fn = create_performance_regression_test(thresholds)

        passing_result = BenchmarkResult(
            name="mcts_search",
            iterations=100,
            total_time_ms=500.0,
            mean_time_ms=5.0,
            min_time_ms=4.0,
            max_time_ms=6.0,
            std_time_ms=0.5,
            iterations_per_second=200.0,  # Above threshold
        )

        failing_result = BenchmarkResult(
            name="mcts_search",
            iterations=100,
            total_time_ms=2000.0,
            mean_time_ms=20.0,
            min_time_ms=15.0,
            max_time_ms=25.0,
            std_time_ms=2.0,
            iterations_per_second=50.0,  # Below threshold
        )

        assert check_fn(passing_result) is True
        assert check_fn(failing_result) is False

    def test_batch_regression_check(self):
        """Test batch processing regression check."""
        from src.performance.optimization import (
            BenchmarkResult,
            PerformanceThresholds,
            create_performance_regression_test,
        )

        thresholds = PerformanceThresholds(batch_processing_items_per_second=50.0)
        check_fn = create_performance_regression_test(thresholds)

        result = BenchmarkResult(
            name="batch_inference",
            iterations=100,
            total_time_ms=1000.0,
            mean_time_ms=10.0,
            min_time_ms=5.0,
            max_time_ms=15.0,
            std_time_ms=2.0,
            iterations_per_second=100.0,  # Above threshold
        )

        assert check_fn(result) is True
