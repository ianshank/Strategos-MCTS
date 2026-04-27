"""
Performance Optimization Utilities.

Provides:
- Batch processing for efficient inference
- Memory optimization utilities
- MCTS benchmarking functions
- Performance regression detection
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

from src.observability.logging import get_logger

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchResult:
    """Result from batch processing."""

    results: list[Any]
    batch_count: int
    total_items: int
    elapsed_time_ms: float
    items_per_second: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_count": self.batch_count,
            "total_items": self.total_items,
            "elapsed_time_ms": self.elapsed_time_ms,
            "items_per_second": self.items_per_second,
        }


@dataclass
class BenchmarkResult:
    """Result from performance benchmark."""

    name: str
    iterations: int
    total_time_ms: float
    mean_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    iterations_per_second: float
    memory_delta_mb: float = 0.0
    passed_threshold: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "mean_time_ms": self.mean_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "std_time_ms": self.std_time_ms,
            "iterations_per_second": self.iterations_per_second,
            "memory_delta_mb": self.memory_delta_mb,
            "passed_threshold": self.passed_threshold,
        }


class BatchProcessor:
    """
    Efficient batch processing for inference tasks.

    Features:
    - Configurable batch sizes
    - Progress tracking
    - Error handling per batch
    - Memory-aware processing
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_concurrent: int = 4,
        memory_limit_mb: float | None = None,
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            max_concurrent: Maximum concurrent batches
            memory_limit_mb: Memory limit for processing (optional)
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.memory_limit_mb = memory_limit_mb

        self._total_processed = 0
        self._total_batches = 0
        self._total_time_ms = 0.0

        # Cache process for memory checks (avoid repeated creation)
        self._process = psutil.Process() if _HAS_PSUTIL else None

    def batch_iterator(self, items: list[T]) -> Iterator[tuple[int, list[T]]]:
        """
        Iterate over items in batches.

        Args:
            items: List of items to process

        Yields:
            Tuple of (batch_index, batch_items)
        """
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            yield i // self.batch_size, batch

    def process(
        self,
        items: list[T],
        process_fn: Callable[[list[T]], list[R]],
    ) -> BatchResult:
        """
        Process items in batches.

        Args:
            items: Items to process
            process_fn: Function to process a batch

        Returns:
            BatchResult with all results and timing info
        """
        start_time = time.perf_counter()
        all_results: list[R] = []
        batch_count = 0

        for batch_idx, batch in self.batch_iterator(items):
            # Check memory limit
            if self.memory_limit_mb is not None and self._process is not None:
                memory_mb = self._process.memory_info().rss / (1024 * 1024)
                if memory_mb > self.memory_limit_mb:
                    gc.collect()
                    logger.warning(
                        "Memory limit approached (%.1f MB), running GC",
                        memory_mb,
                    )

            # Process batch
            try:
                batch_results = process_fn(batch)
                all_results.extend(batch_results)
                batch_count += 1
            except Exception as e:
                logger.error("Error processing batch %d: %s", batch_idx, e)
                # Add None placeholders for failed items
                all_results.extend([None] * len(batch))  # type: ignore[list-item]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        items_per_second = len(items) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        self._total_processed += len(items)
        self._total_batches += batch_count
        self._total_time_ms += elapsed_ms

        return BatchResult(
            results=all_results,
            batch_count=batch_count,
            total_items=len(items),
            elapsed_time_ms=elapsed_ms,
            items_per_second=items_per_second,
        )

    async def process_async(
        self,
        items: list[T],
        process_fn: Callable[[list[T]], Any],
    ) -> BatchResult:
        """
        Process items in batches asynchronously.

        Args:
            items: Items to process
            process_fn: Async function to process a batch

        Returns:
            BatchResult with all results and timing info
        """

        start_time = time.perf_counter()
        all_results: list[Any] = []
        batch_count = 0

        for batch_idx, batch in self.batch_iterator(items):
            try:
                batch_results = await process_fn(batch)
                all_results.extend(batch_results)
                batch_count += 1
            except Exception as e:
                logger.error("Error processing batch %d: %s", batch_idx, e)
                all_results.extend([None] * len(batch))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        items_per_second = len(items) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        return BatchResult(
            results=all_results,
            batch_count=batch_count,
            total_items=len(items),
            elapsed_time_ms=elapsed_ms,
            items_per_second=items_per_second,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_processed": self._total_processed,
            "total_batches": self._total_batches,
            "total_time_ms": self._total_time_ms,
            "avg_batch_time_ms": (self._total_time_ms / self._total_batches if self._total_batches > 0 else 0.0),
            "avg_items_per_second": (
                self._total_processed / (self._total_time_ms / 1000) if self._total_time_ms > 0 else 0.0
            ),
        }


class MemoryOptimizer:
    """
    Memory optimization utilities.

    Features:
    - Memory usage tracking
    - Garbage collection management
    - Memory leak detection
    - Object pool management
    """

    def __init__(
        self,
        target_memory_mb: float | None = None,
        max_samples: int = 100,
    ):
        """
        Initialize memory optimizer.

        Args:
            target_memory_mb: Target memory usage (triggers GC if exceeded)
            max_samples: Maximum number of memory samples to keep
        """
        self.target_memory_mb = target_memory_mb
        self._gc_runs = 0
        self._memory_samples: list[float] = []
        self._max_samples = max_samples

        if _HAS_PSUTIL:
            self._process = psutil.Process()
        else:
            self._process = None

    def get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB."""
        if self._process is None:
            return 0.0
        return float(self._process.memory_info().rss / (1024 * 1024))

    def sample_memory(self) -> float:
        """Sample current memory and track it."""
        memory_mb = self.get_memory_usage_mb()
        if len(self._memory_samples) >= self._max_samples:
            self._memory_samples.pop(0)
        self._memory_samples.append(memory_mb)
        return memory_mb

    def check_and_collect(self) -> bool:
        """
        Check memory and run GC if needed.

        Returns:
            True if GC was run
        """
        if self.target_memory_mb is None:
            return False

        memory_mb = self.get_memory_usage_mb()
        if memory_mb > self.target_memory_mb:
            gc.collect()
            self._gc_runs += 1
            new_memory = self.get_memory_usage_mb()
            logger.debug(
                "GC run: %.1f MB -> %.1f MB (freed %.1f MB)",
                memory_mb,
                new_memory,
                memory_mb - new_memory,
            )
            return True
        return False

    def detect_memory_leak(self, threshold_mb_per_sample: float = 1.0) -> bool:
        """
        Detect potential memory leak.

        Args:
            threshold_mb_per_sample: MB growth per sample to consider a leak

        Returns:
            True if potential leak detected
        """
        if len(self._memory_samples) < 10:
            return False

        # Check if memory is consistently increasing
        samples = self._memory_samples[-10:]
        growth = samples[-1] - samples[0]
        avg_growth = growth / len(samples)

        if avg_growth > threshold_mb_per_sample:
            logger.warning(
                "Potential memory leak detected: %.2f MB growth per sample",
                avg_growth,
            )
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get memory optimizer statistics."""
        stats = {
            "gc_runs": self._gc_runs,
            "current_memory_mb": self.get_memory_usage_mb(),
            "target_memory_mb": self.target_memory_mb,
        }

        if self._memory_samples:
            stats["min_memory_mb"] = min(self._memory_samples)
            stats["max_memory_mb"] = max(self._memory_samples)
            stats["avg_memory_mb"] = sum(self._memory_samples) / len(self._memory_samples)

        return stats


def benchmark_iterations_per_second(
    fn: Callable[[], Any],
    iterations: int = 100,
    warmup_iterations: int = 10,
    name: str = "benchmark",
    target_ips: float | None = None,
) -> BenchmarkResult:
    """
    Benchmark a function to measure iterations per second.

    Args:
        fn: Function to benchmark (no arguments)
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations (not timed)
        name: Benchmark name
        target_ips: Target iterations per second (for pass/fail)

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup_iterations):
        fn()

    # Get initial memory
    memory_before = 0.0
    if _HAS_PSUTIL:
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)

    # Run benchmark
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    # Get final memory
    memory_after = 0.0
    if _HAS_PSUTIL:
        memory_after = process.memory_info().rss / (1024 * 1024)

    # Calculate statistics
    total_time = sum(times_ms)
    mean_time = total_time / iterations
    min_time = min(times_ms)
    max_time = max(times_ms)

    # Standard deviation
    variance = sum((t - mean_time) ** 2 for t in times_ms) / iterations
    std_time = variance**0.5

    iterations_per_second = iterations / (total_time / 1000) if total_time > 0 else 0

    # Check against target
    passed = True
    if target_ips is not None:
        passed = iterations_per_second >= target_ips

    result = BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_time,
        mean_time_ms=mean_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        std_time_ms=std_time,
        iterations_per_second=iterations_per_second,
        memory_delta_mb=memory_after - memory_before,
        passed_threshold=passed,
    )

    logger.info(
        "Benchmark '%s': %.1f iter/sec (mean=%.2fms, std=%.2fms) %s",
        name,
        iterations_per_second,
        mean_time,
        std_time,
        "PASSED" if passed else f"FAILED (target: {target_ips})",
    )

    return result


async def benchmark_async_iterations_per_second(
    fn: Callable[[], Any],
    iterations: int = 100,
    warmup_iterations: int = 10,
    name: str = "async_benchmark",
    target_ips: float | None = None,
) -> BenchmarkResult:
    """
    Benchmark an async function to measure iterations per second.

    Args:
        fn: Async function to benchmark
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations
        name: Benchmark name
        target_ips: Target iterations per second

    Returns:
        BenchmarkResult with timing statistics
    """
    import asyncio

    # Warmup
    for _ in range(warmup_iterations):
        result = fn()
        if asyncio.iscoroutine(result):
            await result

    # Get initial memory
    memory_before = 0.0
    if _HAS_PSUTIL:
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)

    # Run benchmark
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        if asyncio.iscoroutine(result):
            await result
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    # Get final memory
    memory_after = 0.0
    if _HAS_PSUTIL:
        memory_after = process.memory_info().rss / (1024 * 1024)

    # Calculate statistics
    total_time = sum(times_ms)
    mean_time = total_time / iterations
    min_time = min(times_ms)
    max_time = max(times_ms)

    variance = sum((t - mean_time) ** 2 for t in times_ms) / iterations
    std_time = variance**0.5

    iterations_per_second = iterations / (total_time / 1000) if total_time > 0 else 0

    passed = True
    if target_ips is not None:
        passed = iterations_per_second >= target_ips

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_time,
        mean_time_ms=mean_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        std_time_ms=std_time,
        iterations_per_second=iterations_per_second,
        memory_delta_mb=memory_after - memory_before,
        passed_threshold=passed,
    )


@dataclass
class PerformanceThresholds:
    """Performance thresholds for regression testing."""

    mcts_iterations_per_second: float = 1000.0
    embedding_cache_hit_rate: float = 0.8
    llm_request_latency_p95_ms: float = 2000.0
    memory_usage_mb: float = 1000.0
    batch_processing_items_per_second: float = 100.0


def create_performance_regression_test(
    thresholds: PerformanceThresholds | None = None,
) -> Callable[[BenchmarkResult], bool]:
    """
    Create a performance regression test function.

    Args:
        thresholds: Performance thresholds

    Returns:
        Function that checks if benchmark passes thresholds
    """
    thresholds = thresholds or PerformanceThresholds()

    def check_regression(result: BenchmarkResult) -> bool:
        """Check if benchmark result passes regression thresholds."""
        if "mcts" in result.name.lower():
            return result.iterations_per_second >= thresholds.mcts_iterations_per_second
        elif "batch" in result.name.lower():
            return result.iterations_per_second >= thresholds.batch_processing_items_per_second
        else:
            # Default: just check if it passed its own threshold
            return result.passed_threshold

    return check_regression
