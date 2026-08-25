"""
Load and performance tests for MCTS Framework.

Tests:
- Concurrent request handling
- Memory stability under sustained load
- MCTS scaling benchmarks
- Throughput and latency measurements
"""

import asyncio
from dataclasses import dataclass
import gc
import os
import statistics
import sys
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# The reference LangGraph framework under test lives in ``examples/``; add it to
# the path so the load suite can import it regardless of the working dir.
_EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

# Skip tests only if the reference framework genuinely cannot be imported (e.g.
# the optional ``langgraph`` extra is absent). The module guards its own
# optional deps, so this succeeds in a standard [dev] environment.
try:
    from langgraph_multi_agent_mcts import LangGraphMultiAgentFramework  # noqa: F401

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    memory_start_mb: float
    memory_end_mb: float
    memory_growth_mb: float


def calculate_percentile(data: list[float], percentile: float) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.performance
@pytest.mark.skipif(not AGENTS_AVAILABLE, reason="HRMAgent and TRMAgent not available")
class TestConcurrentRequestHandling:
    """Test system behavior under concurrent load."""

    @pytest.fixture
    def mock_framework(self):
        """Create a mock framework for load testing."""
        from langgraph_multi_agent_mcts import LangGraphMultiAgentFramework

        mock_adapter = AsyncMock()
        mock_adapter.generate = AsyncMock(return_value=Mock(text="Response", tokens_used=10))
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.error = Mock()

        with patch("langgraph_multi_agent_mcts.OpenAIEmbeddings"):
            framework = LangGraphMultiAgentFramework(
                model_adapter=mock_adapter,
                logger=mock_logger,
                mcts_iterations=10,
            )
            return framework

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_concurrent_requests_50(self, mock_framework):
        """Test handling 50 concurrent requests."""
        num_concurrent = 50
        queries = [f"Test query {i}" for i in range(num_concurrent)]

        latencies = []
        errors = []
        start_time = time.perf_counter()

        async def process_query(query: str, index: int):
            try:
                req_start = time.perf_counter()
                result = await mock_framework.process(query=query, use_rag=False, use_mcts=False)
                req_end = time.perf_counter()
                latencies.append((req_end - req_start) * 1000)  # ms
                return result
            except Exception as e:
                errors.append((index, str(e)))
                return None

        tasks = [process_query(q, i) for i, q in enumerate(queries)]
        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time
        successful = sum(1 for r in results if r is not None)

        # Performance requirements
        assert successful >= num_concurrent * 0.95, f"Success rate {successful}/{num_concurrent} < 95%"

        # Calculate metrics
        if latencies:
            mean_latency = statistics.mean(latencies)
            p95_latency = calculate_percentile(latencies, 95)
            throughput = num_concurrent / total_time

            print(f"\n--- Concurrent Requests Test (n={num_concurrent}) ---")
            print(f"Total time: {total_time:.2f}s")
            print(f"Success rate: {successful}/{num_concurrent} ({100 * successful / num_concurrent:.1f}%)")
            print(f"Mean latency: {mean_latency:.2f}ms")
            print(f"P95 latency: {p95_latency:.2f}ms")
            print(f"Throughput: {throughput:.2f} req/s")

            # SLA checks
            assert p95_latency < 15000, f"P95 latency {p95_latency}ms > 15s SLA"
            assert throughput > 0.5, f"Throughput {throughput} < 0.5 req/s"

    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_sustained_load(self, mock_framework):
        """Test system under sustained load for 30 seconds."""
        duration_seconds = 30
        requests_per_second = 5

        latencies = []
        errors = []
        start_time = time.perf_counter()

        async def generate_load():
            nonlocal latencies, errors
            request_id = 0

            while time.perf_counter() - start_time < duration_seconds:
                req_start = time.perf_counter()
                try:
                    await mock_framework.process(
                        query=f"Sustained load query {request_id}", use_rag=False, use_mcts=False
                    )
                    latencies.append((time.perf_counter() - req_start) * 1000)
                except Exception as e:
                    errors.append(str(e))

                request_id += 1
                # Control request rate
                await asyncio.sleep(1.0 / requests_per_second)

        await generate_load()

        total_requests = len(latencies) + len(errors)
        success_rate = len(latencies) / total_requests if total_requests > 0 else 0

        print(f"\n--- Sustained Load Test ({duration_seconds}s @ {requests_per_second} RPS) ---")
        print(f"Total requests: {total_requests}")
        print(f"Success rate: {success_rate:.2%}")
        if latencies:
            print(f"Mean latency: {statistics.mean(latencies):.2f}ms")
            print(f"P95 latency: {calculate_percentile(latencies, 95):.2f}ms")

        assert success_rate >= 0.99, f"Success rate {success_rate:.2%} < 99%"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_burst_handling(self, mock_framework):
        """Test system handles burst traffic."""
        # Normal load
        burst_rps = 20
        burst_duration = 5

        latencies = []

        async def process_request():
            start = time.perf_counter()
            await mock_framework.process(query="Burst test", use_rag=False, use_mcts=False)
            latencies.append((time.perf_counter() - start) * 1000)

        # Burst phase
        burst_tasks = []
        for _ in range(burst_rps * burst_duration):
            burst_tasks.append(process_request())

        await asyncio.gather(*burst_tasks, return_exceptions=True)

        success_rate = len(latencies) / (burst_rps * burst_duration)

        print(f"\n--- Burst Test ({burst_rps} RPS for {burst_duration}s) ---")
        print(f"Success rate: {success_rate:.2%}")
        if latencies:
            print(f"P95 latency: {calculate_percentile(latencies, 95):.2f}ms")

        # Should handle burst with > 90% success
        assert success_rate >= 0.90


@pytest.mark.performance
@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
@pytest.mark.skipif(not AGENTS_AVAILABLE, reason="HRMAgent and TRMAgent not available")
class TestMemoryStability:
    """Test memory stability under load."""

    @pytest.fixture
    def mock_framework(self):
        """Create framework instance."""
        from langgraph_multi_agent_mcts import LangGraphMultiAgentFramework

        mock_adapter = AsyncMock()
        mock_logger = Mock()

        with patch("langgraph_multi_agent_mcts.OpenAIEmbeddings"):
            return LangGraphMultiAgentFramework(
                model_adapter=mock_adapter,
                logger=mock_logger,
            )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_does_not_leak(self, mock_framework):
        """Test memory stability over many iterations."""
        process = psutil.Process()

        # Baseline
        gc.collect()
        baseline_mb = process.memory_info().rss / (1024 * 1024)

        # Run many iterations
        num_iterations = 100
        for i in range(num_iterations):
            # Simulate framework usage
            from langgraph_multi_agent_mcts import MCTSNode, MCTSState

            root = MCTSNode(state=MCTSState(f"root_{i}"))
            for j in range(10):
                child = root.add_child(f"action_{j}", MCTSState(f"state_{j}"))
                child.visits = j + 1
                child.value_sum = j * (j + 1) * 0.5
            del root

            if i % 20 == 0:
                gc.collect()

        gc.collect()
        final_mb = process.memory_info().rss / (1024 * 1024)
        growth_mb = final_mb - baseline_mb

        print(f"\n--- Memory Leak Test ({num_iterations} iterations) ---")
        print(f"Baseline: {baseline_mb:.2f}MB")
        print(f"Final: {final_mb:.2f}MB")
        print(f"Growth: {growth_mb:.2f}MB")

        # Memory growth should be minimal
        assert growth_mb < 50, f"Memory grew {growth_mb:.2f}MB over {num_iterations} iterations"

    @pytest.mark.asyncio
    async def test_large_tree_memory_usage(self):
        """Test memory usage with large MCTS trees."""
        from langgraph_multi_agent_mcts import MCTSNode, MCTSState

        process = psutil.Process()
        gc.collect()
        baseline_mb = process.memory_info().rss / (1024 * 1024)

        # Create large tree
        root = MCTSNode(state=MCTSState("root"))
        nodes = [root]

        # Build tree with 10,000 nodes
        for i in range(10000):
            parent = nodes[i % len(nodes)]
            child = parent.add_child(f"action_{i}", MCTSState(f"state_{i}"))
            child.visits = i % 100
            child.value_sum = (i % 100) * (i % 100) * 0.01
            nodes.append(child)

        gc.collect()
        peak_mb = process.memory_info().rss / (1024 * 1024)
        tree_memory = peak_mb - baseline_mb

        print("\n--- Large Tree Memory Test (10,000 nodes) ---")
        print(f"Tree memory usage: {tree_memory:.2f}MB")

        # Clean up
        del nodes
        del root
        gc.collect()

        final_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Memory after cleanup: {final_mb - baseline_mb:.2f}MB")

        # Should not use excessive memory
        assert tree_memory < 100, f"Tree used {tree_memory:.2f}MB (expected < 100MB)"


@pytest.mark.performance
@pytest.mark.skipif(not AGENTS_AVAILABLE, reason="HRMAgent and TRMAgent not available")
class TestMCTSScaling:
    """Test MCTS performance scaling."""

    @pytest.fixture
    def mock_framework(self):
        """Create framework with mock dependencies."""
        from langgraph_multi_agent_mcts import LangGraphMultiAgentFramework

        mock_adapter = AsyncMock()
        mock_adapter.generate = AsyncMock(return_value=Mock(text="Response", tokens_used=10))
        mock_logger = Mock()
        mock_logger.info = Mock()

        with patch("langgraph_multi_agent_mcts.OpenAIEmbeddings"):
            return LangGraphMultiAgentFramework(
                model_adapter=mock_adapter,
                logger=mock_logger,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("iterations", [10, 50, 100, 500, 1000])
    async def test_mcts_iteration_scaling(self, mock_framework, iterations):
        """Test MCTS performance at different iteration counts."""
        mock_framework.mcts_iterations = iterations

        state = {
            "query": "Test query",
            "hrm_results": {"metadata": {"decomposition_quality_score": 0.7}},
            "trm_results": {"metadata": {"final_quality_score": 0.7}},
        }

        start = time.perf_counter()
        result = await mock_framework.mcts_simulator_node(state)
        elapsed = time.perf_counter() - start

        print(f"\n--- MCTS Scaling: {iterations} iterations ---")
        print(f"Time: {elapsed:.4f}s ({elapsed * 1000:.2f}ms)")
        print(f"Rate: {iterations / elapsed:.2f} iterations/s")
        print(f"Best action: {result['mcts_best_action']}")
        print(f"Root visits: {result['mcts_stats']['root_visits']}")

        # Performance bounds (should scale roughly linearly)
        max_time = iterations * 0.01  # 10ms per iteration max
        assert elapsed < max_time, f"{iterations} iterations took {elapsed:.2f}s (max: {max_time:.2f}s)"

    @pytest.mark.slow
    def test_ucb1_computation_scaling(self):
        """Test UCB1 computation scales with tree size."""
        from langgraph_multi_agent_mcts import MCTSNode, MCTSState

        tree_sizes = [10, 100, 1000, 10000]
        results = {}

        for size in tree_sizes:
            # Build tree
            root = MCTSNode(state=MCTSState("root"))
            root.visits = size * 10
            nodes = []

            for i in range(size):
                child = root.add_child(f"action_{i}", MCTSState(f"state_{i}"))
                child.visits = i + 1
                child.value_sum = (i + 1) * (i + 1) * 0.5
                nodes.append(child)

            # Benchmark UCB1-based child selection
            start = time.perf_counter()
            for _ in range(100):
                root.select_child()
            elapsed = time.perf_counter() - start

            results[size] = elapsed
            print(f"Tree size {size}: {elapsed * 1000:.2f}ms for 100 selections")

        # Scaling should be roughly linear (O(n) for max operation)
        # Allow for some overhead, but 100x tree shouldn't be 100x slower
        scaling_factor = results[10000] / results[100]
        assert scaling_factor < 150, f"Scaling factor {scaling_factor} too high"


@pytest.mark.performance
@pytest.mark.skipif(not AGENTS_AVAILABLE, reason="HRMAgent and TRMAgent not available")
class TestThroughputBenchmarks:
    """Benchmark overall system throughput."""

    @pytest.fixture
    def mock_framework(self):
        """Create lightweight mock framework."""
        from langgraph_multi_agent_mcts import LangGraphMultiAgentFramework

        mock_adapter = AsyncMock()
        mock_adapter.generate = AsyncMock(return_value=Mock(text="OK", tokens_used=5))
        mock_logger = Mock()

        with patch("langgraph_multi_agent_mcts.OpenAIEmbeddings"):
            return LangGraphMultiAgentFramework(model_adapter=mock_adapter, logger=mock_logger, mcts_iterations=10)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_maximum_throughput(self, mock_framework):
        """Determine maximum requests per second."""
        test_duration = 10  # seconds
        start_time = time.perf_counter()
        request_count = 0
        error_count = 0

        async def process_batch():
            nonlocal request_count, error_count
            while time.perf_counter() - start_time < test_duration:
                try:
                    await mock_framework.process(query="Throughput test", use_rag=False, use_mcts=False)
                    request_count += 1
                except Exception:
                    # Count errors but continue processing
                    error_count += 1

        # Run multiple concurrent workers
        num_workers = 10
        workers = [process_batch() for _ in range(num_workers)]
        await asyncio.gather(*workers)

        total_time = time.perf_counter() - start_time
        total_attempts = request_count + error_count
        throughput = total_attempts / total_time if total_time > 0 else 0

        print("\n--- Maximum Throughput Test ---")
        print(f"Workers: {num_workers}")
        print(f"Total requests: {request_count}")
        print(f"Total errors: {error_count}")
        print(f"Total attempts: {total_attempts}")
        print(f"Duration: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} req/s")

        # Should achieve reasonable throughput (lowered threshold for test environment)
        # Note: Real production systems may achieve higher throughput; this tests basic functionality
        assert throughput > 1, f"Throughput {throughput:.2f} req/s is too low"
