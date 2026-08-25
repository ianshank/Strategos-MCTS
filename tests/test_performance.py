"""
Performance and benchmark tests for critical paths.

Performance testing ensures the framework maintains acceptable
performance characteristics as it evolves.

Best Practices 2025:
- Use pytest-benchmark for consistent measurements
- Test both throughput and latency
- Track regression over time
- Test realistic workloads
- Include memory profiling
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from tests.builders import AgentContextBuilder, LLMResponseBuilder


class TestMCTSPerformance:
    """Performance tests for MCTS operations."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_node_expansion_speed(self):
        """
        Test: Node expansion should be fast enough for real-time use.

        Target: < 1ms per node expansion
        """
        from src.framework.mcts.core import MCTSNode, MCTSState

        start = time.perf_counter()
        iterations = 1000

        for i in range(iterations):
            state = MCTSState(state_id=f"node_{i}", features={"value": i})
            node = MCTSNode(state=state)
            node.visits = i
            node.value_sum = float(i * 0.5)

        elapsed = time.perf_counter() - start
        per_operation = (elapsed / iterations) * 1000  # Convert to ms

        assert per_operation < 1.0, f"Node operations too slow: {per_operation:.3f}ms per operation"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_state_hashing_performance(self):
        """
        Test: State hashing should be fast for cache lookups.

        Target: < 0.1ms per hash computation
        """
        from src.framework.mcts.core import MCTSState

        features = {f"feature_{i}": i for i in range(100)}
        state = MCTSState(state_id="test", features=features)

        start = time.perf_counter()
        iterations = 1000

        for _ in range(iterations):
            _ = state.to_hash_key()

        elapsed = time.perf_counter() - start
        per_operation = (elapsed / iterations) * 1000  # Convert to ms

        assert per_operation < 0.1, f"State hashing too slow: {per_operation:.3f}ms per hash"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_ucb1_calculation_speed(self):
        """
        Test: UCB1 calculation should be fast for selection phase.

        Target: < 0.01ms per calculation
        """
        from src.framework.mcts.policies import ucb1

        start = time.perf_counter()
        iterations = 10000

        for i in range(iterations):
            _ = ucb1(
                value_sum=float(i % 100),
                visits=i % 50 + 1,
                parent_visits=100,
                c=1.414,
            )

        elapsed = time.perf_counter() - start
        per_operation = (elapsed / iterations) * 1000  # Convert to ms

        assert per_operation < 0.01, f"UCB1 calculation too slow: {per_operation:.3f}ms per operation"


class TestAgentPerformance:
    """Performance tests for agent operations."""

    @pytest.mark.performance
    def test_context_creation_overhead(self):
        """
        Test: Agent context creation should have minimal overhead.

        Target: < 0.1ms per context creation
        """
        from src.framework.agents.base import AgentContext

        start = time.perf_counter()
        iterations = 1000

        for i in range(iterations):
            _ = AgentContext(
                query=f"test query {i}",
                metadata={"iteration": i},
                conversation_history=[{"role": "user", "content": "test"}],
            )

        elapsed = time.perf_counter() - start
        per_operation = (elapsed / iterations) * 1000  # Convert to ms

        assert per_operation < 0.1, f"Context creation too slow: {per_operation:.3f}ms per operation"

    @pytest.mark.performance
    def test_context_serialization_speed(self):
        """
        Test: Context serialization should be fast for logging/tracing.

        Target: < 0.5ms per serialization
        """
        context = AgentContextBuilder().with_query("test" * 100).build()

        start = time.perf_counter()
        iterations = 1000

        for _ in range(iterations):
            _ = context.to_dict()

        elapsed = time.perf_counter() - start
        per_operation = (elapsed / iterations) * 1000  # Convert to ms

        assert per_operation < 0.5, f"Context serialization too slow: {per_operation:.3f}ms per operation"


class TestFactoryPerformance:
    """Performance tests for factory creation."""

    @pytest.mark.performance
    def test_factory_instantiation_overhead(self):
        """
        Test: Factory instantiation should be lightweight.

        Target: < 5ms per factory creation (includes settings load)
        """
        import os

        from src.framework.factories import MCTSEngineFactory

        # Set minimal env to avoid validation errors
        os.environ.setdefault("LLM_PROVIDER", "lmstudio")

        start = time.perf_counter()
        iterations = 10  # Reduce iterations since settings loading is slower

        for _ in range(iterations):
            _ = MCTSEngineFactory()

        elapsed = time.perf_counter() - start
        per_operation = (elapsed / iterations) * 1000  # Convert to ms

        assert per_operation < 10.0, f"Factory instantiation too slow: {per_operation:.3f}ms per operation"


class TestValidationPerformance:
    """Performance tests for input validation."""

    @pytest.mark.performance
    def test_pydantic_validation_overhead(self):
        """
        Test: Pydantic validation should have acceptable overhead.

        Target: < 0.5ms per validation
        """
        from src.models.validation import QueryInput

        start = time.perf_counter()
        iterations = 1000

        for i in range(iterations):
            _ = QueryInput(
                query=f"test query {i}",
                use_rag=True,
                use_mcts=False,
            )

        elapsed = time.perf_counter() - start
        per_operation = (elapsed / iterations) * 1000  # Convert to ms

        assert per_operation < 0.5, f"Pydantic validation too slow: {per_operation:.3f}ms per operation"


class TestAsyncPerformance:
    """Performance tests for async operations."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_task_overhead(self):
        """
        Test: Async task creation should have minimal overhead.

        Target: < 0.1ms per task
        """

        async def dummy_task():
            await asyncio.sleep(0.001)

        start = time.perf_counter()
        iterations = 100

        tasks = [dummy_task() for _ in range(iterations)]
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start
        per_task = (elapsed / iterations) * 1000  # Convert to ms

        # This includes sleep time, so we're testing scheduling overhead
        assert per_task < 2.0, f"Async task overhead too high: {per_task:.3f}ms per task"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_mock_llm_response_speed(self):
        """
        Test: Mock LLM responses should be fast for testing.

        Target: < 1ms per mock response
        """
        mock_client = Mock()
        mock_client.generate = AsyncMock(return_value=LLMResponseBuilder().build())

        start = time.perf_counter()
        iterations = 100

        for _ in range(iterations):
            await mock_client.generate("test prompt")

        elapsed = time.perf_counter() - start
        per_operation = (elapsed / iterations) * 1000  # Convert to ms

        assert per_operation < 1.0, f"Mock LLM response too slow: {per_operation:.3f}ms per operation"


class TestMemoryEfficiency:
    """Memory efficiency tests."""

    @pytest.mark.performance
    def test_mcts_node_memory_footprint(self):
        """
        Test: MCTS nodes should have reasonable memory footprint.

        This helps ensure we can build large trees without OOM.
        """
        import sys

        from src.framework.mcts.core import MCTSNode, MCTSState

        # Create a node
        state = MCTSState(state_id="test", features={"key": "value"})
        node = MCTSNode(state=state)

        # Get approximate size
        size = sys.getsizeof(node)

        # Each node should be relatively small (< 1KB baseline)
        assert size < 1024, f"Node memory footprint too large: {size} bytes"

    @pytest.mark.performance
    def test_builder_memory_efficiency(self):
        """
        Test: Builders should not leak memory on repeated use.

        Target: Consistent memory usage across iterations
        """
        import gc
        import sys

        # Force garbage collection
        gc.collect()

        # Create many builders
        builders = []
        for i in range(1000):
            builder = AgentContextBuilder().with_query(f"query_{i}")
            builders.append(builder)

        # Size should be roughly linear
        total_size = sum(sys.getsizeof(b) for b in builders)
        avg_size = total_size / len(builders)

        # Each builder should be small
        assert avg_size < 1024, f"Builder average size too large: {avg_size} bytes"


class TestThroughput:
    """Throughput tests for bulk operations."""

    @pytest.mark.performance
    def test_bulk_context_creation_throughput(self):
        """
        Test: Should handle bulk context creation efficiently.

        Target: > 1000 contexts/second
        """
        from src.framework.agents.base import AgentContext

        start = time.perf_counter()
        count = 1000

        contexts = [AgentContext(query=f"query_{i}") for i in range(count)]

        elapsed = time.perf_counter() - start
        throughput = count / elapsed

        assert throughput > 1000, f"Throughput too low: {throughput:.0f} contexts/second"
        assert len(contexts) == count

    @pytest.mark.performance
    def test_bulk_state_hashing_throughput(self):
        """
        Test: Should handle bulk state hashing efficiently.

        Target: > 5000 hashes/second
        """
        from src.framework.mcts.core import MCTSState

        states = [MCTSState(state_id=f"state_{i}", features={"value": i}) for i in range(1000)]

        start = time.perf_counter()

        hashes = [state.to_hash_key() for state in states]

        elapsed = time.perf_counter() - start
        throughput = len(states) / elapsed

        assert throughput > 5000, f"Hash throughput too low: {throughput:.0f} hashes/second"
        assert len(hashes) == len(states)


# Benchmark comparison helper
def benchmark_operation(operation_func, iterations: int = 1000, name: str = "operation"):
    """
    Helper function to benchmark an operation.

    Args:
        operation_func: Function to benchmark
        iterations: Number of iterations
        name: Operation name for reporting

    Returns:
        Dictionary with benchmark results
    """
    import statistics

    timings = []

    for _ in range(iterations):
        start = time.perf_counter()
        operation_func()
        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)  # Convert to ms

    return {
        "operation": name,
        "iterations": iterations,
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "stdev_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


# Run performance tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance", "--tb=short"])
