"""
Performance and Stress Tests for Multi-Agent MCTS Framework.

Tests system behavior under:
- High load conditions
- Concurrent requests
- Memory pressure
- Extended operation
- Edge cases and boundary conditions

These tests help identify performance bottlenecks and ensure
system stability under stress.
"""

import asyncio
import gc
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

# Optional torch import - skip tests if not available
torch = pytest.importorskip("torch", reason="PyTorch required for performance stress tests")

import numpy as np  # noqa: E402 - import after torch check for type annotations

from tests.mocks.mock_external_services import create_mock_llm  # noqa: E402

# ============================================================================
# LOAD TESTING
# ============================================================================


class TestLoadPerformance:
    """Tests system performance under various load conditions."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self):
        """Test system handles concurrent queries efficiently."""
        from src.models.validation import QueryInput

        num_concurrent = 20
        queries = []

        for i in range(num_concurrent):
            queries.append(
                QueryInput(
                    query=f"Tactical query number {i} requiring analysis",
                    use_rag=True,
                    use_mcts=False,
                    thread_id=f"concurrent_test_{i}",
                )
            )

        mock_llm = create_mock_llm()
        mock_llm.set_responses(
            [f"Response {i}: Analysis complete. Confidence: 0.85" for i in range(num_concurrent * 2)]
        )

        async def process_query(query: QueryInput) -> dict[str, Any]:
            start = time.perf_counter()
            response = await mock_llm.generate(query.query)
            elapsed = time.perf_counter() - start
            return {
                "thread_id": query.thread_id,
                "response": response.content,
                "latency_ms": elapsed * 1000,
            }

        # Process all queries concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*[process_query(q) for q in queries])
        total_time = time.perf_counter() - start_time

        # Validate results
        assert len(results) == num_concurrent
        assert all(r["response"] is not None for r in results)

        # Calculate metrics
        latencies = [r["latency_ms"] for r in results]
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        throughput = num_concurrent / total_time

        # Performance assertions
        assert avg_latency < 100  # Average under 100ms
        assert p95_latency < 200  # P95 under 200ms
        assert throughput > 10  # At least 10 queries/second

    @pytest.mark.performance
    def test_meta_controller_batch_processing(self):
        """Test meta-controller handles batch predictions efficiently."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="BatchTest", seed=42)

        # Create batch of features
        batch_size = 100
        features_batch = [
            MetaControllerFeatures(
                hrm_confidence=0.5 + (i % 50) / 100,
                trm_confidence=0.4 + (i % 40) / 100,
                mcts_value=0.6 + (i % 30) / 100,
                consensus_score=0.5 + (i % 50) / 100,
                last_agent=["hrm", "trm", "mcts"][i % 3],
                iteration=i % 10,
                query_length=100 + i * 10,
                has_rag_context=i % 2 == 0,
            )
            for i in range(batch_size)
        ]

        # Warm up
        controller.predict(features_batch[0])

        # Measure batch processing time
        start = time.perf_counter()
        predictions = [controller.predict(f) for f in features_batch]
        elapsed = time.perf_counter() - start

        # Validate predictions
        assert len(predictions) == batch_size
        assert all(p.agent in ["hrm", "trm", "mcts"] for p in predictions)

        # Performance metrics
        throughput = batch_size / elapsed
        avg_latency = (elapsed / batch_size) * 1000

        assert throughput > 100  # At least 100 predictions/second
        assert avg_latency < 50  # Under 50ms per prediction

    @pytest.mark.performance
    def test_hrm_trm_sustained_throughput(self):
        """Test HRM-TRM pipeline maintains throughput over time."""
        from src.agents.hrm_agent import create_hrm_agent
        from src.agents.trm_agent import create_trm_agent
        from src.training.system_config import HRMConfig, TRMConfig

        hrm_config = HRMConfig(h_dim=32, l_dim=16, num_h_layers=1, num_l_layers=1)
        trm_config = TRMConfig(latent_dim=32, hidden_dim=64, num_recursions=3)

        hrm_agent = create_hrm_agent(hrm_config, device="cpu", use_ponder_net=False)
        trm_agent = create_trm_agent(trm_config, output_dim=32, device="cpu")

        hrm_agent.eval()
        trm_agent.eval()

        test_input = torch.randn(2, 8, 32)

        # Run for extended period
        duration_seconds = 2
        iterations = 0
        latencies = []

        start_time = time.perf_counter()

        with torch.no_grad():
            while time.perf_counter() - start_time < duration_seconds:
                iter_start = time.perf_counter()
                hrm_out = hrm_agent(test_input, max_steps=2)
                trm_agent(hrm_out.final_state, num_recursions=2)
                latencies.append((time.perf_counter() - iter_start) * 1000)
                iterations += 1

        elapsed = time.perf_counter() - start_time
        throughput = iterations / elapsed
        avg_latency = statistics.mean(latencies)

        # Should maintain consistent throughput
        assert throughput > 10  # At least 10 iterations/second
        assert avg_latency < 200  # Under 200ms average


# ============================================================================
# MCTS STRESS TESTS
# ============================================================================


class TestMCTSStress:
    """Stress tests for MCTS engine."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_iteration_mcts(self):
        """Test MCTS with high iteration count."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class FastPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return rng.uniform(0.3, 0.9)

        engine = MCTSEngine(
            seed=42,
            exploration_weight=1.414,
            cache_size_limit=5000,
        )

        root_state = MCTSState(state_id="stress_root", features={})
        root = MCTSNode(state=root_state, rng=engine.rng)

        actions = ["a", "b", "c", "d", "e"]

        def action_gen(state: MCTSState) -> list[str]:
            return actions

        def state_trans(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        # High iteration count
        num_iterations = 1000

        start = time.perf_counter()
        best_action, stats = await engine.search(
            root=root,
            num_iterations=num_iterations,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=FastPolicy(),
            max_rollout_depth=10,
        )
        elapsed = time.perf_counter() - start

        assert best_action in actions
        assert stats["iterations"] == num_iterations
        assert stats["root_visits"] >= num_iterations

        # Performance check
        iterations_per_second = num_iterations / elapsed
        assert iterations_per_second > 100  # At least 100 iter/s

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_deep_tree_exploration(self):
        """Test MCTS with deep tree exploration."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class DepthAwarePolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                # Reward deeper exploration
                depth = state.state_id.count("_")
                return min(1.0, 0.3 + depth * 0.1 + rng.uniform(-0.1, 0.1))

        engine = MCTSEngine(
            seed=42,
            progressive_widening_k=0.5,  # More aggressive widening
            progressive_widening_alpha=0.3,
        )

        root = MCTSNode(
            state=MCTSState(state_id="deep", features={}),
            rng=engine.rng,
        )

        def action_gen(state: MCTSState) -> list[str]:
            return ["down", "left", "right"]

        def state_trans(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        _, stats = await engine.search(
            root=root,
            num_iterations=500,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=DepthAwarePolicy(),
            max_rollout_depth=20,
        )

        # Should have explored multiple levels
        tree_depth = engine.get_tree_depth(root)
        assert tree_depth > 0

        # Should have reasonable node count
        node_count = engine.count_nodes(root)
        assert node_count > 10

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_wide_action_space(self):
        """Test MCTS with wide action space."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class SimplePolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return rng.uniform(0.0, 1.0)

        engine = MCTSEngine(
            seed=42,
            progressive_widening_k=2.0,  # Control widening
            progressive_widening_alpha=0.5,
        )

        root = MCTSNode(
            state=MCTSState(state_id="wide", features={}),
            rng=engine.rng,
        )

        # Wide action space
        wide_actions = [f"action_{i}" for i in range(50)]

        def action_gen(state: MCTSState) -> list[str]:
            return wide_actions

        def state_trans(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        start = time.perf_counter()
        best_action, stats = await engine.search(
            root=root,
            num_iterations=300,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=SimplePolicy(),
        )
        elapsed = time.perf_counter() - start

        assert best_action in wide_actions

        # Should complete in reasonable time despite wide action space
        assert elapsed < 10  # Under 10 seconds


# ============================================================================
# MEMORY STRESS TESTS
# ============================================================================


class TestMemoryStress:
    """Tests memory behavior under stress."""

    @pytest.mark.performance
    def test_large_batch_memory(self):
        """Test memory usage with large batches."""
        from src.agents.hrm_agent import create_hrm_agent
        from src.training.system_config import HRMConfig

        hrm_config = HRMConfig(h_dim=64, l_dim=32, num_h_layers=2, num_l_layers=1)
        hrm_agent = create_hrm_agent(hrm_config, device="cpu", use_ponder_net=False)
        hrm_agent.eval()

        # Force garbage collection
        gc.collect()

        # Large batch
        batch_size = 32
        seq_len = 64
        large_input = torch.randn(batch_size, seq_len, hrm_config.h_dim)

        # Process multiple times
        with torch.no_grad():
            for _ in range(10):
                hrm_agent(large_input, max_steps=3)

        # Should complete without memory issues
        gc.collect()

    @pytest.mark.performance
    def test_repeated_model_creation(self):
        """Test memory doesn't leak with repeated model creation."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        gc.collect()

        test_features = MetaControllerFeatures(
            hrm_confidence=0.7,
            trm_confidence=0.6,
            mcts_value=0.8,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )

        # Create and destroy many controllers
        for i in range(50):
            controller = RNNMetaController(name=f"Controller_{i}", seed=i)
            controller.predict(test_features)
            del controller

        gc.collect()
        # Should complete without memory issues

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_mcts_cache_memory(self):
        """Test MCTS cache doesn't grow unbounded."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class CacheTestPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return rng.uniform(0.0, 1.0)

        # Small cache limit
        engine = MCTSEngine(
            seed=42,
            cache_size_limit=100,
        )

        root = MCTSNode(
            state=MCTSState(state_id="cache_mem", features={}),
            rng=engine.rng,
        )

        def action_gen(state: MCTSState) -> list[str]:
            return ["a", "b", "c", "d", "e"]

        def state_trans(state: MCTSState, action: str) -> MCTSState:
            # Generate unique states to fill cache
            return MCTSState(
                state_id=f"{state.state_id}_{action}_{time.time_ns()}",
                features={},
            )

        # Run many iterations to exceed cache limit
        await engine.search(
            root=root,
            num_iterations=500,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=CacheTestPolicy(),
        )

        # Cache should not exceed limit
        assert len(engine._simulation_cache) <= engine.cache_size_limit

        # Should have evictions
        assert engine.cache_evictions > 0


# ============================================================================
# CONCURRENT STRESS TESTS
# ============================================================================


class TestConcurrencyStress:
    """Tests system behavior under concurrent access."""

    @pytest.mark.performance
    def test_thread_safe_controller(self):
        """Test meta-controller is thread-safe."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="ThreadSafeTest", seed=42)

        test_features = MetaControllerFeatures(
            hrm_confidence=0.7,
            trm_confidence=0.6,
            mcts_value=0.8,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )

        results = []
        errors = []

        def predict_task(task_id: int):
            try:
                pred = controller.predict(test_features)
                results.append((task_id, pred.agent, pred.confidence))
            except Exception as e:
                errors.append((task_id, str(e)))

        # Run concurrent predictions
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(predict_task, i) for i in range(100)]
            for future in futures:
                future.result()

        # Should have no errors
        assert len(errors) == 0
        assert len(results) == 100

        # All predictions should be valid
        assert all(r[1] in ["hrm", "trm", "mcts"] for r in results)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_mcts_searches(self):
        """Test multiple concurrent MCTS searches."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class ConcurrentPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                await asyncio.sleep(0.001)  # Simulate some async work
                return rng.uniform(0.0, 1.0)

        async def run_search(search_id: int) -> dict[str, Any]:
            engine = MCTSEngine(seed=42 + search_id)
            root = MCTSNode(
                state=MCTSState(state_id=f"concurrent_{search_id}", features={}),
                rng=engine.rng,
            )

            def action_gen(state: MCTSState) -> list[str]:
                return ["a", "b", "c"]

            def state_trans(state: MCTSState, action: str) -> MCTSState:
                return MCTSState(state_id=f"{state.state_id}_{action}", features={})

            start = time.perf_counter()
            best_action, stats = await engine.search(
                root=root,
                num_iterations=50,
                action_generator=action_gen,
                state_transition=state_trans,
                rollout_policy=ConcurrentPolicy(),
            )
            elapsed = time.perf_counter() - start

            return {
                "search_id": search_id,
                "best_action": best_action,
                "iterations": stats["iterations"],
                "elapsed": elapsed,
            }

        # Run multiple searches concurrently
        num_searches = 5
        results = await asyncio.gather(*[run_search(i) for i in range(num_searches)])

        # All searches should complete successfully
        assert len(results) == num_searches
        assert all(r["best_action"] in ["a", "b", "c"] for r in results)
        assert all(r["iterations"] == 50 for r in results)


# ============================================================================
# EDGE CASE STRESS TESTS
# ============================================================================


class TestEdgeCaseStress:
    """Tests system behavior with edge cases."""

    @pytest.mark.performance
    def test_extreme_confidence_values(self):
        """Test handling of extreme confidence values."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="ExtremeTest", seed=42)

        extreme_cases = [
            # All zeros
            MetaControllerFeatures(
                hrm_confidence=0.0,
                trm_confidence=0.0,
                mcts_value=0.0,
                consensus_score=0.0,
                last_agent="none",
                iteration=0,
                query_length=0,
                has_rag_context=False,
            ),
            # All ones
            MetaControllerFeatures(
                hrm_confidence=1.0,
                trm_confidence=1.0,
                mcts_value=1.0,
                consensus_score=1.0,
                last_agent="mcts",
                iteration=100,
                query_length=10000,
                has_rag_context=True,
            ),
            # Mixed extremes
            MetaControllerFeatures(
                hrm_confidence=1.0,
                trm_confidence=0.0,
                mcts_value=0.5,
                consensus_score=0.0,
                last_agent="hrm",
                iteration=50,
                query_length=5000,
                has_rag_context=False,
            ),
        ]

        for features in extreme_cases:
            pred = controller.predict(features)

            assert pred.agent in ["hrm", "trm", "mcts"]
            assert 0.0 <= pred.confidence <= 1.0
            assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-5

    @pytest.mark.performance
    def test_rapid_state_transitions(self):
        """Test rapid state transitions in meta-controller."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="RapidTest", seed=42)

        # Rapidly changing states
        predictions = []
        for i in range(100):
            features = MetaControllerFeatures(
                hrm_confidence=(i % 100) / 100,
                trm_confidence=((i + 33) % 100) / 100,
                mcts_value=((i + 66) % 100) / 100,
                consensus_score=(i % 50) / 50,
                last_agent=["hrm", "trm", "mcts"][i % 3],
                iteration=i,
                query_length=i * 10,
                has_rag_context=i % 2 == 0,
            )

            pred = controller.predict(features)
            predictions.append(pred)

        # All predictions should be valid
        assert len(predictions) == 100
        assert all(p.agent in ["hrm", "trm", "mcts"] for p in predictions)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_mcts_single_action(self):
        """Test MCTS with single available action."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class SingleActionPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return 0.8

        engine = MCTSEngine(seed=42)
        root = MCTSNode(
            state=MCTSState(state_id="single", features={}),
            rng=engine.rng,
        )

        def action_gen(state: MCTSState) -> list[str]:
            return ["only_action"]

        def state_trans(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        best_action, stats = await engine.search(
            root=root,
            num_iterations=50,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=SingleActionPolicy(),
        )

        assert best_action == "only_action"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_mcts_terminal_state(self):
        """Test MCTS handles terminal states correctly."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class TerminalPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                if "terminal" in state.state_id:
                    return 1.0
                return 0.5

        engine = MCTSEngine(seed=42)

        # Start with terminal state
        root = MCTSNode(
            state=MCTSState(state_id="terminal", features={}),
            rng=engine.rng,
        )
        root.terminal = True

        def action_gen(state: MCTSState) -> list[str]:
            return []  # No actions from terminal

        def state_trans(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        best_action, stats = await engine.search(
            root=root,
            num_iterations=10,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=TerminalPolicy(),
        )

        # Should handle terminal state gracefully
        assert stats["iterations"] == 10


# ============================================================================
# SUSTAINED LOAD TESTS
# ============================================================================


class TestSustainedLoad:
    """Tests system stability under sustained load."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_sustained_predictions(self):
        """Test sustained prediction load over time."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="SustainedTest", seed=42)

        test_features = MetaControllerFeatures(
            hrm_confidence=0.7,
            trm_confidence=0.6,
            mcts_value=0.8,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )

        # Run for extended period
        duration_seconds = 3
        predictions = 0
        latencies = []

        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_seconds:
            iter_start = time.perf_counter()
            controller.predict(test_features)
            latencies.append((time.perf_counter() - iter_start) * 1000)
            predictions += 1

        elapsed = time.perf_counter() - start_time

        # Calculate metrics
        throughput = predictions / elapsed
        avg_latency = statistics.mean(latencies)
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

        # Stability checks
        assert throughput > 50  # Maintain throughput
        assert avg_latency < 50  # Stable latency
        assert p99_latency < 100  # No major outliers

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_mcts_operations(self):
        """Test sustained MCTS operations over time."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class StablePolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return rng.uniform(0.4, 0.8)

        def action_gen(state: MCTSState) -> list[str]:
            return ["a", "b", "c"]

        def state_trans(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        # Run multiple searches
        num_searches = 20
        search_times = []

        for i in range(num_searches):
            engine = MCTSEngine(seed=42 + i)
            root = MCTSNode(
                state=MCTSState(state_id=f"sustained_{i}", features={}),
                rng=engine.rng,
            )

            start = time.perf_counter()
            await engine.search(
                root=root,
                num_iterations=100,
                action_generator=action_gen,
                state_transition=state_trans,
                rollout_policy=StablePolicy(),
            )
            search_times.append(time.perf_counter() - start)

        # Check for consistent performance
        avg_time = statistics.mean(search_times)
        std_time = statistics.stdev(search_times)

        # Coefficient of variation should be low (consistent performance)
        cv = std_time / avg_time
        assert cv < 0.5  # Less than 50% variation


# ============================================================================
# RECOVERY STRESS TESTS
# ============================================================================


class TestRecoveryStress:
    """Tests system recovery from stress conditions."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_recovery_after_high_load(self):
        """Test system recovers after high load period."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="RecoveryTest", seed=42)

        test_features = MetaControllerFeatures(
            hrm_confidence=0.7,
            trm_confidence=0.6,
            mcts_value=0.8,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )

        # Baseline performance
        baseline_latencies = []
        for _ in range(50):
            start = time.perf_counter()
            controller.predict(test_features)
            baseline_latencies.append((time.perf_counter() - start) * 1000)

        baseline_avg = statistics.mean(baseline_latencies)

        # High load period
        for _ in range(500):
            controller.predict(test_features)

        # Recovery period
        gc.collect()
        await asyncio.sleep(0.1)

        # Post-recovery performance
        recovery_latencies = []
        for _ in range(50):
            start = time.perf_counter()
            controller.predict(test_features)
            recovery_latencies.append((time.perf_counter() - start) * 1000)

        recovery_avg = statistics.mean(recovery_latencies)

        # Should recover to near-baseline performance
        assert recovery_avg < baseline_avg * 2  # Within 2x of baseline

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_recovery(self):
        """Test MCTS cache recovers after clearing."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class CacheRecoveryPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return rng.uniform(0.0, 1.0)

        engine = MCTSEngine(seed=42, cache_size_limit=500)

        def action_gen(state: MCTSState) -> list[str]:
            return ["a", "b", "c"]

        def state_trans(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        # Build up cache
        root = MCTSNode(
            state=MCTSState(state_id="cache_recovery", features={}),
            rng=engine.rng,
        )

        await engine.search(
            root=root,
            num_iterations=200,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=CacheRecoveryPolicy(),
        )

        initial_cache_size = len(engine._simulation_cache)
        assert initial_cache_size > 0

        # Clear cache
        engine.clear_cache()
        assert len(engine._simulation_cache) == 0

        # Run again - should rebuild cache
        root2 = MCTSNode(
            state=MCTSState(state_id="cache_recovery_2", features={}),
            rng=engine.rng,
        )

        await engine.search(
            root=root2,
            num_iterations=200,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=CacheRecoveryPolicy(),
        )

        # Cache should be rebuilt
        assert len(engine._simulation_cache) > 0
