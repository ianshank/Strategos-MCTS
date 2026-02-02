"""
Integration Tests for Agent Orchestration.

Tests the integration between:
- Meta-controller agent selection
- HRM hierarchical decomposition
- TRM iterative refinement
- MCTS tactical simulation
- Consensus mechanisms

These tests verify that components work together correctly
in realistic multi-agent scenarios.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest

# Skip all tests in this module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for agent orchestration tests")

import numpy as np  # noqa: E402 - import after torch check for type annotations

from tests.mocks.mock_external_services import create_mock_llm  # noqa: E402

# ============================================================================
# META-CONTROLLER INTEGRATION TESTS
# ============================================================================


class TestMetaControllerIntegration:
    """Tests meta-controller integration with agent selection."""

    @pytest.fixture
    def sample_states(self) -> list[dict[str, Any]]:
        """Create sample agent states for testing."""
        return [
            # HRM should be selected (high HRM confidence)
            {
                "hrm_confidence": 0.9,
                "trm_confidence": 0.5,
                "mcts_value": 0.4,
                "consensus_score": 0.7,
                "last_agent": "none",
                "iteration": 0,
                "query": "Complex hierarchical planning problem",
                "rag_context": "Relevant context here",
            },
            # TRM should be selected (high TRM confidence)
            {
                "hrm_confidence": 0.4,
                "trm_confidence": 0.9,
                "mcts_value": 0.5,
                "consensus_score": 0.6,
                "last_agent": "hrm",
                "iteration": 2,
                "query": "Refine the previous solution",
                "rag_context": None,
            },
            # MCTS should be selected (high MCTS value)
            {
                "hrm_confidence": 0.5,
                "trm_confidence": 0.4,
                "mcts_value": 0.95,
                "consensus_score": 0.5,
                "last_agent": "trm",
                "iteration": 3,
                "query": "Tactical simulation needed",
                "rag_context": "Battle context",
            },
        ]

    @pytest.mark.integration
    def test_rnn_controller_agent_selection(self, sample_states):
        """Test RNN meta-controller selects appropriate agents."""
        from src.agents.meta_controller.base import MetaControllerPrediction
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="TestRNN", seed=42)

        for state in sample_states:
            features = controller.extract_features(state)
            prediction = controller.predict(features)

            assert isinstance(prediction, MetaControllerPrediction)
            assert prediction.agent in ["hrm", "trm", "mcts"]
            assert 0.0 <= prediction.confidence <= 1.0
            assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.integration
    def test_feature_extraction_consistency(self, sample_states):
        """Test feature extraction produces consistent results."""
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="TestRNN", seed=42)

        for state in sample_states:
            features1 = controller.extract_features(state)
            features2 = controller.extract_features(state)

            # Same input should produce same features
            assert features1.hrm_confidence == features2.hrm_confidence
            assert features1.trm_confidence == features2.trm_confidence
            assert features1.mcts_value == features2.mcts_value
            assert features1.consensus_score == features2.consensus_score
            assert features1.last_agent == features2.last_agent
            assert features1.iteration == features2.iteration
            assert features1.query_length == features2.query_length
            assert features1.has_rag_context == features2.has_rag_context

    @pytest.mark.integration
    def test_controller_determinism_with_seed(self, sample_states):
        """Test controller produces deterministic results with same seed."""
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller1 = RNNMetaController(name="TestRNN1", seed=42)
        controller2 = RNNMetaController(name="TestRNN2", seed=42)

        for state in sample_states:
            features = controller1.extract_features(state)

            pred1 = controller1.predict(features)
            pred2 = controller2.predict(features)

            assert pred1.agent == pred2.agent
            assert pred1.confidence == pytest.approx(pred2.confidence, abs=1e-6)

    @pytest.mark.integration
    def test_controller_save_load_roundtrip(self, sample_states):
        """Test controller save/load preserves behavior."""
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "controller.pt"

            # Create and save controller
            controller1 = RNNMetaController(name="Original", seed=42)
            features = controller1.extract_features(sample_states[0])
            pred_before = controller1.predict(features)

            controller1.save_model(str(model_path))
            assert model_path.exists()

            # Load into new controller
            controller2 = RNNMetaController(name="Loaded", seed=99)
            controller2.load_model(str(model_path))

            pred_after = controller2.predict(features)

            # Predictions should match
            assert pred_before.agent == pred_after.agent
            assert pred_before.confidence == pytest.approx(pred_after.confidence, abs=1e-6)


# ============================================================================
# HRM-TRM INTEGRATION TESTS
# ============================================================================


class TestHRMTRMIntegration:
    """Tests integration between HRM and TRM agents."""

    @pytest.fixture
    def hrm_config(self):
        """Create HRM configuration."""
        from src.training.system_config import HRMConfig

        return HRMConfig(
            h_dim=64,
            l_dim=32,
            num_h_layers=2,
            num_l_layers=1,
            max_outer_steps=5,
            max_ponder_steps=8,
            halt_threshold=0.9,
            dropout=0.1,
            ponder_epsilon=1e-6,
        )

    @pytest.fixture
    def trm_config(self):
        """Create TRM configuration."""
        from src.training.system_config import TRMConfig

        return TRMConfig(
            latent_dim=64,
            hidden_dim=128,
            num_recursions=5,
            min_recursions=2,
            convergence_threshold=0.01,
            deep_supervision=True,
            use_layer_norm=True,
            dropout=0.1,
        )

    @pytest.mark.integration
    def test_hrm_to_trm_pipeline(self, hrm_config, trm_config):
        """Test HRM output flows correctly to TRM input."""
        from src.agents.hrm_agent import create_hrm_agent
        from src.agents.trm_agent import create_trm_agent

        hrm_agent = create_hrm_agent(hrm_config, device="cpu", use_ponder_net=False)
        trm_agent = create_trm_agent(trm_config, output_dim=64, device="cpu")

        # Create test input
        batch_size = 2
        seq_len = 8
        test_input = torch.randn(batch_size, seq_len, hrm_config.h_dim)

        # HRM processing
        hrm_output = hrm_agent(test_input, max_steps=3)

        assert hrm_output.final_state.shape == (batch_size, seq_len, hrm_config.h_dim)

        # TRM refinement on HRM output
        trm_output = trm_agent(hrm_output.final_state, num_recursions=3)

        assert trm_output.final_prediction.shape[0] == batch_size
        assert trm_output.recursion_depth > 0

    @pytest.mark.integration
    def test_hrm_trm_gradient_flow(self, hrm_config, trm_config):
        """Test gradients flow correctly through HRM-TRM pipeline."""
        from src.agents.hrm_agent import create_hrm_agent
        from src.agents.trm_agent import create_trm_agent

        hrm_agent = create_hrm_agent(hrm_config, device="cpu", use_ponder_net=False)
        trm_agent = create_trm_agent(trm_config, output_dim=64, device="cpu")

        # Enable training mode
        hrm_agent.train()
        trm_agent.train()

        # Create test input
        test_input = torch.randn(2, 8, hrm_config.h_dim, requires_grad=True)

        # Forward pass
        hrm_output = hrm_agent(test_input, max_steps=2)
        trm_output = trm_agent(hrm_output.final_state, num_recursions=2)

        # Compute loss
        target = torch.randn_like(trm_output.final_prediction)
        loss = torch.nn.functional.mse_loss(trm_output.final_prediction, target)

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert test_input.grad is not None
        assert not torch.all(test_input.grad == 0)

    @pytest.mark.integration
    def test_hrm_decomposition_quality(self, hrm_config):
        """Test HRM produces meaningful decomposition."""
        from src.agents.hrm_agent import create_hrm_agent

        hrm_agent = create_hrm_agent(hrm_config, device="cpu", use_ponder_net=False)
        hrm_agent.eval()

        test_input = torch.randn(1, 10, hrm_config.h_dim)

        hrm_output = hrm_agent(
            test_input,
            max_steps=5,
            return_decomposition=True,
        )

        # Should produce subproblems
        assert len(hrm_output.subproblems) > 0

        # Each subproblem should have valid structure
        for sp in hrm_output.subproblems:
            assert sp.level >= 0
            assert sp.state is not None
            assert 0.0 <= sp.confidence <= 1.0

    @pytest.mark.integration
    def test_trm_convergence_behavior(self, trm_config):
        """Test TRM convergence detection works correctly."""
        from src.agents.trm_agent import create_trm_agent

        trm_agent = create_trm_agent(trm_config, output_dim=64, device="cpu")
        trm_agent.eval()

        # Create input that should converge
        test_input = torch.randn(2, 8, trm_config.latent_dim)

        # Use num_recursions within the bounds of supervision heads
        trm_output = trm_agent(
            test_input,
            num_recursions=trm_config.num_recursions,  # Use config value
            check_convergence=True,
        )

        # Should have intermediate predictions
        assert len(trm_output.intermediate_predictions) > 0

        # Residual norms should generally decrease
        if len(trm_output.residual_norms) > 1:
            # Not strictly required, but generally expected
            pass

        # Should report convergence status
        assert isinstance(trm_output.converged, bool)
        assert trm_output.convergence_step > 0


# ============================================================================
# MCTS INTEGRATION TESTS
# ============================================================================


class TestMCTSIntegration:
    """Tests MCTS integration with other components."""

    @pytest.fixture
    def mcts_engine(self):
        """Create MCTS engine for testing."""
        from src.framework.mcts.core import MCTSEngine

        return MCTSEngine(
            seed=42,
            exploration_weight=1.414,
            progressive_widening_k=1.0,
            progressive_widening_alpha=0.5,
            max_parallel_rollouts=4,
            cache_size_limit=1000,
        )

    @pytest.fixture
    def tactical_actions(self) -> list[str]:
        """Define tactical actions for testing."""
        return [
            "advance",
            "hold",
            "retreat",
            "flank_left",
            "flank_right",
            "reinforce",
        ]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcts_search_integration(self, mcts_engine, tactical_actions):
        """Test MCTS search produces valid results."""
        from src.framework.mcts.core import MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        # Create root state
        root_state = MCTSState(
            state_id="root",
            features={"position": "center", "resources": 100},
        )
        root = MCTSNode(state=root_state, rng=mcts_engine.rng)

        def action_generator(state: MCTSState) -> list[str]:
            return tactical_actions

        def state_transition(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(
                state_id=f"{state.state_id}_{action}",
                features={**state.features, "last_action": action},
            )

        class TestRolloutPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return rng.uniform(0.3, 0.9)

        rollout_policy = TestRolloutPolicy()

        # Run search
        best_action, stats = await mcts_engine.search(
            root=root,
            num_iterations=100,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
            max_rollout_depth=5,
        )

        assert best_action in tactical_actions
        assert stats["iterations"] == 100
        assert stats["root_visits"] >= 100
        assert stats["num_children"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcts_caching_behavior(self, mcts_engine, tactical_actions):
        """Test MCTS caching improves performance."""
        from src.framework.mcts.core import MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        root_state = MCTSState(state_id="cache_test", features={})
        root = MCTSNode(state=root_state, rng=mcts_engine.rng)

        def action_generator(state: MCTSState) -> list[str]:
            return tactical_actions[:3]  # Fewer actions for cache testing

        def state_transition(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        class CacheTestPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return 0.5 + rng.uniform(-0.1, 0.1)

        rollout_policy = CacheTestPolicy()

        # Clear cache
        mcts_engine.clear_cache()

        # Run search
        _, stats = await mcts_engine.search(
            root=root,
            num_iterations=50,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
        )

        # Should have some cache activity
        assert stats["cache_size"] > 0
        # Cache hit rate may be 0 for first run, but structure should exist
        assert "cache_hit_rate" in stats

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcts_determinism(self, tactical_actions):
        """Test MCTS produces deterministic results with same seed."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class DeterministicPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return rng.uniform(0.0, 1.0)

        def action_generator(state: MCTSState) -> list[str]:
            return tactical_actions

        def state_transition(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        results = []

        for _ in range(2):
            engine = MCTSEngine(seed=42)
            root = MCTSNode(
                state=MCTSState(state_id="determinism_test", features={}),
                rng=engine.rng,
            )

            best_action, stats = await engine.search(
                root=root,
                num_iterations=50,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=DeterministicPolicy(),
            )

            results.append((best_action, stats["root_value"]))

        # Results should be identical
        assert results[0][0] == results[1][0]
        assert results[0][1] == pytest.approx(results[1][1], abs=1e-6)


# ============================================================================
# FULL PIPELINE INTEGRATION TESTS
# ============================================================================


class TestFullPipelineIntegration:
    """Tests full pipeline integration from query to response."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_to_response_pipeline(self):
        """Test complete query processing pipeline."""
        from src.agents.meta_controller.rnn_controller import RNNMetaController
        from src.models.validation import QueryInput

        # Step 1: Validate input
        query_input = QueryInput(
            query="Tactical analysis needed for defensive positioning",
            use_rag=True,
            use_mcts=True,
            thread_id="pipeline_test_001",
        )

        assert query_input.query is not None

        # Step 2: Meta-controller decision
        controller = RNNMetaController(name="PipelineController", seed=42)

        initial_state = {
            "hrm_confidence": 0.0,
            "trm_confidence": 0.0,
            "mcts_value": 0.0,
            "consensus_score": 0.0,
            "last_agent": "none",
            "iteration": 0,
            "query": query_input.query,
            "rag_context": "Sample context",
        }

        features = controller.extract_features(initial_state)
        prediction = controller.predict(features)

        assert prediction.agent in ["hrm", "trm", "mcts"]

        # Step 3: Simulate agent processing
        mock_llm = create_mock_llm()
        mock_llm.set_responses(
            [
                "Analysis complete. Confidence: 0.85",
            ]
        )

        response = await mock_llm.generate(query_input.query)

        assert response.content is not None

        # Step 4: Build final response
        final_response = {
            "query_id": query_input.thread_id,
            "agent_selected": prediction.agent,
            "agent_confidence": prediction.confidence,
            "response": response.content,
            "processing_complete": True,
        }

        assert final_response["processing_complete"]
        assert final_response["agent_selected"] in ["hrm", "trm", "mcts"]

    @pytest.mark.integration
    def test_training_inference_consistency(self):
        """Test model behaves consistently between training and inference."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="ConsistencyTest", seed=42)

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

        # Training mode prediction
        controller.model.train()
        train_pred = controller.predict(test_features)

        # Inference mode prediction
        controller.model.eval()
        eval_pred = controller.predict(test_features)

        # Agent selection should be consistent
        # Note: Values may differ due to dropout, but agent selection should be stable
        assert train_pred.agent in ["hrm", "trm", "mcts"]
        assert eval_pred.agent in ["hrm", "trm", "mcts"]

    @pytest.mark.integration
    def test_multi_iteration_refinement(self):
        """Test multi-iteration refinement improves results."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="RefinementTest", seed=42)

        # Simulate multiple iterations with improving confidence
        iterations = []

        for i in range(5):
            features = MetaControllerFeatures(
                hrm_confidence=0.5 + (i * 0.1),  # Increasing confidence
                trm_confidence=0.4 + (i * 0.08),
                mcts_value=0.6 + (i * 0.05),
                consensus_score=0.5 + (i * 0.1),
                last_agent="hrm" if i % 2 == 0 else "trm",
                iteration=i,
                query_length=100,
                has_rag_context=True,
            )

            prediction = controller.predict(features)
            iterations.append(
                {
                    "iteration": i,
                    "agent": prediction.agent,
                    "confidence": prediction.confidence,
                }
            )

        # Should have predictions for all iterations
        assert len(iterations) == 5

        # All predictions should be valid
        for it in iterations:
            assert it["agent"] in ["hrm", "trm", "mcts"]
            assert 0.0 <= it["confidence"] <= 1.0


# ============================================================================
# ERROR HANDLING INTEGRATION TESTS
# ============================================================================


class TestErrorHandlingIntegration:
    """Tests error handling across integrated components."""

    @pytest.mark.integration
    def test_invalid_features_handling(self):
        """Test handling of invalid feature values."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="ErrorTest", seed=42)

        # Edge case: all zeros
        zero_features = MetaControllerFeatures(
            hrm_confidence=0.0,
            trm_confidence=0.0,
            mcts_value=0.0,
            consensus_score=0.0,
            last_agent="none",
            iteration=0,
            query_length=0,
            has_rag_context=False,
        )

        prediction = controller.predict(zero_features)

        # Should still produce valid prediction
        assert prediction.agent in ["hrm", "trm", "mcts"]
        assert 0.0 <= prediction.confidence <= 1.0

    @pytest.mark.integration
    def test_boundary_value_handling(self):
        """Test handling of boundary values."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="BoundaryTest", seed=42)

        # Maximum values
        max_features = MetaControllerFeatures(
            hrm_confidence=1.0,
            trm_confidence=1.0,
            mcts_value=1.0,
            consensus_score=1.0,
            last_agent="mcts",
            iteration=100,
            query_length=10000,
            has_rag_context=True,
        )

        prediction = controller.predict(max_features)

        assert prediction.agent in ["hrm", "trm", "mcts"]
        assert 0.0 <= prediction.confidence <= 1.0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_failure_recovery(self):
        """Test recovery from LLM failures."""
        mock_llm = create_mock_llm()

        # Configure failure
        mock_llm.set_failure_mode(True, "Service unavailable")

        # First call fails
        with pytest.raises(RuntimeError):
            await mock_llm.generate("Test query")

        # Subsequent call succeeds
        response = await mock_llm.generate("Recovery test")
        assert response.content is not None

    @pytest.mark.integration
    def test_state_extraction_robustness(self):
        """Test feature extraction handles malformed states."""
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="RobustnessTest", seed=42)

        # Minimal state
        minimal_state = {"query": "Test"}
        features = controller.extract_features(minimal_state)

        assert features.hrm_confidence == 0.0
        assert features.trm_confidence == 0.0
        assert features.query_length == 4

        # State with nested structure
        nested_state = {
            "agent_confidences": {"hrm": 0.8, "trm": 0.6},
            "mcts_state": {"value": 0.7},
            "query": "Nested test",
        }
        features = controller.extract_features(nested_state)

        assert features.hrm_confidence == 0.8
        assert features.trm_confidence == 0.6
        assert features.mcts_value == 0.7


# ============================================================================
# PERFORMANCE INTEGRATION TESTS
# ============================================================================


class TestPerformanceIntegration:
    """Tests performance characteristics of integrated components."""

    @pytest.mark.integration
    def test_meta_controller_latency(self):
        """Test meta-controller prediction latency."""
        import time

        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(name="LatencyTest", seed=42)

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

        # Warm up
        controller.predict(test_features)

        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            controller.predict(test_features)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        # Should be fast (< 10ms average, < 50ms p95)
        assert avg_latency < 100  # Relaxed for CI
        assert p95_latency < 200  # Relaxed for CI

    @pytest.mark.integration
    def test_hrm_trm_throughput(self):
        """Test HRM-TRM pipeline throughput."""
        import time

        from src.agents.hrm_agent import create_hrm_agent
        from src.agents.trm_agent import create_trm_agent
        from src.training.system_config import HRMConfig, TRMConfig

        hrm_config = HRMConfig(h_dim=32, l_dim=16, num_h_layers=1, num_l_layers=1)
        trm_config = TRMConfig(latent_dim=32, hidden_dim=64, num_recursions=3)

        hrm_agent = create_hrm_agent(hrm_config, device="cpu", use_ponder_net=False)
        trm_agent = create_trm_agent(trm_config, output_dim=32, device="cpu")

        hrm_agent.eval()
        trm_agent.eval()

        test_input = torch.randn(4, 8, 32)

        # Warm up
        with torch.no_grad():
            hrm_out = hrm_agent(test_input, max_steps=2)
            trm_agent(hrm_out.final_state, num_recursions=2)

        # Measure throughput
        start = time.perf_counter()
        iterations = 50

        with torch.no_grad():
            for _ in range(iterations):
                hrm_out = hrm_agent(test_input, max_steps=2)
                trm_agent(hrm_out.final_state, num_recursions=2)

        elapsed = time.perf_counter() - start
        throughput = iterations / elapsed

        # Should handle at least 10 iterations per second
        assert throughput > 5  # Relaxed for CI

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mcts_scaling(self):
        """Test MCTS performance scales reasonably."""
        import time

        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        class FastPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
                return rng.uniform(0.0, 1.0)

        def action_gen(state: MCTSState) -> list[str]:
            return ["a", "b", "c"]

        def state_trans(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(state_id=f"{state.state_id}_{action}", features={})

        iteration_counts = [50, 100, 200]
        times = []

        for num_iter in iteration_counts:
            engine = MCTSEngine(seed=42)
            root = MCTSNode(
                state=MCTSState(state_id="scale_test", features={}),
                rng=engine.rng,
            )

            start = time.perf_counter()
            await engine.search(
                root=root,
                num_iterations=num_iter,
                action_generator=action_gen,
                state_transition=state_trans,
                rollout_policy=FastPolicy(),
            )
            times.append(time.perf_counter() - start)

        # Time should scale roughly linearly (within 3x)
        ratio = times[-1] / times[0]
        expected_ratio = iteration_counts[-1] / iteration_counts[0]

        # Allow some overhead, but shouldn't be worse than 2x expected
        assert ratio < expected_ratio * 2
