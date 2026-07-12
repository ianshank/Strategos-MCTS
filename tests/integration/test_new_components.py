"""
Integration tests for new components.

Tests end-to-end functionality of:
- GameState implementations
- Observability facade
- Logging decorators
- Dataset format handlers

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 10
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

# Set environment variables before importing modules
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

# Import with graceful fallback
try:
    from src.framework.mcts.game_states import (
        DecisionState,
        PlanningState,
        ReasoningState,
        create_game_state,
    )

    GAME_STATES_AVAILABLE = True
    GAME_STATES_SKIP_REASON = ""
except ImportError as exc:  # typically torch missing (neural extra not installed)
    GAME_STATES_AVAILABLE = False
    GAME_STATES_SKIP_REASON = f"GameStates not available: {exc}"

try:
    from src.observability.facade import (
        ObservabilityConfig,
        ObservabilityFacade,
        get_observability,
    )

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

try:
    from src.observability.decorators import (
        cached,
        debug_on_error,
        logged,
        retry,
        timed,
        validate_args,
    )

    DECORATORS_AVAILABLE = True
except ImportError:
    DECORATORS_AVAILABLE = False

try:
    from src.data.dataset_loader import DatasetSample, UnifiedDatasetLoader

    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False


pytestmark = pytest.mark.integration


# =============================================================================
# GameState Integration Tests
# =============================================================================


@pytest.mark.skipif(not GAME_STATES_AVAILABLE, reason=GAME_STATES_SKIP_REASON)
class TestGameStateIntegration:
    """Integration tests for GameState implementations."""

    def test_reasoning_state_full_workflow(self):
        """Test complete reasoning workflow from start to conclusion."""
        state = ReasoningState(
            problem="What is 2 + 2?",
            max_steps=10,
        )

        # Simulate reasoning workflow
        steps_taken = 0
        current_state = state

        while not current_state.is_terminal() and steps_taken < 15:
            actions = current_state.get_legal_actions()
            assert len(actions) > 0, "Should always have legal actions"

            # Take first available action
            action = actions[0]
            current_state = current_state.apply_action(action)
            steps_taken += 1

            # Validate state invariants
            assert len(current_state.reasoning_steps) == steps_taken
            assert 0 <= current_state.confidence <= 1.0

        # Should have terminated
        assert current_state.is_terminal() or steps_taken >= 15

        # Get reward
        reward = current_state.get_reward()
        assert 0 <= reward <= 1.0

    def test_planning_state_resource_management(self):
        """Test planning state with resource constraints."""
        state = PlanningState(
            goal="Complete task",
            current_state="Start",
            available_actions=["analyze", "execute", "verify", "finish"],
            resources={"time": 10.0, "compute": 5.0},
            max_actions=10,
        )

        # Execute actions until resources exhausted or goal reached
        current_state = state
        while not current_state.is_terminal():
            actions = current_state.get_legal_actions()
            if not actions:
                break

            # Take first action
            action = actions[0]
            current_state = current_state.apply_action(action)

            # Validate resource deduction
            for resource in state.resources:
                assert current_state.resources.get(resource, 0) <= state.resources.get(resource, 0)

        assert current_state.is_terminal()

    def test_decision_state_evaluation_workflow(self):
        """Test decision state with option evaluation."""
        options = [
            {"id": "option_a", "value": 10},
            {"id": "option_b", "value": 20},
            {"id": "option_c", "value": 15},
        ]

        state = DecisionState(
            context="Select best option",
            options=options,
            max_evaluations=5,
        )

        # Evaluate options; the step bound turns a termination regression into a
        # fast failure instead of an infinite loop
        current_state = state
        steps_taken = 0
        while not current_state.is_terminal() and steps_taken < 15:
            actions = current_state.get_legal_actions()
            if not actions:
                break

            # Prioritize evaluation actions
            eval_actions = [a for a in actions if a.get("type") == "evaluate"]
            if eval_actions:
                action = eval_actions[0]
            else:
                action = actions[0]

            current_state = current_state.apply_action(action)
            steps_taken += 1

        assert current_state.is_terminal(), f"workflow did not terminate within {steps_taken} steps"

        # Should have evaluated some options
        assert len(current_state.evaluated_options) > 0

    def test_create_game_state_factory(self):
        """Test factory function for creating states."""
        # Create reasoning state
        reasoning = create_game_state("reasoning", problem="Test problem")
        assert isinstance(reasoning, ReasoningState)
        assert reasoning.problem == "Test problem"

        # Create planning state
        planning = create_game_state(
            "planning",
            goal="Test goal",
            current_state="Initial",
            available_actions=["action1"],
        )
        assert isinstance(planning, PlanningState)

        # Create decision state
        decision = create_game_state(
            "decision",
            context="Test context",
            options=[{"id": "opt1"}],
        )
        assert isinstance(decision, DecisionState)

        # Invalid type should raise
        with pytest.raises(ValueError):
            create_game_state("invalid_type")

    def test_game_state_tensor_conversion(self):
        """Test tensor conversion for neural network input."""
        state = ReasoningState(
            problem="Test problem",
            reasoning_steps=["Step 1", "Step 2"],
            confidence=0.5,
        )

        tensor = state.to_tensor()

        # Should be a valid tensor
        assert tensor is not None
        assert tensor.dim() == 1
        assert tensor.shape[0] > 0

        # Check some features
        # Feature 0: step progress
        assert 0 <= tensor[0].item() <= 1.0
        # Feature 1: confidence
        assert abs(tensor[1].item() - 0.5) < 0.01

    def test_game_state_hashing(self):
        """Test deterministic hashing for caching."""
        state1 = ReasoningState(
            problem="Test",
            reasoning_steps=["Step 1"],
            confidence=0.5,
        )

        state2 = ReasoningState(
            problem="Test",
            reasoning_steps=["Step 1"],
            confidence=0.5,
        )

        state3 = ReasoningState(
            problem="Different",
            reasoning_steps=["Step 1"],
            confidence=0.5,
        )

        # Same state should have same hash
        assert state1.get_hash() == state2.get_hash()

        # Different state should have different hash
        assert state1.get_hash() != state3.get_hash()


# =============================================================================
# Observability Facade Integration Tests
# =============================================================================


@pytest.mark.skipif(not OBSERVABILITY_AVAILABLE, reason="Observability not available")
class TestObservabilityIntegration:
    """Integration tests for observability facade."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton between tests."""
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    def test_singleton_pattern(self):
        """Test singleton instance creation."""
        obs1 = get_observability()
        obs2 = get_observability()

        assert obs1 is obs2

    def test_correlation_id_tracking(self):
        """Test correlation ID propagation."""
        obs = get_observability()
        obs.set_correlation_id("test-correlation-123")

        # Should be included in log extra
        extra = obs._get_log_extra(custom_field="value")

        assert "correlation_id" in extra
        assert extra["correlation_id"] == "test-correlation-123"
        assert extra["custom_field"] == "value"

    def test_trace_context_manager(self):
        """Test tracing context manager."""
        obs = get_observability()

        with obs.trace("test_operation", attributes={"key": "value"}):
            # Should complete without error
            pass

    @pytest.mark.asyncio
    async def test_async_trace_context_manager(self):
        """Test async tracing context manager."""
        obs = get_observability()

        async with obs.trace_async("async_operation"):
            await asyncio.sleep(0.01)
            # Should complete without error

    def test_profile_context_manager(self):
        """Test profiling context manager."""
        obs = get_observability()

        with obs.profile("test_profile") as metrics:
            # Simulate some work
            sum(range(1000))

        assert metrics.name == "test_profile"
        assert metrics.success is True
        assert metrics.duration_ms > 0

    def test_profile_captures_errors(self):
        """Test profiling captures errors."""
        obs = get_observability()

        with pytest.raises(ValueError):
            with obs.profile("failing_operation") as metrics:
                raise ValueError("Test error")

        assert metrics.success is False
        assert "Test error" in metrics.error

    def test_config_from_settings(self):
        """Test config creation from settings."""
        config = ObservabilityConfig.from_settings()

        assert isinstance(config.log_level, str)
        assert isinstance(config.metrics_enabled, bool)
        assert isinstance(config.tracing_enabled, bool)


# =============================================================================
# Decorator Integration Tests
# =============================================================================


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="Decorators not available")
class TestDecoratorIntegration:
    """Integration tests for logging decorators."""

    def test_logged_decorator_sync(self):
        """Test logged decorator on sync function."""
        call_count = 0

        @logged()
        def test_function(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        result = test_function(1, 2)

        assert result == 3
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_logged_decorator_async(self):
        """Test logged decorator on async function."""
        call_count = 0

        @logged()
        async def async_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        result = await async_function(5)

        assert result == 10
        assert call_count == 1

    def test_timed_decorator(self):
        """Test timed decorator."""

        @timed(threshold_ms=1000)
        def fast_function():
            return "done"

        result = fast_function()
        assert result == "done"

    def test_retry_decorator_success(self):
        """Test retry decorator with eventual success."""
        attempt_count = 0

        @retry(max_attempts=3, initial_delay=0.01)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = flaky_function()

        assert result == "success"
        assert attempt_count == 2

    def test_retry_decorator_failure(self):
        """Test retry decorator with all attempts failing."""

        @retry(max_attempts=2, initial_delay=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

    def test_cached_decorator(self):
        """Test cached decorator."""
        call_count = 0

        @cached(ttl_seconds=60)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call (cached)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

        # Different argument
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_validate_args_decorator(self):
        """Test argument validation decorator."""

        @validate_args(
            x=lambda v: v > 0,
            name=lambda v: len(v) > 0,
        )
        def validated_function(x: int, name: str) -> str:
            return f"{name}: {x}"

        # Valid args
        result = validated_function(5, "test")
        assert result == "test: 5"

        # Invalid x
        with pytest.raises(ValueError):
            validated_function(-1, "test")

        # Invalid name
        with pytest.raises(ValueError):
            validated_function(5, "")

    def test_debug_on_error_decorator(self):
        """Test debug on error decorator."""

        @debug_on_error(log_locals=True, reraise=True)
        def function_with_error():
            _local_var = "test_value"  # noqa: F841 - intentionally unused for testing
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            function_with_error()


# =============================================================================
# Dataset Format Handler Integration Tests
# =============================================================================


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset loader not available")
class TestDatasetFormatIntegration:
    """Integration tests for dataset format handlers."""

    @pytest.fixture
    def sample_loader(self):
        """Create loader with sample data."""
        loader = UnifiedDatasetLoader()

        # Add sample data
        samples = [
            DatasetSample(
                id="sample_1",
                text="Test text 1",
                domain="test",
                difficulty="easy",
                labels=["label1"],
                metadata={"source": "test"},
            ),
            DatasetSample(
                id="sample_2",
                text="Test text 2",
                domain="test",
                difficulty="medium",
                labels=["label2"],
                metadata={"source": "test"},
            ),
        ]

        loader._all_samples = samples
        return loader

    def test_export_csv(self, sample_loader):
        """Test CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"

            result = sample_loader.export_for_training(str(output_path), format="csv")

            assert Path(result).exists()

            # Verify content
            with open(result, encoding="utf-8") as f:
                content = f.read()
                assert "sample_1" in content
                assert "sample_2" in content
                assert "Test text 1" in content

    def test_export_and_load_csv(self, sample_loader):
        """Test CSV round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"

            # Export
            sample_loader.export_for_training(str(output_path), format="csv")

            # Load
            loaded_samples = UnifiedDatasetLoader.load_from_csv(output_path)

            assert len(loaded_samples) == 2
            assert loaded_samples[0].id == "sample_1"
            assert loaded_samples[0].text == "Test text 1"
            assert loaded_samples[0].domain == "test"

    def test_export_jsonl(self, sample_loader):
        """Test JSONL export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"

            result = sample_loader.export_for_training(str(output_path), format="jsonl")

            assert Path(result).exists()

            # Verify line count
            with open(result, encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 2

    def test_unsupported_format_raises(self, sample_loader):
        """Test unsupported format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.xml"

            with pytest.raises(NotImplementedError):
                sample_loader.export_for_training(str(output_path), format="xml")


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflows:
    """End-to-end integration tests."""

    @pytest.mark.skipif(
        not (GAME_STATES_AVAILABLE and OBSERVABILITY_AVAILABLE),
        reason="Required modules not available",
    )
    def test_mcts_with_observability(self):
        """Test MCTS workflow with full observability."""
        obs = get_observability()
        obs.set_correlation_id("e2e-test-123")

        with obs.profile("mcts_workflow") as metrics:
            # Create state
            state = ReasoningState(
                problem="Integration test problem",
                max_steps=5,
            )

            # Run through reasoning steps
            current = state
            step_count = 0

            while not current.is_terminal() and step_count < 10:
                with obs.trace(f"step_{step_count}"):
                    actions = current.get_legal_actions()
                    if actions:
                        current = current.apply_action(actions[0])
                        step_count += 1

            # Get final reward
            reward = current.get_reward()
            metrics.metadata["reward"] = reward
            metrics.metadata["steps"] = step_count

        assert metrics.success is True
        assert metrics.duration_ms > 0

    @pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="Decorators not available")
    def test_decorated_pipeline(self):
        """Test pipeline with multiple decorators."""

        @logged()
        @timed(threshold_ms=5000)
        def step_1(data: dict) -> dict:
            data["step_1"] = True
            return data

        @logged()
        @cached(ttl_seconds=60)
        def step_2(data: dict) -> dict:
            data["step_2"] = True
            return data

        @logged()
        @retry(max_attempts=2, initial_delay=0.01)
        def step_3(data: dict) -> dict:
            data["step_3"] = True
            return data

        # Run pipeline
        result = step_1({})
        result = step_2(result)
        result = step_3(result)

        assert result["step_1"] is True
        assert result["step_2"] is True
        assert result["step_3"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
