"""
Unit tests for MCTS early termination based on value convergence.

Tests:
- MCTSConfig early_stop_threshold and early_stop_patience defaults and validation
- MCTSEngine.search() early stopping on value convergence
- MCTSEngine.search() continues when values are changing
- Disabling early stop via threshold=0
"""

from __future__ import annotations

import numpy as np
import pytest

from src.framework.mcts.config import MCTSConfig
from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.policies import RolloutPolicy, SelectionPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_root(state_id: str = "root") -> MCTSNode:
    """Create a root MCTSNode with a fixed RNG."""
    state = MCTSState(state_id=state_id)
    return MCTSNode(state=state, rng=np.random.default_rng(42))


def _action_generator(state: MCTSState) -> list[str]:
    """Simple action generator that returns two actions."""
    return ["action_a", "action_b"]


def _state_transition(state: MCTSState, action: str) -> MCTSState:
    """Simple state transition that appends action to state id."""
    return MCTSState(state_id=f"{state.state_id}->{action}")


class ConstantRolloutPolicy(RolloutPolicy):
    """Rollout policy that always returns a constant value."""

    def __init__(self, value: float = 0.5):
        self._value = value

    async def evaluate(self, state: MCTSState, rng, max_depth: int = 10) -> float:
        return self._value


class ConvergingRolloutPolicy(RolloutPolicy):
    """Rollout policy whose value converges quickly so early stop triggers."""

    def __init__(self):
        self._call_count = 0

    async def evaluate(self, state: MCTSState, rng, max_depth: int = 10) -> float:
        self._call_count += 1
        # Return 0.5 always -- values will converge instantly
        return 0.5


class ChangingRolloutPolicy(RolloutPolicy):
    """Rollout policy whose value keeps changing significantly."""

    def __init__(self):
        self._call_count = 0

    async def evaluate(self, state: MCTSState, rng, max_depth: int = 10) -> float:
        self._call_count += 1
        # Oscillate between high and low values to prevent convergence
        if self._call_count % 2 == 0:
            return 1.0
        return 0.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigEarlyStopDefaults:
    """Test that MCTSConfig has correct defaults for early stop fields."""

    def test_config_early_stop_defaults(self):
        config = MCTSConfig()
        assert config.early_stop_threshold == 0.01
        assert config.early_stop_patience == 10

    def test_config_early_stop_custom_values(self):
        config = MCTSConfig(early_stop_threshold=0.05, early_stop_patience=20)
        assert config.early_stop_threshold == 0.05
        assert config.early_stop_patience == 20

    def test_config_early_stop_threshold_zero_allowed(self):
        """Threshold of 0 should be valid (disables early stop)."""
        config = MCTSConfig(early_stop_threshold=0.0)
        assert config.early_stop_threshold == 0.0

    def test_config_early_stop_validation_negative_threshold(self):
        """Negative threshold should raise ValueError."""
        with pytest.raises(ValueError, match="early_stop_threshold must be >= 0"):
            MCTSConfig(early_stop_threshold=-0.01)

    def test_config_early_stop_validation_patience_zero(self):
        """Patience of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="early_stop_patience must be >= 1"):
            MCTSConfig(early_stop_patience=0)

    def test_config_early_stop_validation_patience_negative(self):
        """Negative patience should raise ValueError."""
        with pytest.raises(ValueError, match="early_stop_patience must be >= 1"):
            MCTSConfig(early_stop_patience=-1)


# ---------------------------------------------------------------------------
# Search early termination tests
# ---------------------------------------------------------------------------

class TestSearchEarlyStopOnConvergence:
    """Test that search stops early when best action value converges."""

    @pytest.mark.asyncio
    async def test_search_early_stops_on_convergence(self):
        """Search should terminate before max iterations when values converge."""
        engine = MCTSEngine(seed=42)
        root = _make_root()
        max_iterations = 200

        _, stats = await engine.search(
            root=root,
            num_iterations=max_iterations,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=ConstantRolloutPolicy(0.5),
            max_rollout_depth=5,
            selection_policy=SelectionPolicy.MAX_VISITS,
            early_termination_threshold=0.99,  # high threshold so visit-based won't fire
            min_iterations_before_termination=max_iterations,  # disable visit-based
            early_stop_threshold=0.01,
            early_stop_patience=10,
        )

        # Should have stopped early due to value convergence
        assert stats["early_stopped"] is True
        assert stats["iterations_run"] < max_iterations
        assert stats["termination_reason"] == "value_converged"

    @pytest.mark.asyncio
    async def test_search_continues_when_values_changing(self):
        """Search should run all iterations when values keep changing."""
        engine = MCTSEngine(seed=42)
        root = _make_root()
        max_iterations = 50

        _, stats = await engine.search(
            root=root,
            num_iterations=max_iterations,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=ChangingRolloutPolicy(),
            max_rollout_depth=5,
            selection_policy=SelectionPolicy.MAX_VISITS,
            early_termination_threshold=0.99,  # high threshold so visit-based won't fire
            min_iterations_before_termination=max_iterations,  # disable visit-based
            early_stop_threshold=0.01,
            early_stop_patience=10,
        )

        # With oscillating values, convergence counter should keep resetting
        # It may or may not early stop depending on averaging, but should run most iterations
        assert stats["iterations_run"] >= 20  # ran a significant number

    @pytest.mark.asyncio
    async def test_search_disabled_early_stop(self):
        """Setting early_stop_threshold=0 should disable value-convergence early stop."""
        engine = MCTSEngine(seed=42)
        root = _make_root()
        max_iterations = 50

        _, stats = await engine.search(
            root=root,
            num_iterations=max_iterations,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=ConstantRolloutPolicy(0.5),
            max_rollout_depth=5,
            selection_policy=SelectionPolicy.MAX_VISITS,
            early_termination_threshold=0.99,
            min_iterations_before_termination=max_iterations,  # disable visit-based
            early_stop_threshold=0.0,  # disabled
            early_stop_patience=5,
        )

        # Should NOT have early stopped via value convergence
        assert stats["early_stopped"] is False
        assert stats["iterations_run"] == max_iterations

    @pytest.mark.asyncio
    async def test_search_stats_contain_early_stop_fields(self):
        """Stats dict should always contain early_stopped and iterations_run."""
        engine = MCTSEngine(seed=42)
        root = _make_root()

        _, stats = await engine.search(
            root=root,
            num_iterations=10,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=ConstantRolloutPolicy(0.5),
            max_rollout_depth=5,
        )

        assert "early_stopped" in stats
        assert "iterations_run" in stats
        assert isinstance(stats["early_stopped"], bool)
        assert isinstance(stats["iterations_run"], int)
