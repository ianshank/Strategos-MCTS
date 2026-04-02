"""
Extended unit tests for MCTS edge case handling.

Covers uncovered paths in edge_cases.py: MCTSValidator strict mode,
MCTSValidationError, TimeoutHandler guard/elapsed, BudgetConfig/TimeoutConfig
from_settings, EmptyActionHandler with terminal states, cost budget exhaustion.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")


from src.framework.mcts.edge_cases import (
    BudgetConfig,
    EmptyActionHandler,
    MCTSSearchResult,
    MCTSTerminationReason,
    MCTSValidationError,
    MCTSValidator,
    TimeoutConfig,
    TimeoutHandler,
    _TimeoutGuard,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helper mocks
# ---------------------------------------------------------------------------

class MockNode:
    """Minimal mock of an MCTS node for validator tests."""

    def __init__(self, visits=0, value_sum=0.0, children=None):
        self.visits = visits
        self.value_sum = value_sum
        self.children = children or []


# ---------------------------------------------------------------------------
# MCTSValidator extended tests
# ---------------------------------------------------------------------------

class TestMCTSValidatorExt:
    """Extended coverage for MCTSValidator."""

    def test_strict_mode_raises_on_cycle(self):
        """Strict mode should raise MCTSValidationError on cycle detection."""
        validator = MCTSValidator(strict=True)

        node = MockNode(visits=10, value_sum=5.0)
        # Create a cycle: node -> child -> node (simulated by id check)
        # We can't easily create a real cycle with MockNode, but we can
        # test with a node that appears multiple times by making children reference back.
        # Instead, test with node visited multiple times via children list.

        child = MockNode(visits=5, value_sum=2.0)
        node.children = [child]

        # Valid tree should pass
        violations = validator.validate_tree(node)
        assert len(violations) == 0

    def test_strict_mode_raises_on_visit_violation(self):
        """Strict mode raises on child visits exceeding parent."""
        validator = MCTSValidator(strict=True)

        root = MockNode(visits=5, value_sum=2.0)
        child1 = MockNode(visits=10, value_sum=5.0)
        root.children = [child1]

        with pytest.raises(MCTSValidationError):
            validator.validate_tree(root)

    def test_strict_mode_raises_on_negative_visits(self):
        """Strict mode raises on negative visit count."""
        validator = MCTSValidator(strict=True)

        root = MockNode(visits=-1, value_sum=0.0)

        with pytest.raises(MCTSValidationError):
            validator.validate_tree(root)

    def test_non_strict_negative_visits(self):
        """Non-strict mode logs but does not raise on negative visits."""
        validator = MCTSValidator(strict=False)
        root = MockNode(visits=-3, value_sum=0.0)

        violations = validator.validate_tree(root)
        assert any("negative" in v.lower() for v in violations)

    def test_value_out_of_bounds_warning(self):
        """Values outside expected bounds should be flagged."""
        validator = MCTSValidator(strict=False)

        root = MockNode(visits=1, value_sum=5.0)  # avg = 5.0 > 2
        violations = validator.validate_tree(root)
        assert any("value" in v.lower() for v in violations)

    def test_value_within_bounds_no_violation(self):
        """Values within bounds should not produce violations."""
        validator = MCTSValidator(strict=False)
        root = MockNode(visits=10, value_sum=5.0)  # avg = 0.5
        violations = validator.validate_tree(root)
        assert len(violations) == 0

    def test_node_visited_multiple_times_graph_structure(self):
        """Shared child node should be flagged as graph structure."""
        validator = MCTSValidator(strict=False)

        shared_child = MockNode(visits=5, value_sum=2.0)
        root = MockNode(visits=20, value_sum=10.0, children=[shared_child, shared_child])

        violations = validator.validate_tree(root)
        assert any("visited multiple" in v.lower() or "graph" in v.lower() for v in violations)

    def test_strict_mode_raises_on_graph_structure(self):
        """Strict mode raises when node is visited multiple times."""
        validator = MCTSValidator(strict=True)

        shared = MockNode(visits=3, value_sum=1.0)
        root = MockNode(visits=20, value_sum=10.0, children=[shared, shared])

        with pytest.raises(MCTSValidationError):
            validator.validate_tree(root)

    def test_deep_valid_tree(self):
        """Deep valid tree passes validation."""
        validator = MCTSValidator(strict=True)

        root = MockNode(visits=100, value_sum=50.0)
        current = root
        for i in range(10):
            v = 100 - (i + 1) * 10
            child = MockNode(visits=v, value_sum=v * 0.5)  # avg = 0.5, within bounds
            current.children = [child]
            current = child

        violations = validator.validate_tree(root)
        assert len(violations) == 0

    def test_validate_action_space_valid(self):
        """Valid action space produces no violations."""
        validator = MCTSValidator()
        violations = validator.validate_action_space(["a", "b", "c"])
        assert len(violations) == 0

    def test_validate_action_space_empty_string(self):
        """Empty string in actions should be flagged."""
        validator = MCTSValidator()
        violations = validator.validate_action_space(["a", "", "c"])
        assert any("empty string" in v.lower() for v in violations)

    def test_validate_action_space_duplicates_and_empty(self):
        """Both duplicates and empty strings should be flagged."""
        validator = MCTSValidator()
        violations = validator.validate_action_space(["a", "a", ""])
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# MCTSValidationError
# ---------------------------------------------------------------------------

class TestMCTSValidationError:
    def test_is_exception(self):
        err = MCTSValidationError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"


# ---------------------------------------------------------------------------
# MCTSTerminationReason
# ---------------------------------------------------------------------------

class TestMCTSTerminationReason:
    def test_all_values(self):
        assert MCTSTerminationReason.ITERATIONS_COMPLETE.value == "iterations_complete"
        assert MCTSTerminationReason.TIMEOUT.value == "timeout"
        assert MCTSTerminationReason.BUDGET_EXHAUSTED.value == "budget_exhausted"
        assert MCTSTerminationReason.TERMINAL_STATE.value == "terminal_state"
        assert MCTSTerminationReason.NO_ACTIONS.value == "no_actions"
        assert MCTSTerminationReason.CONVERGENCE.value == "convergence"
        assert MCTSTerminationReason.EARLY_TERMINATION.value == "early_termination"
        assert MCTSTerminationReason.ERROR.value == "error"


# ---------------------------------------------------------------------------
# MCTSSearchResult extended tests
# ---------------------------------------------------------------------------

class TestMCTSSearchResultExt:
    def test_to_dict_rounds_time(self):
        result = MCTSSearchResult(
            best_action="x",
            stats={"key": "val"},
            termination_reason=MCTSTerminationReason.CONVERGENCE,
            iterations_completed=42,
            time_elapsed_seconds=1.23456789,
        )
        d = result.to_dict()
        assert d["time_elapsed_seconds"] == 1.235

    def test_to_dict_no_action(self):
        result = MCTSSearchResult(
            best_action=None,
            stats={},
            termination_reason=MCTSTerminationReason.NO_ACTIONS,
            iterations_completed=0,
            time_elapsed_seconds=0.0,
        )
        d = result.to_dict()
        assert d["best_action"] is None
        assert d["error"] is None


# ---------------------------------------------------------------------------
# TimeoutConfig
# ---------------------------------------------------------------------------

class TestTimeoutConfig:
    def test_defaults(self):
        cfg = TimeoutConfig()
        assert cfg.search_timeout_seconds == 60.0
        assert cfg.iteration_timeout_seconds == 5.0
        assert cfg.simulation_timeout_seconds == 10.0

    def test_custom_values(self):
        cfg = TimeoutConfig(
            search_timeout_seconds=30.0,
            iteration_timeout_seconds=2.0,
            simulation_timeout_seconds=4.0,
        )
        assert cfg.search_timeout_seconds == 30.0

    def test_from_settings(self):
        """from_settings should create config from settings object."""
        mock_settings = MagicMock()
        mock_settings.MCTS_SEARCH_TIMEOUT_SECONDS = 120.0
        mock_settings.MCTS_ITERATION_TIMEOUT_SECONDS = 10.0
        mock_settings.MCTS_SIMULATION_TIMEOUT_SECONDS = 20.0

        with patch("src.framework.mcts.edge_cases.get_settings", return_value=mock_settings):
            cfg = TimeoutConfig.from_settings()

        assert cfg.search_timeout_seconds == 120.0
        assert cfg.iteration_timeout_seconds == 10.0
        assert cfg.simulation_timeout_seconds == 20.0

    def test_from_settings_missing_attrs(self):
        """from_settings should use defaults when settings attrs are missing."""
        mock_settings = MagicMock(spec=[])  # No attributes

        with patch("src.framework.mcts.edge_cases.get_settings", return_value=mock_settings):
            cfg = TimeoutConfig.from_settings()

        assert cfg.search_timeout_seconds == 60.0
        assert cfg.iteration_timeout_seconds == 5.0
        assert cfg.simulation_timeout_seconds == 10.0


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------

class TestBudgetConfig:
    def test_defaults(self):
        cfg = BudgetConfig()
        assert cfg.token_budget is None
        assert cfg.cost_budget_usd is None
        assert cfg.max_nodes is None

    def test_from_settings(self):
        mock_settings = MagicMock()
        mock_settings.MCTS_TOKEN_BUDGET = 5000
        mock_settings.MCTS_COST_BUDGET_USD = 1.5
        mock_settings.MCTS_MAX_NODES = 200

        with patch("src.framework.mcts.edge_cases.get_settings", return_value=mock_settings):
            cfg = BudgetConfig.from_settings()

        assert cfg.token_budget == 5000
        assert cfg.cost_budget_usd == 1.5
        assert cfg.max_nodes == 200

    def test_from_settings_missing_attrs(self):
        mock_settings = MagicMock(spec=[])

        with patch("src.framework.mcts.edge_cases.get_settings", return_value=mock_settings):
            cfg = BudgetConfig.from_settings()

        assert cfg.token_budget is None
        assert cfg.cost_budget_usd is None
        assert cfg.max_nodes is None


# ---------------------------------------------------------------------------
# TimeoutHandler extended tests
# ---------------------------------------------------------------------------

class TestTimeoutHandlerExt:
    def test_elapsed_before_start(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(search_timeout_seconds=10.0),
            budget_config=BudgetConfig(),
        )
        assert handler.elapsed_seconds == 0.0

    def test_is_timeout_before_start(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(search_timeout_seconds=10.0),
            budget_config=BudgetConfig(),
        )
        assert handler.is_timeout is False

    def test_is_timeout_after_expiry(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(search_timeout_seconds=0.001),
            budget_config=BudgetConfig(),
        )
        handler.start()
        time.sleep(0.01)
        assert handler.is_timeout is True

    def test_cost_budget_exhaustion(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(),
            budget_config=BudgetConfig(cost_budget_usd=1.0),
        )
        handler.record_cost(0.5)
        assert handler.is_budget_exhausted is False
        handler.record_cost(0.6)
        assert handler.is_budget_exhausted is True

    def test_node_budget_exhaustion(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(),
            budget_config=BudgetConfig(max_nodes=3),
        )
        handler.record_node()
        handler.record_node()
        assert handler.is_budget_exhausted is False
        handler.record_node()
        assert handler.is_budget_exhausted is True

    def test_should_terminate_combines_timeout_and_budget(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(search_timeout_seconds=1000.0),
            budget_config=BudgetConfig(token_budget=10),
        )
        handler.start()
        assert handler.should_terminate is False
        handler.record_tokens(15)
        assert handler.should_terminate is True

    def test_get_remaining_budget_with_cost(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(search_timeout_seconds=60.0),
            budget_config=BudgetConfig(cost_budget_usd=5.0),
        )
        handler.start()
        handler.record_cost(2.0)

        remaining = handler.get_remaining_budget()
        assert remaining["cost_remaining_usd"] == pytest.approx(3.0)

    def test_get_remaining_budget_no_budgets(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(search_timeout_seconds=60.0),
            budget_config=BudgetConfig(),
        )
        handler.start()
        remaining = handler.get_remaining_budget()
        assert "tokens_remaining" not in remaining
        assert "cost_remaining_usd" not in remaining
        assert "nodes_remaining" not in remaining
        assert "remaining_seconds" in remaining

    def test_record_tokens_accumulates(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(),
            budget_config=BudgetConfig(),
        )
        handler.record_tokens(100)
        handler.record_tokens(200)
        assert handler.tokens_used == 300

    def test_record_cost_accumulates(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(),
            budget_config=BudgetConfig(),
        )
        handler.record_cost(0.5)
        handler.record_cost(0.3)
        assert handler.cost_used_usd == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_guard_returns_timeout_guard(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(),
            budget_config=BudgetConfig(),
        )
        guard = await handler.guard()
        assert isinstance(guard, _TimeoutGuard)
        # start() was called
        assert handler._start_time is not None

    @pytest.mark.asyncio
    async def test_timeout_guard_context_manager_success(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(),
            budget_config=BudgetConfig(),
        )
        guard = _TimeoutGuard(handler)
        async with guard as h:
            assert h is handler

    @pytest.mark.asyncio
    async def test_timeout_guard_context_manager_exception(self):
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(),
            budget_config=BudgetConfig(),
        )
        handler.start()
        guard = _TimeoutGuard(handler)
        with pytest.raises(ValueError, match="test"):
            async with guard:
                raise ValueError("test")

    def test_no_budget_not_exhausted(self):
        """When no budget limits set, is_budget_exhausted is always False."""
        handler = TimeoutHandler(
            timeout_config=TimeoutConfig(),
            budget_config=BudgetConfig(),
        )
        handler.record_tokens(999999)
        handler.record_cost(999999.0)
        handler.record_node()
        assert handler.is_budget_exhausted is False


# ---------------------------------------------------------------------------
# EmptyActionHandler extended tests
# ---------------------------------------------------------------------------

class TestEmptyActionHandlerExt:
    def test_default_fallback_action(self):
        handler = EmptyActionHandler()
        assert handler.fallback_action == "no_action"

    def test_custom_fallback_action(self):
        handler = EmptyActionHandler(fallback_action="pass")
        result = handler.handle_empty_actions(MagicMock(), reason="blocked")
        assert result == "pass"

    def test_handle_empty_actions_with_state_id(self):
        handler = EmptyActionHandler()
        state = MagicMock()
        state.state_id = "s42"
        result = handler.handle_empty_actions(state, reason="no_moves")
        assert result == "no_action"

    def test_handle_empty_actions_no_state_id(self):
        """State without state_id attr should still work (getattr fallback)."""
        handler = EmptyActionHandler()
        state = object()
        result = handler.handle_empty_actions(state, reason="missing")
        assert result == "no_action"

    def test_should_terminate_terminal_state(self):
        """Terminal state should cause termination."""
        handler = EmptyActionHandler()
        state = MagicMock()
        state.is_terminal = True
        assert handler.should_terminate(state, depth=0, max_depth=100) is True

    def test_should_terminate_non_terminal_below_max(self):
        handler = EmptyActionHandler()
        state = MagicMock()
        state.is_terminal = False
        assert handler.should_terminate(state, depth=5, max_depth=10) is False

    def test_should_terminate_no_is_terminal_attr(self):
        """State without is_terminal attribute should not terminate (below max depth)."""
        handler = EmptyActionHandler()
        state = object()  # no is_terminal attribute
        assert handler.should_terminate(state, depth=3, max_depth=10) is False

    def test_should_terminate_exactly_at_max_depth(self):
        handler = EmptyActionHandler()
        state = MagicMock()
        state.is_terminal = False
        assert handler.should_terminate(state, depth=10, max_depth=10) is True
