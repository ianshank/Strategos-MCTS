"""
Extended unit tests for GameState implementations to improve coverage.

Covers missed lines in:
- ReasoningState: mid/late actions, verify/generalize/specialize/backtrack/clarification,
  canonical form, action_to_index edge cases, to_tensor encoding
- PlanningState: get_legal_actions, _get_action_cost, apply_action, is_terminal,
  get_reward, to_tensor, get_hash, action_to_index
- DecisionState: get_legal_actions edge cases, apply_action paths, is_terminal,
  get_reward, to_tensor with evaluated options, get_hash
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Set environment variables before importing modules
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

try:
    import torch

    from src.framework.mcts.game_states import (
        DecisionState,
        PlanningState,
        ReasoningState,
        create_game_state,
    )

    GAME_STATES_AVAILABLE = True
except ImportError:
    GAME_STATES_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not GAME_STATES_AVAILABLE, reason="GameStates module not available"),
]


class TestReasoningStateMidActions:
    """Test ReasoningState mid-stage legal actions (lines 84-96)."""

    def test_mid_stage_actions(self):
        """Mid reasoning stage should return verify, generalize, infer."""
        state = ReasoningState(
            problem="Test problem",
            reasoning_steps=["s1", "s2", "s3", "s4"],  # 4 steps, 0.3*10=3 so >3 -> mid
            confidence=0.5,
            max_steps=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "verify" in action_types
        assert "generalize" in action_types
        assert "infer" in action_types

    def test_mid_stage_backtrack_low_confidence(self):
        """Mid stage with low confidence and >1 step should allow backtracking."""
        state = ReasoningState(
            problem="Test problem",
            reasoning_steps=["s1", "s2", "s3", "s4"],  # 4 steps, mid stage
            confidence=0.2,  # < 0.3
            max_steps=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "backtrack" in action_types

    def test_mid_stage_no_backtrack_high_confidence(self):
        """Mid stage with high confidence should not allow backtracking."""
        state = ReasoningState(
            problem="Test problem",
            reasoning_steps=["s1", "s2", "s3", "s4"],
            confidence=0.5,  # >= 0.3
            max_steps=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "backtrack" not in action_types

    def test_late_stage_actions_high_confidence(self):
        """Late reasoning stage with high confidence should only have conclude."""
        state = ReasoningState(
            problem="Test problem",
            reasoning_steps=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],  # 8 steps, 0.7*10=7 -> late
            confidence=0.9,  # >= 0.8
            max_steps=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "conclude" in action_types
        # High confidence, no verify
        assert "verify" not in action_types

    def test_late_stage_actions_low_confidence(self):
        """Late reasoning stage with low confidence should have conclude and verify."""
        state = ReasoningState(
            problem="Test problem",
            reasoning_steps=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
            confidence=0.5,  # < 0.8
            max_steps=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "conclude" in action_types
        assert "verify" in action_types


class TestReasoningStateClarification:
    """Test ask_clarification action (line 99-100)."""

    def test_unclear_problem_adds_clarification(self):
        """Problems with 'unclear' should add ask_clarification action."""
        state = ReasoningState(
            problem="This is an unclear problem statement",
            max_steps=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "ask_clarification" in action_types

    def test_ambiguous_problem_adds_clarification(self):
        """Problems with 'ambiguous' should add ask_clarification action."""
        state = ReasoningState(
            problem="This is an ambiguous question",
            max_steps=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "ask_clarification" in action_types

    def test_clear_problem_no_clarification(self):
        """Clear problems should not add ask_clarification."""
        state = ReasoningState(
            problem="What is 2 + 2?",
            max_steps=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "ask_clarification" not in action_types


class TestReasoningStateApplyActions:
    """Test all action types in apply_action (lines 136-160)."""

    def test_apply_verify_action(self):
        """Verify action adds step and increases confidence."""
        state = ReasoningState(
            problem="Test",
            current_hypothesis="hypothesis A",
            confidence=0.3,
            max_steps=10,
        )
        new_state = state.apply_action({"type": "verify"})
        assert "[VERIFY]" in new_state.reasoning_steps[-1]
        assert new_state.confidence == pytest.approx(0.5)  # 0.3 + 0.2

    def test_apply_generalize_action(self):
        """Generalize action adds step and increases confidence."""
        state = ReasoningState(
            problem="Test",
            confidence=0.3,
            max_steps=10,
        )
        new_state = state.apply_action({"type": "generalize"})
        assert "[GENERALIZE]" in new_state.reasoning_steps[-1]
        assert new_state.confidence == pytest.approx(0.4)  # 0.3 + 0.1

    def test_apply_specialize_action(self):
        """Specialize action adds step and increases confidence."""
        state = ReasoningState(
            problem="Test",
            confidence=0.3,
            max_steps=10,
        )
        new_state = state.apply_action({"type": "specialize"})
        assert "[SPECIALIZE]" in new_state.reasoning_steps[-1]
        assert new_state.confidence == pytest.approx(0.45)  # 0.3 + 0.15

    def test_apply_ask_clarification_action(self):
        """ask_clarification action adds step without changing confidence."""
        state = ReasoningState(
            problem="Test",
            confidence=0.5,
            max_steps=10,
        )
        new_state = state.apply_action({"type": "ask_clarification"})
        assert "[CLARIFY]" in new_state.reasoning_steps[-1]
        assert new_state.confidence == 0.5

    def test_apply_backtrack_empty_steps(self):
        """Backtrack with no steps should not crash."""
        state = ReasoningState(
            problem="Test",
            reasoning_steps=[],
            confidence=0.5,
            max_steps=10,
        )
        new_state = state.apply_action({"type": "backtrack"})
        assert len(new_state.reasoning_steps) == 0
        assert new_state.confidence == pytest.approx(0.4)  # 0.5 - 0.1

    def test_apply_backtrack_with_steps(self):
        """Backtrack removes last step and reduces confidence."""
        state = ReasoningState(
            problem="Test",
            reasoning_steps=["step1", "step2"],
            confidence=0.5,
            max_steps=10,
        )
        new_state = state.apply_action({"type": "backtrack"})
        assert len(new_state.reasoning_steps) == 1
        assert new_state.reasoning_steps[0] == "step1"
        assert new_state.confidence == pytest.approx(0.4)

    def test_apply_backtrack_confidence_floor(self):
        """Backtrack confidence should not go below 0."""
        state = ReasoningState(
            problem="Test",
            reasoning_steps=["step1"],
            confidence=0.05,
            max_steps=10,
        )
        new_state = state.apply_action({"type": "backtrack"})
        assert new_state.confidence == pytest.approx(0.0)

    def test_apply_default_action_type(self):
        """Action without type defaults to infer."""
        state = ReasoningState(problem="Test", max_steps=10)
        new_state = state.apply_action({})
        assert "[INFER]" in new_state.reasoning_steps[-1]

    def test_confidence_capped_at_one(self):
        """Confidence should not exceed 1.0."""
        state = ReasoningState(
            problem="Test",
            confidence=0.95,
            max_steps=10,
        )
        new_state = state.apply_action({"type": "verify"})  # +0.2
        assert new_state.confidence == 1.0


class TestReasoningStateCanonicalAndIndex:
    """Test canonical form and action_to_index edge cases (lines 246, 258-259)."""

    def test_get_canonical_form_returns_self(self):
        """Canonical form returns same state for reasoning."""
        state = ReasoningState(problem="Test", max_steps=10)
        canonical = state.get_canonical_form(player=1)
        assert canonical is state

    def test_action_to_index_known_type(self):
        """Known action type returns correct index."""
        state = ReasoningState(problem="Test", max_steps=10)
        assert state.action_to_index({"type": "decompose"}) == 0
        assert state.action_to_index({"type": "infer"}) == 1
        assert state.action_to_index({"type": "verify"}) == 2
        assert state.action_to_index({"type": "conclude"}) == 5

    def test_action_to_index_unknown_type(self):
        """Unknown action type returns 0."""
        state = ReasoningState(problem="Test", max_steps=10)
        assert state.action_to_index({"type": "unknown_action"}) == 0

    def test_action_to_index_missing_type(self):
        """Missing type defaults to infer (index 1)."""
        state = ReasoningState(problem="Test", max_steps=10)
        assert state.action_to_index({}) == 1  # defaults to "infer"


class TestReasoningStateToTensor:
    """Test to_tensor encoding details (line 233)."""

    def test_to_tensor_encodes_progress(self):
        """Tensor should encode step progress in feature[0]."""
        state = ReasoningState(
            problem="Test",
            reasoning_steps=["[INFER] step1", "[VERIFY] step2"],
            confidence=0.7,
            max_steps=10,
        )
        tensor = state.to_tensor()
        assert tensor[0] == pytest.approx(2.0 / 10.0)
        assert tensor[1] == pytest.approx(0.7)

    def test_to_tensor_action_type_distribution(self):
        """Tensor should encode action type distribution."""
        state = ReasoningState(
            problem="Test" * 300,  # long problem for feature[10]
            reasoning_steps=["[INFER] step1", "[INFER] step2", "[VERIFY] step3"],
            max_steps=10,
        )
        tensor = state.to_tensor()
        # infer is at index 1 in _action_types, so feature[3] = count/total
        # 2 infer steps out of 3 = 0.667
        assert tensor[3] == pytest.approx(2.0 / 3.0, abs=0.01)
        # feature[10] = problem_length / 1000 capped at 1.0
        assert tensor[10] == pytest.approx(1.0)

    def test_to_tensor_terminal_indicator(self):
        """Tensor feature[11] should be 1.0 for terminal state."""
        state = ReasoningState(
            problem="Test",
            reasoning_steps=["[CONCLUDE] done"],
            max_steps=10,
        )
        tensor = state.to_tensor()
        assert tensor[11] == 1.0


class TestReasoningStateReward:
    """Test reward calculation details."""

    def test_reward_with_backtrack_penalty(self):
        """Reward should be penalized for backtracking steps."""
        state = ReasoningState(
            problem="Test",
            reasoning_steps=["[BACKTRACK] undo", "[BACKTRACK] undo2", "[CONCLUDE] done"],
            confidence=0.8,
            max_steps=10,
        )
        reward = state.get_reward()
        # base = 0.8, efficiency = (1.0 - 3/10)*0.2 = 0.14, backtrack = 2*0.05 = 0.1
        expected = 0.8 + 0.14 - 0.1
        assert reward == pytest.approx(expected, abs=0.01)


# ============================================================================
# PlanningState extended tests
# ============================================================================


class TestPlanningStateGetLegalActions:
    """Test PlanningState.get_legal_actions (lines 292-314)."""

    def test_actions_filtered_by_resources(self):
        """Actions requiring more resources than available should be filtered."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze", "execute"],
            resources={"time": 0.5, "compute": 0.1},  # low resources
            time_remaining=30.0,
            max_actions=10,
        )
        actions = state.get_legal_actions()
        action_names = [a["name"] for a in actions]
        # analyze costs time=1.0, compute=0.5 -> can't afford
        # execute costs time=2.0, compute=1.0 -> can't afford
        assert "analyze" not in action_names
        assert "execute" not in action_names
        # Should fallback to "wait"
        assert "wait" in action_names

    def test_actions_respect_max_actions(self):
        """No actions when max_actions reached."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze"],
            resources={"time": 10.0, "compute": 5.0},
            completed_actions=[{"name": f"a{i}"} for i in range(10)],
            time_remaining=30.0,
            max_actions=10,
        )
        actions = state.get_legal_actions()
        action_names = [a["name"] for a in actions]
        # Can't do analyze because completed_actions >= max_actions
        assert "analyze" not in action_names
        assert "wait" in action_names

    def test_actions_with_sufficient_resources(self):
        """Actions available when resources are sufficient."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze", "verify"],
            resources={"time": 10.0, "compute": 5.0},
            time_remaining=30.0,
            max_actions=10,
        )
        actions = state.get_legal_actions()
        action_names = [a["name"] for a in actions]
        assert "analyze" in action_names
        assert "verify" in action_names

    def test_actions_include_cost(self):
        """Each action should include its cost."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze"],
            resources={"time": 10.0, "compute": 5.0},
            time_remaining=30.0,
            max_actions=10,
        )
        actions = state.get_legal_actions()
        analyze_action = next(a for a in actions if a["name"] == "analyze")
        assert analyze_action["cost"] == {"time": 1.0, "compute": 0.5}


class TestPlanningStateGetActionCost:
    """Test PlanningState._get_action_cost (lines 319-327)."""

    def test_known_action_costs(self):
        """Known actions should return their specific costs."""
        state = PlanningState(goal="T", current_state="S", time_remaining=30.0, max_actions=10)
        assert state._get_action_cost("analyze") == {"time": 1.0, "compute": 0.5}
        assert state._get_action_cost("execute") == {"time": 2.0, "compute": 1.0}
        assert state._get_action_cost("verify") == {"time": 0.5, "compute": 0.2}
        assert state._get_action_cost("optimize") == {"time": 1.5, "compute": 0.8}
        assert state._get_action_cost("wait") == {}
        assert state._get_action_cost("finish") == {}

    def test_unknown_action_default_cost(self):
        """Unknown actions should return default cost."""
        state = PlanningState(goal="T", current_state="S", time_remaining=30.0, max_actions=10)
        assert state._get_action_cost("unknown_action") == {"time": 1.0}


class TestPlanningStateApplyAction:
    """Test PlanningState.apply_action (lines 331-346)."""

    def test_apply_action_deducts_resources(self):
        """Applying action should deduct resources from cost."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze"],
            resources={"time": 10.0, "compute": 5.0},
            time_remaining=30.0,
            max_actions=10,
        )
        action = {"name": "analyze", "cost": {"time": 1.0, "compute": 0.5}, "step": 0}
        new_state = state.apply_action(action)
        assert new_state.resources["time"] == pytest.approx(9.0)
        assert new_state.resources["compute"] == pytest.approx(4.5)

    def test_apply_action_records_completed(self):
        """Applied action should be recorded in completed_actions."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze"],
            resources={"time": 10.0},
            time_remaining=30.0,
            max_actions=10,
        )
        action = {"name": "analyze", "cost": {"time": 1.0}, "step": 0}
        new_state = state.apply_action(action)
        assert len(new_state.completed_actions) == 1
        assert new_state.completed_actions[0] == action

    def test_apply_action_updates_current_state(self):
        """Current state description should be updated."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze"],
            resources={"time": 10.0},
            time_remaining=30.0,
            max_actions=10,
        )
        action = {"name": "analyze", "cost": {}, "step": 0}
        new_state = state.apply_action(action)
        assert new_state.current_state == "Start -> analyze"

    def test_apply_action_updates_time_remaining(self):
        """Time remaining should decrease by action's time cost."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["execute"],
            resources={"time": 10.0, "compute": 5.0},
            time_remaining=20.0,
            max_actions=10,
        )
        action = {"name": "execute", "cost": {"time": 2.0, "compute": 1.0}, "step": 0}
        new_state = state.apply_action(action)
        assert new_state.time_remaining == pytest.approx(18.0)

    def test_apply_wait_action(self):
        """Wait action with empty cost should not change resources."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=[],
            resources={"time": 10.0},
            time_remaining=30.0,
            max_actions=10,
        )
        action = {"name": "wait", "cost": {}, "step": 0}
        new_state = state.apply_action(action)
        assert new_state.resources["time"] == 10.0


class TestPlanningStateIsTerminal:
    """Test PlanningState.is_terminal (lines 360-368)."""

    def test_terminal_finish_action(self):
        """State is terminal when last action is 'finish'."""
        state = PlanningState(
            goal="Test",
            current_state="Done",
            completed_actions=[{"name": "analyze"}, {"name": "finish"}],
            time_remaining=30.0,
            max_actions=10,
        )
        assert state.is_terminal() is True

    def test_terminal_time_exhausted(self):
        """State is terminal when time_remaining <= 0."""
        state = PlanningState(
            goal="Test",
            current_state="Running",
            time_remaining=0.0,
            max_actions=10,
        )
        assert state.is_terminal() is True

    def test_terminal_negative_time(self):
        """State is terminal when time_remaining is negative."""
        state = PlanningState(
            goal="Test",
            current_state="Running",
            time_remaining=-1.0,
            max_actions=10,
        )
        assert state.is_terminal() is True

    def test_terminal_max_actions_reached(self):
        """State is terminal when completed_actions >= max_actions."""
        state = PlanningState(
            goal="Test",
            current_state="Running",
            completed_actions=[{"name": f"a{i}"} for i in range(5)],
            time_remaining=10.0,
            max_actions=5,
        )
        assert state.is_terminal() is True

    def test_not_terminal_ongoing(self):
        """State is not terminal when still running."""
        state = PlanningState(
            goal="Test",
            current_state="Running",
            completed_actions=[{"name": "a1"}],
            time_remaining=10.0,
            max_actions=5,
        )
        assert state.is_terminal() is False


class TestPlanningStateGetReward:
    """Test PlanningState.get_reward (lines 372-387)."""

    def test_reward_non_terminal(self):
        """Non-terminal state should return 0."""
        state = PlanningState(
            goal="Test",
            current_state="Running",
            time_remaining=10.0,
            max_actions=10,
        )
        assert state.get_reward() == 0.0

    def test_reward_with_finish(self):
        """Finishing with 'finish' gives higher base reward."""
        state = PlanningState(
            goal="Test",
            current_state="Done",
            completed_actions=[{"name": "finish"}],
            resources={"time": 0.0},
            time_remaining=0.0,
            max_actions=10,
        )
        reward = state.get_reward()
        # base=0.8, efficiency=(1 - 1/10)*0.2=0.18, resource_penalty=0
        assert reward == pytest.approx(0.98, abs=0.01)

    def test_reward_without_finish(self):
        """Terminating without 'finish' gives lower base reward."""
        state = PlanningState(
            goal="Test",
            current_state="Timeout",
            completed_actions=[{"name": "analyze"}],
            resources={"time": 0.0},
            time_remaining=0.0,
            max_actions=10,
        )
        reward = state.get_reward()
        # base=0.3, efficiency=(1 - 1/10)*0.2=0.18
        assert reward == pytest.approx(0.48, abs=0.01)

    def test_reward_resource_penalty(self):
        """Wasted resources should reduce reward."""
        state = PlanningState(
            goal="Test",
            current_state="Done",
            completed_actions=[{"name": "finish"}],
            resources={"time": 5.0, "compute": 3.0},  # 8.0 total, penalty = (8-1)*0.1 = 0.7
            time_remaining=0.0,
            max_actions=10,
        )
        reward = state.get_reward()
        # base=0.8, efficiency=(1-1/10)*0.2=0.18, penalty=0.7
        expected = max(0.0, min(1.0, 0.8 + 0.18 - 0.7))
        assert reward == pytest.approx(expected, abs=0.01)


class TestPlanningStateToTensor:
    """Test PlanningState.to_tensor (lines 391-413)."""

    def _mock_settings(self):
        """Create mock settings with MCTS_SEARCH_TIMEOUT."""
        mock = MagicMock()
        mock.STATE_FEATURE_DIM = 128
        mock.MCTS_SEARCH_TIMEOUT = 60.0
        return mock

    def test_to_tensor_shape(self):
        """Tensor should have correct shape."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze", "verify"],
            resources={"time": 5.0, "compute": 3.0},
            time_remaining=30.0,
            max_actions=10,
        )
        with patch("src.framework.mcts.game_states.get_settings", return_value=self._mock_settings()):
            tensor = state.to_tensor()
        assert tensor.dim() == 1
        assert tensor.shape[0] >= 14

    def test_to_tensor_progress(self):
        """Tensor should encode progress."""
        state = PlanningState(
            goal="Test",
            current_state="Mid",
            completed_actions=[{"name": "a1"}, {"name": "a2"}, {"name": "a3"}],
            resources={"time": 5.0},
            available_actions=["analyze"],
            time_remaining=30.0,
            max_actions=10,
        )
        with patch("src.framework.mcts.game_states.get_settings", return_value=self._mock_settings()):
            tensor = state.to_tensor()
        assert tensor[0] == pytest.approx(3.0 / 10.0)

    def test_to_tensor_resource_encoding(self):
        """Resources should be encoded in tensor."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            resources={"time": 5.0, "compute": 3.0},
            time_remaining=30.0,
            max_actions=10,
        )
        with patch("src.framework.mcts.game_states.get_settings", return_value=self._mock_settings()):
            tensor = state.to_tensor()
        # Resources at indices 2+
        assert tensor[2] == pytest.approx(min(1.0, 5.0 / 10.0))

    def test_to_tensor_terminal_indicator(self):
        """Terminal indicator at feature[13]."""
        state = PlanningState(
            goal="Test",
            current_state="Done",
            completed_actions=[{"name": "finish"}],
            time_remaining=0.0,
            max_actions=10,
        )
        with patch("src.framework.mcts.game_states.get_settings", return_value=self._mock_settings()):
            tensor = state.to_tensor()
        assert tensor[13] == 1.0


class TestPlanningStateHashAndIndex:
    """Test PlanningState.get_hash and action_to_index (lines 417-428)."""

    def test_get_hash_deterministic(self):
        """Hash should be deterministic."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            completed_actions=[{"name": "a1"}],
            resources={"time": 5.0, "compute": 3.0},
            time_remaining=30.0,
            max_actions=10,
        )
        h1 = state.get_hash()
        h2 = state.get_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_get_hash_different_states(self):
        """Different states produce different hashes."""
        s1 = PlanningState(goal="Goal1", current_state="S", time_remaining=30.0, max_actions=10)
        s2 = PlanningState(goal="Goal2", current_state="S", time_remaining=30.0, max_actions=10)
        assert s1.get_hash() != s2.get_hash()

    def test_action_to_index_known(self):
        """Known action returns its index in available_actions."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze", "execute", "verify"],
            time_remaining=30.0,
            max_actions=10,
        )
        assert state.action_to_index({"name": "analyze"}) == 0
        assert state.action_to_index({"name": "execute"}) == 1
        assert state.action_to_index({"name": "verify"}) == 2

    def test_action_to_index_unknown(self):
        """Unknown action returns len(available_actions)."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze", "execute"],
            time_remaining=30.0,
            max_actions=10,
        )
        assert state.action_to_index({"name": "unknown"}) == 2

    def test_action_to_index_default_wait(self):
        """Missing name defaults to 'wait'."""
        state = PlanningState(
            goal="Test",
            current_state="Start",
            available_actions=["analyze"],
            time_remaining=30.0,
            max_actions=10,
        )
        # "wait" not in available_actions, so returns len
        assert state.action_to_index({}) == 1


# ============================================================================
# DecisionState extended tests
# ============================================================================


class TestDecisionStateGetLegalActions:
    """Test DecisionState.get_legal_actions edge cases (lines 462-471)."""

    def test_compare_available_with_two_evaluated(self):
        """Compare action should be available with >= 2 evaluated options."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
            evaluated_options={"a": 0.5, "b": 0.7},
            max_evaluations=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "compare" in action_types
        compare_action = next(a for a in actions if a["type"] == "compare")
        assert set(compare_action["options"]) == {"a", "b"}

    def test_decide_available_with_evaluated(self):
        """Decide action should be available when options are evaluated."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}, {"id": "b"}],
            evaluated_options={"a": 0.5},
            max_evaluations=10,
        )
        actions = state.get_legal_actions()
        action_types = [a["type"] for a in actions]
        assert "decide" in action_types
        decide_action = next(a for a in actions if a["type"] == "decide")
        assert decide_action["best_option"] == "a"

    def test_decide_selects_best_option(self):
        """Decide action should select option with highest score."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}, {"id": "b"}],
            evaluated_options={"a": 0.3, "b": 0.9},
            max_evaluations=10,
        )
        actions = state.get_legal_actions()
        decide_action = next(a for a in actions if a["type"] == "decide")
        assert decide_action["best_option"] == "b"

    def test_fallback_decide_when_no_actions(self):
        """When no evaluate/compare/decide actions, fallback decide with None."""
        state = DecisionState(
            context="Test",
            options=[],
            evaluated_options={},
            max_evaluations=10,
        )
        actions = state.get_legal_actions()
        assert len(actions) == 1
        assert actions[0]["type"] == "decide"
        assert actions[0]["best_option"] is None


class TestDecisionStateApplyAction:
    """Test DecisionState.apply_action paths (lines 497-500)."""

    def test_apply_decide_action_records_final(self):
        """Decide action should record final=True in history."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}],
            evaluated_options={"a": 0.8},
            max_evaluations=10,
        )
        new_state = state.apply_action({"type": "decide", "best_option": "a"})
        assert new_state.decision_history[-1]["final"] is True
        assert new_state.decision_history[-1]["selected"] == "a"

    def test_apply_compare_action(self):
        """Compare action should record comparison in history."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}, {"id": "b"}],
            evaluated_options={"a": 0.5, "b": 0.7},
            max_evaluations=10,
        )
        new_state = state.apply_action({"type": "compare", "options": ["a", "b"]})
        assert new_state.decision_history[-1]["action"] == "compare"
        assert new_state.decision_history[-1]["result"] == "compared"


class TestDecisionStateIsTerminal:
    """Test DecisionState.is_terminal edge cases."""

    def test_terminal_max_evaluations(self):
        """Terminal when evaluated_options >= max_evaluations."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}, {"id": "b"}],
            evaluated_options={"a": 0.5, "b": 0.7},
            max_evaluations=2,
        )
        assert state.is_terminal() is True

    def test_not_terminal_partial_evaluation(self):
        """Not terminal when evaluated < max and no final decision."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}, {"id": "b"}],
            evaluated_options={"a": 0.5},
            max_evaluations=5,
        )
        assert state.is_terminal() is False


class TestDecisionStateGetReward:
    """Test DecisionState.get_reward (lines 518, 521)."""

    def test_reward_non_terminal(self):
        """Non-terminal returns 0."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}],
            max_evaluations=10,
        )
        assert state.get_reward() == 0.0

    def test_reward_no_evaluated_options(self):
        """Terminal with no evaluated options returns 0."""
        state = DecisionState(
            context="Test",
            options=[],
            evaluated_options={},
            decision_history=[{"final": True}],
            max_evaluations=10,
        )
        assert state.get_reward() == 0.0

    def test_reward_with_efficiency_bonus(self):
        """Reward includes efficiency bonus for fewer evaluations."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
            evaluated_options={"a": 0.8},
            decision_history=[{"final": True}],
            max_evaluations=10,
        )
        reward = state.get_reward()
        # best_score = 0.8
        # efficiency = 1.0 - 1/4 = 0.75, bonus = 0.75 * 0.1 = 0.075
        expected = min(1.0, 0.8 + 0.075)
        assert reward == pytest.approx(expected, abs=0.01)


class TestDecisionStateToTensor:
    """Test DecisionState.to_tensor (lines 544-545, 560-563)."""

    def test_to_tensor_with_evaluated_options(self):
        """Tensor should encode evaluated option statistics."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}, {"id": "b"}],
            evaluated_options={"a": 0.6, "b": 0.8},
            max_evaluations=10,
        )
        tensor = state.to_tensor()
        # features[0] = progress = 2/2 = 1.0
        assert tensor[0] == pytest.approx(1.0)
        # features[1] = max score = 0.8
        assert tensor[1] == pytest.approx(0.8)
        # features[2] = avg score = 0.7
        assert tensor[2] == pytest.approx(0.7)
        # features[3] = remaining fraction = 0/2 = 0.0
        assert tensor[3] == pytest.approx(0.0)

    def test_to_tensor_no_evaluated(self):
        """Tensor with no evaluated options should have zeros for scores."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}],
            max_evaluations=10,
        )
        tensor = state.to_tensor()
        assert tensor[1] == 0.0
        assert tensor[2] == 0.0


class TestDecisionStateHashExt:
    """Test DecisionState.get_hash (lines 560-563)."""

    def test_hash_includes_evaluated_and_history(self):
        """Hash should incorporate evaluated options and history."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}],
            evaluated_options={"a": 0.500},
            decision_history=[{"action": "evaluate", "option": "a"}],
            max_evaluations=10,
        )
        h = state.get_hash()
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_different_for_different_evaluations(self):
        """Different evaluations produce different hashes."""
        s1 = DecisionState(
            context="Test",
            evaluated_options={"a": 0.5},
            max_evaluations=10,
        )
        s2 = DecisionState(
            context="Test",
            evaluated_options={"a": 0.9},
            max_evaluations=10,
        )
        assert s1.get_hash() != s2.get_hash()
