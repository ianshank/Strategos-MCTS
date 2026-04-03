"""
Unit tests for M&A Due Diligence reward function.

Tests cover:
- Reward evaluation and clamping to [0, 1]
- Individual component computation (information gain, risk discovery, timeline efficiency)
- get_components debugging output
- Phase completion reward
- Configurable weights and parameters
- Diminishing returns for repeated actions
"""

from __future__ import annotations

import pytest

from src.enterprise.config.enterprise_settings import MADueDiligenceConfig
from src.enterprise.use_cases.ma_due_diligence.reward import MADueDiligenceReward
from src.enterprise.use_cases.ma_due_diligence.state import (
    AnalyzedDocument,
    IdentifiedRisk,
    MADueDiligenceState,
    RiskLevel,
)


def _make_config(**overrides) -> MADueDiligenceConfig:
    """Create a config with defaults, applying overrides."""
    return MADueDiligenceConfig(**overrides)


def _make_state(**kwargs) -> MADueDiligenceState:
    """Create a state with sensible defaults."""
    defaults = {
        "state_id": "test-state",
        "target_company": "TargetCo",
        "acquirer_company": "AcquirerCo",
    }
    defaults.update(kwargs)
    return MADueDiligenceState(**defaults)


def _make_risk(risk_id: str = "R001", severity: RiskLevel = RiskLevel.MEDIUM) -> IdentifiedRisk:
    return IdentifiedRisk(
        risk_id=risk_id,
        category="financial",
        description="Test risk",
        severity=severity,
        probability=0.5,
        impact=0.5,
    )


@pytest.mark.unit
class TestMADueDiligenceRewardInit:
    """Tests for reward function initialization."""

    def test_init_with_explicit_config(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        assert reward._config is config

    def test_init_with_custom_weights(self) -> None:
        config = _make_config()
        custom_weights = {"information_gain": 0.5, "risk_discovery": 0.3, "timeline_efficiency": 0.2}
        reward = MADueDiligenceReward(config=config, weights=custom_weights)
        assert reward._weights == custom_weights

    def test_init_uses_config_weights_by_default(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        assert reward._weights == config.reward_weights


@pytest.mark.unit
class TestRewardEvaluate:
    """Tests for the evaluate method."""

    def test_evaluate_returns_float(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        result = reward.evaluate(state, "deep_dive_revenue", {})
        assert isinstance(result, float)

    def test_evaluate_clamped_to_zero_one(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        for action in ["deep_dive_revenue", "review_contracts", "unknown_action"]:
            result = reward.evaluate(state, action, {})
            assert 0.0 <= result <= 1.0

    def test_evaluate_known_action_higher_than_unknown(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        known = reward.evaluate(state, "check_litigation_history", {})
        unknown = reward.evaluate(state, "some_unknown_action", {})
        assert known > unknown

    def test_evaluate_with_different_weights(self) -> None:
        config = _make_config()
        w1 = {"information_gain": 1.0, "risk_discovery": 0.0, "timeline_efficiency": 0.0}
        w2 = {"information_gain": 0.0, "risk_discovery": 1.0, "timeline_efficiency": 0.0}
        r1 = MADueDiligenceReward(config=config, weights=w1)
        r2 = MADueDiligenceReward(config=config, weights=w2)
        state = _make_state()
        # These should generally differ since different components are weighted
        v1 = r1.evaluate(state, "deep_dive_revenue", {})
        v2 = r2.evaluate(state, "deep_dive_revenue", {})
        # Both valid but potentially different
        assert 0.0 <= v1 <= 1.0
        assert 0.0 <= v2 <= 1.0


@pytest.mark.unit
class TestGetComponents:
    """Tests for the get_components debugging method."""

    def test_get_components_keys(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        components = reward.get_components(state, "review_contracts", {})
        assert "information_gain" in components
        assert "risk_discovery" in components
        assert "timeline_efficiency" in components
        assert "total_reward" in components

    def test_total_reward_matches_evaluate(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        action = "deep_dive_revenue"
        components = reward.get_components(state, action, {})
        direct = reward.evaluate(state, action, {})
        assert components["total_reward"] == pytest.approx(direct)


@pytest.mark.unit
class TestInformationGain:
    """Tests for the information gain component."""

    def test_known_action_base_gain(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        # With no action history, no decay applied
        components = reward.get_components(state, "check_litigation_history", {})
        assert components["information_gain"] == pytest.approx(0.85)

    def test_unknown_action_default_gain(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        components = reward.get_components(state, "totally_unknown", {})
        assert components["information_gain"] == pytest.approx(0.5)

    def test_diminishing_returns(self) -> None:
        config = _make_config(reward_decay_factor=0.8)
        reward = MADueDiligenceReward(config=config)
        state = _make_state(action_history=["deep_dive_revenue", "deep_dive_something"])
        # "deep" prefix matches both history items for "deep_dive_revenue"
        components = reward.get_components(state, "deep_dive_revenue", {})
        # Base 0.8 * 0.8^2 = 0.512
        assert components["information_gain"] == pytest.approx(0.8 * 0.8**2)

    def test_no_diminishing_returns_fresh_category(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state(action_history=["deep_dive_revenue"])
        # "review_contracts" starts with "review" - not in history
        components = reward.get_components(state, "review_contracts", {})
        assert components["information_gain"] == pytest.approx(0.75)


@pytest.mark.unit
class TestRiskDiscovery:
    """Tests for the risk discovery component."""

    def test_high_risk_action_base(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        components = reward.get_components(state, "check_litigation_history", {})
        # base 0.85 * multiplier 1.2 (0 risks < low_threshold=3) => 1.02, clamped to 1.0
        assert components["risk_discovery"] == pytest.approx(1.0)

    def test_medium_risk_action_base(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        components = reward.get_components(state, "deep_dive_revenue", {})
        # base 0.65 * multiplier 1.2 = 0.78
        assert components["risk_discovery"] == pytest.approx(0.78)

    def test_low_risk_action_base(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        components = reward.get_components(state, "escalate_to_expert", {})
        # base 0.4 * multiplier 1.2 = 0.48
        assert components["risk_discovery"] == pytest.approx(0.48)

    def test_many_risks_reduces_multiplier(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        # Add many risks to exceed high_threshold (default 6)
        for i in range(7):
            state.risks_identified.append(_make_risk(f"R{i}"))
        components = reward.get_components(state, "check_litigation_history", {})
        # base 0.85 * 0.8 = 0.68
        assert components["risk_discovery"] == pytest.approx(0.68)

    def test_medium_risk_count_multiplier(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        # 4 risks: between low_threshold(3) and high_threshold(6)
        for i in range(4):
            state.risks_identified.append(_make_risk(f"R{i}"))
        components = reward.get_components(state, "check_litigation_history", {})
        # base 0.85 * 1.0 = 0.85
        assert components["risk_discovery"] == pytest.approx(0.85)


@pytest.mark.unit
class TestTimelineEfficiency:
    """Tests for the timeline efficiency component."""

    def test_standard_action_early(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state(action_history=[])
        components = reward.get_components(state, "deep_dive_revenue", {})
        # 0 actions, depth_ratio=0 < early_threshold(0.6), penalty=1.0
        # standard action: 0.6 * 1.0 = 0.6
        assert components["timeline_efficiency"] == pytest.approx(0.6)

    def test_proceed_to_next_phase_with_coverage(self) -> None:
        config = _make_config(min_phase_coverage_actions=3)
        reward = MADueDiligenceReward(config=config)
        # 5 substantive actions in recent history (not phase transitions)
        history = ["action1", "action2", "action3", "action4", "action5"]
        state = _make_state(action_history=history)
        components = reward.get_components(state, "proceed_to_next_phase", {})
        # phase_actions >= 3, so 0.9 * time_penalty
        # 5 actions, max_analysis_depth=10*5=50, depth_ratio=5/50=0.1 < 0.6 => penalty=1.0
        assert components["timeline_efficiency"] == pytest.approx(0.9)

    def test_proceed_to_next_phase_without_coverage(self) -> None:
        config = _make_config(min_phase_coverage_actions=3)
        reward = MADueDiligenceReward(config=config)
        state = _make_state(action_history=["action1"])
        components = reward.get_components(state, "proceed_to_next_phase", {})
        # phase_actions=1 < 3, so 0.5 * time_penalty=1.0
        assert components["timeline_efficiency"] == pytest.approx(0.5)

    def test_revisit_previous_phase(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state(action_history=[])
        components = reward.get_components(state, "revisit_previous_phase", {})
        # 0.4 * 1.0
        assert components["timeline_efficiency"] == pytest.approx(0.4)

    def test_late_stage_penalty(self) -> None:
        config = _make_config(max_analysis_depth=10)  # max = 10*5 = 50
        reward = MADueDiligenceReward(config=config)
        # 45 actions => depth_ratio=45/50=0.9 > late_threshold(0.8)
        state = _make_state(action_history=["a"] * 45)
        components = reward.get_components(state, "deep_dive_revenue", {})
        # standard: 0.6 * late_penalty(0.3) = 0.18
        assert components["timeline_efficiency"] == pytest.approx(0.18)

    def test_mid_stage_penalty(self) -> None:
        config = _make_config(max_analysis_depth=10)  # max = 50
        reward = MADueDiligenceReward(config=config)
        # 35 actions => depth_ratio=35/50=0.7, between 0.6 and 0.8
        state = _make_state(action_history=["a"] * 35)
        components = reward.get_components(state, "deep_dive_revenue", {})
        # standard: 0.6 * mid_penalty(0.6) = 0.36
        assert components["timeline_efficiency"] == pytest.approx(0.36)


@pytest.mark.unit
class TestPhaseCompletionReward:
    """Tests for the phase completion bonus reward."""

    def test_empty_state(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        bonus = reward.get_phase_completion_reward(state)
        assert bonus == pytest.approx(0.0)

    def test_diverse_actions_bonus(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        actions = [f"action_{i}" for i in range(8)]
        state = _make_state(action_history=actions)
        bonus = reward.get_phase_completion_reward(state)
        # diversity: min(8/8, 1.0) * 0.2 = 0.2, risks=0, docs=0
        assert bonus == pytest.approx(0.2)

    def test_risk_coverage_bonus(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        for i in range(5):
            state.risks_identified.append(_make_risk(f"R{i}"))
        bonus = reward.get_phase_completion_reward(state)
        # diversity: 0, risks: min(5/5, 1.0)*0.15 = 0.15, docs: 0
        assert bonus == pytest.approx(0.15)

    def test_document_coverage_bonus(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        state = _make_state()
        for i in range(10):
            state.documents_analyzed.append(AnalyzedDocument(doc_id=f"D{i}", doc_type="contract"))
        bonus = reward.get_phase_completion_reward(state)
        # diversity: 0, risks: 0, docs: min(10/10, 1.0)*0.1 = 0.1
        assert bonus == pytest.approx(0.1)

    def test_combined_bonus(self) -> None:
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        actions = [f"unique_action_{i}" for i in range(8)]
        state = _make_state(action_history=actions)
        for i in range(5):
            state.risks_identified.append(_make_risk(f"R{i}"))
        for i in range(10):
            state.documents_analyzed.append(AnalyzedDocument(doc_id=f"D{i}", doc_type="contract"))
        bonus = reward.get_phase_completion_reward(state)
        assert bonus == pytest.approx(0.2 + 0.15 + 0.1)

    def test_capped_values(self) -> None:
        """Bonus components should be capped even with large inputs."""
        config = _make_config()
        reward = MADueDiligenceReward(config=config)
        actions = [f"action_{i}" for i in range(50)]  # Way more than 8
        state = _make_state(action_history=actions)
        for i in range(100):
            state.risks_identified.append(_make_risk(f"R{i}"))
        for i in range(200):
            state.documents_analyzed.append(AnalyzedDocument(doc_id=f"D{i}", doc_type="contract"))
        bonus = reward.get_phase_completion_reward(state)
        # Max: 0.2 + 0.15 + 0.1 = 0.45
        assert bonus == pytest.approx(0.45)
