"""Unit tests for Clinical Trial Design use case.

Tests cover ClinicalTrialDesign and ClinicalTrialState from
src/enterprise/use_cases/clinical_trial/use_case.py.
"""

from __future__ import annotations

import pytest

from src.enterprise.config.enterprise_settings import ClinicalTrialConfig
from src.enterprise.use_cases.clinical_trial.use_case import (
    ClinicalTrialDesign,
    ClinicalTrialState,
)


@pytest.fixture
def config() -> ClinicalTrialConfig:
    return ClinicalTrialConfig()


@pytest.fixture
def use_case(config: ClinicalTrialConfig) -> ClinicalTrialDesign:
    return ClinicalTrialDesign(config=config)


@pytest.mark.unit
class TestClinicalTrialState:
    """Tests for ClinicalTrialState dataclass."""

    def test_defaults(self):
        state = ClinicalTrialState(state_id="ct_1")
        assert state.domain == "clinical_trial"
        assert state.trial_phase == 1
        assert state.sample_size == 0
        assert state.statistical_power == 0.8
        assert state.alpha_level == 0.05
        assert state.action_history == []
        assert state.design_iterations == 0

    def test_custom_values(self):
        state = ClinicalTrialState(
            state_id="ct_2",
            trial_phase=3,
            indication="oncology",
            therapeutic_area="immuno-oncology",
            sample_size=500,
            duration_months=24,
        )
        assert state.trial_phase == 3
        assert state.indication == "oncology"
        assert state.sample_size == 500
        assert state.duration_months == 24


@pytest.mark.unit
class TestClinicalTrialDesign:
    """Tests for ClinicalTrialDesign use case."""

    def test_name(self, use_case):
        assert use_case.name == "clinical_trial_design"

    def test_domain(self, use_case):
        assert use_case.domain == "healthcare"

    def test_get_initial_state(self, use_case):
        state = use_case.get_initial_state(
            "Design phase 2 trial for diabetes drug",
            {"phase": 2, "indication": "T2D", "initial_sample_size": 200},
        )
        assert state.state_id.startswith("ct_")
        assert state.trial_phase == 2
        assert state.indication == "T2D"
        assert state.sample_size == 200
        assert state.features["query"] == "Design phase 2 trial for diabetes drug"

    def test_get_initial_state_defaults(self, use_case):
        state = use_case.get_initial_state("basic query", {})
        assert state.trial_phase == 2  # default from context.get
        assert state.sample_size == 100  # default
        assert state.duration_months == 12  # default

    def test_get_available_actions(self, use_case):
        state = ClinicalTrialState(state_id="ct_1")
        actions = use_case.get_available_actions(state)
        assert "adjust_sample_size_up" in actions
        assert "adjust_sample_size_down" in actions
        assert "extend_duration" in actions
        assert "shorten_duration" in actions
        assert "finalize_design" in actions
        assert len(actions) == 12  # All actions available initially

    def test_actions_filter_recent_history(self, use_case):
        state = ClinicalTrialState(
            state_id="ct_1",
            action_history=["extend_duration", "adjust_sample_size_up", "finalize_design"],
        )
        actions = use_case.get_available_actions(state)
        # Recent 3 actions should be filtered out
        assert "extend_duration" not in actions
        assert "adjust_sample_size_up" not in actions
        assert "finalize_design" not in actions

    def test_apply_action_sample_size_up(self, use_case, config):
        state = ClinicalTrialState(state_id="ct_1", sample_size=100)
        new_state = use_case.apply_action(state, "adjust_sample_size_up")
        expected = int(100 * config.sample_size_increase_factor)
        assert new_state.sample_size == expected
        assert "adjust_sample_size_up" in new_state.action_history
        assert new_state.design_iterations == 1
        assert state.sample_size == 100  # Original unchanged

    def test_apply_action_sample_size_down(self, use_case, config):
        state = ClinicalTrialState(state_id="ct_1", sample_size=100)
        new_state = use_case.apply_action(state, "adjust_sample_size_down")
        expected = max(config.min_sample_size, int(100 * config.sample_size_decrease_factor))
        assert new_state.sample_size == expected

    def test_apply_action_sample_size_down_min_clamped(self, use_case, config):
        state = ClinicalTrialState(state_id="ct_1", sample_size=config.min_sample_size)
        new_state = use_case.apply_action(state, "adjust_sample_size_down")
        assert new_state.sample_size >= config.min_sample_size

    def test_apply_action_extend_duration(self, use_case, config):
        state = ClinicalTrialState(state_id="ct_1", duration_months=12)
        new_state = use_case.apply_action(state, "extend_duration")
        assert new_state.duration_months == 12 + config.duration_adjustment_months

    def test_apply_action_shorten_duration(self, use_case, config):
        state = ClinicalTrialState(state_id="ct_1", duration_months=12)
        new_state = use_case.apply_action(state, "shorten_duration")
        expected = max(config.min_trial_duration_months, 12 - config.duration_adjustment_months)
        assert new_state.duration_months == expected

    def test_apply_action_shorten_duration_min_clamped(self, use_case, config):
        state = ClinicalTrialState(
            state_id="ct_1",
            duration_months=config.min_trial_duration_months,
        )
        new_state = use_case.apply_action(state, "shorten_duration")
        assert new_state.duration_months >= config.min_trial_duration_months

    def test_apply_action_unknown_action(self, use_case):
        """Unknown actions should not crash, just record in history."""
        state = ClinicalTrialState(state_id="ct_1", sample_size=100)
        new_state = use_case.apply_action(state, "unknown_action")
        assert "unknown_action" in new_state.action_history
        assert new_state.sample_size == 100  # Unchanged

    def test_apply_action_tracks_count(self, use_case):
        state = ClinicalTrialState(state_id="ct_1")
        new_state = use_case.apply_action(state, "extend_duration")
        assert new_state.features["action_count"] == 1

    def test_state_id_changes_on_action(self, use_case):
        state = ClinicalTrialState(state_id="ct_1")
        new_state = use_case.apply_action(state, "extend_duration")
        assert new_state.state_id != state.state_id
        assert new_state.state_id.startswith("ct_1_")

    def test_default_config(self):
        """ClinicalTrialDesign can be created without explicit config."""
        uc = ClinicalTrialDesign()
        assert uc.name == "clinical_trial_design"
        assert uc.config is not None
