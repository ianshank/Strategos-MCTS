"""Unit tests for Regulatory Compliance use case.

Tests cover RegulatoryCompliance, RegulatoryComplianceState, and ComplianceGap
from src/enterprise/use_cases/regulatory_compliance/use_case.py.
"""

from __future__ import annotations

import pytest

from src.enterprise.config.enterprise_settings import RegulatoryComplianceConfig
from src.enterprise.use_cases.regulatory_compliance.use_case import (
    ComplianceGap,
    RegulatoryCompliance,
    RegulatoryComplianceState,
)


@pytest.fixture
def config() -> RegulatoryComplianceConfig:
    return RegulatoryComplianceConfig()


@pytest.fixture
def use_case(config: RegulatoryComplianceConfig) -> RegulatoryCompliance:
    return RegulatoryCompliance(config=config)


@pytest.mark.unit
class TestComplianceGap:
    """Tests for ComplianceGap dataclass."""

    def test_create_gap(self):
        gap = ComplianceGap(
            gap_id="g1",
            jurisdiction="US",
            regulation="SOX",
            description="Missing audit trail",
            severity="high",
            remediation_effort="medium",
        )
        assert gap.gap_id == "g1"
        assert gap.jurisdiction == "US"
        assert gap.severity == "high"
        assert gap.estimated_cost == 0.0
        assert gap.deadline is None

    def test_gap_with_optional_fields(self):
        gap = ComplianceGap(
            gap_id="g2",
            jurisdiction="EU",
            regulation="GDPR",
            description="Data retention policy",
            severity="critical",
            remediation_effort="high",
            estimated_cost=50000.0,
            deadline="2026-12-31",
        )
        assert gap.estimated_cost == 50000.0
        assert gap.deadline == "2026-12-31"


@pytest.mark.unit
class TestRegulatoryComplianceState:
    """Tests for RegulatoryComplianceState dataclass."""

    def test_defaults(self):
        state = RegulatoryComplianceState(state_id="rc_1")
        assert state.domain == "regulatory_compliance"
        assert state.organization_type == ""
        assert state.operating_jurisdictions == []
        assert state.regulations_analyzed == []
        assert state.gaps_identified == []
        assert state.overall_compliance_score == 0.0
        assert state.action_history == []

    def test_custom_state(self):
        state = RegulatoryComplianceState(
            state_id="rc_2",
            organization_type="bank",
            industry="financial_services",
            operating_jurisdictions=["US", "EU"],
        )
        assert state.organization_type == "bank"
        assert state.industry == "financial_services"
        assert len(state.operating_jurisdictions) == 2


@pytest.mark.unit
class TestRegulatoryCompliance:
    """Tests for RegulatoryCompliance use case."""

    def test_name(self, use_case):
        assert use_case.name == "regulatory_compliance"

    def test_domain(self, use_case):
        assert use_case.domain == "legal"

    def test_get_initial_state(self, use_case):
        state = use_case.get_initial_state(
            "Assess GDPR compliance",
            {"organization_type": "tech_company", "industry": "software"},
        )
        assert state.state_id.startswith("rc_")
        assert state.organization_type == "tech_company"
        assert state.industry == "software"
        assert state.features["query"] == "Assess GDPR compliance"

    def test_get_initial_state_defaults(self, use_case, config):
        state = use_case.get_initial_state("basic query", {})
        assert state.organization_type == "corporation"
        assert state.operating_jurisdictions == config.jurisdictions

    def test_get_initial_state_custom_jurisdictions(self, use_case):
        state = use_case.get_initial_state(
            "check compliance",
            {"jurisdictions": ["JP", "KR"]},
        )
        assert state.operating_jurisdictions == ["JP", "KR"]

    def test_get_available_actions(self, use_case):
        state = RegulatoryComplianceState(
            state_id="rc_1",
            operating_jurisdictions=["US", "EU"],
        )
        actions = use_case.get_available_actions(state)
        # Base actions + jurisdiction-specific
        assert "analyze_gdpr_requirements" in actions
        assert "analyze_sox_requirements" in actions
        assert "identify_gaps" in actions
        assert "deep_dive_us" in actions
        assert "deep_dive_eu" in actions

    def test_actions_filter_recent_history(self, use_case):
        state = RegulatoryComplianceState(
            state_id="rc_1",
            operating_jurisdictions=[],
            action_history=["identify_gaps", "prioritize_gaps", "assess_enforcement_risk"],
        )
        actions = use_case.get_available_actions(state)
        assert "identify_gaps" not in actions
        assert "prioritize_gaps" not in actions
        assert "assess_enforcement_risk" not in actions

    def test_apply_action_analyze_regulation(self, use_case):
        state = RegulatoryComplianceState(state_id="rc_1")
        new_state = use_case.apply_action(state, "analyze_gdpr_requirements")
        assert "GDPR" in new_state.regulations_analyzed
        assert "analyze_gdpr_requirements" in new_state.action_history
        assert new_state.features["regulations_analyzed"] == 1

    def test_apply_action_tracks_action_count(self, use_case):
        state = RegulatoryComplianceState(state_id="rc_1")
        new_state = use_case.apply_action(state, "identify_gaps")
        assert new_state.features["action_count"] == 1

    def test_apply_action_does_not_duplicate_regulations(self, use_case):
        state = RegulatoryComplianceState(
            state_id="rc_1",
            regulations_analyzed=["GDPR"],
        )
        new_state = use_case.apply_action(state, "analyze_gdpr_requirements")
        assert new_state.regulations_analyzed.count("GDPR") == 1

    def test_apply_action_sox(self, use_case):
        state = RegulatoryComplianceState(state_id="rc_1")
        new_state = use_case.apply_action(state, "analyze_sox_requirements")
        assert "SOX" in new_state.regulations_analyzed

    def test_apply_action_hipaa(self, use_case):
        state = RegulatoryComplianceState(state_id="rc_1")
        new_state = use_case.apply_action(state, "analyze_hipaa_requirements")
        assert "HIPAA" in new_state.regulations_analyzed

    def test_apply_action_non_analyze(self, use_case):
        """Non-analyze actions should not modify regulations_analyzed."""
        state = RegulatoryComplianceState(state_id="rc_1")
        new_state = use_case.apply_action(state, "identify_gaps")
        assert len(new_state.regulations_analyzed) == 0

    def test_state_id_changes(self, use_case):
        state = RegulatoryComplianceState(state_id="rc_1")
        new_state = use_case.apply_action(state, "identify_gaps")
        assert new_state.state_id != state.state_id

    def test_original_state_unchanged(self, use_case):
        state = RegulatoryComplianceState(state_id="rc_1")
        use_case.apply_action(state, "analyze_gdpr_requirements")
        assert len(state.regulations_analyzed) == 0
        assert len(state.action_history) == 0

    def test_default_config(self):
        """RegulatoryCompliance can be created without explicit config."""
        uc = RegulatoryCompliance()
        assert uc.name == "regulatory_compliance"
        assert uc.config is not None
