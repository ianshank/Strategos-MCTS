"""
Unit tests for M&A Due Diligence state definitions.

Tests cover:
- DueDiligencePhase enum methods
- RiskLevel enum and weight mapping
- IdentifiedRisk dataclass and risk score calculation
- AnalyzedDocument dataclass and serialization
- SynergyOpportunity dataclass and expected value
- MADueDiligenceState methods and computed fields
"""

from __future__ import annotations

import pytest

from src.enterprise.use_cases.ma_due_diligence.state import (
    AnalyzedDocument,
    DueDiligencePhase,
    IdentifiedRisk,
    MADueDiligenceState,
    RiskLevel,
    SynergyOpportunity,
)


@pytest.mark.unit
class TestDueDiligencePhase:
    """Tests for DueDiligencePhase enum."""

    def test_all_phases_exist(self) -> None:
        phases = list(DueDiligencePhase)
        assert len(phases) == 8

    def test_from_string_valid(self) -> None:
        assert DueDiligencePhase.from_string("initial_screening") == DueDiligencePhase.INITIAL_SCREENING
        assert DueDiligencePhase.from_string("FINANCIAL_ANALYSIS") == DueDiligencePhase.FINANCIAL_ANALYSIS
        assert DueDiligencePhase.from_string("Final_Recommendation") == DueDiligencePhase.FINAL_RECOMMENDATION

    def test_from_string_invalid(self) -> None:
        with pytest.raises(KeyError):
            DueDiligencePhase.from_string("nonexistent_phase")

    def test_to_index_ordering(self) -> None:
        assert DueDiligencePhase.INITIAL_SCREENING.to_index() == 0
        assert DueDiligencePhase.FINANCIAL_ANALYSIS.to_index() == 1
        assert DueDiligencePhase.FINAL_RECOMMENDATION.to_index() == 7

    def test_to_index_monotonic(self) -> None:
        phases = list(DueDiligencePhase)
        for i, phase in enumerate(phases):
            assert phase.to_index() == i


@pytest.mark.unit
class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_values(self) -> None:
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_to_weight(self) -> None:
        assert RiskLevel.LOW.to_weight() == 0.25
        assert RiskLevel.MEDIUM.to_weight() == 0.5
        assert RiskLevel.HIGH.to_weight() == 0.75
        assert RiskLevel.CRITICAL.to_weight() == 1.0

    def test_weights_are_monotonically_increasing(self) -> None:
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        weights = [level.to_weight() for level in levels]
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]


@pytest.mark.unit
class TestIdentifiedRisk:
    """Tests for IdentifiedRisk dataclass."""

    def _make_risk(self, **kwargs) -> IdentifiedRisk:
        defaults = {
            "risk_id": "R001",
            "category": "financial",
            "description": "Revenue decline risk",
            "severity": RiskLevel.HIGH,
            "probability": 0.6,
            "impact": 0.8,
        }
        defaults.update(kwargs)
        return IdentifiedRisk(**defaults)

    def test_default_initialization(self) -> None:
        risk = self._make_risk()
        assert risk.risk_id == "R001"
        assert risk.mitigation_possible is True
        assert risk.mitigation_cost is None
        assert risk.source_document is None
        assert risk.identified_at_phase is None

    def test_get_risk_score(self) -> None:
        risk = self._make_risk(severity=RiskLevel.HIGH, probability=0.6, impact=0.8)
        expected = 0.75 * 0.6 * 0.8  # weight * probability * impact
        assert risk.get_risk_score() == pytest.approx(expected)

    def test_get_risk_score_critical(self) -> None:
        risk = self._make_risk(severity=RiskLevel.CRITICAL, probability=1.0, impact=1.0)
        assert risk.get_risk_score() == pytest.approx(1.0)

    def test_get_risk_score_zero_probability(self) -> None:
        risk = self._make_risk(probability=0.0)
        assert risk.get_risk_score() == pytest.approx(0.0)

    def test_to_dict(self) -> None:
        risk = self._make_risk()
        d = risk.to_dict()
        assert d["risk_id"] == "R001"
        assert d["category"] == "financial"
        assert d["severity"] == "high"
        assert d["probability"] == 0.6
        assert d["impact"] == 0.8
        assert d["mitigation_possible"] is True
        assert d["mitigation_cost"] is None
        assert "risk_score" in d
        assert d["risk_score"] == pytest.approx(risk.get_risk_score())

    def test_optional_fields(self) -> None:
        risk = self._make_risk(
            mitigation_possible=False,
            mitigation_cost=500_000.0,
            source_document="annual_report_2024.pdf",
            identified_at_phase=DueDiligencePhase.FINANCIAL_ANALYSIS,
        )
        assert risk.mitigation_possible is False
        assert risk.mitigation_cost == 500_000.0
        assert risk.source_document == "annual_report_2024.pdf"
        assert risk.identified_at_phase == DueDiligencePhase.FINANCIAL_ANALYSIS


@pytest.mark.unit
class TestAnalyzedDocument:
    """Tests for AnalyzedDocument dataclass."""

    def test_default_initialization(self) -> None:
        doc = AnalyzedDocument(doc_id="D001", doc_type="financial_statement")
        assert doc.key_findings == []
        assert doc.risks_identified == []
        assert doc.confidence == 0.0
        assert doc.analysis_depth == "surface"
        assert doc.metadata == {}

    def test_to_dict(self) -> None:
        doc = AnalyzedDocument(
            doc_id="D001",
            doc_type="contract",
            key_findings=["finding1", "finding2"],
            risks_identified=["R001"],
            confidence=0.85,
            analysis_depth="deep",
        )
        d = doc.to_dict()
        assert d["doc_id"] == "D001"
        assert d["doc_type"] == "contract"
        assert d["key_findings_count"] == 2
        assert d["risks_identified_count"] == 1
        assert d["confidence"] == 0.85
        assert d["analysis_depth"] == "deep"

    def test_to_dict_empty(self) -> None:
        doc = AnalyzedDocument(doc_id="D002", doc_type="legal_filing")
        d = doc.to_dict()
        assert d["key_findings_count"] == 0
        assert d["risks_identified_count"] == 0


@pytest.mark.unit
class TestSynergyOpportunity:
    """Tests for SynergyOpportunity dataclass."""

    def _make_synergy(self, **kwargs) -> SynergyOpportunity:
        defaults = {
            "synergy_id": "S001",
            "category": "revenue",
            "description": "Cross-sell opportunity",
            "estimated_value": 10_000_000.0,
            "probability": 0.7,
            "timeline_months": 12,
        }
        defaults.update(kwargs)
        return SynergyOpportunity(**defaults)

    def test_default_initialization(self) -> None:
        syn = self._make_synergy()
        assert syn.dependencies == []
        assert syn.risks == []

    def test_get_expected_value(self) -> None:
        syn = self._make_synergy(estimated_value=10_000_000.0, probability=0.7)
        assert syn.get_expected_value() == pytest.approx(7_000_000.0)

    def test_get_expected_value_zero_probability(self) -> None:
        syn = self._make_synergy(probability=0.0)
        assert syn.get_expected_value() == pytest.approx(0.0)

    def test_to_dict(self) -> None:
        syn = self._make_synergy()
        d = syn.to_dict()
        assert d["synergy_id"] == "S001"
        assert d["category"] == "revenue"
        assert d["estimated_value"] == 10_000_000.0
        assert d["probability"] == 0.7
        assert d["expected_value"] == pytest.approx(7_000_000.0)
        assert d["timeline_months"] == 12


@pytest.mark.unit
class TestMADueDiligenceState:
    """Tests for MADueDiligenceState dataclass."""

    def _make_state(self, **kwargs) -> MADueDiligenceState:
        defaults = {
            "state_id": "test-state-001",
            "target_company": "TargetCo",
            "acquirer_company": "AcquirerCo",
            "deal_value": 500_000_000.0,
        }
        defaults.update(kwargs)
        return MADueDiligenceState(**defaults)

    def _make_risk(
        self,
        risk_id: str = "R001",
        severity: RiskLevel = RiskLevel.MEDIUM,
        probability: float = 0.5,
        impact: float = 0.5,
    ) -> IdentifiedRisk:
        return IdentifiedRisk(
            risk_id=risk_id,
            category="financial",
            description="Test risk",
            severity=severity,
            probability=probability,
            impact=impact,
        )

    def _make_synergy(
        self, synergy_id: str = "S001", estimated_value: float = 1_000_000.0, probability: float = 0.8
    ) -> SynergyOpportunity:
        return SynergyOpportunity(
            synergy_id=synergy_id,
            category="cost",
            description="Test synergy",
            estimated_value=estimated_value,
            probability=probability,
            timeline_months=6,
        )

    def test_default_initialization(self) -> None:
        state = self._make_state()
        assert state.domain == "ma_due_diligence"
        assert state.phase == DueDiligencePhase.INITIAL_SCREENING
        assert state.documents_analyzed == []
        assert state.risks_identified == []
        assert state.synergies_found == []
        assert state.overall_risk_score == 0.0
        assert state.action_history == []

    def test_compute_risk_score_no_risks(self) -> None:
        state = self._make_state()
        assert state.compute_risk_score() == 0.0

    def test_compute_risk_score_single_risk(self) -> None:
        state = self._make_state()
        risk = self._make_risk(severity=RiskLevel.HIGH, probability=0.6, impact=0.8)
        state.risks_identified.append(risk)
        score = state.compute_risk_score()
        expected = 0.75 * 0.6 * 0.8  # single risk, max_possible = 1.0
        assert score == pytest.approx(expected)
        assert state.overall_risk_score == pytest.approx(expected)

    def test_compute_risk_score_multiple_risks(self) -> None:
        state = self._make_state()
        state.risks_identified.append(self._make_risk("R1", RiskLevel.LOW, 0.5, 0.5))
        state.risks_identified.append(self._make_risk("R2", RiskLevel.CRITICAL, 1.0, 1.0))
        score = state.compute_risk_score()
        total = (0.25 * 0.5 * 0.5) + (1.0 * 1.0 * 1.0)
        expected = total / 2.0
        assert score == pytest.approx(expected)

    def test_compute_synergy_confidence_no_synergies(self) -> None:
        state = self._make_state()
        assert state.compute_synergy_confidence() == 0.0

    def test_compute_synergy_confidence(self) -> None:
        state = self._make_state()
        state.synergies_found.append(self._make_synergy("S1", 1_000_000.0, 0.8))
        state.synergies_found.append(self._make_synergy("S2", 2_000_000.0, 0.5))
        conf = state.compute_synergy_confidence()
        total_expected = (1_000_000.0 * 0.8) + (2_000_000.0 * 0.5)
        total_estimated = 1_000_000.0 + 2_000_000.0
        assert conf == pytest.approx(total_expected / total_estimated)

    def test_compute_synergy_confidence_zero_estimated(self) -> None:
        state = self._make_state()
        state.synergies_found.append(self._make_synergy("S1", 0.0, 0.5))
        assert state.compute_synergy_confidence() == 0.0

    def test_get_critical_risks(self) -> None:
        state = self._make_state()
        state.risks_identified.append(self._make_risk("R1", RiskLevel.LOW))
        state.risks_identified.append(self._make_risk("R2", RiskLevel.CRITICAL))
        state.risks_identified.append(self._make_risk("R3", RiskLevel.CRITICAL))
        critical = state.get_critical_risks()
        assert len(critical) == 2
        assert all(r.severity == RiskLevel.CRITICAL for r in critical)

    def test_get_critical_risks_none(self) -> None:
        state = self._make_state()
        state.risks_identified.append(self._make_risk("R1", RiskLevel.LOW))
        assert state.get_critical_risks() == []

    def test_get_high_value_synergies(self) -> None:
        state = self._make_state()
        state.synergies_found.append(self._make_synergy("S1", 100.0, 0.5))  # expected = 50
        state.synergies_found.append(self._make_synergy("S2", 1000.0, 0.9))  # expected = 900
        high = state.get_high_value_synergies(min_value=100.0)
        assert len(high) == 1
        assert high[0].synergy_id == "S2"

    def test_get_high_value_synergies_default(self) -> None:
        state = self._make_state()
        state.synergies_found.append(self._make_synergy("S1", 100.0, 0.5))
        assert len(state.get_high_value_synergies()) == 1  # > 0.0

    def test_get_progress_percentage(self) -> None:
        state = self._make_state()
        assert state.get_progress_percentage() == pytest.approx(1 / 8 * 100)

        state.phase = DueDiligencePhase.FINAL_RECOMMENDATION
        assert state.get_progress_percentage() == pytest.approx(100.0)

    def test_to_summary(self) -> None:
        state = self._make_state()
        state.risks_identified.append(self._make_risk("R1", RiskLevel.CRITICAL, 0.9, 0.9))
        state.synergies_found.append(self._make_synergy("S1", 1_000_000.0, 0.8))
        state.documents_analyzed.append(AnalyzedDocument(doc_id="D1", doc_type="contract"))
        state.jurisdictions_checked = ["US", "EU"]
        state.action_history = ["action1", "action2"]

        summary = state.to_summary()
        assert summary["state_id"] == "test-state-001"
        assert summary["phase"] == "INITIAL_SCREENING"
        assert summary["target_company"] == "TargetCo"
        assert summary["documents_analyzed"] == 1
        assert summary["risks_identified"] == 1
        assert summary["critical_risks"] == 1
        assert summary["synergies_found"] == 1
        assert summary["jurisdictions_checked"] == ["US", "EU"]
        assert summary["action_count"] == 2
        assert "progress_pct" in summary
        assert "overall_risk_score" in summary
        assert "synergy_confidence" in summary

    def test_update_features(self) -> None:
        state = self._make_state()
        state.risks_identified.append(self._make_risk())
        state.synergies_found.append(self._make_synergy())
        state.documents_analyzed.append(AnalyzedDocument(doc_id="D1", doc_type="contract"))
        state.action_history = ["a1", "a2"]

        state.update_features()
        assert state.features["phase"] == "INITIAL_SCREENING"
        assert state.features["phase_idx"] == 0
        assert state.features["documents_count"] == 1
        assert state.features["risks_count"] == 1
        assert state.features["synergies_count"] == 1
        assert state.features["action_count"] == 2
        assert "risk_score" in state.features
        assert "synergy_confidence" in state.features
