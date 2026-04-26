"""
M&A Due Diligence domain state definitions.

Defines the state representation for M&A due diligence processes,
including phases, risks, synergies, and analysis progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from ...base.use_case import BaseDomainState
from ...config.enterprise_settings import get_enterprise_settings


class DueDiligencePhase(Enum):
    """Phases of M&A due diligence process."""

    INITIAL_SCREENING = auto()
    FINANCIAL_ANALYSIS = auto()
    LEGAL_REVIEW = auto()
    OPERATIONAL_ASSESSMENT = auto()
    TECHNOLOGY_EVALUATION = auto()
    SYNERGY_EXPLORATION = auto()
    RISK_CONSOLIDATION = auto()
    FINAL_RECOMMENDATION = auto()

    @classmethod
    def from_string(cls, value: str) -> DueDiligencePhase:
        """Convert string to enum value."""
        return cls[value.upper()]

    def to_index(self) -> int:
        """Get phase index for ordering."""
        phases = list(DueDiligencePhase)
        return phases.index(self)


class RiskLevel(Enum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def to_weight(self) -> float:
        """Convert to numeric weight for calculations."""
        return get_enterprise_settings().ma_due_diligence.risk_level_weights[self.value]


@dataclass
class IdentifiedRisk:
    """Represents an identified risk in due diligence."""

    risk_id: str
    category: str  # financial, legal, operational, technology
    description: str
    severity: RiskLevel
    probability: float  # 0-1, likelihood of materialization
    impact: float  # 0-1, impact if materialized
    mitigation_possible: bool = True
    mitigation_cost: float | None = None
    source_document: str | None = None
    identified_at_phase: DueDiligencePhase | None = None

    def get_risk_score(self) -> float:
        """Calculate composite risk score."""
        return self.severity.to_weight() * self.probability * self.impact

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "risk_id": self.risk_id,
            "category": self.category,
            "description": self.description,
            "severity": self.severity.value,
            "probability": self.probability,
            "impact": self.impact,
            "mitigation_possible": self.mitigation_possible,
            "mitigation_cost": self.mitigation_cost,
            "risk_score": self.get_risk_score(),
        }


@dataclass
class AnalyzedDocument:
    """Represents an analyzed document."""

    doc_id: str
    doc_type: str  # contract, financial_statement, legal_filing, etc.
    key_findings: list[str] = field(default_factory=list)
    risks_identified: list[str] = field(default_factory=list)  # risk_ids
    confidence: float = 0.0
    analysis_depth: str = "surface"  # surface, moderate, deep
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "key_findings_count": len(self.key_findings),
            "risks_identified_count": len(self.risks_identified),
            "confidence": self.confidence,
            "analysis_depth": self.analysis_depth,
        }


@dataclass
class SynergyOpportunity:
    """Represents a potential synergy from the acquisition."""

    synergy_id: str
    category: str  # revenue, cost, operational, strategic
    description: str
    estimated_value: float  # in base currency
    probability: float  # 0-1
    timeline_months: int  # time to realize
    dependencies: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def get_expected_value(self) -> float:
        """Calculate expected value with probability weighting."""
        return self.estimated_value * self.probability

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "synergy_id": self.synergy_id,
            "category": self.category,
            "description": self.description,
            "estimated_value": self.estimated_value,
            "probability": self.probability,
            "expected_value": self.get_expected_value(),
            "timeline_months": self.timeline_months,
        }


@dataclass
class MADueDiligenceState(BaseDomainState):
    """
    Domain state for M&A Due Diligence.

    Tracks progress through due diligence phases and
    accumulates findings, risks, and synergies.

    Attributes:
        phase: Current due diligence phase
        target_company: Name of target company
        acquirer_company: Name of acquiring company
        deal_value: Deal value in base currency
        documents_analyzed: List of analyzed documents
        risks_identified: List of identified risks
        synergies_found: List of synergy opportunities
        jurisdictions_checked: Jurisdictions reviewed for compliance
        compliance_issues: Compliance issues found
        overall_risk_score: Computed aggregate risk score
        synergy_confidence: Confidence in synergy estimates
        recommendation_confidence: Confidence in final recommendation
        action_history: History of MCTS actions taken
    """

    domain: str = "ma_due_diligence"

    # Current phase
    phase: DueDiligencePhase = DueDiligencePhase.INITIAL_SCREENING

    # Target company info
    target_company: str = ""
    acquirer_company: str = ""
    deal_value: float | None = None
    deal_rationale: str = ""

    # Analysis progress
    documents_analyzed: list[AnalyzedDocument] = field(default_factory=list)
    risks_identified: list[IdentifiedRisk] = field(default_factory=list)
    synergies_found: list[SynergyOpportunity] = field(default_factory=list)

    # Compliance tracking
    jurisdictions_checked: list[str] = field(default_factory=list)
    compliance_issues: list[dict[str, Any]] = field(default_factory=list)

    # Scores (computed)
    overall_risk_score: float = 0.0
    synergy_confidence: float = 0.0
    recommendation_confidence: float = 0.0

    # Path tracking for MCTS
    action_history: list[str] = field(default_factory=list)

    def compute_risk_score(self) -> float:
        """
        Compute weighted risk score from identified risks.

        Returns:
            Normalized risk score between 0 and 1
        """
        if not self.risks_identified:
            return 0.0

        total_weighted_risk = sum(r.get_risk_score() for r in self.risks_identified)

        # Normalize to 0-1 (max possible is len * 1.0)
        max_possible = len(self.risks_identified) * 1.0
        self.overall_risk_score = total_weighted_risk / max_possible if max_possible > 0 else 0.0
        return self.overall_risk_score

    def compute_synergy_confidence(self) -> float:
        """
        Compute confidence in synergy estimates.

        Returns:
            Confidence score between 0 and 1
        """
        if not self.synergies_found:
            return 0.0

        # Weight by expected value
        total_expected = sum(s.get_expected_value() for s in self.synergies_found)
        total_estimated = sum(s.estimated_value for s in self.synergies_found)

        if total_estimated == 0:
            return 0.0

        self.synergy_confidence = total_expected / total_estimated
        return self.synergy_confidence

    def get_critical_risks(self) -> list[IdentifiedRisk]:
        """Get all critical severity risks."""
        return [r for r in self.risks_identified if r.severity == RiskLevel.CRITICAL]

    def get_high_value_synergies(self, min_value: float = 0.0) -> list[SynergyOpportunity]:
        """Get synergies above a minimum expected value."""
        return [s for s in self.synergies_found if s.get_expected_value() > min_value]

    def get_progress_percentage(self) -> float:
        """Calculate overall due diligence progress."""
        total_phases = len(DueDiligencePhase)
        current_phase_idx = self.phase.to_index()
        return (current_phase_idx + 1) / total_phases * 100

    def to_summary(self) -> dict[str, Any]:
        """Generate summary for reporting."""
        return {
            "state_id": self.state_id,
            "phase": self.phase.name,
            "progress_pct": self.get_progress_percentage(),
            "target_company": self.target_company,
            "acquirer_company": self.acquirer_company,
            "deal_value": self.deal_value,
            "documents_analyzed": len(self.documents_analyzed),
            "risks_identified": len(self.risks_identified),
            "critical_risks": len(self.get_critical_risks()),
            "synergies_found": len(self.synergies_found),
            "overall_risk_score": self.compute_risk_score(),
            "synergy_confidence": self.compute_synergy_confidence(),
            "jurisdictions_checked": self.jurisdictions_checked,
            "compliance_issues": len(self.compliance_issues),
            "action_count": len(self.action_history),
        }

    def update_features(self) -> None:
        """Update features dictionary for MCTS."""
        self.features.update(
            {
                "phase": self.phase.name,
                "phase_idx": self.phase.to_index(),
                "risk_score": self.compute_risk_score(),
                "synergy_confidence": self.compute_synergy_confidence(),
                "documents_count": len(self.documents_analyzed),
                "risks_count": len(self.risks_identified),
                "synergies_count": len(self.synergies_found),
                "action_count": len(self.action_history),
            }
        )
