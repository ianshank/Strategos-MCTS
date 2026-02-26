"""
Domain-specific agents for M&A Due Diligence.

Each agent specializes in a specific aspect of the due diligence process,
following patterns established in the HRM and TRM agent architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.adapters.llm.base import LLMClient

from src.adapters.llm.base import LLMResponse

from ...config.enterprise_settings import MADueDiligenceConfig, get_enterprise_settings
from .state import (
    IdentifiedRisk,
    MADueDiligenceState,
    RiskLevel,
    SynergyOpportunity,
)


def _get_default_config() -> MADueDiligenceConfig:
    """Get default configuration from enterprise settings."""
    return get_enterprise_settings().ma_due_diligence


@dataclass
class AgentConfig:
    """Base configuration for domain agents."""

    confidence_threshold: float = field(default_factory=lambda: _get_default_config().agent_config.confidence_threshold)
    max_retries: int = field(default_factory=lambda: _get_default_config().agent_config.max_retries)
    timeout_seconds: float = field(default_factory=lambda: _get_default_config().agent_config.timeout_seconds)


@dataclass
class DocumentAnalysisAgentConfig(AgentConfig):
    """Configuration for document analysis agent."""

    max_docs_per_batch: int = field(default_factory=lambda: _get_default_config().max_docs_per_batch)
    extraction_confidence_threshold: float = field(
        default_factory=lambda: _get_default_config().extraction_confidence_threshold
    )
    enable_ocr: bool = True
    analysis_depth: str = "moderate"  # surface, moderate, deep


class DocumentAnalysisAgent:
    """
    Agent for extracting key terms from M&A documents.

    Pattern: HRM (hierarchical decomposition of document types)

    Responsible for:
    - Analyzing financial statements
    - Extracting key contract terms
    - Identifying red flags in documents
    - Summarizing legal filings
    """

    def __init__(
        self,
        config: DocumentAnalysisAgentConfig | None = None,
        llm_client: LLMClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config or DocumentAnalysisAgentConfig()
        self._llm = llm_client
        self._logger = logger or logging.getLogger(__name__)
        self._last_confidence: float = 0.0

    @property
    def name(self) -> str:
        return "document_analysis"

    async def process(
        self,
        query: str,
        domain_state: MADueDiligenceState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Analyze documents and extract key information.

        Args:
            query: Analysis query
            domain_state: Current due diligence state
            context: Additional context

        Returns:
            Analysis results including findings and risks
        """
        self._logger.info(
            f"DocumentAnalysisAgent processing: {query[:100]}...",
            extra={"phase": domain_state.phase.name},
        )

        # Build analysis prompt
        prompt = self._build_analysis_prompt(query, domain_state, context)

        # Call LLM if available
        if self._llm:
            try:
                result = await self._llm.generate(prompt=prompt, temperature=0.3)
                assert isinstance(result, LLMResponse), "Expected LLMResponse, not stream"
                findings = self._parse_findings(result.text)
            except Exception as e:
                self._logger.error(f"LLM call failed: {e}")
                findings = self._generate_mock_findings(domain_state)
        else:
            # Generate mock findings for testing
            findings = self._generate_mock_findings(domain_state)

        self._last_confidence = findings.get("confidence", 0.7)

        return {
            "agent": self.name,
            "response": findings.get("summary", ""),
            "key_terms": findings.get("key_terms", []),
            "risks": findings.get("risks", []),
            "documents_analyzed": findings.get("documents_analyzed", []),
            "confidence": self._last_confidence,
        }

    def get_confidence(self) -> float:
        return self._last_confidence

    def _build_analysis_prompt(
        self,
        query: str,
        state: MADueDiligenceState,
        context: dict[str, Any],
    ) -> str:
        """Build hierarchical document analysis prompt."""
        return f"""
Analyze the following M&A due diligence query and extract key information.

Target Company: {state.target_company}
Acquirer: {state.acquirer_company}
Current Phase: {state.phase.name}
Documents Already Analyzed: {len(state.documents_analyzed)}
Risks Already Identified: {len(state.risks_identified)}

Query: {query}

Document Content (if available):
{context.get("document_content", "No specific document provided")}

MCTS Recommended Action: {context.get("mcts_action", "None")}

Extract and provide:
1. Key financial terms and conditions
2. Legal obligations and covenants
3. Risk indicators (categorize by: financial, legal, operational, technology)
4. Red flags requiring deeper investigation
5. Synergy opportunities if identified

Respond in structured JSON format with:
- summary: Brief analysis summary
- key_terms: List of important terms found
- risks: List of risks with severity (low/medium/high/critical)
- confidence: Your confidence in this analysis (0-1)
"""

    def _parse_findings(self, content: str) -> dict[str, Any]:
        """Parse LLM response into structured findings."""
        # In production, this would include robust JSON parsing
        return {
            "summary": content[:500] if content else "",
            "key_terms": [],
            "risks": [],
            "documents_analyzed": [],
            "confidence": 0.75,
        }

    def _generate_mock_findings(self, state: MADueDiligenceState) -> dict[str, Any]:
        """Generate mock findings for testing."""
        return {
            "summary": f"Analysis of {state.target_company} documents in {state.phase.name} phase.",
            "key_terms": [
                f"Revenue: ${state.deal_value:,.0f}" if state.deal_value else "Revenue: TBD",
                "Change of control clause identified",
                "Material adverse change provisions",
            ],
            "risks": [
                {"description": "Pending litigation", "severity": "medium"},
                {"description": "Key customer concentration", "severity": "high"},
            ],
            "documents_analyzed": [
                {"doc_id": "DOC001", "doc_type": "financial_statement"},
            ],
            "confidence": 0.72,
        }


class RiskIdentificationAgent:
    """
    Agent for identifying hidden risks and red flags.

    Pattern: TRM (iterative refinement of risk categories)

    Responsible for:
    - Deep dive into financial risks
    - Legal and regulatory risk assessment
    - Operational risk identification
    - Technology and IP risk evaluation
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_client: LLMClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        enterprise_config = _get_default_config()
        self._config = config or {"max_refinement_rounds": enterprise_config.max_refinement_rounds}
        self._llm = llm_client
        self._logger = logger or logging.getLogger(__name__)
        self._last_confidence: float = 0.0

    @property
    def name(self) -> str:
        return "risk_identification"

    async def process(
        self,
        query: str,
        domain_state: MADueDiligenceState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Identify and categorize risks through iterative refinement.

        Args:
            query: Risk identification query
            domain_state: Current state
            context: Additional context

        Returns:
            Identified risks with severity and mitigation potential
        """
        self._logger.info(
            f"RiskIdentificationAgent processing: {query[:100]}...",
            extra={"existing_risks": len(domain_state.risks_identified)},
        )

        risks: list[IdentifiedRisk] = []
        refinement_rounds = self._config.get("max_refinement_rounds", 3)

        for round_num in range(refinement_rounds):
            round_risks = await self._identify_risks_round(query, domain_state, context, round_num, risks)
            risks.extend(round_risks)

            # Check convergence (no new risks found)
            if not round_risks:
                self._logger.debug(f"Risk identification converged at round {round_num}")
                break

        self._last_confidence = self._compute_confidence(risks)

        return {
            "agent": self.name,
            "risks": [self._risk_to_dict(r) for r in risks],
            "risk_count": len(risks),
            "critical_count": len([r for r in risks if r.severity == RiskLevel.CRITICAL]),
            "risk_score": self._compute_risk_score(risks),
            "confidence": self._last_confidence,
        }

    def get_confidence(self) -> float:
        return self._last_confidence

    async def _identify_risks_round(
        self,
        query: str,
        state: MADueDiligenceState,
        context: dict[str, Any],
        round_num: int,
        existing_risks: list[IdentifiedRisk],
    ) -> list[IdentifiedRisk]:
        """Run one round of risk identification."""
        # Generate mock risks for demonstration
        if round_num == 0:
            return [
                IdentifiedRisk(
                    risk_id=f"RISK_{len(existing_risks) + 1:03d}",
                    category="financial",
                    description="Working capital requirements may exceed projections",
                    severity=RiskLevel.MEDIUM,
                    probability=0.6,
                    impact=0.7,
                    identified_at_phase=state.phase,
                ),
                IdentifiedRisk(
                    risk_id=f"RISK_{len(existing_risks) + 2:03d}",
                    category="legal",
                    description="Pending regulatory approval required",
                    severity=RiskLevel.HIGH,
                    probability=0.4,
                    impact=0.9,
                    identified_at_phase=state.phase,
                ),
            ]
        return []

    def _compute_confidence(self, risks: list[IdentifiedRisk]) -> float:
        """Compute confidence based on risk identification completeness."""
        if not risks:
            return 0.5
        # Higher confidence with more diverse risk categories
        categories = {r.category for r in risks}
        return min(0.5 + len(categories) * 0.1, 0.9)

    def _compute_risk_score(self, risks: list[IdentifiedRisk]) -> float:
        """Compute aggregate risk score."""
        if not risks:
            return 0.0
        return sum(r.get_risk_score() for r in risks) / len(risks)

    def _risk_to_dict(self, risk: IdentifiedRisk) -> dict[str, Any]:
        """Convert risk to dictionary."""
        return risk.to_dict()


class SynergyExplorationAgent:
    """
    Agent for exploring potential synergies.

    Pattern: MCTS (explore combination scenarios)

    Responsible for:
    - Revenue synergy identification
    - Cost synergy analysis
    - Operational efficiency opportunities
    - Strategic value assessment
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_client: LLMClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        enterprise_config = _get_default_config()
        self._config = config or {"exploration_depth": enterprise_config.synergy_exploration_depth}
        self._llm = llm_client
        self._logger = logger or logging.getLogger(__name__)
        self._last_confidence: float = 0.0

    @property
    def name(self) -> str:
        return "synergy_exploration"

    async def process(
        self,
        query: str,
        domain_state: MADueDiligenceState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Explore and quantify synergy opportunities.

        Args:
            query: Synergy exploration query
            domain_state: Current state
            context: Additional context

        Returns:
            Identified synergies with valuations
        """
        self._logger.info(
            f"SynergyExplorationAgent processing: {query[:100]}...",
            extra={"deal_value": domain_state.deal_value},
        )

        synergies = self._identify_synergies(domain_state, context)

        total_value = sum(s.estimated_value for s in synergies)
        expected_value = sum(s.get_expected_value() for s in synergies)

        self._last_confidence = expected_value / total_value if total_value > 0 else 0.5

        return {
            "agent": self.name,
            "synergies": [s.to_dict() for s in synergies],
            "synergy_count": len(synergies),
            "total_estimated_value": total_value,
            "total_expected_value": expected_value,
            "categories": list({s.category for s in synergies}),
            "confidence": self._last_confidence,
        }

    def get_confidence(self) -> float:
        return self._last_confidence

    def _identify_synergies(
        self,
        state: MADueDiligenceState,
        context: dict[str, Any],
    ) -> list[SynergyOpportunity]:
        """Identify synergy opportunities."""
        base_value = state.deal_value or 100_000_000

        return [
            SynergyOpportunity(
                synergy_id="SYN_001",
                category="cost",
                description="Consolidated IT infrastructure",
                estimated_value=base_value * 0.02,
                probability=0.8,
                timeline_months=12,
            ),
            SynergyOpportunity(
                synergy_id="SYN_002",
                category="revenue",
                description="Cross-selling opportunities",
                estimated_value=base_value * 0.05,
                probability=0.6,
                timeline_months=24,
            ),
            SynergyOpportunity(
                synergy_id="SYN_003",
                category="operational",
                description="Supply chain optimization",
                estimated_value=base_value * 0.03,
                probability=0.7,
                timeline_months=18,
            ),
        ]


class ComplianceCheckAgent:
    """
    Agent for verifying regulatory compliance.

    Pattern: Rule-based with LLM fallback

    Responsible for:
    - Multi-jurisdictional compliance review
    - Regulatory requirement mapping
    - Compliance gap identification
    - Remediation planning
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_client: LLMClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._enterprise_config = _get_default_config()
        self._config = config or {"jurisdictions": self._enterprise_config.jurisdictions}
        self._llm = llm_client
        self._logger = logger or logging.getLogger(__name__)
        self._last_confidence: float = 0.0

    @property
    def name(self) -> str:
        return "compliance_check"

    async def process(
        self,
        query: str,
        domain_state: MADueDiligenceState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Check regulatory compliance across jurisdictions.

        Args:
            query: Compliance check query
            domain_state: Current state
            context: Additional context

        Returns:
            Compliance status and issues by jurisdiction
        """
        self._logger.info(
            f"ComplianceCheckAgent processing: {query[:100]}...",
            extra={"jurisdictions": self._config.get("jurisdictions", [])},
        )

        jurisdictions = self._config.get("jurisdictions", ["US", "EU"])
        compliance_results = {}

        for jurisdiction in jurisdictions:
            result = await self._check_jurisdiction(jurisdiction, domain_state, context)
            compliance_results[jurisdiction] = result

        issues = [issue for result in compliance_results.values() for issue in result.get("issues", [])]

        self._last_confidence = self._compute_confidence(compliance_results)

        return {
            "agent": self.name,
            "jurisdictions_checked": list(compliance_results.keys()),
            "compliance_by_jurisdiction": compliance_results,
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i.get("severity") == "critical"]),
            "confidence": self._last_confidence,
        }

    def get_confidence(self) -> float:
        return self._last_confidence

    async def _check_jurisdiction(
        self,
        jurisdiction: str,
        state: MADueDiligenceState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Check compliance for a specific jurisdiction."""
        # Rule-based compliance checks (would be more comprehensive in production)
        issues = []

        # Get thresholds from configuration
        hsr_threshold = self._enterprise_config.hsr_filing_threshold
        eu_threshold = self._enterprise_config.eu_merger_threshold

        if jurisdiction == "US":
            if state.deal_value and state.deal_value > hsr_threshold:
                issues.append(
                    {
                        "regulation": "HSR Act",
                        "description": "Hart-Scott-Rodino filing required",
                        "severity": "high",
                        "remediation": "File HSR notification",
                        "threshold": hsr_threshold,
                    }
                )
        elif jurisdiction == "EU" and state.deal_value and state.deal_value > eu_threshold:
            issues.append(
                {
                    "regulation": "EU Merger Regulation",
                    "description": "EU merger notification may be required",
                    "severity": "high",
                    "remediation": "Assess EU turnover thresholds",
                    "threshold": eu_threshold,
                }
            )

        return {
            "jurisdiction": jurisdiction,
            "compliant": len(issues) == 0,
            "issues": issues,
            "checks_performed": 5,
        }

    def _compute_confidence(self, results: dict[str, Any]) -> float:
        """Compute confidence based on checks performed."""
        total_checks = sum(r.get("checks_performed", 0) for r in results.values())
        return float(min(0.5 + total_checks * 0.05, 0.9))
