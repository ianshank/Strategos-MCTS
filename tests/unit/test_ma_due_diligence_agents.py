"""
Tests for src/enterprise/use_cases/ma_due_diligence/agents.py

Covers DocumentAnalysisAgent, RiskIdentificationAgent,
SynergyExplorationAgent, and ComplianceCheckAgent -- including
their configs, process methods, confidence scoring, and
internal helpers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enterprise.use_cases.ma_due_diligence.agents import (
    AgentConfig,
    ComplianceCheckAgent,
    DocumentAnalysisAgent,
    DocumentAnalysisAgentConfig,
    RiskIdentificationAgent,
    SynergyExplorationAgent,
)
from src.enterprise.use_cases.ma_due_diligence.state import (
    DueDiligencePhase,
    MADueDiligenceState,
    RiskLevel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> MADueDiligenceState:
    defaults = {
        "state_id": "test_state",
        "domain": "ma_due_diligence",
        "phase": DueDiligencePhase.INITIAL_SCREENING,
        "target_company": "TargetCo",
        "acquirer_company": "AcquirerCo",
        "deal_value": 200_000_000,
    }
    defaults.update(overrides)
    return MADueDiligenceState(**defaults)


# ---------------------------------------------------------------------------
# AgentConfig / DocumentAnalysisAgentConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentConfigs:
    """Tests for agent configuration dataclasses."""

    def test_agent_config_defaults(self):
        config = AgentConfig()
        assert 0 < config.confidence_threshold <= 1.0
        assert config.max_retries >= 1
        assert config.timeout_seconds > 0

    def test_document_analysis_config_defaults(self):
        config = DocumentAnalysisAgentConfig()
        assert config.max_docs_per_batch >= 1
        assert 0 < config.extraction_confidence_threshold <= 1.0
        assert config.enable_ocr is True
        assert config.analysis_depth == "moderate"

    def test_document_analysis_config_custom(self):
        config = DocumentAnalysisAgentConfig(
            confidence_threshold=0.8,
            max_docs_per_batch=20,
            enable_ocr=False,
            analysis_depth="deep",
        )
        assert config.confidence_threshold == 0.8
        assert config.max_docs_per_batch == 20
        assert config.enable_ocr is False
        assert config.analysis_depth == "deep"


# ---------------------------------------------------------------------------
# DocumentAnalysisAgent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDocumentAnalysisAgent:
    """Tests for DocumentAnalysisAgent."""

    def test_name(self):
        agent = DocumentAnalysisAgent()
        assert agent.name == "document_analysis"

    def test_initial_confidence_is_zero(self):
        agent = DocumentAnalysisAgent()
        assert agent.get_confidence() == 0.0

    @pytest.mark.asyncio
    async def test_process_without_llm(self):
        agent = DocumentAnalysisAgent()
        state = _make_state()
        result = await agent.process("analyze docs", state, {})

        assert result["agent"] == "document_analysis"
        assert "response" in result
        assert "key_terms" in result
        assert "risks" in result
        assert "documents_analyzed" in result
        assert "confidence" in result
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    async def test_process_sets_confidence(self):
        agent = DocumentAnalysisAgent()
        state = _make_state()
        await agent.process("query", state, {})
        assert agent.get_confidence() > 0

    @pytest.mark.asyncio
    async def test_process_with_llm_success(self):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "LLM analysis output"
        mock_llm.generate.return_value = mock_response

        # Patch isinstance check for LLMResponse
        with patch("src.enterprise.use_cases.ma_due_diligence.agents.LLMResponse", type(mock_response)):
            agent = DocumentAnalysisAgent(llm_client=mock_llm)
            state = _make_state()
            result = await agent.process("analyze", state, {})

        assert result["confidence"] > 0
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_llm_failure_falls_back_to_mock(self):
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RuntimeError("API error")

        agent = DocumentAnalysisAgent(llm_client=mock_llm)
        state = _make_state()
        result = await agent.process("analyze", state, {})

        # Should fall back to mock findings
        assert result["confidence"] > 0
        assert "response" in result

    def test_build_analysis_prompt(self):
        agent = DocumentAnalysisAgent()
        state = _make_state()
        prompt = agent._build_analysis_prompt("my query", state, {"document_content": "some content"})
        assert "TargetCo" in prompt
        assert "AcquirerCo" in prompt
        assert "my query" in prompt
        assert "some content" in prompt

    def test_parse_findings(self):
        agent = DocumentAnalysisAgent()
        result = agent._parse_findings("some content here")
        assert result["summary"] == "some content here"
        assert result["confidence"] == 0.75

    def test_parse_findings_empty(self):
        agent = DocumentAnalysisAgent()
        result = agent._parse_findings("")
        assert result["summary"] == ""

    def test_generate_mock_findings(self):
        agent = DocumentAnalysisAgent()
        state = _make_state()
        findings = agent._generate_mock_findings(state)
        assert "TargetCo" in findings["summary"]
        assert len(findings["key_terms"]) > 0
        assert len(findings["risks"]) > 0
        assert findings["confidence"] > 0

    def test_generate_mock_findings_no_deal_value(self):
        agent = DocumentAnalysisAgent()
        state = _make_state(deal_value=None)
        findings = agent._generate_mock_findings(state)
        assert "TBD" in findings["key_terms"][0]


# ---------------------------------------------------------------------------
# RiskIdentificationAgent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRiskIdentificationAgent:
    """Tests for RiskIdentificationAgent."""

    def test_name(self):
        agent = RiskIdentificationAgent()
        assert agent.name == "risk_identification"

    def test_initial_confidence(self):
        agent = RiskIdentificationAgent()
        assert agent.get_confidence() == 0.0

    @pytest.mark.asyncio
    async def test_process_identifies_risks(self):
        agent = RiskIdentificationAgent()
        state = _make_state()
        result = await agent.process("identify risks", state, {})

        assert result["agent"] == "risk_identification"
        assert result["risk_count"] >= 2
        assert "risks" in result
        assert "risk_score" in result
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_process_convergence(self):
        """Should stop when no new risks found."""
        agent = RiskIdentificationAgent(config={"max_refinement_rounds": 5})
        state = _make_state()
        result = await agent.process("risks", state, {})
        # Round 0 produces 2 risks, round 1 produces 0 => converge
        assert result["risk_count"] == 2

    @pytest.mark.asyncio
    async def test_process_with_single_round(self):
        agent = RiskIdentificationAgent(config={"max_refinement_rounds": 1})
        state = _make_state()
        result = await agent.process("risks", state, {})
        assert result["risk_count"] == 2

    def test_compute_confidence_no_risks(self):
        agent = RiskIdentificationAgent()
        conf = agent._compute_confidence([])
        assert conf == 0.5

    def test_compute_confidence_diverse_categories(self):
        from src.enterprise.use_cases.ma_due_diligence.state import IdentifiedRisk

        agent = RiskIdentificationAgent()
        risks = [
            IdentifiedRisk(
                risk_id="R1", category="financial", description="x", severity=RiskLevel.LOW, probability=0.5, impact=0.5
            ),
            IdentifiedRisk(
                risk_id="R2", category="legal", description="y", severity=RiskLevel.LOW, probability=0.5, impact=0.5
            ),
            IdentifiedRisk(
                risk_id="R3",
                category="operational",
                description="z",
                severity=RiskLevel.LOW,
                probability=0.5,
                impact=0.5,
            ),
        ]
        conf = agent._compute_confidence(risks)
        assert conf == min(0.5 + 3 * 0.1, 0.9)

    def test_compute_risk_score_empty(self):
        agent = RiskIdentificationAgent()
        assert agent._compute_risk_score([]) == 0.0

    def test_compute_risk_score(self):
        from src.enterprise.use_cases.ma_due_diligence.state import IdentifiedRisk

        agent = RiskIdentificationAgent()
        risks = [
            IdentifiedRisk(
                risk_id="R1",
                category="financial",
                description="x",
                severity=RiskLevel.MEDIUM,
                probability=0.6,
                impact=0.7,
            ),
        ]
        score = agent._compute_risk_score(risks)
        expected = RiskLevel.MEDIUM.to_weight() * 0.6 * 0.7
        assert abs(score - expected) < 1e-6

    def test_risk_to_dict(self):
        from src.enterprise.use_cases.ma_due_diligence.state import IdentifiedRisk

        agent = RiskIdentificationAgent()
        risk = IdentifiedRisk(
            risk_id="R1",
            category="financial",
            description="test",
            severity=RiskLevel.HIGH,
            probability=0.8,
            impact=0.9,
        )
        d = agent._risk_to_dict(risk)
        assert d["risk_id"] == "R1"
        assert d["severity"] == "high"

    @pytest.mark.asyncio
    async def test_process_sets_confidence(self):
        agent = RiskIdentificationAgent()
        state = _make_state()
        await agent.process("identify", state, {})
        # Two risks with categories financial and legal => 0.5 + 2*0.1 = 0.7
        assert agent.get_confidence() == 0.7

    @pytest.mark.asyncio
    async def test_critical_count(self):
        agent = RiskIdentificationAgent()
        state = _make_state()
        result = await agent.process("risks", state, {})
        # Default mock produces MEDIUM and HIGH, no critical
        assert result["critical_count"] == 0


# ---------------------------------------------------------------------------
# SynergyExplorationAgent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSynergyExplorationAgent:
    """Tests for SynergyExplorationAgent."""

    def test_name(self):
        agent = SynergyExplorationAgent()
        assert agent.name == "synergy_exploration"

    def test_initial_confidence(self):
        agent = SynergyExplorationAgent()
        assert agent.get_confidence() == 0.0

    @pytest.mark.asyncio
    async def test_process_returns_synergies(self):
        agent = SynergyExplorationAgent()
        state = _make_state()
        result = await agent.process("find synergies", state, {})

        assert result["agent"] == "synergy_exploration"
        assert result["synergy_count"] == 3
        assert len(result["synergies"]) == 3
        assert result["total_estimated_value"] > 0
        assert result["total_expected_value"] > 0
        assert len(result["categories"]) > 0

    @pytest.mark.asyncio
    async def test_process_sets_confidence(self):
        agent = SynergyExplorationAgent()
        state = _make_state()
        await agent.process("synergies", state, {})
        assert agent.get_confidence() > 0

    def test_identify_synergies_default_deal_value(self):
        agent = SynergyExplorationAgent()
        state = _make_state(deal_value=None)
        synergies = agent._identify_synergies(state, {})
        # Should use 100M as default
        assert len(synergies) == 3
        assert synergies[0].estimated_value == 100_000_000 * 0.02

    def test_identify_synergies_custom_deal_value(self):
        agent = SynergyExplorationAgent()
        state = _make_state(deal_value=500_000_000)
        synergies = agent._identify_synergies(state, {})
        assert synergies[0].estimated_value == 500_000_000 * 0.02
        assert synergies[1].estimated_value == 500_000_000 * 0.05

    @pytest.mark.asyncio
    async def test_confidence_calculation(self):
        agent = SynergyExplorationAgent()
        state = _make_state(deal_value=100_000_000)
        result = await agent.process("synergies", state, {})
        # confidence = total_expected / total_estimated
        synergies = agent._identify_synergies(state, {})
        total_est = sum(s.estimated_value for s in synergies)
        total_exp = sum(s.get_expected_value() for s in synergies)
        expected_conf = total_exp / total_est
        assert abs(result["confidence"] - expected_conf) < 1e-6


# ---------------------------------------------------------------------------
# ComplianceCheckAgent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComplianceCheckAgent:
    """Tests for ComplianceCheckAgent."""

    def test_name(self):
        agent = ComplianceCheckAgent()
        assert agent.name == "compliance_check"

    def test_initial_confidence(self):
        agent = ComplianceCheckAgent()
        assert agent.get_confidence() == 0.0

    @pytest.mark.asyncio
    async def test_process_checks_jurisdictions(self):
        agent = ComplianceCheckAgent(config={"jurisdictions": ["US", "EU"]})
        state = _make_state(deal_value=200_000_000)
        result = await agent.process("check compliance", state, {})

        assert result["agent"] == "compliance_check"
        assert set(result["jurisdictions_checked"]) == {"US", "EU"}
        assert "compliance_by_jurisdiction" in result
        assert "total_issues" in result
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_us_hsr_filing_triggered(self):
        agent = ComplianceCheckAgent(config={"jurisdictions": ["US"]})
        # Deal value above HSR threshold (default ~101M)
        state = _make_state(deal_value=200_000_000)
        result = await agent.process("compliance", state, {})

        us_result = result["compliance_by_jurisdiction"]["US"]
        assert not us_result["compliant"]
        assert len(us_result["issues"]) == 1
        assert us_result["issues"][0]["regulation"] == "HSR Act"

    @pytest.mark.asyncio
    async def test_us_no_issue_below_threshold(self):
        agent = ComplianceCheckAgent(config={"jurisdictions": ["US"]})
        state = _make_state(deal_value=50_000_000)
        result = await agent.process("compliance", state, {})

        us_result = result["compliance_by_jurisdiction"]["US"]
        assert us_result["compliant"]
        assert len(us_result["issues"]) == 0

    @pytest.mark.asyncio
    async def test_eu_merger_notification(self):
        agent = ComplianceCheckAgent(config={"jurisdictions": ["EU"]})
        state = _make_state(deal_value=200_000_000)
        result = await agent.process("compliance", state, {})

        eu_result = result["compliance_by_jurisdiction"]["EU"]
        assert len(eu_result["issues"]) == 1
        assert "EU Merger Regulation" in eu_result["issues"][0]["regulation"]

    @pytest.mark.asyncio
    async def test_eu_no_issue_below_threshold(self):
        agent = ComplianceCheckAgent(config={"jurisdictions": ["EU"]})
        state = _make_state(deal_value=100_000_000)
        result = await agent.process("compliance", state, {})

        eu_result = result["compliance_by_jurisdiction"]["EU"]
        assert eu_result["compliant"]

    @pytest.mark.asyncio
    async def test_no_deal_value(self):
        agent = ComplianceCheckAgent(config={"jurisdictions": ["US", "EU"]})
        state = _make_state(deal_value=None)
        result = await agent.process("compliance", state, {})
        assert result["total_issues"] == 0

    @pytest.mark.asyncio
    async def test_unknown_jurisdiction(self):
        agent = ComplianceCheckAgent(config={"jurisdictions": ["JP"]})
        state = _make_state(deal_value=200_000_000)
        result = await agent.process("compliance", state, {})
        jp_result = result["compliance_by_jurisdiction"]["JP"]
        assert jp_result["compliant"]

    def test_compute_confidence(self):
        agent = ComplianceCheckAgent()
        results = {
            "US": {"checks_performed": 5},
            "EU": {"checks_performed": 5},
        }
        conf = agent._compute_confidence(results)
        # 0.5 + 10 * 0.05 = 1.0, capped at 0.9
        assert conf == 0.9

    def test_compute_confidence_low(self):
        agent = ComplianceCheckAgent()
        results = {
            "US": {"checks_performed": 1},
        }
        conf = agent._compute_confidence(results)
        assert conf == 0.55

    @pytest.mark.asyncio
    async def test_process_sets_confidence(self):
        agent = ComplianceCheckAgent(config={"jurisdictions": ["US"]})
        state = _make_state()
        await agent.process("check", state, {})
        assert agent.get_confidence() > 0

    @pytest.mark.asyncio
    async def test_critical_issues_counted(self):
        agent = ComplianceCheckAgent(config={"jurisdictions": ["US"]})
        state = _make_state(deal_value=200_000_000)
        result = await agent.process("compliance", state, {})
        # HSR issue has severity "high" not "critical"
        assert result["critical_issues"] == 0

    @pytest.mark.asyncio
    async def test_default_jurisdictions_used(self):
        agent = ComplianceCheckAgent(config={})
        state = _make_state()
        result = await agent.process("compliance", state, {})
        assert set(result["jurisdictions_checked"]) == {"US", "EU"}
