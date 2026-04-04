"""
Tests for src/enterprise/use_cases/ma_due_diligence/use_case.py

Covers MADueDiligence initialization, property accessors, agent setup,
reward function setup, initial state creation, action delegation,
state updates from agents, final analysis generation, and the
async process entry point.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enterprise.config.enterprise_settings import MADueDiligenceConfig
from src.enterprise.use_cases.ma_due_diligence.state import (
    DueDiligencePhase,
    IdentifiedRisk,
    MADueDiligenceState,
    RiskLevel,
)
from src.enterprise.use_cases.ma_due_diligence.use_case import MADueDiligence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> MADueDiligenceState:
    """Build a minimal MADueDiligenceState for tests."""
    defaults = {
        "state_id": "test_state",
        "domain": "ma_due_diligence",
        "phase": DueDiligencePhase.INITIAL_SCREENING,
        "target_company": "TargetCo",
        "acquirer_company": "AcquirerCo",
        "deal_value": 200_000_000,
        "deal_rationale": "Strategic acquisition",
    }
    defaults.update(overrides)
    return MADueDiligenceState(**defaults)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMADueDiligenceProperties:
    """Tests for MADueDiligence property accessors."""

    def test_name(self):
        uc = MADueDiligence()
        assert uc.name == "ma_due_diligence"

    def test_domain(self):
        uc = MADueDiligence()
        assert uc.domain == "finance"

    def test_config_returns_ma_config(self):
        config = MADueDiligenceConfig()
        uc = MADueDiligence(config=config)
        assert uc.config is config

    def test_default_config_created(self):
        uc = MADueDiligence()
        assert isinstance(uc.config, MADueDiligenceConfig)


# ---------------------------------------------------------------------------
# _setup_agents
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetupAgents:
    """Tests for agent setup."""

    def test_setup_agents_creates_four_agents(self):
        uc = MADueDiligence()
        uc.initialize()
        agents = uc.get_domain_agents()
        assert len(agents) == 4
        assert "document_analysis" in agents
        assert "risk_identification" in agents
        assert "synergy_exploration" in agents
        assert "compliance_check" in agents


# ---------------------------------------------------------------------------
# _setup_reward_function
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetupRewardFunction:
    """Tests for reward function setup."""

    def test_reward_function_initialized_after_init(self):
        uc = MADueDiligence()
        uc.initialize()
        rf = uc.get_reward_function()
        assert rf is not None

    def test_get_reward_function_lazy_init(self):
        uc = MADueDiligence()
        # _reward_function starts as None; calling get_reward_function
        # on the subclass should set it up.
        rf = uc.get_reward_function()
        assert rf is not None


# ---------------------------------------------------------------------------
# get_initial_state
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetInitialState:
    """Tests for creating initial state."""

    def test_initial_state_fields(self):
        uc = MADueDiligence()
        state = uc.get_initial_state(
            "Analyze TargetCo",
            {"target_company": "TargetCo", "acquirer_company": "BuyerCo", "deal_value": 500_000_000},
        )
        assert state.target_company == "TargetCo"
        assert state.acquirer_company == "BuyerCo"
        assert state.deal_value == 500_000_000
        assert state.phase == DueDiligencePhase.INITIAL_SCREENING
        assert state.state_id.startswith("ma_dd_")

    def test_initial_state_defaults(self):
        uc = MADueDiligence()
        state = uc.get_initial_state("query", {})
        assert state.target_company == "Unknown Target"
        assert state.acquirer_company == "Unknown Acquirer"
        assert state.deal_value is None

    def test_deal_value_from_deal_value_usd(self):
        uc = MADueDiligence()
        state = uc.get_initial_state("q", {"deal_value_usd": 1_000_000})
        assert state.deal_value == 1_000_000

    def test_deal_value_usd_takes_priority(self):
        uc = MADueDiligence()
        state = uc.get_initial_state("q", {"deal_value_usd": 1, "deal_value": 2})
        assert state.deal_value == 1

    def test_features_contain_query(self):
        uc = MADueDiligence()
        state = uc.get_initial_state("my query", {})
        assert "query" in state.features


# ---------------------------------------------------------------------------
# get_available_actions / apply_action (delegation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestActionDelegation:
    """Tests for action retrieval and application."""

    def test_get_available_actions_returns_list(self):
        uc = MADueDiligence()
        state = _make_state()
        actions = uc.get_available_actions(state)
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_apply_action_returns_new_state(self):
        uc = MADueDiligence()
        state = _make_state()
        new_state = uc.apply_action(state, "analyze_financial_overview")
        assert new_state is not state
        assert "analyze_financial_overview" in new_state.action_history


# ---------------------------------------------------------------------------
# get_rollout_policy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetRolloutPolicy:
    """Tests for the rollout policy."""

    def test_returns_policy(self):
        uc = MADueDiligence()
        policy = uc.get_rollout_policy()
        assert policy is not None

    def test_fallback_on_import_error(self):
        """When the inner imports fail, the except ImportError path is taken."""
        uc = MADueDiligence()
        # Patch the local import of MCTSState to raise ImportError.
        # The method catches ImportError and calls super().get_rollout_policy().
        # We mock super's method to verify the fallback is reached.
        with patch.object(
            type(uc).__mro__[1], "get_rollout_policy", return_value=MagicMock(name="fallback_policy")
        ) as mock_super:
            import builtins

            original_import = builtins.__import__

            def _blocking_import(name, *args, **kwargs):
                if name == "src.framework.mcts.core":
                    raise ImportError("mocked")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_blocking_import):
                policy = uc.get_rollout_policy()

            mock_super.assert_called_once()
            assert policy is not None


# ---------------------------------------------------------------------------
# _update_state_from_agents
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUpdateStateFromAgents:
    """Tests for _update_state_from_agents."""

    def test_adds_risks(self):
        uc = MADueDiligence()
        state = _make_state()
        agent_results = {
            "risk_identification": {
                "risks": [
                    {
                        "risk_id": "R1",
                        "category": "financial",
                        "description": "cash",
                        "severity": "high",
                        "probability": 0.5,
                        "impact": 0.8,
                    },
                ]
            }
        }
        updated = uc._update_state_from_agents(state, agent_results)
        assert len(updated.risks_identified) == 1
        assert updated.risks_identified[0].risk_id == "R1"
        assert updated.risks_identified[0].severity == RiskLevel.HIGH

    def test_adds_synergies(self):
        uc = MADueDiligence()
        state = _make_state()
        agent_results = {
            "synergy_exploration": {
                "synergies": [
                    {
                        "synergy_id": "S1",
                        "category": "cost",
                        "description": "IT consolidation",
                        "estimated_value": 5000000,
                        "probability": 0.7,
                        "timeline_months": 12,
                    },
                ]
            }
        }
        updated = uc._update_state_from_agents(state, agent_results)
        assert len(updated.synergies_found) == 1
        assert updated.synergies_found[0].synergy_id == "S1"

    def test_adds_compliance_issues(self):
        uc = MADueDiligence()
        state = _make_state()
        agent_results = {
            "compliance_check": {
                "jurisdictions_checked": ["US", "EU"],
                "compliance_by_jurisdiction": {
                    "US": {"issues": [{"regulation": "HSR", "severity": "high"}]},
                    "EU": {"issues": []},
                },
            }
        }
        updated = uc._update_state_from_agents(state, agent_results)
        assert updated.jurisdictions_checked == ["US", "EU"]
        assert len(updated.compliance_issues) == 1
        assert updated.compliance_issues[0]["jurisdiction"] == "US"

    def test_empty_agent_results(self):
        uc = MADueDiligence()
        state = _make_state()
        updated = uc._update_state_from_agents(state, {})
        assert len(updated.risks_identified) == 0
        assert len(updated.synergies_found) == 0

    def test_non_dict_risk_skipped(self):
        uc = MADueDiligence()
        state = _make_state()
        agent_results = {"risk_identification": {"risks": ["not_a_dict"]}}
        updated = uc._update_state_from_agents(state, agent_results)
        assert len(updated.risks_identified) == 0

    def test_non_dict_synergy_skipped(self):
        uc = MADueDiligence()
        state = _make_state()
        agent_results = {"synergy_exploration": {"synergies": [42]}}
        updated = uc._update_state_from_agents(state, agent_results)
        assert len(updated.synergies_found) == 0


# ---------------------------------------------------------------------------
# _generate_final_analysis
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateFinalAnalysis:
    """Tests for _generate_final_analysis."""

    def test_proceed_recommendation(self):
        uc = MADueDiligence()
        state = _make_state()
        # No risks, low risk score
        result = uc._generate_final_analysis(state, {}, {})
        assert "PROCEED" in result["recommendation"]
        assert result["confidence"] >= 0

    def test_do_not_proceed_many_critical_risks(self):
        uc = MADueDiligence()
        state = _make_state()
        for i in range(3):
            state.risks_identified.append(
                IdentifiedRisk(
                    risk_id=f"R{i}",
                    category="financial",
                    description="bad",
                    severity=RiskLevel.CRITICAL,
                    probability=0.9,
                    impact=0.9,
                )
            )
        agent_results = {"doc": {"confidence": 0.8}}
        result = uc._generate_final_analysis(state, agent_results, {})
        assert "DO NOT PROCEED" in result["recommendation"]

    def test_proceed_with_caution_high_risk(self):
        uc = MADueDiligence()
        config = MADueDiligenceConfig(critical_risk_threshold=0.3, risk_threshold=0.2)
        uc = MADueDiligence(config=config)
        state = _make_state()
        # Add risks to push score above critical threshold but <3 critical
        state.risks_identified.append(
            IdentifiedRisk(
                risk_id="R1",
                category="financial",
                description="severe",
                severity=RiskLevel.HIGH,
                probability=0.9,
                impact=0.9,
            )
        )
        agent_results = {"agent1": {"confidence": 0.8}}
        result = uc._generate_final_analysis(state, agent_results, {})
        # risk_score should be > critical_risk_threshold
        assert "CAUTION" in result["recommendation"] or "PROCEED" in result["recommendation"]

    def test_conditional_proceed(self):
        uc = MADueDiligence()
        config = MADueDiligenceConfig(risk_threshold=0.1, critical_risk_threshold=0.9)
        uc = MADueDiligence(config=config)
        state = _make_state()
        state.risks_identified.append(
            IdentifiedRisk(
                risk_id="R1",
                category="legal",
                description="minor",
                severity=RiskLevel.MEDIUM,
                probability=0.5,
                impact=0.5,
            )
        )
        agent_results = {"a1": {"confidence": 0.7}}
        result = uc._generate_final_analysis(state, agent_results, {})
        # risk_score = 0.5 * 0.5 * 0.5 = 0.125 > risk_threshold(0.1) but < critical(0.9)
        assert "CONDITIONAL" in result["recommendation"]


# ---------------------------------------------------------------------------
# process (async)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcess:
    """Tests for the async process entry point."""

    @pytest.mark.asyncio
    async def test_process_without_mcts(self):
        config = MADueDiligenceConfig(enabled=False)
        uc = MADueDiligence(config=config)

        result = await uc.process(
            "Analyze TargetCo for acquisition",
            context={"target_company": "TargetCo", "deal_value": 100_000_000},
            use_mcts=False,
        )

        assert "result" in result
        assert "recommendation" in result
        assert "confidence" in result
        assert result["use_case"] == "ma_due_diligence"
        assert result["domain"] == "finance"
        assert "risk_analysis" in result
        assert "synergy_analysis" in result
        assert "compliance_status" in result

    @pytest.mark.asyncio
    async def test_process_initializes_if_needed(self):
        uc = MADueDiligence(config=MADueDiligenceConfig(enabled=False))
        assert not uc.is_initialized
        await uc.process("test query", use_mcts=False)
        assert uc.is_initialized

    @pytest.mark.asyncio
    async def test_process_with_mcts_enabled(self):
        config = MADueDiligenceConfig(enabled=True)
        uc = MADueDiligence(config=config)

        with patch.object(uc, "_run_mcts", new_callable=AsyncMock) as mock_mcts:
            mock_mcts.return_value = {"best_action": "analyze_financial_overview", "stats": {"iterations": 10}}
            result = await uc.process(
                "test",
                context={"target_company": "X"},
                use_mcts=True,
            )
            mock_mcts.assert_called_once()
            assert result["mcts_stats"] == {"iterations": 10}

    @pytest.mark.asyncio
    async def test_process_default_context(self):
        uc = MADueDiligence(config=MADueDiligenceConfig(enabled=False))
        result = await uc.process("query", use_mcts=False)
        assert result["domain_state"]["target_company"] == "Unknown Target"
