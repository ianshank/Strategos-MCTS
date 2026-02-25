"""
M&A Due Diligence Use Case Implementation.

Complete implementation of the M&A due diligence use case that integrates:
- MCTS-guided exploration of due diligence pathways
- Domain-specific agents for document analysis, risk identification, etc.
- Dynamic configuration via settings
- LangGraph integration for orchestration
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.adapters.llm.base import LLMClient
    from src.framework.mcts.policies import RolloutPolicy

from ...base.use_case import BaseUseCase, RewardFunctionProtocol
from ...config.enterprise_settings import MADueDiligenceConfig
from .actions import apply_action, get_available_actions
from .agents import (
    ComplianceCheckAgent,
    DocumentAnalysisAgent,
    DocumentAnalysisAgentConfig,
    RiskIdentificationAgent,
    SynergyExplorationAgent,
)
from .reward import MADueDiligenceReward
from .state import (
    DueDiligencePhase,
    IdentifiedRisk,
    MADueDiligenceState,
    RiskLevel,
    SynergyOpportunity,
)


class MADueDiligence(BaseUseCase[MADueDiligenceState]):
    """
    M&A Due Diligence Use Case.

    Combines MCTS exploration with specialized domain agents to conduct
    comprehensive due diligence on acquisition targets.

    Features:
    - Multi-phase due diligence workflow
    - Parallel agent execution for efficiency
    - MCTS-guided pathway exploration
    - Comprehensive risk and synergy analysis
    - Multi-jurisdictional compliance checking

    Example:
        >>> config = MADueDiligenceConfig()
        >>> use_case = MADueDiligence(config=config)
        >>> result = await use_case.process(
        ...     "Analyze TargetCo Inc. for acquisition",
        ...     context={"deal_value": 100_000_000}
        ... )
    """

    def __init__(
        self,
        config: MADueDiligenceConfig | None = None,
        llm_client: LLMClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the M&A Due Diligence use case.

        Args:
            config: Use case configuration
            llm_client: Optional LLM client for agent operations
            logger: Optional logger instance
        """
        config = config or MADueDiligenceConfig()
        super().__init__(config=config, llm_client=llm_client, logger=logger)

    @property
    def name(self) -> str:
        return "ma_due_diligence"

    @property
    def domain(self) -> str:
        return "finance"

    @property
    def config(self) -> MADueDiligenceConfig:
        config: MADueDiligenceConfig = self._config
        return config

    def _setup_agents(self) -> None:
        """Initialize domain-specific agents."""
        self._logger.info("Setting up M&A Due Diligence agents")

        # Document Analysis Agent
        doc_config = DocumentAnalysisAgentConfig(
            confidence_threshold=self._config.confidence_threshold,
            max_docs_per_batch=self._config.max_documents_per_analysis,
        )
        self._agents["document_analysis"] = DocumentAnalysisAgent(
            config=doc_config,
            llm_client=self._llm_client,
            logger=self._logger,
        )

        # Risk Identification Agent
        risk_config = {
            "max_refinement_rounds": 3,
            "risk_threshold": self._config.risk_threshold,
            "critical_threshold": self._config.critical_risk_threshold,
        }
        self._agents["risk_identification"] = RiskIdentificationAgent(
            config=risk_config,
            llm_client=self._llm_client,
            logger=self._logger,
        )

        # Synergy Exploration Agent
        synergy_config = {
            "exploration_depth": self._config.synergy_exploration_depth,
            "min_synergy_value": self._config.min_synergy_value,
        }
        self._agents["synergy_exploration"] = SynergyExplorationAgent(
            config=synergy_config,
            llm_client=self._llm_client,
            logger=self._logger,
        )

        # Compliance Check Agent
        compliance_config = {
            "jurisdictions": self._config.jurisdictions,
        }
        self._agents["compliance_check"] = ComplianceCheckAgent(
            config=compliance_config,
            llm_client=self._llm_client,
            logger=self._logger,
        )

        self._logger.info(f"Initialized {len(self._agents)} agents")

    def _setup_reward_function(self) -> None:
        """Initialize the reward function."""
        self._reward_function = MADueDiligenceReward(
            config=self._config,
            weights=self._config.action_weights,
        )

    def get_initial_state(
        self,
        query: str,
        context: dict[str, Any],
    ) -> MADueDiligenceState:
        """
        Create initial due diligence state from query.

        Args:
            query: User query describing the due diligence request
            context: Context including target company info, deal value, etc.

        Returns:
            Initialized MADueDiligenceState
        """
        state_id = f"ma_dd_{uuid.uuid4().hex[:8]}"

        state = MADueDiligenceState(
            state_id=state_id,
            domain=self.domain,
            phase=DueDiligencePhase.INITIAL_SCREENING,
            target_company=context.get("target_company", "Unknown Target"),
            acquirer_company=context.get("acquirer_company", "Unknown Acquirer"),
            deal_value=context.get("deal_value_usd") or context.get("deal_value"),
            deal_rationale=context.get("deal_rationale", "Strategic acquisition"),
            jurisdictions_checked=[],
            features={
                "query": query,
                "initial_context": str(context)[:500],
            },
            metadata={
                "created_at": str(uuid.uuid1()),
                "config_enabled": self._config.enabled,
            },
        )

        state.update_features()
        self._logger.info(
            f"Created initial state: {state_id}",
            extra=state.to_summary(),
        )

        return state

    def get_available_actions(self, state: MADueDiligenceState) -> list[str]:
        """
        Return available actions for MCTS expansion.

        Args:
            state: Current due diligence state

        Returns:
            List of available action strings
        """
        return get_available_actions(
            state,
            include_meta=True,
            exclude_recent=True,
            recent_window=5,
        )

    def apply_action(
        self,
        state: MADueDiligenceState,
        action: str,
    ) -> MADueDiligenceState:
        """
        Apply action to state, returning new state.

        Args:
            state: Current state
            action: Action to apply

        Returns:
            New state after action
        """
        new_state = apply_action(state, action)
        self._logger.debug(
            f"Applied action '{action}' to state",
            extra={"new_phase": new_state.phase.name},
        )
        return new_state

    def get_reward_function(self) -> RewardFunctionProtocol:
        """Return the reward function for MCTS."""
        if self._reward_function is None:
            self._setup_reward_function()
        assert self._reward_function is not None, "Reward function not initialized"
        return self._reward_function

    def get_rollout_policy(self) -> RolloutPolicy:
        """
        Return MCTS rollout policy optimized for M&A due diligence.

        The policy balances:
        - Domain heuristics (phase completion, risk coverage)
        - Random exploration for discovering unexpected pathways
        """
        try:
            from src.framework.mcts.core import MCTSState
            from src.framework.mcts.policies import HybridRolloutPolicy

            def heuristic_fn(mcts_state: MCTSState) -> float:
                """Evaluate MCTS state using M&A domain heuristics."""
                features = mcts_state.features

                # Base value
                base = 0.5

                # Bonus for deeper analysis
                action_count = features.get("action_count", 0)
                depth_bonus = min(action_count / 30, 0.15)

                # Bonus for phase progression
                phase_idx = features.get("phase_idx", 0)
                phase_bonus = phase_idx * 0.04

                # Bonus for risk coverage
                risk_count = features.get("risks_count", 0)
                risk_bonus = min(risk_count / 10, 0.1)

                # Penalty for high risk score
                risk_score = features.get("risk_score", 0)
                risk_penalty = risk_score * 0.05

                return float(min(base + depth_bonus + phase_bonus + risk_bonus - risk_penalty, 1.0))

            return HybridRolloutPolicy(
                heuristic_fn=heuristic_fn,
                heuristic_weight=0.7,
                random_weight=0.3,
            )
        except ImportError:
            # Fallback to base implementation
            return super().get_rollout_policy()

    async def process(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        use_mcts: bool = True,
    ) -> dict[str, Any]:
        """
        Main entry point for M&A due diligence processing.

        Args:
            query: Due diligence request
            context: Context including target company, deal value, etc.
            use_mcts: Whether to use MCTS for exploration

        Returns:
            Comprehensive due diligence results
        """
        if not self._initialized:
            self.initialize()

        context = context or {}

        self._logger.info(
            f"Starting M&A due diligence: {query[:100]}...",
            extra={
                "target": context.get("target_company"),
                "deal_value": context.get("deal_value"),
                "use_mcts": use_mcts,
            },
        )

        # Create initial state
        state = self.get_initial_state(query, context)

        # Run MCTS exploration if enabled
        mcts_result = {}
        if use_mcts and self._config.enabled:
            mcts_result = await self._run_mcts(state, context)

        # Process with domain agents
        agent_results = await self._process_with_agents(query, state, context, mcts_result)

        # Update state with agent findings
        state = self._update_state_from_agents(state, agent_results)

        # Generate final analysis
        final_result = self._generate_final_analysis(state, agent_results, mcts_result)

        return {
            "result": final_result.get("recommendation", ""),
            "recommendation": final_result.get("recommendation"),
            "confidence": final_result.get("confidence", 0.0),
            "domain_state": state.to_summary(),
            "agent_results": agent_results,
            "mcts_stats": mcts_result.get("stats", {}),
            "risk_analysis": {
                "total_risks": len(state.risks_identified),
                "critical_risks": len(state.get_critical_risks()),
                "risk_score": state.compute_risk_score(),
            },
            "synergy_analysis": {
                "total_synergies": len(state.synergies_found),
                "total_estimated_value": sum(s.estimated_value for s in state.synergies_found),
                "total_expected_value": sum(s.get_expected_value() for s in state.synergies_found),
            },
            "compliance_status": {
                "jurisdictions_checked": state.jurisdictions_checked,
                "issues_found": len(state.compliance_issues),
            },
            "use_case": self.name,
            "domain": self.domain,
        }

    def _update_state_from_agents(
        self,
        state: MADueDiligenceState,
        agent_results: dict[str, Any],
    ) -> MADueDiligenceState:
        """Update state with findings from agents."""
        # Add risks from risk identification agent
        if "risk_identification" in agent_results:
            risk_data = agent_results["risk_identification"]
            for risk_dict in risk_data.get("risks", []):
                if isinstance(risk_dict, dict):
                    risk = IdentifiedRisk(
                        risk_id=risk_dict.get("risk_id", f"RISK_{len(state.risks_identified)}"),
                        category=risk_dict.get("category", "unknown"),
                        description=risk_dict.get("description", ""),
                        severity=RiskLevel(risk_dict.get("severity", "medium")),
                        probability=risk_dict.get("probability", 0.5),
                        impact=risk_dict.get("impact", 0.5),
                    )
                    state.risks_identified.append(risk)

        # Add synergies from synergy exploration agent
        if "synergy_exploration" in agent_results:
            synergy_data = agent_results["synergy_exploration"]
            for syn_dict in synergy_data.get("synergies", []):
                if isinstance(syn_dict, dict):
                    synergy = SynergyOpportunity(
                        synergy_id=syn_dict.get("synergy_id", f"SYN_{len(state.synergies_found)}"),
                        category=syn_dict.get("category", "unknown"),
                        description=syn_dict.get("description", ""),
                        estimated_value=syn_dict.get("estimated_value", 0),
                        probability=syn_dict.get("probability", 0.5),
                        timeline_months=syn_dict.get("timeline_months", 12),
                    )
                    state.synergies_found.append(synergy)

        # Update compliance from compliance check agent
        if "compliance_check" in agent_results:
            compliance_data = agent_results["compliance_check"]
            state.jurisdictions_checked = compliance_data.get("jurisdictions_checked", [])
            for jurisdiction, result in compliance_data.get("compliance_by_jurisdiction", {}).items():
                for issue in result.get("issues", []):
                    state.compliance_issues.append(
                        {
                            "jurisdiction": jurisdiction,
                            **issue,
                        }
                    )

        state.update_features()
        return state

    def _generate_final_analysis(
        self,
        state: MADueDiligenceState,
        agent_results: dict[str, Any],
        mcts_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate final due diligence recommendation."""
        risk_score = state.compute_risk_score()
        synergy_confidence = state.compute_synergy_confidence()

        # Compute overall confidence
        agent_confidences = [
            result.get("confidence", 0.5)
            for result in agent_results.values()
            if isinstance(result, dict) and "confidence" in result
        ]
        avg_agent_confidence = sum(agent_confidences) / len(agent_confidences) if agent_confidences else 0.5

        # Generate recommendation
        critical_risks = state.get_critical_risks()

        if len(critical_risks) > 2:
            recommendation = "DO NOT PROCEED - Multiple critical risks identified"
            confidence = min(avg_agent_confidence, 0.6)
        elif risk_score > self._config.critical_risk_threshold:
            recommendation = "PROCEED WITH CAUTION - High risk profile requires significant due diligence"
            confidence = avg_agent_confidence * 0.8
        elif risk_score > self._config.risk_threshold:
            recommendation = "CONDITIONAL PROCEED - Address identified risks before closing"
            confidence = avg_agent_confidence * 0.9
        else:
            recommendation = "PROCEED - Risk profile acceptable, synergies identified"
            confidence = avg_agent_confidence

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "risk_score": risk_score,
            "synergy_confidence": synergy_confidence,
            "critical_risks": [r.to_dict() for r in critical_risks],
            "top_synergies": [s.to_dict() for s in state.get_high_value_synergies()[:3]],
        }
