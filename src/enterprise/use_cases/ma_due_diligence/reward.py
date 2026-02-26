"""
MCTS reward function for M&A Due Diligence.

Defines how actions are evaluated during MCTS exploration.
Reward = Information_Gain + Risk_Discovery + Timeline_Efficiency
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...config.enterprise_settings import MADueDiligenceConfig, get_enterprise_settings

if TYPE_CHECKING:
    pass

from ...base.use_case import RewardFunctionProtocol
from .state import MADueDiligenceState


def _get_default_config() -> MADueDiligenceConfig:
    """Get default configuration from enterprise settings."""
    return get_enterprise_settings().ma_due_diligence


class MADueDiligenceReward(RewardFunctionProtocol):
    """
    Reward function for M&A Due Diligence MCTS.

    The reward function balances three objectives:
    1. Information gain: Reward for uncovering new information
    2. Risk discovery: Reward for identifying risks
    3. Timeline efficiency: Reward for progressing through phases efficiently

    All weights are configurable via settings - no hardcoded values.
    """

    def __init__(
        self,
        config: MADueDiligenceConfig | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize reward function.

        Args:
            config: Use case configuration
            weights: Optional custom weights for components
        """
        self._config = config or _get_default_config()

        # Get weights from configuration
        self._weights = weights or self._config.reward_weights.copy()

    def evaluate(  # type: ignore[override]
        self,
        state: MADueDiligenceState,
        action: str,
        context: dict[str, Any],
    ) -> float:
        """
        Evaluate action in state, return reward in [0, 1].

        Args:
            state: Current domain state
            action: Action being evaluated
            context: Additional context

        Returns:
            Reward value between 0 (worst) and 1 (best)
        """
        info_gain = self._compute_information_gain(state, action, context)
        risk_discovery = self._compute_risk_discovery(state, action, context)
        timeline_efficiency = self._compute_timeline_efficiency(state, action)

        reward = (
            self._weights["information_gain"] * info_gain
            + self._weights["risk_discovery"] * risk_discovery
            + self._weights["timeline_efficiency"] * timeline_efficiency
        )

        return max(0.0, min(1.0, reward))

    def get_components(  # type: ignore[override]
        self,
        state: MADueDiligenceState,
        action: str,
        context: dict[str, Any],
    ) -> dict[str, float]:
        """
        Get individual reward components for debugging.

        Args:
            state: Current domain state
            action: Action being evaluated
            context: Additional context

        Returns:
            Dictionary mapping component names to their values
        """
        return {
            "information_gain": self._compute_information_gain(state, action, context),
            "risk_discovery": self._compute_risk_discovery(state, action, context),
            "timeline_efficiency": self._compute_timeline_efficiency(state, action),
            "total_reward": self.evaluate(state, action, context),
        }

    def _compute_information_gain(
        self,
        state: MADueDiligenceState,
        action: str,
        context: dict[str, Any],
    ) -> float:
        """
        Compute information gain from action.

        Higher reward for actions that uncover new information,
        with diminishing returns for repeated similar actions.
        """
        # Base information gain by action type
        action_base_gain = {
            "deep_dive_revenue": 0.8,
            "analyze_cost_structure": 0.75,
            "review_contracts": 0.75,
            "check_litigation_history": 0.85,
            "verify_ip_ownership": 0.7,
            "assess_regulatory_compliance": 0.8,
            "evaluate_operations": 0.65,
            "assess_supply_chain": 0.6,
            "review_hr_structure": 0.55,
            "analyze_it_infrastructure": 0.7,
            "assess_tech_stack": 0.75,
            "review_security_posture": 0.8,
            "evaluate_scalability": 0.65,
            "check_tech_debt": 0.7,
            "identify_revenue_synergies": 0.75,
            "identify_cost_synergies": 0.7,
            "escalate_to_expert": 0.6,
            "request_additional_docs": 0.5,
        }

        base = action_base_gain.get(action, 0.5)

        # Diminishing returns for repeated similar actions (configurable decay factor)
        similar_count = sum(1 for a in state.action_history if action.split("_")[0] in a)  # Same action category
        decay_factor = self._config.reward_decay_factor
        decay = decay_factor**similar_count

        return base * decay

    def _compute_risk_discovery(
        self,
        state: MADueDiligenceState,
        action: str,
        context: dict[str, Any],
    ) -> float:
        """
        Compute reward for risk discovery potential.

        Higher reward for actions likely to uncover risks,
        especially when few risks have been found.
        """
        # Actions with high risk discovery potential
        high_risk_actions = {
            "check_litigation_history",
            "assess_regulatory_compliance",
            "review_security_posture",
            "verify_ip_ownership",
            "evaluate_operations",
            "check_tech_debt",
        }

        medium_risk_actions = {
            "deep_dive_revenue",
            "analyze_cost_structure",
            "review_contracts",
            "assess_supply_chain",
        }

        if action in high_risk_actions:
            base_reward = 0.85
        elif action in medium_risk_actions:
            base_reward = 0.65
        else:
            base_reward = 0.4

        # Higher reward if few risks found so far (configurable thresholds)
        risk_count = len(state.risks_identified)
        low_threshold = self._config.risk_count_low_threshold
        high_threshold = self._config.risk_count_high_threshold

        if risk_count < low_threshold:
            multiplier = self._config.risk_low_count_multiplier
        elif risk_count < high_threshold:
            multiplier = self._config.risk_medium_count_multiplier
        else:
            multiplier = self._config.risk_high_count_multiplier

        return min(base_reward * multiplier, 1.0)

    def _compute_timeline_efficiency(
        self,
        state: MADueDiligenceState,
        action: str,
    ) -> float:
        """
        Compute reward for timeline-efficient actions.

        Rewards progression through phases while penalizing
        excessive time in any single phase.
        """
        action_count = len(state.action_history)
        max_analysis_depth = self._config.max_analysis_depth * 5

        # Get configurable thresholds
        early_threshold = self._config.timeline_depth_early_threshold
        late_threshold = self._config.timeline_depth_late_threshold

        # Penalize if taking too long (configurable penalties)
        depth_ratio = action_count / max_analysis_depth if max_analysis_depth > 0 else 0
        if depth_ratio > late_threshold:
            time_penalty = self._config.timeline_late_penalty
        elif depth_ratio > early_threshold:
            time_penalty = self._config.timeline_mid_penalty
        else:
            time_penalty = self._config.timeline_early_penalty

        # Reward phase progression
        if action == "proceed_to_next_phase":
            # Check if current phase has enough coverage (configurable minimum)
            phase_actions = sum(
                1 for a in state.action_history[-10:] if a not in ["proceed_to_next_phase", "revisit_previous_phase"]
            )
            min_coverage = self._config.min_phase_coverage_actions
            if phase_actions >= min_coverage:
                return 0.9 * time_penalty
            else:
                return 0.5 * time_penalty  # Too early to progress

        if action == "revisit_previous_phase":
            return 0.4 * time_penalty  # Generally discourage backtracking

        # Standard actions
        return 0.6 * time_penalty

    def get_phase_completion_reward(self, state: MADueDiligenceState) -> float:
        """
        Calculate bonus reward for completing a phase thoroughly.

        Args:
            state: Current state

        Returns:
            Bonus reward for phase completion
        """
        # Count distinct action types in current phase
        recent_actions = state.action_history[-15:]  # Last 15 actions
        distinct_actions = len(set(recent_actions))

        # Bonus for diverse actions
        diversity_bonus = min(distinct_actions / 8, 1.0) * 0.2

        # Bonus for risk coverage
        risk_coverage = min(len(state.risks_identified) / 5, 1.0) * 0.15

        # Bonus for document coverage
        doc_coverage = min(len(state.documents_analyzed) / 10, 1.0) * 0.1

        return diversity_bonus + risk_coverage + doc_coverage
