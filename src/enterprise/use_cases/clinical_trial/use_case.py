"""
Clinical Trial Design Optimizer Implementation.

MCTS-guided optimization of clinical trial designs to maximize
approval probability while minimizing cost and timeline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any
import uuid

if TYPE_CHECKING:
    from src.adapters.llm.base import LLMClient

from ...base.use_case import BaseDomainState, BaseUseCase
from ...config.enterprise_settings import ClinicalTrialConfig


@dataclass
class ClinicalTrialState(BaseDomainState):
    """State for clinical trial design optimization."""

    domain: str = "clinical_trial"

    # Trial parameters
    trial_phase: int = 1  # 1, 2, or 3
    indication: str = ""
    therapeutic_area: str = ""

    # Design parameters
    sample_size: int = 0
    duration_months: int = 0
    primary_endpoint: str = ""
    secondary_endpoints: list[str] = field(default_factory=list)

    # Cohort parameters
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    stratification_factors: list[str] = field(default_factory=list)

    # Statistical parameters
    statistical_power: float = 0.8
    alpha_level: float = 0.05
    expected_effect_size: float = 0.0

    # Optimization metrics
    approval_probability: float = 0.0
    estimated_cost: float = 0.0
    estimated_timeline_months: int = 0

    # MCTS tracking
    action_history: list[str] = field(default_factory=list)
    design_iterations: int = 0


class ClinicalTrialDesign(BaseUseCase[ClinicalTrialState]):
    """
    Clinical Trial Design Optimizer.

    Uses MCTS to explore the space of possible trial designs,
    optimizing for approval probability, cost, and timeline.
    """

    def __init__(
        self,
        config: ClinicalTrialConfig | None = None,
        llm_client: LLMClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        config = config or ClinicalTrialConfig()
        super().__init__(config=config, llm_client=llm_client, logger=logger)

    @property
    def name(self) -> str:
        return "clinical_trial_design"

    @property
    def domain(self) -> str:
        return "healthcare"

    def get_initial_state(
        self,
        query: str,
        context: dict[str, Any],
    ) -> ClinicalTrialState:
        """Create initial clinical trial state."""
        return ClinicalTrialState(
            state_id=f"ct_{uuid.uuid4().hex[:8]}",
            trial_phase=context.get("phase", 2),
            indication=context.get("indication", ""),
            therapeutic_area=context.get("therapeutic_area", ""),
            sample_size=context.get("initial_sample_size", 100),
            duration_months=context.get("duration_months", 12),
            statistical_power=self._config.min_statistical_power,
            alpha_level=self._config.alpha_level,
            features={"query": query},
        )

    def get_available_actions(self, state: ClinicalTrialState) -> list[str]:
        """Return available trial design actions."""
        actions = [
            "adjust_sample_size_up",
            "adjust_sample_size_down",
            "add_secondary_endpoint",
            "remove_secondary_endpoint",
            "modify_inclusion_criteria",
            "modify_exclusion_criteria",
            "add_stratification_factor",
            "extend_duration",
            "shorten_duration",
            "optimize_primary_endpoint",
            "run_power_analysis",
            "finalize_design",
        ]
        # Use configurable recent action window for deduplication
        recent_window = self._config.recent_action_window
        return [a for a in actions if a not in state.action_history[-recent_window:]]

    def apply_action(
        self,
        state: ClinicalTrialState,
        action: str,
    ) -> ClinicalTrialState:
        """Apply action to trial design state."""
        new_state = copy.deepcopy(state)
        new_state.state_id = f"{state.state_id}_{hash(action) % 10000}"
        new_state.action_history.append(action)
        new_state.design_iterations += 1

        # Get adjustment factors from configuration
        increase_factor = self._config.sample_size_increase_factor
        decrease_factor = self._config.sample_size_decrease_factor
        min_sample = self._config.min_sample_size
        duration_adjustment = self._config.duration_adjustment_months
        min_duration = self._config.min_trial_duration_months

        # Apply action effects using configurable factors
        if action == "adjust_sample_size_up":
            new_state.sample_size = int(new_state.sample_size * increase_factor)
        elif action == "adjust_sample_size_down":
            new_state.sample_size = max(min_sample, int(new_state.sample_size * decrease_factor))
        elif action == "extend_duration":
            new_state.duration_months += duration_adjustment
        elif action == "shorten_duration":
            new_state.duration_months = max(min_duration, new_state.duration_months - duration_adjustment)

        new_state.features["action_count"] = len(new_state.action_history)
        return new_state
