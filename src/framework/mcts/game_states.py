"""
Concrete GameState implementations for Neural MCTS.

Provides ready-to-use state implementations for common domains:
- ReasoningState: Multi-step reasoning tasks
- PlanningState: Sequential planning tasks
- DecisionState: Decision-making with multiple alternatives

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 5
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ReasoningState:
    """
    State for multi-step reasoning tasks.

    Represents a reasoning chain where each step builds on previous steps.
    Suitable for tasks like mathematical reasoning, logical deduction, etc.

    Attributes:
        problem: Original problem statement
        reasoning_steps: List of reasoning steps taken so far
        current_hypothesis: Current hypothesis or intermediate answer
        confidence: Confidence score for current state (0.0-1.0)
        max_steps: Maximum allowed reasoning steps
        metadata: Additional state metadata
    """

    problem: str
    reasoning_steps: list[str] = field(default_factory=list)
    current_hypothesis: str = ""
    confidence: float = 0.0
    max_steps: int = field(default_factory=lambda: get_settings().MCTS_MAX_DEPTH)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Action space configuration
    _action_types: list[str] = field(
        default_factory=lambda: [
            "decompose",  # Break problem into sub-problems
            "infer",  # Make logical inference
            "verify",  # Verify current hypothesis
            "generalize",  # Generalize from examples
            "specialize",  # Apply specific knowledge
            "conclude",  # Conclude reasoning chain
            "backtrack",  # Backtrack to previous state
            "ask_clarification",  # Request clarification
        ]
    )

    def get_legal_actions(self) -> list[dict[str, Any]]:
        """
        Return list of legal actions from this state.

        Returns actions based on current reasoning state:
        - Early: decompose, infer, specialize
        - Mid: verify, generalize, infer
        - Late: conclude, verify
        """
        actions = []

        step_count = len(self.reasoning_steps)

        # Early reasoning: focus on decomposition and inference
        if step_count < self.max_steps * 0.3:
            for action_type in ["decompose", "infer", "specialize"]:
                actions.append({"type": action_type, "step": step_count})

        # Mid reasoning: verification and generalization
        elif step_count < self.max_steps * 0.7:
            for action_type in ["verify", "generalize", "infer"]:
                actions.append({"type": action_type, "step": step_count})

            # Allow backtracking if confidence is low
            if self.confidence < 0.3 and step_count > 1:
                actions.append({"type": "backtrack", "step": step_count})

        # Late reasoning: conclude or verify
        else:
            actions.append({"type": "conclude", "step": step_count})
            if self.confidence < 0.8:
                actions.append({"type": "verify", "step": step_count})

        # Always allow asking for clarification
        if "unclear" in self.problem.lower() or "ambiguous" in self.problem.lower():
            actions.append({"type": "ask_clarification", "step": step_count})

        logger.debug(
            "Generated %d legal actions for step %d",
            len(actions),
            step_count,
            extra={"state_hash": self.get_hash()[:8]},
        )

        return actions

    def apply_action(self, action: dict[str, Any]) -> ReasoningState:
        """
        Apply action and return new state.

        Args:
            action: Action dict with 'type' and optional parameters

        Returns:
            New ReasoningState with action applied
        """
        action_type = action.get("type", "infer")
        new_steps = self.reasoning_steps.copy()

        # Determine new hypothesis and confidence based on action
        new_hypothesis = self.current_hypothesis
        new_confidence = self.confidence

        if action_type == "decompose":
            new_steps.append(f"[DECOMPOSE] Breaking down: {self.problem[:50]}...")
            new_confidence = min(1.0, self.confidence + 0.1)

        elif action_type == "infer":
            new_steps.append(f"[INFER] Drawing inference from step {len(new_steps)}")
            new_confidence = min(1.0, self.confidence + 0.15)

        elif action_type == "verify":
            new_steps.append(f"[VERIFY] Verifying hypothesis: {new_hypothesis[:30]}...")
            new_confidence = min(1.0, self.confidence + 0.2)

        elif action_type == "generalize":
            new_steps.append("[GENERALIZE] Generalizing from current findings")
            new_confidence = min(1.0, self.confidence + 0.1)

        elif action_type == "specialize":
            new_steps.append("[SPECIALIZE] Applying domain-specific knowledge")
            new_confidence = min(1.0, self.confidence + 0.15)

        elif action_type == "conclude":
            new_steps.append(f"[CONCLUDE] Final answer: {new_hypothesis}")
            new_confidence = self.confidence  # Keep current confidence

        elif action_type == "backtrack":
            # Remove last step and reduce confidence
            if new_steps:
                new_steps = new_steps[:-1]
            new_confidence = max(0.0, self.confidence - 0.1)

        elif action_type == "ask_clarification":
            new_steps.append("[CLARIFY] Requesting clarification")
            new_confidence = self.confidence

        new_metadata = self.metadata.copy()
        new_metadata["last_action"] = action_type

        return ReasoningState(
            problem=self.problem,
            reasoning_steps=new_steps,
            current_hypothesis=new_hypothesis,
            confidence=new_confidence,
            max_steps=self.max_steps,
            metadata=new_metadata,
        )

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        # Terminal if concluded or max steps reached
        if self.reasoning_steps:
            last_step = self.reasoning_steps[-1]
            if "[CONCLUDE]" in last_step:
                return True

        return len(self.reasoning_steps) >= self.max_steps

    def get_reward(self, player: int = 1) -> float:
        """
        Get reward for the reasoning chain.

        Args:
            player: Not used for single-player reasoning (default 1)

        Returns:
            Reward based on confidence and reasoning quality
        """
        if not self.is_terminal():
            return 0.0

        # Base reward from confidence
        reward = self.confidence

        # Bonus for reaching conclusion efficiently
        efficiency_bonus = max(0, 1.0 - len(self.reasoning_steps) / self.max_steps) * 0.2

        # Penalty for backtracking
        backtrack_count = sum(1 for s in self.reasoning_steps if "[BACKTRACK]" in s)
        backtrack_penalty = backtrack_count * 0.05

        return max(0.0, min(1.0, reward + efficiency_bonus - backtrack_penalty))

    def to_tensor(self) -> torch.Tensor:
        """
        Convert state to tensor for neural network input.

        Creates a feature vector encoding:
        - Problem embedding (placeholder - use actual embedding in production)
        - Step count normalized
        - Confidence
        - Action type distribution
        """
        settings = get_settings()
        feature_dim = getattr(settings, "STATE_FEATURE_DIM", 128)

        features = torch.zeros(feature_dim)

        # Encode step progress
        features[0] = len(self.reasoning_steps) / self.max_steps

        # Encode confidence
        features[1] = self.confidence

        # Encode action type distribution in recent steps
        for i, action_type in enumerate(self._action_types):
            count = sum(1 for s in self.reasoning_steps if f"[{action_type.upper()}]" in s)
            if i + 2 < feature_dim:
                features[i + 2] = min(1.0, count / max(1, len(self.reasoning_steps)))

        # Encode problem length (normalized)
        features[10] = min(1.0, len(self.problem) / 1000)

        # Encode if terminal
        features[11] = 1.0 if self.is_terminal() else 0.0

        return features

    def get_canonical_form(self, player: int) -> ReasoningState:
        """Get state (no player perspective needed for reasoning)."""
        return self

    def get_hash(self) -> str:
        """Get unique hash for this state."""
        hash_data = f"{self.problem}|{','.join(self.reasoning_steps)}|{self.confidence}"
        return hashlib.sha256(hash_data.encode()).hexdigest()[:16]

    def action_to_index(self, action: dict[str, Any]) -> int:
        """Map action to index in action space."""
        action_type = action.get("type", "infer")
        try:
            return self._action_types.index(action_type)
        except ValueError:
            return 0


@dataclass
class PlanningState:
    """
    State for sequential planning tasks.

    Represents a planning problem with goals, resources, and constraints.
    Suitable for task planning, resource allocation, scheduling, etc.

    Attributes:
        goal: Target goal description
        current_state: Current state description
        available_actions: List of available action types
        completed_actions: Actions already taken
        resources: Available resources (dict)
        constraints: Planning constraints
        time_remaining: Remaining time budget
        max_actions: Maximum actions allowed
    """

    goal: str
    current_state: str
    available_actions: list[str] = field(default_factory=list)
    completed_actions: list[dict[str, Any]] = field(default_factory=list)
    resources: dict[str, float] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    time_remaining: float = field(default_factory=lambda: get_settings().MCTS_SEARCH_TIMEOUT)
    max_actions: int = field(default_factory=lambda: get_settings().MCTS_MAX_DEPTH)

    def get_legal_actions(self) -> list[dict[str, Any]]:
        """Return legal actions based on current resources and constraints."""
        actions = []

        for action_name in self.available_actions:
            # Check resource requirements (simplified)
            cost = self._get_action_cost(action_name)

            # Check if we have enough resources
            can_afford = all(self.resources.get(resource, 0) >= amount for resource, amount in cost.items())

            if can_afford and len(self.completed_actions) < self.max_actions:
                actions.append(
                    {
                        "name": action_name,
                        "cost": cost,
                        "step": len(self.completed_actions),
                    }
                )

        # Always allow "wait" or "finish" actions
        if not actions:
            actions.append({"name": "wait", "cost": {}, "step": len(self.completed_actions)})

        return actions

    def _get_action_cost(self, action_name: str) -> dict[str, float]:
        """Get resource cost for an action (configurable)."""
        # Default costs - override in subclass for domain-specific costs
        default_costs = {
            "analyze": {"time": 1.0, "compute": 0.5},
            "execute": {"time": 2.0, "compute": 1.0},
            "verify": {"time": 0.5, "compute": 0.2},
            "optimize": {"time": 1.5, "compute": 0.8},
            "wait": {},
            "finish": {},
        }
        return default_costs.get(action_name, {"time": 1.0})

    def apply_action(self, action: dict[str, Any]) -> PlanningState:
        """Apply action and return new state."""
        action_name = action.get("name", "wait")
        cost = action.get("cost", {})

        # Deduct resources
        new_resources = self.resources.copy()
        for resource, amount in cost.items():
            new_resources[resource] = new_resources.get(resource, 0) - amount

        # Record completed action
        new_completed = self.completed_actions.copy()
        new_completed.append(action)

        # Update current state description
        new_current_state = f"{self.current_state} -> {action_name}"

        return PlanningState(
            goal=self.goal,
            current_state=new_current_state,
            available_actions=self.available_actions,
            completed_actions=new_completed,
            resources=new_resources,
            constraints=self.constraints,
            time_remaining=self.time_remaining - cost.get("time", 0),
            max_actions=self.max_actions,
        )

    def is_terminal(self) -> bool:
        """Check if planning is complete or resources exhausted."""
        # Check if goal achieved (simplified - check for "finish" action)
        if self.completed_actions and self.completed_actions[-1].get("name") == "finish":
            return True

        # Check resource exhaustion
        if self.time_remaining <= 0:
            return True

        # Check max actions
        return len(self.completed_actions) >= self.max_actions

    def get_reward(self, player: int = 1) -> float:
        """Get reward based on goal achievement and efficiency."""
        if not self.is_terminal():
            return 0.0

        # Base reward for completion
        completed_with_finish = self.completed_actions and self.completed_actions[-1].get("name") == "finish"
        base_reward = 0.8 if completed_with_finish else 0.3

        # Efficiency bonus
        efficiency = 1.0 - len(self.completed_actions) / self.max_actions
        efficiency_bonus = efficiency * 0.2

        # Resource utilization (using resources efficiently is good)
        total_resources = sum(self.resources.values())
        resource_penalty = max(0, total_resources - 1.0) * 0.1  # Penalty for wasted resources

        return max(0.0, min(1.0, base_reward + efficiency_bonus - resource_penalty))

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor."""
        settings = get_settings()
        feature_dim = getattr(settings, "STATE_FEATURE_DIM", 128)

        features = torch.zeros(feature_dim)

        # Progress
        features[0] = len(self.completed_actions) / self.max_actions

        # Time remaining (normalized)
        features[1] = self.time_remaining / get_settings().MCTS_SEARCH_TIMEOUT

        # Resource levels
        for i, (_resource, amount) in enumerate(list(self.resources.items())[:10]):
            if i + 2 < feature_dim:
                features[i + 2] = min(1.0, amount / 10.0)

        # Number of available actions
        features[12] = len(self.available_actions) / 20.0

        # Terminal state indicator
        features[13] = 1.0 if self.is_terminal() else 0.0

        return features

    def get_hash(self) -> str:
        """Get unique hash for this state."""
        action_str = "|".join(str(a) for a in self.completed_actions)
        resource_str = "|".join(f"{k}:{v}" for k, v in sorted(self.resources.items()))
        hash_data = f"{self.goal}|{action_str}|{resource_str}"
        return hashlib.sha256(hash_data.encode()).hexdigest()[:16]

    def action_to_index(self, action: dict[str, Any]) -> int:
        """Map action to index."""
        action_name = action.get("name", "wait")
        try:
            return self.available_actions.index(action_name)
        except ValueError:
            return len(self.available_actions)  # Unknown action


@dataclass
class DecisionState:
    """
    State for decision-making tasks with multiple alternatives.

    Represents a decision point with options to evaluate.
    Suitable for recommendation, selection, multi-criteria decisions.

    Attributes:
        context: Decision context description
        options: Available options to choose from
        evaluated_options: Options already evaluated with scores
        criteria: Evaluation criteria with weights
        decision_history: History of sub-decisions made
        max_evaluations: Maximum evaluations allowed
    """

    context: str
    options: list[dict[str, Any]] = field(default_factory=list)
    evaluated_options: dict[str, float] = field(default_factory=dict)
    criteria: dict[str, float] = field(default_factory=dict)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    max_evaluations: int = field(default_factory=lambda: get_settings().MCTS_ITERATIONS)

    def get_legal_actions(self) -> list[dict[str, Any]]:
        """Return legal actions: evaluate, compare, or decide."""
        actions = []

        # Evaluate unevaluated options
        for option in self.options:
            option_id = option.get("id", str(option))
            if option_id not in self.evaluated_options:
                actions.append({"type": "evaluate", "option_id": option_id, "option": option})

        # Compare if we have at least 2 evaluated options
        if len(self.evaluated_options) >= 2:
            actions.append({"type": "compare", "options": list(self.evaluated_options.keys())})

        # Decide if we have evaluated enough or reached limit
        if self.evaluated_options:
            actions.append(
                {
                    "type": "decide",
                    "best_option": max(self.evaluated_options, key=lambda k: self.evaluated_options[k]),
                }
            )

        return actions if actions else [{"type": "decide", "best_option": None}]

    def apply_action(self, action: dict[str, Any]) -> DecisionState:
        """Apply action and return new state."""
        action_type = action.get("type", "evaluate")

        new_evaluated = self.evaluated_options.copy()
        new_history = self.decision_history.copy()

        if action_type == "evaluate":
            option_id = action.get("option_id", "unknown")
            # Simulate evaluation score (in production, this would call actual evaluation)
            score = hash(option_id) % 100 / 100.0
            new_evaluated[option_id] = score
            new_history.append({"action": "evaluate", "option": option_id, "score": score})

        elif action_type == "compare":
            new_history.append({"action": "compare", "options": action.get("options", []), "result": "compared"})

        elif action_type == "decide":
            new_history.append({"action": "decide", "selected": action.get("best_option"), "final": True})

        return DecisionState(
            context=self.context,
            options=self.options,
            evaluated_options=new_evaluated,
            criteria=self.criteria,
            decision_history=new_history,
            max_evaluations=self.max_evaluations,
        )

    def is_terminal(self) -> bool:
        """Check if decision is made."""
        if self.decision_history:
            return bool(self.decision_history[-1].get("final", False))
        return len(self.evaluated_options) >= self.max_evaluations

    def get_reward(self, player: int = 1) -> float:
        """Get reward based on decision quality."""
        if not self.is_terminal():
            return 0.0

        if not self.evaluated_options:
            return 0.0

        # Reward is the score of the selected option
        best_score = max(self.evaluated_options.values())

        # Efficiency bonus for fewer evaluations
        efficiency = 1.0 - len(self.evaluated_options) / max(1, len(self.options))
        efficiency_bonus = efficiency * 0.1

        return min(1.0, best_score + efficiency_bonus)

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor."""
        settings = get_settings()
        feature_dim = getattr(settings, "STATE_FEATURE_DIM", 128)

        features = torch.zeros(feature_dim)

        # Progress
        features[0] = len(self.evaluated_options) / max(1, len(self.options))

        # Best score so far
        if self.evaluated_options:
            features[1] = max(self.evaluated_options.values())
            features[2] = sum(self.evaluated_options.values()) / len(self.evaluated_options)
        else:
            features[1] = 0.0
            features[2] = 0.0

        # Number of options remaining
        features[3] = (len(self.options) - len(self.evaluated_options)) / max(1, len(self.options))

        # Decision made indicator
        features[4] = 1.0 if self.is_terminal() else 0.0

        return features

    def get_hash(self) -> str:
        """Get unique hash for this state."""
        eval_str = "|".join(f"{k}:{v:.3f}" for k, v in sorted(self.evaluated_options.items()))
        history_str = "|".join(str(h) for h in self.decision_history)
        hash_data = f"{self.context}|{eval_str}|{history_str}"
        return hashlib.sha256(hash_data.encode()).hexdigest()[:16]

    def action_to_index(self, action: dict[str, Any]) -> int:
        """Map action to index."""
        action_type = action.get("type", "evaluate")
        type_map = {"evaluate": 0, "compare": 1, "decide": 2}
        return type_map.get(action_type, 0)


# Factory function for creating states from configuration
def create_game_state(
    state_type: str,
    **kwargs: Any,
) -> ReasoningState | PlanningState | DecisionState:
    """
    Factory function to create game states from configuration.

    Args:
        state_type: Type of state ('reasoning', 'planning', 'decision')
        **kwargs: State-specific parameters

    Returns:
        Configured state instance

    Raises:
        ValueError: If state_type is not recognized
    """
    state_classes = {
        "reasoning": ReasoningState,
        "planning": PlanningState,
        "decision": DecisionState,
    }

    if state_type not in state_classes:
        raise ValueError(f"Unknown state type: {state_type}. Valid types: {list(state_classes.keys())}")

    state_class = state_classes[state_type]

    # Filter kwargs to only include valid fields for the dataclass
    valid_fields = {f.name for f in dataclasses.fields(state_class)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    result: ReasoningState | PlanningState | DecisionState = state_class(**filtered_kwargs)
    return result


__all__ = [
    "ReasoningState",
    "PlanningState",
    "DecisionState",
    "create_game_state",
]
