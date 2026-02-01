"""
Process Reward Model (PRM) for Step-Level Evaluation.

Based on OpenAI's "Let's Verify Step by Step" research, PRMs provide
per-step credit assignment rather than sparse final-answer signals,
enabling precise guidance through the MCTS tree.

Key features:
- Step-level scoring for reasoning trajectories
- Monte Carlo estimation for step quality (ReST-MCTS* approach)
- Integration with MCTS selection and expansion phases
- Automatic training data collection from tree search

References:
- "Let's Verify Step by Step" (OpenAI)
- "ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search"
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class ReasoningStep:
    """
    A single step in a reasoning trajectory.

    Represents one reasoning action/thought in a chain of reasoning,
    along with metadata for PRM evaluation.
    """

    content: str
    """The actual reasoning content/text"""

    step_index: int
    """Position in the reasoning chain (0-indexed)"""

    step_type: str = "reasoning"
    """Type of step: 'reasoning', 'action', 'observation', 'reflection'"""

    confidence: float = 0.0
    """Model's self-reported confidence in this step"""

    thinking_tokens: int = 0
    """Number of tokens used for extended thinking"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (e.g., tool calls, intermediate results)"""

    def to_hash_key(self) -> str:
        """Generate a unique hash for this step."""
        combined = f"{self.step_index}:{self.step_type}:{self.content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


@dataclass
class ReasoningTrajectory:
    """
    A complete reasoning trajectory (sequence of steps).

    Used for training PRMs and evaluating reasoning quality.
    """

    steps: list[ReasoningStep] = field(default_factory=list)
    """Ordered list of reasoning steps"""

    query: str = ""
    """Original query/problem that initiated the reasoning"""

    final_answer: str | None = None
    """Final answer produced by the trajectory"""

    is_correct: bool | None = None
    """Whether the final answer is correct (ground truth)"""

    outcome_reward: float = 0.0
    """Sparse outcome-based reward (0 or 1 typically)"""

    step_rewards: list[float] = field(default_factory=list)
    """Dense step-level rewards from PRM"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional trajectory metadata"""

    def __post_init__(self):
        """Validate trajectory consistency."""
        if self.step_rewards and len(self.step_rewards) != len(self.steps):
            raise ValueError(
                f"step_rewards length ({len(self.step_rewards)}) must match "
                f"steps length ({len(self.steps)})"
            )

    def add_step(self, step: ReasoningStep) -> None:
        """Add a step to the trajectory."""
        step.step_index = len(self.steps)
        self.steps.append(step)

    def get_prefix(self, up_to_step: int) -> ReasoningTrajectory:
        """Get a prefix of the trajectory up to (not including) the given step."""
        return ReasoningTrajectory(
            steps=self.steps[:up_to_step],
            query=self.query,
            metadata=self.metadata.copy(),
        )

    def to_text(self, include_query: bool = True) -> str:
        """Convert trajectory to readable text format."""
        parts = []
        if include_query and self.query:
            parts.append(f"Query: {self.query}\n")

        for i, step in enumerate(self.steps):
            prefix = f"Step {i + 1} [{step.step_type}]: "
            parts.append(f"{prefix}{step.content}")

        if self.final_answer:
            parts.append(f"\nFinal Answer: {self.final_answer}")

        return "\n".join(parts)

    def to_hash_key(self) -> str:
        """Generate a unique hash for this trajectory."""
        step_hashes = [s.to_hash_key() for s in self.steps]
        combined = f"{self.query}:" + ":".join(step_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()


@dataclass
class PRMScore:
    """
    Score output from a Process Reward Model.

    Contains both the step-level score and metadata for debugging/analysis.
    """

    step_score: float
    """Quality score for the step (typically 0.0 to 1.0)"""

    cumulative_score: float
    """Cumulative score considering all previous steps"""

    confidence: float = 1.0
    """Model's confidence in this score"""

    reasoning: str = ""
    """Optional reasoning for the score (for interpretability)"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional scoring metadata"""


class ProcessRewardModel(ABC):
    """
    Abstract base class for Process Reward Models.

    PRMs score individual reasoning steps rather than just final outcomes,
    enabling more precise credit assignment and better MCTS guidance.
    """

    @abstractmethod
    async def score_step(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory,
    ) -> PRMScore:
        """
        Score a single reasoning step in context.

        Args:
            step: The step to score
            trajectory: Full trajectory context (step should be in trajectory)

        Returns:
            PRMScore with step quality assessment
        """
        pass

    @abstractmethod
    async def score_trajectory(
        self,
        trajectory: ReasoningTrajectory,
    ) -> list[PRMScore]:
        """
        Score all steps in a trajectory.

        Args:
            trajectory: Complete reasoning trajectory

        Returns:
            List of PRMScores, one for each step
        """
        pass

    async def estimate_trajectory_value(
        self,
        trajectory: ReasoningTrajectory,
        aggregation: str = "product",
    ) -> float:
        """
        Estimate the value/quality of a trajectory.

        Args:
            trajectory: The trajectory to evaluate
            aggregation: How to aggregate step scores:
                - "product": Multiply all step probabilities
                - "min": Take minimum step score
                - "mean": Average all step scores
                - "last": Use only the last step's score

        Returns:
            Estimated trajectory value (0.0 to 1.0)
        """
        scores = await self.score_trajectory(trajectory)

        if not scores:
            return 0.0

        step_scores = [s.step_score for s in scores]

        if aggregation == "product":
            return float(np.prod(step_scores))
        elif aggregation == "min":
            return float(np.min(step_scores))
        elif aggregation == "mean":
            return float(np.mean(step_scores))
        elif aggregation == "last":
            return scores[-1].step_score
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")


class LLMProcessRewardModel(ProcessRewardModel):
    """
    LLM-based Process Reward Model.

    Uses an LLM to evaluate the quality of each reasoning step.
    Suitable for domains where heuristic evaluation is insufficient.
    """

    def __init__(
        self,
        evaluate_fn: Callable[[str], Any],
        cache_size: int = 1000,
        temperature: float = 0.2,
    ):
        """
        Initialize LLM-based PRM.

        Args:
            evaluate_fn: Async function to call LLM with prompt
            cache_size: Maximum number of cached evaluations
            temperature: LLM temperature for evaluation
        """
        self.evaluate_fn = evaluate_fn
        self.temperature = temperature

        # LRU cache for evaluations
        self._cache: OrderedDict[str, PRMScore] = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    async def score_step(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory,
    ) -> PRMScore:
        """Score a step using LLM evaluation."""
        # Build cache key
        context_key = trajectory.get_prefix(step.step_index).to_hash_key()
        cache_key = f"{context_key}:{step.to_hash_key()}"

        # Check cache
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            self.cache_hits += 1
            return self._cache[cache_key]

        self.cache_misses += 1

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(step, trajectory)

        # Get LLM evaluation
        result = await self.evaluate_fn(prompt)

        # Parse score from result
        score = self._parse_score(result, step)

        # Cache result
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = score

        return score

    async def score_trajectory(
        self,
        trajectory: ReasoningTrajectory,
    ) -> list[PRMScore]:
        """Score all steps in trajectory."""
        scores = []
        for step in trajectory.steps:
            score = await self.score_step(step, trajectory)
            scores.append(score)
        return scores

    def _build_evaluation_prompt(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory,
    ) -> str:
        """Build the LLM prompt for step evaluation."""
        # Get context (steps before current step)
        context = trajectory.get_prefix(step.step_index)

        return f"""Evaluate the quality of the following reasoning step.

Problem: {trajectory.query}

Previous reasoning steps:
{context.to_text(include_query=False) if context.steps else "(None - this is the first step)"}

Current step to evaluate:
Step {step.step_index + 1} [{step.step_type}]: {step.content}

Rate this step on a scale of 0.0 to 1.0, where:
- 1.0: Correct, relevant, and advances toward the solution
- 0.8: Mostly correct with minor issues
- 0.6: Partially correct or somewhat relevant
- 0.4: Contains significant errors or is tangential
- 0.2: Mostly incorrect or irrelevant
- 0.0: Completely wrong or harmful to the solution

Respond with:
SCORE: [0.0-1.0]
REASONING: [Brief explanation]
"""

    def _parse_score(self, result: Any, step: ReasoningStep) -> PRMScore:
        """Parse LLM output into PRMScore."""
        # Handle different result types
        if isinstance(result, dict):
            text = result.get("text", str(result))
        elif hasattr(result, "text"):
            text = result.text
        else:
            text = str(result)

        # Parse score from text
        score = 0.5  # Default
        reasoning = ""

        lines = text.strip().split("\n")
        for line in lines:
            if line.upper().startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return PRMScore(
            step_score=score,
            cumulative_score=score,  # Will be updated during aggregation
            confidence=step.confidence,
            reasoning=reasoning,
        )

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class MonteCarloProcessRewardModel(ProcessRewardModel):
    """
    Monte Carlo-based Process Reward Model (ReST-MCTS* approach).

    Estimates step quality using Monte Carlo rollouts from the current step.
    Step quality = P(correct final answer | step taken)

    This is the approach used in ReST-MCTS* for automatic PRM training.
    """

    def __init__(
        self,
        rollout_fn: Callable[[ReasoningTrajectory], Any],
        verify_fn: Callable[[str, str], bool],
        num_rollouts: int = 8,
        discount_factor: float = 0.99,
    ):
        """
        Initialize Monte Carlo PRM.

        Args:
            rollout_fn: Function to complete a trajectory from current state
            verify_fn: Function to verify if answer is correct
            num_rollouts: Number of rollouts per step
            discount_factor: Discount for future steps
        """
        self.rollout_fn = rollout_fn
        self.verify_fn = verify_fn
        self.num_rollouts = num_rollouts
        self.discount_factor = discount_factor

        # Statistics
        self.total_rollouts = 0
        self.successful_rollouts = 0

    async def score_step(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory,
    ) -> PRMScore:
        """
        Score step using Monte Carlo estimation.

        Performs rollouts from the step to estimate P(correct | step).
        """
        # Get trajectory prefix up to and including this step
        prefix = trajectory.get_prefix(step.step_index + 1)

        # Perform rollouts
        successes = 0
        for _ in range(self.num_rollouts):
            # Complete trajectory from current step
            completed = await self.rollout_fn(prefix)

            self.total_rollouts += 1

            # Check if completion is correct
            if completed.final_answer and completed.is_correct is not None:
                if completed.is_correct:
                    successes += 1
                    self.successful_rollouts += 1
            elif completed.final_answer and trajectory.query:
                # Use verify_fn if ground truth not available
                if self.verify_fn(trajectory.query, completed.final_answer):
                    successes += 1
                    self.successful_rollouts += 1

        # Estimate step quality as success rate
        step_score = successes / self.num_rollouts

        # Apply discount based on step position
        discount = self.discount_factor ** step.step_index
        cumulative_score = step_score * discount

        return PRMScore(
            step_score=step_score,
            cumulative_score=cumulative_score,
            confidence=1.0 - (1.0 / (self.num_rollouts + 1)),  # Confidence based on sample size
            reasoning=f"Monte Carlo estimate: {successes}/{self.num_rollouts} successful rollouts",
            metadata={
                "num_rollouts": self.num_rollouts,
                "successes": successes,
                "discount": discount,
            },
        )

    async def score_trajectory(
        self,
        trajectory: ReasoningTrajectory,
    ) -> list[PRMScore]:
        """Score all steps in trajectory using Monte Carlo."""
        scores = []
        for step in trajectory.steps:
            score = await self.score_step(step, trajectory)
            scores.append(score)
        return scores


class HeuristicProcessRewardModel(ProcessRewardModel):
    """
    Heuristic-based Process Reward Model.

    Uses domain-specific heuristics for fast step evaluation.
    Useful when LLM evaluation is too expensive or slow.
    """

    def __init__(
        self,
        heuristics: Sequence[Callable[[ReasoningStep, ReasoningTrajectory], float]],
        weights: Sequence[float] | None = None,
    ):
        """
        Initialize heuristic PRM.

        Args:
            heuristics: List of heuristic functions (step, trajectory) -> score
            weights: Weights for each heuristic (defaults to uniform)
        """
        self.heuristics = list(heuristics)
        self.weights = list(weights) if weights else [1.0] * len(heuristics)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    async def score_step(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory,
    ) -> PRMScore:
        """Score step using heuristics."""
        scores = []
        for heuristic, weight in zip(self.heuristics, self.weights, strict=True):
            try:
                h_score = heuristic(step, trajectory)
                scores.append(h_score * weight)
            except Exception:
                scores.append(0.5 * weight)  # Default on error

        step_score = sum(scores)

        return PRMScore(
            step_score=step_score,
            cumulative_score=step_score,
            confidence=0.8,  # Lower confidence for heuristics
            reasoning=f"Heuristic evaluation with {len(self.heuristics)} heuristics",
        )

    async def score_trajectory(
        self,
        trajectory: ReasoningTrajectory,
    ) -> list[PRMScore]:
        """Score all steps using heuristics."""
        scores = []
        for step in trajectory.steps:
            score = await self.score_step(step, trajectory)
            scores.append(score)
        return scores


class EnsemblePRM(ProcessRewardModel):
    """
    Ensemble of multiple Process Reward Models.

    Combines multiple PRMs for more robust step evaluation.
    """

    def __init__(
        self,
        models: Sequence[ProcessRewardModel],
        weights: Sequence[float] | None = None,
        aggregation: str = "weighted_mean",
    ):
        """
        Initialize ensemble PRM.

        Args:
            models: List of PRM instances
            weights: Weights for each model (defaults to uniform)
            aggregation: How to combine scores:
                - "weighted_mean": Weighted average
                - "min": Minimum across models
                - "max": Maximum across models
        """
        self.models = list(models)
        self.weights = list(weights) if weights else [1.0] * len(models)
        self.aggregation = aggregation

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    async def score_step(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory,
    ) -> PRMScore:
        """Score step using ensemble of models."""
        # Collect scores from all models
        all_scores: list[PRMScore] = []
        for model in self.models:
            score = await model.score_step(step, trajectory)
            all_scores.append(score)

        # Aggregate scores
        step_scores = [s.step_score for s in all_scores]

        if self.aggregation == "weighted_mean":
            final_score = sum(s * w for s, w in zip(step_scores, self.weights, strict=True))
        elif self.aggregation == "min":
            final_score = min(step_scores)
        elif self.aggregation == "max":
            final_score = max(step_scores)
        else:
            final_score = np.mean(step_scores)

        # Combine confidence (lower if models disagree)
        confidence = 1.0 - np.std(step_scores)

        return PRMScore(
            step_score=final_score,
            cumulative_score=final_score,
            confidence=confidence,
            reasoning=f"Ensemble of {len(self.models)} models",
            metadata={
                "individual_scores": step_scores,
                "aggregation": self.aggregation,
            },
        )

    async def score_trajectory(
        self,
        trajectory: ReasoningTrajectory,
    ) -> list[PRMScore]:
        """Score all steps using ensemble."""
        scores = []
        for step in trajectory.steps:
            score = await self.score_step(step, trajectory)
            scores.append(score)
        return scores


# ============================================================================
# PRM Integration with MCTS
# ============================================================================


@dataclass
class PRMEnhancedMCTSConfig:
    """Configuration for PRM-enhanced MCTS."""

    # PRM weighting in selection
    prm_selection_weight: float = 0.5
    """Weight for PRM score in UCB selection (0-1)"""

    # Expansion filtering
    prm_expansion_threshold: float = 0.3
    """Minimum PRM score for a candidate to be expanded"""

    prm_expansion_top_k: int = 5
    """Maximum candidates to keep after PRM filtering"""

    # Backpropagation
    use_prm_for_backprop: bool = True
    """Use PRM step scores for backpropagation instead of outcome rewards"""

    prm_backprop_discount: float = 0.95
    """Discount factor for PRM scores during backpropagation"""

    # Caching
    cache_prm_scores: bool = True
    """Cache PRM scores for repeated evaluations"""


class PRMMCTSIntegration:
    """
    Integrates Process Reward Model with MCTS.

    Provides methods for PRM-enhanced selection, expansion filtering,
    and backpropagation.
    """

    def __init__(
        self,
        prm: ProcessRewardModel,
        config: PRMEnhancedMCTSConfig | None = None,
    ):
        """
        Initialize PRM-MCTS integration.

        Args:
            prm: Process Reward Model instance
            config: Configuration for integration
        """
        self.prm = prm
        self.config = config or PRMEnhancedMCTSConfig()

        # Score cache
        self._score_cache: dict[str, float] = {}

    async def enhanced_uct_score(
        self,
        node_value: float,
        node_visits: int,
        parent_visits: int,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory,
        exploration_weight: float = 1.414,
    ) -> float:
        """
        Compute PRM-enhanced UCT score.

        Combines standard UCB1 with PRM step quality score.

        Args:
            node_value: Node's average value
            node_visits: Number of visits to this node
            parent_visits: Number of visits to parent
            step: Reasoning step for this node
            trajectory: Full trajectory context
            exploration_weight: UCB exploration constant

        Returns:
            Combined UCT + PRM score
        """
        import math

        # Standard UCB1
        if node_visits == 0:
            base_uct = float("inf")
        else:
            exploitation = node_value / node_visits
            exploration = exploration_weight * math.sqrt(
                math.log(parent_visits) / node_visits
            )
            base_uct = exploitation + exploration

        # Get PRM score
        prm_score = await self._get_cached_prm_score(step, trajectory)

        # Combine scores
        alpha = self.config.prm_selection_weight
        if base_uct == float("inf"):
            return base_uct

        return (1 - alpha) * base_uct + alpha * prm_score

    async def filter_expansion_candidates(
        self,
        candidates: list[ReasoningStep],
        trajectory: ReasoningTrajectory,
    ) -> list[tuple[ReasoningStep, float]]:
        """
        Filter expansion candidates using PRM scores.

        Args:
            candidates: List of candidate steps to evaluate
            trajectory: Current trajectory context

        Returns:
            List of (step, score) tuples, sorted by score descending
        """
        scored_candidates: list[tuple[ReasoningStep, float]] = []

        for candidate in candidates:
            score = await self._get_cached_prm_score(candidate, trajectory)

            # Filter by threshold
            if score >= self.config.prm_expansion_threshold:
                scored_candidates.append((candidate, score))

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Keep top-k
        return scored_candidates[: self.config.prm_expansion_top_k]

    async def compute_backprop_value(
        self,
        trajectory: ReasoningTrajectory,
        outcome_value: float,
    ) -> float:
        """
        Compute backpropagation value using PRM scores.

        Instead of sparse outcome rewards, uses dense PRM step scores.

        Args:
            trajectory: Complete trajectory to terminal state
            outcome_value: Outcome-based reward

        Returns:
            Value to backpropagate (PRM-weighted)
        """
        if not self.config.use_prm_for_backprop:
            return outcome_value

        # Get PRM scores for all steps
        prm_scores = await self.prm.score_trajectory(trajectory)

        if not prm_scores:
            return outcome_value

        # Compute discounted sum of step scores
        discounted_sum = 0.0
        discount = 1.0

        for score in prm_scores:
            discounted_sum += discount * score.step_score
            discount *= self.config.prm_backprop_discount

        # Normalize to [0, 1] range
        max_possible = sum(
            self.config.prm_backprop_discount ** i
            for i in range(len(prm_scores))
        )

        prm_value = discounted_sum / max_possible if max_possible > 0 else 0.5

        # Combine with outcome value (outcome still matters for correctness)
        combined = 0.7 * prm_value + 0.3 * outcome_value

        return combined

    async def _get_cached_prm_score(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory,
    ) -> float:
        """Get PRM score with caching."""
        if not self.config.cache_prm_scores:
            score = await self.prm.score_step(step, trajectory)
            return score.step_score

        cache_key = f"{trajectory.to_hash_key()}:{step.to_hash_key()}"

        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        score = await self.prm.score_step(step, trajectory)
        self._score_cache[cache_key] = score.step_score

        return score.step_score

    def clear_cache(self) -> None:
        """Clear the score cache."""
        self._score_cache.clear()


# ============================================================================
# PRM Training Data Collection
# ============================================================================


@dataclass
class PRMTrainingExample:
    """
    Training example for Process Reward Model.

    Contains a step and its label (from Monte Carlo estimation or ground truth).
    """

    trajectory_prefix: str
    """Context: all steps before this one"""

    step_content: str
    """The step being evaluated"""

    step_type: str
    """Type of step"""

    label: float
    """Ground truth score (0-1)"""

    trajectory_outcome: float
    """Final outcome of the full trajectory"""

    metadata: dict[str, Any] = field(default_factory=dict)


class PRMTrainingCollector:
    """
    Collects training data for Process Reward Models from MCTS.

    Implements the ReST-MCTS* approach where MCTS generates
    training data for PRM, creating a self-improvement loop.
    """

    def __init__(
        self,
        verify_fn: Callable[[str, str], bool],
    ):
        """
        Initialize training data collector.

        Args:
            verify_fn: Function to verify if answer is correct
        """
        self.verify_fn = verify_fn
        self.collected_examples: list[PRMTrainingExample] = []
        self.trajectory_outcomes: dict[str, float] = {}

    def record_trajectory(
        self,
        trajectory: ReasoningTrajectory,
        outcome: float | None = None,
    ) -> None:
        """
        Record a completed trajectory for training data extraction.

        Args:
            trajectory: Complete trajectory with final answer
            outcome: Optional explicit outcome score
        """
        # Determine outcome
        if outcome is None:
            if trajectory.is_correct is not None:
                outcome = 1.0 if trajectory.is_correct else 0.0
            elif trajectory.final_answer and trajectory.query:
                is_correct = self.verify_fn(trajectory.query, trajectory.final_answer)
                outcome = 1.0 if is_correct else 0.0
            else:
                return  # Cannot determine outcome

        # Store trajectory outcome
        traj_key = trajectory.to_hash_key()
        self.trajectory_outcomes[traj_key] = outcome

        # Generate training examples for each step
        for step in trajectory.steps:
            prefix = trajectory.get_prefix(step.step_index)

            example = PRMTrainingExample(
                trajectory_prefix=prefix.to_text(),
                step_content=step.content,
                step_type=step.step_type,
                label=outcome,  # Will be refined with Monte Carlo
                trajectory_outcome=outcome,
                metadata={
                    "step_index": step.step_index,
                    "trajectory_hash": traj_key,
                },
            )
            self.collected_examples.append(example)

    def compute_monte_carlo_labels(
        self,
        step_hash: str,
    ) -> float:
        """
        Compute Monte Carlo label for a step.

        Averages outcomes of all trajectories that went through this step.

        Args:
            step_hash: Hash of the step

        Returns:
            Monte Carlo estimated step quality
        """
        relevant_examples = [
            ex for ex in self.collected_examples
            if ex.metadata.get("step_hash") == step_hash
        ]

        if not relevant_examples:
            return 0.5  # Default if no data

        outcomes = [ex.trajectory_outcome for ex in relevant_examples]
        return float(np.mean(outcomes))

    def export_training_data(self) -> list[dict[str, Any]]:
        """
        Export collected training data.

        Returns:
            List of training examples as dictionaries
        """
        return [
            {
                "prefix": ex.trajectory_prefix,
                "step": ex.step_content,
                "step_type": ex.step_type,
                "label": ex.label,
                "outcome": ex.trajectory_outcome,
                **ex.metadata,
            }
            for ex in self.collected_examples
        ]

    def clear(self) -> None:
        """Clear collected data."""
        self.collected_examples.clear()
        self.trajectory_outcomes.clear()
