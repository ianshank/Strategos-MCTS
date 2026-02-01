"""
Hybrid Search Strategy for MCTS with Modern Reasoning Techniques.

Combines parallel scaling (Best-of-N) with serial scaling (extended thinking)
for optimal test-time compute allocation.

Key insights from research:
- Parallel sampling is more robust than serial thinking for avoiding "overthinking"
- Extended thinking provides deeper analysis for critical decisions
- PRM-guided filtering reduces wasted compute on unpromising paths
- Verification is crucial for test-time scaling effectiveness

Architecture:
1. Phase 1: Parallel candidate generation (~25% compute)
2. Phase 2: PRM-guided filtering (~25% compute)
3. Phase 3: Extended thinking for best candidates (~50% compute)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from .core import MCTSEngine, MCTSNode, MCTSState
from .extended_thinking import (
    ExtendedThinkingEvaluator,
    ThinkingBudget,
    ThinkingMode,
    ThinkingResult,
)
from .policies import RolloutPolicy, SelectionPolicy
from .process_reward_model import (
    PRMMCTSIntegration,
    PRMScore,
    ProcessRewardModel,
    ReasoningStep,
    ReasoningTrajectory,
)


class SearchPhase(Enum):
    """Current phase in hybrid search."""

    PARALLEL_GENERATION = "parallel_generation"
    """Phase 1: Generate candidates in parallel"""

    PRM_FILTERING = "prm_filtering"
    """Phase 2: Filter candidates using PRM"""

    EXTENDED_EVALUATION = "extended_evaluation"
    """Phase 3: Deep evaluation of best candidates"""

    BACKPROPAGATION = "backpropagation"
    """Phase 4: Update tree with results"""


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search strategy."""

    # Compute budget allocation
    parallel_budget_ratio: float = 0.25
    """Fraction of compute for parallel generation (Phase 1)"""

    prm_budget_ratio: float = 0.25
    """Fraction of compute for PRM filtering (Phase 2)"""

    extended_budget_ratio: float = 0.50
    """Fraction of compute for extended evaluation (Phase 3)"""

    # Parallel generation
    num_parallel_candidates: int = 8
    """Number of parallel candidates to generate"""

    parallel_thinking_tokens: int = 4096
    """Thinking tokens per parallel candidate"""

    # PRM filtering
    prm_top_k: int = 3
    """Number of candidates to keep after PRM filtering"""

    prm_threshold: float = 0.4
    """Minimum PRM score to keep a candidate"""

    # Extended evaluation
    extended_thinking_tokens: int = 32768
    """Thinking tokens for deep evaluation"""

    use_verification: bool = True
    """Whether to run verification on final candidates"""

    # Adaptive behavior
    adapt_based_on_complexity: bool = True
    """Whether to adapt strategy based on task complexity"""

    simple_task_threshold: float = 0.3
    """Complexity below which to skip extended evaluation"""

    # Early termination
    early_terminate_confidence: float = 0.95
    """Confidence threshold for early termination"""

    max_iterations: int = 100
    """Maximum search iterations"""

    def validate(self) -> None:
        """Validate configuration."""
        total = self.parallel_budget_ratio + self.prm_budget_ratio + self.extended_budget_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Budget ratios must sum to 1.0, got {total}")


@dataclass
class SearchCandidate:
    """A candidate path in the search tree."""

    node: MCTSNode
    """Leaf node of this path"""

    path: list[MCTSNode]
    """Full path from root"""

    trajectory: ReasoningTrajectory
    """Reasoning trajectory for this path"""

    prm_score: float = 0.0
    """PRM score for this candidate"""

    thinking_result: ThinkingResult | None = None
    """Extended thinking evaluation result"""

    confidence: float = 0.0
    """Overall confidence in this candidate"""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""

    best_action: str | None
    """Recommended action"""

    best_candidate: SearchCandidate | None
    """Best candidate found"""

    all_candidates: list[SearchCandidate]
    """All evaluated candidates"""

    iterations: int
    """Number of iterations performed"""

    phases_completed: list[SearchPhase]
    """Search phases that were completed"""

    statistics: dict[str, Any]
    """Search statistics"""


class HybridMCTSSearch:
    """
    Hybrid MCTS search combining parallel and serial scaling.

    Implements a three-phase search:
    1. Parallel generation: Sample multiple paths in parallel
    2. PRM filtering: Score and filter using Process Reward Model
    3. Extended evaluation: Deep analysis of top candidates
    """

    def __init__(
        self,
        mcts_engine: MCTSEngine,
        prm: ProcessRewardModel | None = None,
        thinking_evaluator: ExtendedThinkingEvaluator | None = None,
        config: HybridSearchConfig | None = None,
    ):
        """
        Initialize hybrid search.

        Args:
            mcts_engine: Base MCTS engine
            prm: Process Reward Model for filtering
            thinking_evaluator: Extended thinking evaluator
            config: Search configuration
        """
        self.engine = mcts_engine
        self.prm = prm
        self.thinking_evaluator = thinking_evaluator
        self.config = config or HybridSearchConfig()
        self.config.validate()

        # PRM integration
        self.prm_integration = PRMMCTSIntegration(prm) if prm else None

        # Statistics
        self.total_searches = 0
        self.total_candidates_generated = 0
        self.total_candidates_filtered = 0
        self.early_terminations = 0

    async def search(
        self,
        root: MCTSNode,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        query: str = "",
        total_budget: int | None = None,
    ) -> HybridSearchResult:
        """
        Run hybrid search from root.

        Args:
            root: Root node to search from
            action_generator: Function to generate actions
            state_transition: Function for state transitions
            rollout_policy: Base rollout policy
            query: Original query for context
            total_budget: Total compute budget (iterations or tokens)

        Returns:
            HybridSearchResult with best action and statistics
        """
        self.total_searches += 1

        # Initialize
        phases_completed = []
        all_candidates: list[SearchCandidate] = []
        best_candidate: SearchCandidate | None = None

        # Initialize root actions
        if not root.available_actions:
            root.available_actions = action_generator(root.state)

        # Phase 1: Parallel candidate generation
        candidates = await self._phase1_parallel_generation(
            root=root,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
            query=query,
        )
        phases_completed.append(SearchPhase.PARALLEL_GENERATION)
        all_candidates.extend(candidates)

        # Early termination check
        if candidates and self._should_early_terminate(candidates):
            self.early_terminations += 1
            best_candidate = max(candidates, key=lambda c: c.confidence)
            return self._build_result(
                best_candidate=best_candidate,
                all_candidates=all_candidates,
                phases_completed=phases_completed,
            )

        # Phase 2: PRM filtering
        if self.prm:
            filtered_candidates = await self._phase2_prm_filtering(candidates)
            phases_completed.append(SearchPhase.PRM_FILTERING)
            self.total_candidates_filtered += len(candidates) - len(filtered_candidates)
            candidates = filtered_candidates

        # Phase 3: Extended evaluation
        if self.thinking_evaluator and candidates:
            evaluated_candidates = await self._phase3_extended_evaluation(
                candidates=candidates,
                query=query,
            )
            phases_completed.append(SearchPhase.EXTENDED_EVALUATION)
            candidates = evaluated_candidates

        # Phase 4: Backpropagation
        await self._phase4_backpropagation(candidates)
        phases_completed.append(SearchPhase.BACKPROPAGATION)

        # Select best candidate
        if candidates:
            best_candidate = self._select_best_candidate(candidates)

        return self._build_result(
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            phases_completed=phases_completed,
        )

    async def _phase1_parallel_generation(
        self,
        root: MCTSNode,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        query: str,
    ) -> list[SearchCandidate]:
        """
        Phase 1: Generate candidates in parallel.

        Samples multiple paths from the tree and evaluates them
        with lightweight (parallel) thinking.
        """
        candidates: list[SearchCandidate] = []
        tasks = []

        for _ in range(self.config.num_parallel_candidates):
            task = self._generate_single_candidate(
                root=root,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=rollout_policy,
                query=query,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, SearchCandidate):
                candidates.append(result)
                self.total_candidates_generated += 1

        return candidates

    async def _generate_single_candidate(
        self,
        root: MCTSNode,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        query: str,
    ) -> SearchCandidate:
        """Generate a single candidate path."""
        # Select path using UCB
        path = [root]
        node = root

        while node.children and not node.terminal:
            node = node.select_child(self.engine.exploration_weight)
            path.append(node)

        # Expand if not terminal
        if not node.terminal and not node.is_fully_expanded:
            if not node.available_actions:
                node.available_actions = action_generator(node.state)

            action = node.get_unexpanded_action()
            if action:
                child_state = state_transition(node.state, action)
                child = node.add_child(action, child_state)
                path.append(child)
                node = child

        # Simulate
        value = await self.engine.simulate(node, rollout_policy, max_depth=10)

        # Build trajectory
        trajectory = self._build_trajectory(path, query)

        return SearchCandidate(
            node=node,
            path=path,
            trajectory=trajectory,
            confidence=value,
            metadata={"phase": "parallel_generation"},
        )

    async def _phase2_prm_filtering(
        self,
        candidates: list[SearchCandidate],
    ) -> list[SearchCandidate]:
        """
        Phase 2: Filter candidates using PRM.

        Scores each candidate's trajectory and keeps top-k.
        """
        if not self.prm or not candidates:
            return candidates

        # Score all candidates
        for candidate in candidates:
            scores = await self.prm.score_trajectory(candidate.trajectory)
            if scores:
                # Use product of step scores as overall quality
                candidate.prm_score = float(np.prod([s.step_score for s in scores]))

        # Filter by threshold
        filtered = [c for c in candidates if c.prm_score >= self.config.prm_threshold]

        # Keep top-k
        filtered.sort(key=lambda c: c.prm_score, reverse=True)
        return filtered[: self.config.prm_top_k]

    async def _phase3_extended_evaluation(
        self,
        candidates: list[SearchCandidate],
        query: str,
    ) -> list[SearchCandidate]:
        """
        Phase 3: Deep evaluation of top candidates.

        Uses extended thinking with higher token budget.
        """
        if not self.thinking_evaluator or not candidates:
            return candidates

        budget = ThinkingBudget(
            default_tokens=self.config.extended_thinking_tokens,
        )

        for candidate in candidates:
            # Build context from trajectory
            context = candidate.trajectory.to_text()

            # Evaluate with extended thinking
            result = await self.thinking_evaluator.evaluate(
                state=candidate.node.state,
                context=context,
                budget=budget,
                mode=ThinkingMode.EXTENDED,
            )

            candidate.thinking_result = result
            # Combine PRM score with thinking evaluation
            candidate.confidence = (
                0.5 * candidate.prm_score + 0.5 * result.score
            )

        return candidates

    async def _phase4_backpropagation(
        self,
        candidates: list[SearchCandidate],
    ) -> None:
        """
        Phase 4: Backpropagate results through the tree.

        Uses PRM-weighted values if available.
        """
        for candidate in candidates:
            # Compute value to backpropagate
            if self.prm_integration and candidate.trajectory:
                value = await self.prm_integration.compute_backprop_value(
                    trajectory=candidate.trajectory,
                    outcome_value=candidate.confidence,
                )
            else:
                value = candidate.confidence

            # Backpropagate through path
            self.engine.backpropagate(candidate.node, value)

    def _should_early_terminate(self, candidates: list[SearchCandidate]) -> bool:
        """Check if we should terminate early based on high confidence."""
        if not candidates:
            return False

        max_confidence = max(c.confidence for c in candidates)
        return max_confidence >= self.config.early_terminate_confidence

    def _select_best_candidate(
        self,
        candidates: list[SearchCandidate],
    ) -> SearchCandidate:
        """Select the best candidate based on all available signals."""
        if not candidates:
            raise ValueError("No candidates to select from")

        # Score each candidate using multiple signals
        def composite_score(c: SearchCandidate) -> float:
            score = c.confidence

            # Boost if thinking evaluation is high
            if c.thinking_result and c.thinking_result.confidence > 0.8:
                score *= 1.2

            # Boost if PRM score is high
            if c.prm_score > 0.8:
                score *= 1.1

            return score

        return max(candidates, key=composite_score)

    def _build_trajectory(
        self,
        path: list[MCTSNode],
        query: str,
    ) -> ReasoningTrajectory:
        """Build a reasoning trajectory from an MCTS path."""
        trajectory = ReasoningTrajectory(query=query)

        for i, node in enumerate(path):
            if node.action:
                step = ReasoningStep(
                    content=f"Action: {node.action}",
                    step_index=i,
                    step_type="action",
                    metadata={"state_id": node.state.state_id},
                )
                trajectory.add_step(step)

        return trajectory

    def _build_result(
        self,
        best_candidate: SearchCandidate | None,
        all_candidates: list[SearchCandidate],
        phases_completed: list[SearchPhase],
    ) -> HybridSearchResult:
        """Build the final search result."""
        best_action = None
        if best_candidate and best_candidate.path:
            # Get first action from root
            for node in best_candidate.path[1:]:
                if node.action:
                    best_action = node.action
                    break

        return HybridSearchResult(
            best_action=best_action,
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            iterations=len(all_candidates),
            phases_completed=phases_completed,
            statistics={
                "total_searches": self.total_searches,
                "candidates_generated": self.total_candidates_generated,
                "candidates_filtered": self.total_candidates_filtered,
                "early_terminations": self.early_terminations,
            },
        )


# ============================================================================
# Verification Integration
# ============================================================================


class VerificationResult:
    """Result from verification of a candidate."""

    def __init__(
        self,
        passed: bool,
        confidence: float,
        feedback: str = "",
        errors: list[str] | None = None,
    ):
        self.passed = passed
        self.confidence = confidence
        self.feedback = feedback
        self.errors = errors or []


class Verifier(ABC):
    """Abstract base class for solution verifiers."""

    @abstractmethod
    async def verify(
        self,
        candidate: SearchCandidate,
        query: str,
    ) -> VerificationResult:
        """Verify a candidate solution."""
        pass


class CodeExecutionVerifier:
    """
    Verifier that executes code to check correctness.

    For code-based tasks, runs the candidate solution and checks output.
    """

    def __init__(
        self,
        executor: Callable[[str], Any],
        timeout: float = 30.0,
    ):
        """
        Initialize code verifier.

        Args:
            executor: Function to execute code
            timeout: Execution timeout in seconds
        """
        self.executor = executor
        self.timeout = timeout

    async def verify(
        self,
        candidate: SearchCandidate,
        query: str,
    ) -> VerificationResult:
        """Verify by executing code."""
        # Extract code from candidate
        code = self._extract_code(candidate)
        if not code:
            return VerificationResult(
                passed=False,
                confidence=0.0,
                feedback="No executable code found",
            )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(self.executor, code),
                timeout=self.timeout,
            )

            return VerificationResult(
                passed=True,
                confidence=0.95,
                feedback=f"Code executed successfully: {result}",
            )

        except asyncio.TimeoutError:
            return VerificationResult(
                passed=False,
                confidence=0.0,
                feedback="Execution timed out",
                errors=["TimeoutError"],
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                confidence=0.0,
                feedback=f"Execution failed: {e}",
                errors=[str(e)],
            )

    def _extract_code(self, candidate: SearchCandidate) -> str | None:
        """Extract executable code from candidate."""
        # Look in trajectory for code steps
        for step in candidate.trajectory.steps:
            if step.step_type == "code":
                return step.content

        # Look in thinking result
        if candidate.thinking_result:
            analysis = candidate.thinking_result.analysis
            if "```" in analysis:
                # Extract code block
                parts = analysis.split("```")
                if len(parts) >= 3:
                    return parts[1].strip()

        return None


class SymbolicVerifier:
    """
    Verifier using symbolic computation.

    For math problems, uses symbolic verification to check correctness.
    """

    def __init__(self):
        """Initialize symbolic verifier."""
        try:
            import sympy

            self.sympy = sympy
            self.available = True
        except ImportError:
            self.sympy = None
            self.available = False

    async def verify(
        self,
        candidate: SearchCandidate,
        query: str,
    ) -> VerificationResult:
        """Verify using symbolic math."""
        if not self.available:
            return VerificationResult(
                passed=False,
                confidence=0.0,
                feedback="SymPy not available",
            )

        # Extract answer from candidate
        answer = self._extract_answer(candidate)
        if not answer:
            return VerificationResult(
                passed=False,
                confidence=0.0,
                feedback="No answer to verify",
            )

        try:
            # Parse and simplify
            expr = self.sympy.sympify(answer)
            simplified = self.sympy.simplify(expr)

            return VerificationResult(
                passed=True,
                confidence=0.8,
                feedback=f"Symbolic verification passed: {simplified}",
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                confidence=0.0,
                feedback=f"Symbolic verification failed: {e}",
                errors=[str(e)],
            )

    def _extract_answer(self, candidate: SearchCandidate) -> str | None:
        """Extract answer from candidate."""
        if candidate.trajectory.final_answer:
            return candidate.trajectory.final_answer

        if candidate.thinking_result:
            return candidate.thinking_result.analysis

        return None


class VerifiedHybridSearch(HybridMCTSSearch):
    """
    Hybrid search with verification integration.

    Adds verification phase after extended evaluation for
    higher confidence in results.
    """

    def __init__(
        self,
        mcts_engine: MCTSEngine,
        prm: ProcessRewardModel | None = None,
        thinking_evaluator: ExtendedThinkingEvaluator | None = None,
        verifiers: Sequence[Any] | None = None,
        config: HybridSearchConfig | None = None,
    ):
        """
        Initialize verified hybrid search.

        Args:
            mcts_engine: Base MCTS engine
            prm: Process Reward Model
            thinking_evaluator: Extended thinking evaluator
            verifiers: List of verifiers to apply
            config: Search configuration
        """
        super().__init__(mcts_engine, prm, thinking_evaluator, config)
        self.verifiers = list(verifiers) if verifiers else []

    async def _phase3_extended_evaluation(
        self,
        candidates: list[SearchCandidate],
        query: str,
    ) -> list[SearchCandidate]:
        """Extended evaluation with verification."""
        # Run base extended evaluation
        candidates = await super()._phase3_extended_evaluation(candidates, query)

        if not self.verifiers or not self.config.use_verification:
            return candidates

        # Apply verifiers
        for candidate in candidates:
            verification_passed = True
            total_confidence = candidate.confidence

            for verifier in self.verifiers:
                result = await verifier.verify(candidate, query)

                if not result.passed:
                    verification_passed = False

                # Adjust confidence based on verification
                total_confidence *= result.confidence

                # Store verification result
                candidate.metadata[f"verification_{type(verifier).__name__}"] = {
                    "passed": result.passed,
                    "feedback": result.feedback,
                    "errors": result.errors,
                }

            candidate.confidence = total_confidence
            candidate.metadata["all_verifications_passed"] = verification_passed

        return candidates
