"""
Extended Thinking Integration for MCTS Node Evaluation.

Implements adaptive token budget allocation for reasoning-enabled models
(e.g., Claude's extended thinking, OpenAI o1/o3, DeepSeek-R1).

Key features:
- Adaptive thinking budget based on node importance and depth
- Thinking trace extraction for PRM training
- Parallel evaluation (Best-of-N) for high-uncertainty nodes
- Task complexity routing to avoid "overthinking"

References:
- "Scaling Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters"
- "Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners"
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

    from .process_reward_model import ReasoningStep, ReasoningTrajectory


class ThinkingMode(Enum):
    """Thinking mode for model invocation."""

    NONE = "none"
    """No extended thinking - fast mode"""

    MINIMAL = "minimal"
    """Minimal thinking for simple queries (1K-4K tokens)"""

    STANDARD = "standard"
    """Standard thinking for moderate complexity (8K-16K tokens)"""

    EXTENDED = "extended"
    """Extended thinking for complex problems (32K-64K tokens)"""

    DEEP = "deep"
    """Deep thinking for critical decisions (64K+ tokens)"""


@dataclass
class ThinkingBudget:
    """
    Thinking token budget configuration.

    Controls how much compute to allocate for extended thinking.
    """

    min_tokens: int = 1024
    """Minimum thinking tokens for any evaluation"""

    max_tokens: int = 65536
    """Maximum thinking tokens (model-dependent)"""

    default_tokens: int = 8192
    """Default tokens for standard evaluation"""

    # Adaptive scaling parameters
    depth_multiplier: float = 1.2
    """Multiply budget by this factor per tree depth level"""

    uncertainty_multiplier: float = 1.5
    """Multiply budget when uncertainty is high"""

    critical_threshold: float = 0.8
    """UCB score threshold for "critical" nodes (get more budget)"""

    # Mode thresholds
    minimal_threshold: int = 4096
    """Max tokens for minimal mode"""

    standard_threshold: int = 16384
    """Max tokens for standard mode"""

    extended_threshold: int = 65536
    """Max tokens for extended mode"""

    def compute_budget(
        self,
        depth: int,
        visits: int,
        ucb_score: float,
        uncertainty: float = 0.5,
    ) -> int:
        """
        Compute adaptive thinking budget for a node.

        Args:
            depth: Node depth in tree
            visits: Number of node visits
            ucb_score: UCB score of the node
            uncertainty: Uncertainty measure (0-1)

        Returns:
            Number of thinking tokens to allocate
        """
        # Start with default
        budget = self.default_tokens

        # Scale by depth (deeper = more thinking)
        depth_scale = self.depth_multiplier ** min(depth, 10)
        budget = int(budget * depth_scale)

        # Scale by uncertainty
        if uncertainty > 0.5:
            uncertainty_scale = 1.0 + (uncertainty - 0.5) * (self.uncertainty_multiplier - 1.0)
            budget = int(budget * uncertainty_scale)

        # Critical nodes get extra budget
        if ucb_score > self.critical_threshold:
            budget = int(budget * 1.5)

        # Low-visit nodes are exploratory - give more budget
        if visits < 5:
            budget = int(budget * 1.2)

        # Clamp to bounds
        budget = max(self.min_tokens, min(self.max_tokens, budget))

        return budget

    def get_mode(self, tokens: int) -> ThinkingMode:
        """Get thinking mode for given token count."""
        if tokens <= 0:
            return ThinkingMode.NONE
        elif tokens <= self.minimal_threshold:
            return ThinkingMode.MINIMAL
        elif tokens <= self.standard_threshold:
            return ThinkingMode.STANDARD
        elif tokens <= self.extended_threshold:
            return ThinkingMode.EXTENDED
        else:
            return ThinkingMode.DEEP


@dataclass
class ThinkingResult:
    """
    Result from extended thinking evaluation.

    Contains both the thinking trace and final assessment.
    """

    score: float
    """Evaluation score (0.0 to 1.0)"""

    thinking_trace: str
    """Full thinking trace from the model"""

    analysis: str
    """Summarized analysis/reflection"""

    is_terminal: bool = False
    """Whether this represents a terminal/solution state"""

    confidence: float = 0.5
    """Model's confidence in this evaluation"""

    tokens_used: int = 0
    """Actual tokens used for thinking"""

    mode: ThinkingMode = ThinkingMode.STANDARD
    """Thinking mode that was used"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata from evaluation"""


@dataclass
class TaskComplexity:
    """
    Task complexity assessment for routing decisions.

    Used to determine whether extended thinking is beneficial.
    """

    complexity_score: float
    """Overall complexity (0.0 to 1.0)"""

    reasoning_required: bool
    """Whether multi-step reasoning is required"""

    domain: str
    """Domain classification (math, code, logic, general, etc.)"""

    requires_verification: bool
    """Whether verification/double-checking is beneficial"""

    estimated_steps: int
    """Estimated number of reasoning steps needed"""

    overthinking_risk: float
    """Risk of overthinking degrading performance (0.0 to 1.0)"""


class ExtendedThinkingEvaluator(ABC):
    """
    Abstract base class for extended thinking evaluation.

    Implementations should use reasoning-enabled models (o1, Claude thinking, etc.)
    to evaluate MCTS nodes with adaptive compute budgets.
    """

    @abstractmethod
    async def evaluate(
        self,
        state: Any,
        context: str,
        budget: ThinkingBudget,
        mode: ThinkingMode | None = None,
    ) -> ThinkingResult:
        """
        Evaluate a state using extended thinking.

        Args:
            state: State to evaluate
            context: Additional context (previous steps, query, etc.)
            budget: Token budget configuration
            mode: Optional explicit thinking mode

        Returns:
            ThinkingResult with evaluation and trace
        """
        pass

    @abstractmethod
    async def classify_complexity(
        self,
        query: str,
        context: str | None = None,
    ) -> TaskComplexity:
        """
        Classify task complexity for routing decisions.

        Args:
            query: The task/query to classify
            context: Optional additional context

        Returns:
            TaskComplexity assessment
        """
        pass


class ClaudeExtendedThinkingEvaluator(ExtendedThinkingEvaluator):
    """
    Extended thinking evaluator using Claude's thinking capability.

    Uses Claude with extended thinking enabled for deep node evaluation.
    """

    def __init__(
        self,
        client: Any,
        model: str = "claude-sonnet-4-20250514",
        default_budget: ThinkingBudget | None = None,
    ):
        """
        Initialize Claude-based evaluator.

        Args:
            client: Anthropic client instance
            model: Model to use (must support extended thinking)
            default_budget: Default thinking budget
        """
        self.client = client
        self.model = model
        self.default_budget = default_budget or ThinkingBudget()

    async def evaluate(
        self,
        state: Any,
        context: str,
        budget: ThinkingBudget | None = None,
        mode: ThinkingMode | None = None,
    ) -> ThinkingResult:
        """Evaluate state using Claude's extended thinking."""
        budget = budget or self.default_budget

        # Compute token budget based on mode or use default
        if mode is None:
            tokens = budget.default_tokens
            mode = budget.get_mode(tokens)
        else:
            tokens = self._mode_to_tokens(mode, budget)

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(state, context)

        try:
            # Call Claude with extended thinking
            response = await self._invoke_with_thinking(prompt, tokens)

            # Extract thinking trace and result
            thinking_trace = self._extract_thinking(response)
            score, analysis, is_terminal = self._parse_evaluation(response)

            return ThinkingResult(
                score=score,
                thinking_trace=thinking_trace,
                analysis=analysis,
                is_terminal=is_terminal,
                confidence=self._estimate_confidence(response),
                tokens_used=self._count_thinking_tokens(response),
                mode=mode,
                metadata={"model": self.model},
            )

        except Exception as e:
            # Fallback on error
            return ThinkingResult(
                score=0.5,
                thinking_trace="",
                analysis=f"Evaluation failed: {e}",
                is_terminal=False,
                confidence=0.0,
                tokens_used=0,
                mode=ThinkingMode.NONE,
                metadata={"error": str(e)},
            )

    async def classify_complexity(
        self,
        query: str,
        context: str | None = None,
    ) -> TaskComplexity:
        """Classify task complexity using Claude."""
        prompt = f"""Analyze the complexity of this task:

Query: {query}
{f"Context: {context}" if context else ""}

Assess:
1. Overall complexity (0.0-1.0)
2. Whether multi-step reasoning is required
3. Domain (math, code, logic, general, creative)
4. Whether verification is beneficial
5. Estimated reasoning steps needed
6. Risk of overthinking (0.0-1.0)

Format response as:
COMPLEXITY: [0.0-1.0]
REASONING_REQUIRED: [yes/no]
DOMAIN: [domain]
NEEDS_VERIFICATION: [yes/no]
ESTIMATED_STEPS: [number]
OVERTHINKING_RISK: [0.0-1.0]"""

        try:
            # Use minimal thinking for complexity classification
            response = await self._invoke_with_thinking(prompt, 2048)
            return self._parse_complexity(response)
        except Exception:
            # Default moderate complexity
            return TaskComplexity(
                complexity_score=0.5,
                reasoning_required=True,
                domain="general",
                requires_verification=False,
                estimated_steps=3,
                overthinking_risk=0.3,
            )

    def _mode_to_tokens(self, mode: ThinkingMode, budget: ThinkingBudget) -> int:
        """Convert mode to token count."""
        mode_tokens = {
            ThinkingMode.NONE: 0,
            ThinkingMode.MINIMAL: budget.minimal_threshold // 2,
            ThinkingMode.STANDARD: budget.standard_threshold // 2,
            ThinkingMode.EXTENDED: budget.extended_threshold // 2,
            ThinkingMode.DEEP: budget.max_tokens,
        }
        return mode_tokens.get(mode, budget.default_tokens)

    def _build_evaluation_prompt(self, state: Any, context: str) -> str:
        """Build the evaluation prompt."""
        state_str = str(state) if not isinstance(state, str) else state

        return f"""Evaluate the following state in the context of MCTS tree search.

Context:
{context}

Current State:
{state_str}

Provide:
1. A quality score (0.0 to 1.0) for this state
2. Analysis of strengths and weaknesses
3. Whether this is a terminal/solution state

Think through this carefully before responding.

Format:
SCORE: [0.0-1.0]
ANALYSIS: [your analysis]
IS_TERMINAL: [yes/no]"""

    async def _invoke_with_thinking(self, prompt: str, budget_tokens: int) -> Any:
        """Invoke Claude with extended thinking enabled."""
        # Note: This is a placeholder for actual API call
        # Real implementation would use anthropic client with thinking enabled
        try:
            messages = [{"role": "user", "content": prompt}]

            # Configure for extended thinking
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=8192,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                },
                messages=messages,
            )
            return response
        except (AttributeError, TypeError):
            # Fallback for non-thinking capable client
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=4096,
                messages=messages,
            )
            return response

    def _extract_thinking(self, response: Any) -> str:
        """Extract thinking trace from response."""
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "type") and block.type == "thinking":
                    return block.thinking
        return ""

    def _parse_evaluation(self, response: Any) -> tuple[float, str, bool]:
        """Parse evaluation from response."""
        text = ""
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    text = block.text
                    break

        # Parse structured output
        score = 0.5
        analysis = ""
        is_terminal = False

        for line in text.split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                    score = max(0.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("ANALYSIS:"):
                analysis = line.split(":", 1)[1].strip()
            elif line.startswith("IS_TERMINAL:"):
                is_terminal = "yes" in line.lower()

        return score, analysis, is_terminal

    def _estimate_confidence(self, response: Any) -> float:
        """Estimate confidence from response."""
        # Based on thinking length and content
        thinking = self._extract_thinking(response)
        if not thinking:
            return 0.5

        # Longer thinking typically means more thorough analysis
        # But too short might indicate simple problem
        length = len(thinking)
        if length < 500:
            return 0.6
        elif length < 2000:
            return 0.75
        elif length < 5000:
            return 0.85
        else:
            return 0.9

    def _count_thinking_tokens(self, response: Any) -> int:
        """Count thinking tokens used."""
        if hasattr(response, "usage"):
            # Try to get thinking-specific token count
            if hasattr(response.usage, "thinking_tokens"):
                return response.usage.thinking_tokens
            return getattr(response.usage, "input_tokens", 0)
        return 0

    def _parse_complexity(self, response: Any) -> TaskComplexity:
        """Parse complexity classification from response."""
        text = ""
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    text = block.text
                    break

        # Parse with defaults
        complexity = 0.5
        reasoning_required = True
        domain = "general"
        needs_verification = False
        estimated_steps = 3
        overthinking_risk = 0.3

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("COMPLEXITY:"):
                try:
                    complexity = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING_REQUIRED:"):
                reasoning_required = "yes" in line.lower()
            elif line.startswith("DOMAIN:"):
                domain = line.split(":")[1].strip().lower()
            elif line.startswith("NEEDS_VERIFICATION:"):
                needs_verification = "yes" in line.lower()
            elif line.startswith("ESTIMATED_STEPS:"):
                try:
                    estimated_steps = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("OVERTHINKING_RISK:"):
                try:
                    overthinking_risk = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        return TaskComplexity(
            complexity_score=complexity,
            reasoning_required=reasoning_required,
            domain=domain,
            requires_verification=needs_verification,
            estimated_steps=estimated_steps,
            overthinking_risk=overthinking_risk,
        )


class ParallelThinkingEvaluator:
    """
    Evaluator that runs parallel evaluations (Best-of-N).

    Research shows parallel sampling is more robust than extended
    serial thinking for avoiding "overthinking" degradation.
    """

    def __init__(
        self,
        base_evaluator: ExtendedThinkingEvaluator,
        num_samples: int = 3,
        aggregation: str = "max",
    ):
        """
        Initialize parallel evaluator.

        Args:
            base_evaluator: Base evaluator to run in parallel
            num_samples: Number of parallel evaluations
            aggregation: How to aggregate results (max, mean, median, vote)
        """
        self.base_evaluator = base_evaluator
        self.num_samples = num_samples
        self.aggregation = aggregation

    async def evaluate(
        self,
        state: Any,
        context: str,
        budget: ThinkingBudget | None = None,
        mode: ThinkingMode | None = None,
    ) -> ThinkingResult:
        """Run parallel evaluations and aggregate results."""
        budget = budget or ThinkingBudget()

        # Reduce per-sample budget for parallel sampling
        parallel_budget = ThinkingBudget(
            min_tokens=budget.min_tokens,
            max_tokens=budget.max_tokens // self.num_samples,
            default_tokens=budget.default_tokens // self.num_samples,
        )

        # Run evaluations in parallel
        tasks = [
            self.base_evaluator.evaluate(state, context, parallel_budget, mode)
            for _ in range(self.num_samples)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_results = [r for r in results if isinstance(r, ThinkingResult)]

        if not valid_results:
            return ThinkingResult(
                score=0.5,
                thinking_trace="All parallel evaluations failed",
                analysis="Error in parallel evaluation",
                confidence=0.0,
            )

        # Aggregate results
        return self._aggregate_results(valid_results)

    def _aggregate_results(self, results: list[ThinkingResult]) -> ThinkingResult:
        """Aggregate multiple evaluation results."""
        scores = [r.score for r in results]

        if self.aggregation == "max":
            best_idx = int(np.argmax(scores))
            best_result = results[best_idx]
            return ThinkingResult(
                score=best_result.score,
                thinking_trace=best_result.thinking_trace,
                analysis=f"Best of {len(results)}: {best_result.analysis}",
                is_terminal=best_result.is_terminal,
                confidence=best_result.confidence * (1 - np.std(scores) / 2),
                tokens_used=sum(r.tokens_used for r in results),
                metadata={
                    "all_scores": scores,
                    "aggregation": self.aggregation,
                },
            )

        elif self.aggregation == "mean":
            avg_score = float(np.mean(scores))
            # Use result closest to mean
            closest_idx = int(np.argmin([abs(s - avg_score) for s in scores]))
            closest = results[closest_idx]

            return ThinkingResult(
                score=avg_score,
                thinking_trace=closest.thinking_trace,
                analysis=f"Mean of {len(results)}: {closest.analysis}",
                is_terminal=any(r.is_terminal for r in results),
                confidence=1.0 - np.std(scores),  # Higher agreement = higher confidence
                tokens_used=sum(r.tokens_used for r in results),
                metadata={
                    "all_scores": scores,
                    "aggregation": self.aggregation,
                },
            )

        elif self.aggregation == "vote":
            # Majority vote on is_terminal and high score threshold
            threshold = 0.7
            high_score_votes = sum(1 for s in scores if s >= threshold)
            terminal_votes = sum(1 for r in results if r.is_terminal)

            voted_score = float(np.mean(scores)) if high_score_votes > len(results) / 2 else 0.5
            voted_terminal = terminal_votes > len(results) / 2

            best_idx = int(np.argmax(scores))
            best = results[best_idx]

            return ThinkingResult(
                score=voted_score,
                thinking_trace=best.thinking_trace,
                analysis=f"Vote of {len(results)}: {best.analysis}",
                is_terminal=voted_terminal,
                confidence=max(high_score_votes, len(results) - high_score_votes) / len(results),
                tokens_used=sum(r.tokens_used for r in results),
                metadata={
                    "all_scores": scores,
                    "aggregation": self.aggregation,
                    "high_score_votes": high_score_votes,
                },
            )

        else:  # median
            median_score = float(np.median(scores))
            closest_idx = int(np.argmin([abs(s - median_score) for s in scores]))
            closest = results[closest_idx]

            return ThinkingResult(
                score=median_score,
                thinking_trace=closest.thinking_trace,
                analysis=f"Median of {len(results)}: {closest.analysis}",
                is_terminal=closest.is_terminal,
                confidence=1.0 - np.std(scores),
                tokens_used=sum(r.tokens_used for r in results),
                metadata={
                    "all_scores": scores,
                    "aggregation": self.aggregation,
                },
            )


class AdaptiveThinkingRouter:
    """
    Routes tasks to appropriate thinking modes based on complexity.

    Avoids "overthinking" for simple tasks while allocating more
    compute for genuinely complex problems.
    """

    def __init__(
        self,
        evaluator: ExtendedThinkingEvaluator,
        parallel_evaluator: ParallelThinkingEvaluator | None = None,
        overthinking_threshold: float = 0.6,
    ):
        """
        Initialize adaptive router.

        Args:
            evaluator: Base evaluator
            parallel_evaluator: Optional parallel evaluator for uncertain cases
            overthinking_threshold: Complexity threshold above which to use deep thinking
        """
        self.evaluator = evaluator
        self.parallel_evaluator = parallel_evaluator
        self.overthinking_threshold = overthinking_threshold

    async def evaluate(
        self,
        state: Any,
        context: str,
        query: str,
        budget: ThinkingBudget | None = None,
    ) -> ThinkingResult:
        """
        Adaptively evaluate state based on task complexity.

        Args:
            state: State to evaluate
            context: Evaluation context
            query: Original query for complexity classification
            budget: Token budget

        Returns:
            ThinkingResult from appropriate evaluation path
        """
        budget = budget or ThinkingBudget()

        # Classify complexity
        complexity = await self.evaluator.classify_complexity(query, context)

        # Route based on complexity and overthinking risk
        if complexity.overthinking_risk > self.overthinking_threshold:
            # High overthinking risk - use minimal thinking
            mode = ThinkingMode.MINIMAL
            return await self.evaluator.evaluate(state, context, budget, mode)

        elif complexity.complexity_score < 0.3:
            # Simple task - fast path
            mode = ThinkingMode.NONE
            return await self.evaluator.evaluate(state, context, budget, mode)

        elif complexity.complexity_score > 0.7 and self.parallel_evaluator:
            # Complex task - use parallel evaluation
            return await self.parallel_evaluator.evaluate(state, context, budget)

        else:
            # Moderate complexity - standard thinking
            mode = ThinkingMode.STANDARD
            return await self.evaluator.evaluate(state, context, budget, mode)


# ============================================================================
# MCTS Integration
# ============================================================================


class ThinkingEnhancedMCTSEvaluator:
    """
    Integrates extended thinking into MCTS node evaluation.

    Replaces simple rollout with thinking-enabled evaluation for
    more accurate state assessment.
    """

    def __init__(
        self,
        evaluator: ExtendedThinkingEvaluator | AdaptiveThinkingRouter,
        budget: ThinkingBudget | None = None,
        cache_evaluations: bool = True,
        cache_size: int = 1000,
    ):
        """
        Initialize thinking-enhanced MCTS evaluator.

        Args:
            evaluator: Extended thinking evaluator
            budget: Default thinking budget
            cache_evaluations: Whether to cache evaluations
            cache_size: Maximum cache size
        """
        self.evaluator = evaluator
        self.budget = budget or ThinkingBudget()
        self.cache_evaluations = cache_evaluations

        from collections import OrderedDict

        self._cache: OrderedDict[str, ThinkingResult] = OrderedDict()
        self.cache_size = cache_size

        # Statistics
        self.total_evaluations = 0
        self.cache_hits = 0
        self.total_thinking_tokens = 0

    async def evaluate_node(
        self,
        state: Any,
        trajectory_context: str,
        depth: int,
        visits: int,
        ucb_score: float = 0.5,
        query: str = "",
    ) -> ThinkingResult:
        """
        Evaluate an MCTS node using extended thinking.

        Args:
            state: Node state to evaluate
            trajectory_context: Context from trajectory
            depth: Node depth in tree
            visits: Number of node visits
            ucb_score: Current UCB score
            query: Original query (for complexity routing)

        Returns:
            ThinkingResult with evaluation
        """
        # Check cache
        cache_key = f"{hash(str(state))}:{depth}:{visits}"
        if self.cache_evaluations and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            self.cache_hits += 1
            return self._cache[cache_key]

        self.total_evaluations += 1

        # Compute adaptive budget
        tokens = self.budget.compute_budget(
            depth=depth,
            visits=visits,
            ucb_score=ucb_score,
            uncertainty=0.5 if visits < 5 else 0.3,
        )

        adaptive_budget = ThinkingBudget(
            min_tokens=self.budget.min_tokens,
            max_tokens=tokens,
            default_tokens=tokens,
        )

        # Evaluate
        if isinstance(self.evaluator, AdaptiveThinkingRouter):
            result = await self.evaluator.evaluate(
                state=state,
                context=trajectory_context,
                query=query,
                budget=adaptive_budget,
            )
        else:
            mode = adaptive_budget.get_mode(tokens)
            result = await self.evaluator.evaluate(
                state=state,
                context=trajectory_context,
                budget=adaptive_budget,
                mode=mode,
            )

        self.total_thinking_tokens += result.tokens_used

        # Cache result
        if self.cache_evaluations:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[cache_key] = result

        return result

    def get_statistics(self) -> dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_evaluations),
            "total_thinking_tokens": self.total_thinking_tokens,
            "avg_tokens_per_eval": self.total_thinking_tokens / max(1, self.total_evaluations),
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self._cache.clear()
        self.cache_hits = 0
