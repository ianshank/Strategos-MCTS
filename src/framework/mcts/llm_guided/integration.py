"""
Integration layer for LLM-Guided MCTS with HRM, TRM, and Meta-Controller.

Provides adapters and orchestration for:
- HRM (Hierarchical Reasoning Module): Problem decomposition
- TRM (Tiny Recursive Model): Solution refinement
- Meta-Controller: Dynamic agent routing
- LLM-Guided MCTS: Core search algorithm

This module bridges the neural agents with the LLM-based MCTS system,
enabling hybrid reasoning that combines learned representations with
LLM-guided exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from src.observability.logging import get_structured_logger

from .config import LLMGuidedMCTSConfig
from .engine import LLMGuidedMCTSEngine, MCTSSearchResult

if TYPE_CHECKING:
    from src.agents.hrm_agent import HRMAgent
    from src.agents.meta_controller.base import (
        AbstractMetaController,
    )
    from src.agents.trm_agent import TRMAgent

logger = get_structured_logger(__name__)


class AgentType(str, Enum):
    """Available agent types for routing."""

    HRM = "hrm"
    TRM = "trm"
    MCTS = "mcts"
    LLM_MCTS = "llm_mcts"


@dataclass
class SubProblemDecomposition:
    """Result of HRM problem decomposition."""

    original_problem: str
    subproblems: list[str]
    hierarchy_levels: list[int]
    confidences: list[float]
    latent_states: list[Any] | None = None  # Tensor representations

    @property
    def num_subproblems(self) -> int:
        """Number of decomposed subproblems."""
        return len(self.subproblems)

    def get_leaf_problems(self) -> list[str]:
        """Get the lowest-level (leaf) subproblems."""
        if not self.hierarchy_levels:
            return self.subproblems
        max_level = max(self.hierarchy_levels)
        return [sp for sp, level in zip(self.subproblems, self.hierarchy_levels, strict=True) if level == max_level]


@dataclass
class RefinementResult:
    """Result of TRM solution refinement."""

    original_code: str
    refined_code: str
    num_iterations: int
    converged: bool
    improvement_score: float
    intermediate_codes: list[str] = field(default_factory=list)
    residual_norms: list[float] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Result of meta-controller routing decision."""

    selected_agent: AgentType
    confidence: float
    probabilities: dict[str, float]
    reasoning: str = ""


class IntegrationConfig(BaseModel):
    """Configuration for the integration layer."""

    use_hrm_decomposition: bool = Field(
        default=True,
        description="Whether to use HRM for problem decomposition",
    )
    use_trm_refinement: bool = Field(
        default=True,
        description="Whether to use TRM for solution refinement",
    )
    use_meta_controller: bool = Field(
        default=True,
        description="Whether to use meta-controller for routing",
    )
    decomposition_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for accepting decomposition",
    )
    refinement_max_iterations: int = Field(
        default=16,
        ge=1,
        le=100,
        description="Maximum TRM refinement iterations",
    )
    refinement_convergence_threshold: float = Field(
        default=0.01,
        ge=0.0,
        description="L2 norm threshold for TRM convergence",
    )
    fallback_to_mcts_on_low_confidence: bool = Field(
        default=True,
        description="Fall back to MCTS when agent confidence is low",
    )
    low_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold below which to fall back to MCTS",
    )
    enable_parallel_search: bool = Field(
        default=False,
        description="Run multiple agents in parallel when possible",
    )
    combine_results_strategy: str = Field(
        default="best",
        description="Strategy for combining results: 'best', 'vote', 'ensemble'",
    )


@runtime_checkable
class ProblemDecomposer(Protocol):
    """Protocol for problem decomposition agents."""

    async def decompose(
        self,
        problem: str,
        context: str | None = None,
    ) -> SubProblemDecomposition:
        """Decompose a problem into subproblems."""
        ...


@runtime_checkable
class SolutionRefiner(Protocol):
    """Protocol for solution refinement agents."""

    async def refine(
        self,
        code: str,
        problem: str,
        test_cases: list[str] | None = None,
        max_iterations: int = 16,
    ) -> RefinementResult:
        """Refine a solution iteratively."""
        ...


@runtime_checkable
class AgentRouter(Protocol):
    """Protocol for agent routing."""

    def route(
        self,
        problem: str,
        context: dict[str, Any],
    ) -> RoutingDecision:
        """Decide which agent should handle the problem."""
        ...


class HRMAdapter:
    """
    Adapter for HRM (Hierarchical Reasoning Module) integration.

    Wraps the neural HRM agent to provide problem decomposition
    for the LLM-guided MCTS system.
    """

    def __init__(
        self,
        hrm_agent: HRMAgent | None = None,
        llm_client: Any | None = None,
        config: IntegrationConfig | None = None,
    ):
        """
        Initialize HRM adapter.

        Args:
            hrm_agent: Pre-trained HRM neural network (optional)
            llm_client: LLM client for text-based decomposition fallback
            config: Integration configuration
        """
        self._hrm_agent = hrm_agent
        self._llm_client = llm_client
        self._config = config or IntegrationConfig()
        self._logger = get_structured_logger(f"{__name__}.HRMAdapter")

    @property
    def has_neural_agent(self) -> bool:
        """Whether a neural HRM agent is available."""
        return self._hrm_agent is not None

    async def decompose(
        self,
        problem: str,
        context: str | None = None,
    ) -> SubProblemDecomposition:
        """
        Decompose a problem into subproblems.

        Uses neural HRM if available, otherwise falls back to LLM-based
        decomposition.

        Args:
            problem: The problem description
            context: Optional context for decomposition

        Returns:
            SubProblemDecomposition with hierarchical structure
        """
        if self._hrm_agent is not None:
            return await self._decompose_neural(problem, context)
        elif self._llm_client is not None:
            return await self._decompose_llm(problem, context)
        else:
            # No decomposition available, return original problem
            self._logger.warning("No decomposition agent available")
            return SubProblemDecomposition(
                original_problem=problem,
                subproblems=[problem],
                hierarchy_levels=[0],
                confidences=[1.0],
            )

    async def _decompose_neural(
        self,
        problem: str,
        context: str | None = None,
    ) -> SubProblemDecomposition:
        """Decompose using neural HRM agent."""

        # Encode problem text to tensor (requires encoder)
        # For now, use a placeholder - actual implementation would use
        # a text encoder like CodeBERT
        self._logger.info("Decomposing problem with neural HRM")

        # This is a placeholder - actual implementation would:
        # 1. Encode problem text to tensor
        # 2. Run through HRM agent
        # 3. Decode subproblem tensors back to text

        # For now, return a simple decomposition
        return SubProblemDecomposition(
            original_problem=problem,
            subproblems=[problem],
            hierarchy_levels=[0],
            confidences=[0.8],
        )

    async def _decompose_llm(
        self,
        problem: str,
        context: str | None = None,
    ) -> SubProblemDecomposition:
        """Decompose using LLM."""
        import json

        assert self._llm_client is not None, "LLM client required for LLM decomposition"

        prompt = self._build_decomposition_prompt(problem, context)

        try:
            response = await self._llm_client.complete(prompt)
            result = json.loads(response)

            return SubProblemDecomposition(
                original_problem=problem,
                subproblems=result.get("subproblems", [problem]),
                hierarchy_levels=result.get("levels", [0]),
                confidences=result.get("confidences", [0.8]),
            )
        except Exception as e:
            self._logger.error(
                "LLM decomposition failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return SubProblemDecomposition(
                original_problem=problem,
                subproblems=[problem],
                hierarchy_levels=[0],
                confidences=[0.5],
            )

    def _build_decomposition_prompt(
        self,
        problem: str,
        context: str | None = None,
    ) -> str:
        """Build prompt for LLM-based decomposition."""
        ctx = f"\nContext:\n{context}" if context else ""

        return f"""Decompose this programming problem into smaller subproblems.

Problem:
{problem}
{ctx}

Respond with JSON containing:
- subproblems: list of subproblem descriptions
- levels: list of hierarchy levels (0=root, higher=more specific)
- confidences: list of confidence scores (0-1) for each decomposition

Example:
{{"subproblems": ["Parse input", "Process data", "Format output"], "levels": [0, 0, 0], "confidences": [0.9, 0.85, 0.9]}}
"""


class TRMAdapter:
    """
    Adapter for TRM (Tiny Recursive Model) integration.

    Wraps the neural TRM agent to provide solution refinement
    for the LLM-guided MCTS system.
    """

    def __init__(
        self,
        trm_agent: TRMAgent | None = None,
        llm_client: Any | None = None,
        config: IntegrationConfig | None = None,
    ):
        """
        Initialize TRM adapter.

        Args:
            trm_agent: Pre-trained TRM neural network (optional)
            llm_client: LLM client for text-based refinement fallback
            config: Integration configuration
        """
        self._trm_agent = trm_agent
        self._llm_client = llm_client
        self._config = config or IntegrationConfig()
        self._logger = get_structured_logger(f"{__name__}.TRMAdapter")

    @property
    def has_neural_agent(self) -> bool:
        """Whether a neural TRM agent is available."""
        return self._trm_agent is not None

    async def refine(
        self,
        code: str,
        problem: str,
        test_cases: list[str] | None = None,
        max_iterations: int | None = None,
    ) -> RefinementResult:
        """
        Refine a solution iteratively.

        Uses neural TRM if available, otherwise falls back to LLM-based
        refinement.

        Args:
            code: Initial code solution
            problem: Problem description
            test_cases: Optional test cases for validation
            max_iterations: Maximum refinement iterations

        Returns:
            RefinementResult with refined code and metrics
        """
        max_iter = max_iterations or self._config.refinement_max_iterations

        if self._trm_agent is not None:
            return await self._refine_neural(code, problem, test_cases, max_iter)
        elif self._llm_client is not None:
            return await self._refine_llm(code, problem, test_cases, max_iter)
        else:
            # No refinement available, return original code
            self._logger.warning("No refinement agent available")
            return RefinementResult(
                original_code=code,
                refined_code=code,
                num_iterations=0,
                converged=True,
                improvement_score=0.0,
            )

    async def _refine_neural(
        self,
        code: str,
        problem: str,
        test_cases: list[str] | None,
        max_iterations: int,
    ) -> RefinementResult:
        """Refine using neural TRM agent."""
        self._logger.info("Refining solution with neural TRM")

        # Placeholder - actual implementation would:
        # 1. Encode code to tensor
        # 2. Run through TRM agent with iterations
        # 3. Decode refined tensor back to code

        return RefinementResult(
            original_code=code,
            refined_code=code,
            num_iterations=0,
            converged=True,
            improvement_score=0.0,
        )

    async def _refine_llm(
        self,
        code: str,
        problem: str,
        test_cases: list[str] | None,
        max_iterations: int,
    ) -> RefinementResult:
        """Refine using LLM iteratively."""
        import json

        assert self._llm_client is not None, "LLM client required for LLM refinement"

        current_code = code
        intermediate_codes = [code]
        residual_norms = []

        for iteration in range(max_iterations):
            prompt = self._build_refinement_prompt(current_code, problem, test_cases, iteration)

            try:
                response = await self._llm_client.complete(prompt)
                result = json.loads(response)

                new_code = result.get("refined_code", current_code)
                improvement = result.get("improvement_score", 0.0)
                converged = result.get("converged", False)

                intermediate_codes.append(new_code)
                residual_norms.append(improvement)

                if converged or improvement < self._config.refinement_convergence_threshold:
                    return RefinementResult(
                        original_code=code,
                        refined_code=new_code,
                        num_iterations=iteration + 1,
                        converged=True,
                        improvement_score=sum(residual_norms),
                        intermediate_codes=intermediate_codes,
                        residual_norms=residual_norms,
                    )

                current_code = new_code

            except Exception as e:
                self._logger.error(
                    "LLM refinement iteration failed",
                    iteration=iteration,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                break

        return RefinementResult(
            original_code=code,
            refined_code=current_code,
            num_iterations=len(intermediate_codes) - 1,
            converged=False,
            improvement_score=sum(residual_norms) if residual_norms else 0.0,
            intermediate_codes=intermediate_codes,
            residual_norms=residual_norms,
        )

    def _build_refinement_prompt(
        self,
        code: str,
        problem: str,
        test_cases: list[str] | None,
        iteration: int,
    ) -> str:
        """Build prompt for LLM-based refinement."""
        tests = "\n".join(test_cases) if test_cases else "No test cases provided"

        return f"""Refine this code solution (iteration {iteration + 1}).

Problem:
{problem}

Current Code:
```python
{code}
```

Test Cases:
{tests}

Respond with JSON containing:
- refined_code: the improved code
- improvement_score: estimated improvement (0-1)
- converged: whether further improvement is unlikely

Focus on:
- Correctness
- Edge cases
- Efficiency
- Code clarity
"""


class MetaControllerAdapter:
    """
    Adapter for Meta-Controller integration.

    Wraps the meta-controller to provide routing decisions
    for the LLM-guided MCTS system.
    """

    def __init__(
        self,
        meta_controller: AbstractMetaController | None = None,
        config: IntegrationConfig | None = None,
    ):
        """
        Initialize Meta-Controller adapter.

        Args:
            meta_controller: Pre-trained meta-controller (optional)
            config: Integration configuration
        """
        self._meta_controller = meta_controller
        self._config = config or IntegrationConfig()
        self._logger = get_structured_logger(f"{__name__}.MetaControllerAdapter")
        self._routing_history: list[RoutingDecision] = []

    @property
    def has_meta_controller(self) -> bool:
        """Whether a meta-controller is available."""
        return self._meta_controller is not None

    def route(
        self,
        problem: str,
        context: dict[str, Any],
    ) -> RoutingDecision:
        """
        Decide which agent should handle the problem.

        Args:
            problem: Problem description
            context: Current agent state context

        Returns:
            RoutingDecision with selected agent and confidence
        """
        if self._meta_controller is not None:
            return self._route_neural(problem, context)
        else:
            return self._route_heuristic(problem, context)

    def _route_neural(
        self,
        problem: str,
        context: dict[str, Any],
    ) -> RoutingDecision:
        """Route using neural meta-controller."""
        assert self._meta_controller is not None, "Meta-controller required for neural routing"

        features = self._meta_controller.extract_features(context)
        prediction = self._meta_controller.predict(features)

        decision = RoutingDecision(
            selected_agent=AgentType(prediction.agent),
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            reasoning=f"Neural meta-controller selected {prediction.agent}",
        )

        self._routing_history.append(decision)
        return decision

    def _route_heuristic(
        self,
        problem: str,
        context: dict[str, Any],
    ) -> RoutingDecision:
        """Route using heuristic rules when no meta-controller available."""
        # Simple heuristics based on problem characteristics
        problem_length = len(problem)
        has_complex_keywords = any(
            kw in problem.lower() for kw in ["optimize", "complex", "multiple", "hierarchical", "recursive"]
        )
        has_simple_keywords = any(kw in problem.lower() for kw in ["simple", "basic", "single", "straightforward"])

        # Note: Context signals (hrm_confidence, trm_confidence, mcts_value)
        # can be used for more sophisticated routing when neural meta-controller is available

        # Decision logic
        if has_complex_keywords or problem_length > 500:
            selected = AgentType.HRM
            reasoning = "Complex problem detected, using HRM for decomposition"
        elif has_simple_keywords and problem_length < 200:
            selected = AgentType.TRM
            reasoning = "Simple problem detected, using TRM for refinement"
        else:
            selected = AgentType.LLM_MCTS
            reasoning = "Using LLM-guided MCTS for exploration"

        probabilities = {
            "hrm": 0.33,
            "trm": 0.33,
            "mcts": 0.34,
        }
        probabilities[selected.value] = 0.6

        decision = RoutingDecision(
            selected_agent=selected,
            confidence=0.7,
            probabilities=probabilities,
            reasoning=reasoning,
        )

        self._routing_history.append(decision)
        return decision

    def get_routing_statistics(self) -> dict[str, Any]:
        """Get statistics about routing decisions."""
        if not self._routing_history:
            return {"total_decisions": 0}

        agent_counts: dict[str, int] = {}
        total_confidence = 0.0

        for decision in self._routing_history:
            agent = decision.selected_agent.value
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            total_confidence += decision.confidence

        return {
            "total_decisions": len(self._routing_history),
            "agent_distribution": agent_counts,
            "average_confidence": total_confidence / len(self._routing_history),
        }


@dataclass
class UnifiedSearchResult:
    """Result from unified search orchestration."""

    solution_found: bool
    best_code: str
    best_value: float
    agent_used: AgentType
    routing_decision: RoutingDecision | None
    decomposition: SubProblemDecomposition | None
    refinement: RefinementResult | None
    mcts_result: MCTSSearchResult | None
    execution_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedSearchOrchestrator:
    """
    Unified search orchestrator combining HRM, TRM, Meta-Controller, and LLM-MCTS.

    Orchestrates the full search pipeline:
    1. Meta-Controller routes to appropriate agent
    2. HRM decomposes complex problems
    3. LLM-MCTS performs tree search
    4. TRM refines solutions

    The orchestrator can operate in different modes:
    - Sequential: Route -> Decompose -> Search -> Refine
    - Parallel: Run multiple agents and combine results
    - Adaptive: Dynamically switch based on intermediate results
    """

    def __init__(
        self,
        llm_client: Any,
        mcts_config: LLMGuidedMCTSConfig | None = None,
        integration_config: IntegrationConfig | None = None,
        hrm_adapter: HRMAdapter | None = None,
        trm_adapter: TRMAdapter | None = None,
        meta_controller_adapter: MetaControllerAdapter | None = None,
    ):
        """
        Initialize unified search orchestrator.

        Args:
            llm_client: LLM client for LLM-guided MCTS
            mcts_config: Configuration for MCTS engine
            integration_config: Configuration for integration layer
            hrm_adapter: Pre-configured HRM adapter (optional)
            trm_adapter: Pre-configured TRM adapter (optional)
            meta_controller_adapter: Pre-configured meta-controller adapter (optional)
        """
        self._llm_client = llm_client
        self._mcts_config = mcts_config or LLMGuidedMCTSConfig()
        self._config = integration_config or IntegrationConfig()

        # Initialize adapters
        self._hrm = hrm_adapter or HRMAdapter(llm_client=llm_client, config=self._config)
        self._trm = trm_adapter or TRMAdapter(llm_client=llm_client, config=self._config)
        self._router = meta_controller_adapter or MetaControllerAdapter(config=self._config)

        # Initialize MCTS engine
        self._mcts_engine = LLMGuidedMCTSEngine(llm_client, self._mcts_config)

        self._logger = get_structured_logger(f"{__name__}.UnifiedSearchOrchestrator")

    async def search(
        self,
        problem: str,
        test_cases: list[str],
        context: dict[str, Any] | None = None,
        initial_code: str | None = None,
    ) -> UnifiedSearchResult:
        """
        Perform unified search using all available agents.

        Args:
            problem: Problem description
            test_cases: Test cases for validation
            context: Optional context for routing
            initial_code: Optional initial code to refine

        Returns:
            UnifiedSearchResult with solution and metadata
        """
        import time

        start_time = time.perf_counter()
        ctx = context or {}

        # Step 1: Route to appropriate agent
        routing_decision = None
        if self._config.use_meta_controller:
            routing_decision = self._router.route(problem, ctx)
            self._logger.info(
                "Routed to agent",
                agent=routing_decision.selected_agent.value,
                confidence=round(routing_decision.confidence, 2),
            )
        else:
            routing_decision = RoutingDecision(
                selected_agent=AgentType.LLM_MCTS,
                confidence=1.0,
                probabilities={"mcts": 1.0},
                reasoning="Default to LLM-MCTS",
            )

        # Step 2: Decompose if using HRM
        decomposition = None
        search_problems = [(problem, test_cases)]

        if self._config.use_hrm_decomposition and routing_decision.selected_agent == AgentType.HRM:
            decomposition = await self._hrm.decompose(problem, ctx.get("context"))

            if decomposition.num_subproblems > 1:
                # Use leaf problems for search
                leaf_problems = decomposition.get_leaf_problems()
                search_problems = [(sp, test_cases) for sp in leaf_problems]
                self._logger.info(
                    "Decomposed into subproblems",
                    num_subproblems=len(search_problems),
                )

        # Step 3: Run MCTS search on each subproblem
        mcts_results = []
        for sub_problem, sub_tests in search_problems:
            result = await self._mcts_engine.search(
                problem=sub_problem,
                test_cases=sub_tests,
                initial_code=initial_code or "",
            )
            mcts_results.append(result)

        # Combine results if multiple subproblems
        if len(mcts_results) == 1:
            best_mcts_result = mcts_results[0]
        else:
            # Find best result across subproblems
            best_mcts_result = max(mcts_results, key=lambda r: r.best_value)

        # Step 4: Refine if using TRM and solution found
        refinement = None
        final_code = best_mcts_result.best_code

        if (
            self._config.use_trm_refinement
            and best_mcts_result.solution_found
            and routing_decision.selected_agent in [AgentType.TRM, AgentType.LLM_MCTS]
        ):
            refinement = await self._trm.refine(
                code=best_mcts_result.best_code,
                problem=problem,
                test_cases=test_cases,
            )

            if refinement.improvement_score > 0:
                final_code = refinement.refined_code
                self._logger.info(
                    "Refined solution",
                    num_iterations=refinement.num_iterations,
                    improvement_score=round(refinement.improvement_score, 4),
                )

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return UnifiedSearchResult(
            solution_found=best_mcts_result.solution_found,
            best_code=final_code,
            best_value=best_mcts_result.best_value,
            agent_used=routing_decision.selected_agent,
            routing_decision=routing_decision,
            decomposition=decomposition,
            refinement=refinement,
            mcts_result=best_mcts_result,
            execution_time_ms=execution_time_ms,
            metadata={
                "num_subproblems": len(search_problems),
                "mcts_iterations": best_mcts_result.num_iterations,
                "mcts_expansions": best_mcts_result.num_expansions,
                "llm_calls": best_mcts_result.llm_calls,
            },
        )

    async def search_with_all_agents(
        self,
        problem: str,
        test_cases: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[UnifiedSearchResult]:
        """
        Run search with all agents and return all results.

        Useful for comparison and ensemble methods.

        Args:
            problem: Problem description
            test_cases: Test cases for validation
            context: Optional context

        Returns:
            List of results from each agent
        """
        import asyncio

        results = []

        # Define search tasks for each agent type
        async def search_with_agent(agent_type: AgentType) -> UnifiedSearchResult:
            ctx = {**(context or {}), "force_agent": agent_type.value}
            return await self.search(problem, test_cases, ctx)

        if self._config.enable_parallel_search:
            # Run all agents in parallel
            tasks = [
                search_with_agent(AgentType.HRM),
                search_with_agent(AgentType.TRM),
                search_with_agent(AgentType.LLM_MCTS),
            ]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions and keep only successful results
            results = [r for r in gathered if isinstance(r, UnifiedSearchResult)]
        else:
            # Run sequentially
            for agent_type in [AgentType.HRM, AgentType.TRM, AgentType.LLM_MCTS]:
                try:
                    result = await search_with_agent(agent_type)
                    results.append(result)
                except Exception as e:
                    self._logger.error(
                        "Agent failed",
                        agent_type=agent_type.value,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics from all components."""
        return {
            "mcts": self._mcts_engine.get_statistics(),
            "routing": self._router.get_routing_statistics(),
            "hrm_available": self._hrm.has_neural_agent,
            "trm_available": self._trm.has_neural_agent,
            "meta_controller_available": self._router.has_meta_controller,
        }


def create_unified_orchestrator(
    llm_client: Any,
    preset: str = "balanced",
    hrm_agent: HRMAgent | None = None,
    trm_agent: TRMAgent | None = None,
    meta_controller: AbstractMetaController | None = None,
    **kwargs: Any,
) -> UnifiedSearchOrchestrator:
    """
    Factory function to create a configured UnifiedSearchOrchestrator.

    Args:
        llm_client: LLM client for completions
        preset: Configuration preset ('fast', 'balanced', 'thorough')
        hrm_agent: Optional pre-trained HRM agent
        trm_agent: Optional pre-trained TRM agent
        meta_controller: Optional pre-trained meta-controller
        **kwargs: Additional configuration overrides

    Returns:
        Configured UnifiedSearchOrchestrator
    """
    from .config import get_preset_config

    mcts_config = get_preset_config(preset)

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(mcts_config, key):
            setattr(mcts_config, key, value)

    # Create adapters
    integration_config = IntegrationConfig()
    hrm_adapter = HRMAdapter(hrm_agent, llm_client, integration_config)
    trm_adapter = TRMAdapter(trm_agent, llm_client, integration_config)
    meta_controller_adapter = MetaControllerAdapter(meta_controller, integration_config)

    return UnifiedSearchOrchestrator(
        llm_client=llm_client,
        mcts_config=mcts_config,
        integration_config=integration_config,
        hrm_adapter=hrm_adapter,
        trm_adapter=trm_adapter,
        meta_controller_adapter=meta_controller_adapter,
    )
