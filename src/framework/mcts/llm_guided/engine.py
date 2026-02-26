"""
LLM-Guided MCTS Engine.

Main orchestrator that combines:
- MCTS tree search with UCB1 selection
- LLM-based expansion (Generator agent)
- LLM-based evaluation (Reflector agent)
- Code execution for test verification
- Training data collection for neural network distillation
- LangGraph integration for state management
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

# LangGraph imports
try:
    from langgraph.graph import END, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"

from src.observability.logging import get_structured_logger

from . import constants as C
from .agents import GeneratorAgent, LLMClientProtocol, ReflectorAgent
from .config import LLMGuidedMCTSConfig, LLMGuidedMCTSPreset, create_llm_mcts_preset
from .data_collector import TrainingDataCollector
from .executor import CodeExecutionResult, CodeExecutor
from .node import LLMGuidedMCTSNode, NodeState, NodeStatus, create_root_node

if TYPE_CHECKING:
    pass

logger = get_structured_logger(__name__)


# ============================================================================
# LangGraph State
# ============================================================================


class TreeState(TypedDict):
    """State for LangGraph MCTS orchestration."""

    # Tree structure
    root: LLMGuidedMCTSNode
    current_node: LLMGuidedMCTSNode | None

    # Problem definition
    problem: str
    test_cases: list[str]

    # Search state
    iteration: int
    max_iterations: int
    solution_found: bool
    best_solution: str | None

    # Episode tracking
    episode_id: str

    # Statistics
    total_expansions: int
    total_evaluations: int
    total_llm_calls: int
    total_tokens: int

    # Control
    should_terminate: bool


# ============================================================================
# MCTS Search Statistics
# ============================================================================


@dataclass
class MCTSSearchResult:
    """Result from MCTS search."""

    solution_found: bool
    """Whether a solution was found."""

    best_code: str
    """Best code found (solution or highest value)."""

    best_value: float
    """Value of the best code."""

    num_iterations: int
    """Number of iterations completed."""

    num_expansions: int
    """Number of node expansions."""

    num_evaluations: int
    """Number of evaluations performed."""

    tree_depth: int
    """Maximum depth reached in the tree."""

    tree_size: int
    """Total number of nodes in the tree."""

    execution_time_ms: float
    """Total execution time in milliseconds."""

    llm_calls: int
    """Total LLM API calls made."""

    tokens_used: int
    """Total tokens used."""

    episode_id: str
    """Episode identifier."""

    root_visits: int
    """Number of visits to root node."""

    action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Statistics for each action from root."""

    test_results: CodeExecutionResult | None = None
    """Test results for the best solution."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "solution_found": self.solution_found,
            "best_code": self.best_code,
            "best_value": self.best_value,
            "num_iterations": self.num_iterations,
            "num_expansions": self.num_expansions,
            "num_evaluations": self.num_evaluations,
            "tree_depth": self.tree_depth,
            "tree_size": self.tree_size,
            "execution_time_ms": self.execution_time_ms,
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used,
            "episode_id": self.episode_id,
            "root_visits": self.root_visits,
            "action_stats": self.action_stats,
            "test_results": self.test_results.to_dict() if self.test_results else None,
        }


# ============================================================================
# LLM-Guided MCTS Engine
# ============================================================================


class LLMGuidedMCTSEngine:
    """
    Main MCTS engine with LLM guidance.

    Orchestrates:
    1. Selection: UCB1-based node selection
    2. Expansion: LLM generates code variants
    3. Evaluation: Run tests + LLM scoring
    4. Backpropagation: Update tree statistics
    5. Data collection: Save training examples
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: LLMGuidedMCTSConfig | None = None,
        data_collector: TrainingDataCollector | None = None,
    ):
        """
        Initialize the LLM-Guided MCTS engine.

        Args:
            llm_client: LLM client for API calls
            config: MCTS configuration
            data_collector: Optional data collector for training
        """
        self._config = config or create_llm_mcts_preset(LLMGuidedMCTSPreset.BALANCED)
        self._llm_client = llm_client

        # Create agents
        self._generator = GeneratorAgent(llm_client, self._config.generator_config)
        self._reflector = ReflectorAgent(llm_client, self._config.reflector_config)

        # Create executor
        self._executor = CodeExecutor(
            timeout_seconds=self._config.execution_timeout_seconds,
            max_memory_mb=self._config.max_memory_mb,
            allow_network=self._config.allow_network,
        )

        # Data collection
        self._data_collector = data_collector
        if self._config.collect_training_data and data_collector is None:
            self._data_collector = TrainingDataCollector(output_dir=self._config.training_data_dir)

        # Random number generator
        self._rng = np.random.default_rng(self._config.seed)

        # Statistics
        self._total_searches = 0
        self._total_solutions = 0

        logger.info(
            "Initialized LLMGuidedMCTSEngine",
            config_name=self._config.name,
            num_iterations=self._config.num_iterations,
            collect_training_data=self._config.collect_training_data,
        )

    @property
    def config(self) -> LLMGuidedMCTSConfig:
        """Get the current configuration."""
        return self._config

    def reset_seed(self, seed: int) -> None:
        """Reset the random seed for reproducibility."""
        self._config.seed = seed
        self._rng = np.random.default_rng(seed)

    # ========================================================================
    # MCTS Core Operations
    # ========================================================================

    def _select(self, root: LLMGuidedMCTSNode) -> LLMGuidedMCTSNode:
        """
        Selection phase: Traverse tree to find leaf node.

        Uses UCB1 to balance exploration and exploitation.
        """
        node = root

        while node.children and not node.is_terminal:
            selected = node.select_child(self._config.exploration_weight)
            if selected is None:
                break
            node = selected

        return node

    async def _expand(
        self,
        node: LLMGuidedMCTSNode,
        episode_id: str,
    ) -> list[LLMGuidedMCTSNode]:
        """
        Expansion phase: Generate code variants using LLM.

        Args:
            node: Node to expand
            episode_id: Episode identifier

        Returns:
            List of newly created child nodes
        """
        if node.is_terminal:
            return []

        # Generate variants using LLM
        output = await self._generator.run(node.state)

        if not output.variants:
            logger.warning("Generator produced no variants")
            node.status = NodeStatus.TERMINAL_FAILURE
            return []

        # Create child nodes for each variant
        children = []
        for i, variant in enumerate(output.variants):
            child_state = node.state.with_new_code(variant.code)
            action = f"variant_{i}"

            child = node.add_child(
                state=child_state,
                action=action,
                llm_action_probs=output.action_probs,
                episode_id=episode_id,
            )

            child.generated_variants = [
                {"code": v.code, "confidence": v.confidence, "reasoning": v.reasoning} for v in output.variants
            ]

            children.append(child)

        node.status = NodeStatus.EXPANDED

        logger.debug(
            "Expanded node",
            depth=node.depth,
            num_children=len(children),
        )

        return children

    async def _evaluate(
        self,
        node: LLMGuidedMCTSNode,
    ) -> tuple[float, bool]:
        """
        Evaluation phase: Run tests and get LLM value estimate.

        Args:
            node: Node to evaluate

        Returns:
            Tuple of (reward, is_solution)
        """
        # Execute code against tests
        test_result = self._executor.execute(
            code=node.state.code,
            test_cases=node.state.test_cases,
        )
        node.test_results = test_result.to_dict()

        # Check if all tests passed
        if test_result.passed:
            node.status = NodeStatus.TERMINAL_SUCCESS
            node.llm_value_estimate = 1.0
            return 1.0, True

        # Update state with errors for future generations
        if test_result.errors:
            node.state = NodeState(
                code=node.state.code,
                problem=node.state.problem,
                test_cases=node.state.test_cases,
                errors=test_result.errors,
                attempt_history=node.state.attempt_history,
                metadata=node.state.metadata,
            )

        # Get LLM value estimate
        reflection_output = await self._reflector.run(
            node.state,
            test_results=test_result.to_dict(),
        )

        node.llm_value_estimate = reflection_output.value
        node.reflection = reflection_output.reflection

        # Check if reflector thinks it's a solution (shouldn't happen if tests failed)
        if reflection_output.is_solution and test_result.passed:
            node.status = NodeStatus.TERMINAL_SUCCESS
            return 1.0, True

        # Return value as reward (can be negative for very bad code)
        # Scale [0, 1] to [-1, 1] using configurable constants
        reward = reflection_output.value * C.REWARD_SCALING_FACTOR - C.REWARD_OFFSET
        return reward, False

    def _backpropagate(self, node: LLMGuidedMCTSNode, reward: float) -> None:
        """
        Backpropagation phase: Update statistics up to root.

        Args:
            node: Leaf node to start from
            reward: Reward to propagate
        """
        node.backpropagate(reward)

    # ========================================================================
    # Main Search Method
    # ========================================================================

    async def search(
        self,
        problem: str,
        test_cases: list[str],
        initial_code: str = "",
    ) -> MCTSSearchResult:
        """
        Run MCTS search to find solution code.

        Args:
            problem: Problem description
            test_cases: List of test case assertions
            initial_code: Optional starting code

        Returns:
            MCTSSearchResult with best code and statistics
        """
        start_time = time.perf_counter()
        episode_id = str(uuid.uuid4())

        # Start data collection
        if self._data_collector:
            self._data_collector.start_episode(
                episode_id=episode_id,
                problem_type="code_generation",
                difficulty="unknown",
                config_name=self._config.name,
            )

        # Create root node
        root = create_root_node(
            problem=problem,
            initial_code=initial_code,
            test_cases=test_cases,
            episode_id=episode_id,
            seed=self._config.seed,
        )

        # Track statistics
        num_expansions = 0
        num_evaluations = 0
        solution_found = False
        best_solution_node: LLMGuidedMCTSNode | None = None

        logger.info(
            "Starting MCTS search",
            episode_id=episode_id,
            max_iterations=self._config.num_iterations,
            problem_length=len(problem),
            num_tests=len(test_cases),
        )

        try:
            # Main MCTS loop
            for iteration in range(self._config.num_iterations):
                # Selection
                leaf = self._select(root)

                # Check if we need to expand
                if not leaf.is_terminal and leaf.visits > 0:
                    # Expansion
                    children = await self._expand(leaf, episode_id)
                    num_expansions += 1

                    if children:
                        # Select first unexplored child
                        leaf = children[0]

                # Evaluation
                reward, is_solution = await self._evaluate(leaf)
                num_evaluations += 1

                # Record for training
                if self._data_collector:
                    self._data_collector.record_node(leaf)

                # Backpropagation
                self._backpropagate(leaf, reward)

                # Check for solution
                if is_solution:
                    solution_found = True
                    best_solution_node = leaf
                    logger.info(
                        "Solution found!",
                        iteration=iteration + 1,
                        depth=leaf.depth,
                    )
                    if self._config.early_termination_on_solution:
                        break

                # Log progress periodically
                if (iteration + 1) % 10 == 0:
                    logger.debug(
                        "Search progress",
                        iteration=iteration + 1,
                        root_visits=root.visits,
                        num_children=len(root.children),
                    )

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(
                "Search failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

        # Compute MCTS policies for training
        if self._config.save_mcts_policy:
            self._compute_all_mcts_policies(root)

        # Find best node if no solution found
        if not solution_found:
            best_solution_node = self._find_best_node(root)

        # Compute tree statistics
        tree_depth = self._compute_tree_depth(root)
        tree_size = self._count_nodes(root)

        # Record final MCTS policies
        if self._data_collector and root.children:
            self._data_collector.record_mcts_policy(root)

        # Finalize data collection
        if self._data_collector:
            # Execute final solution to verify (result used for logging if needed)
            if best_solution_node:
                _ = self._executor.execute(
                    best_solution_node.state.code,
                    test_cases,
                )

            self._data_collector.finalize_episode(
                outcome=1.0 if solution_found else -1.0,
                solution_found=solution_found,
                solution_code=best_solution_node.state.code if best_solution_node else None,
                total_iterations=root.visits,
                total_llm_calls=self._generator.total_calls + self._reflector.total_calls,
                total_tokens=self._generator.total_tokens + self._reflector.total_tokens,
            )

        # Build result
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Get action statistics from root
        action_stats = {}
        for child in root.children:
            if child.action:
                action_stats[child.action] = {
                    "visits": child.visits,
                    "q_value": child.q_value,
                    "is_solution": child.is_solution,
                }

        result = MCTSSearchResult(
            solution_found=solution_found,
            best_code=best_solution_node.state.code if best_solution_node else "",
            best_value=best_solution_node.q_value if best_solution_node else 0.0,
            num_iterations=root.visits,
            num_expansions=num_expansions,
            num_evaluations=num_evaluations,
            tree_depth=tree_depth,
            tree_size=tree_size,
            execution_time_ms=execution_time_ms,
            llm_calls=self._generator.total_calls + self._reflector.total_calls,
            tokens_used=self._generator.total_tokens + self._reflector.total_tokens,
            episode_id=episode_id,
            root_visits=root.visits,
            action_stats=action_stats,
        )

        # Update statistics
        self._total_searches += 1
        if solution_found:
            self._total_solutions += 1

        logger.info(
            "Search complete",
            episode_id=episode_id,
            solution_found=solution_found,
            iterations=root.visits,
            tree_depth=tree_depth,
            tree_size=tree_size,
            execution_time_ms=round(execution_time_ms, 2),
        )

        return result

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _find_best_node(self, root: LLMGuidedMCTSNode) -> LLMGuidedMCTSNode | None:
        """Find the best node in the tree (highest value with most visits)."""
        best_node = None
        best_score = float("-inf")

        def visit(node: LLMGuidedMCTSNode) -> None:
            nonlocal best_node, best_score

            if node.visits > 0 and node.state.code:
                # Score combines value and visit count
                score = node.q_value + C.BEST_NODE_VISIT_WEIGHT * math.log1p(node.visits)
                if score > best_score:
                    best_score = score
                    best_node = node

            for child in node.children:
                visit(child)

        visit(root)
        return best_node

    def _compute_tree_depth(self, node: LLMGuidedMCTSNode) -> int:
        """Compute maximum depth of the tree."""
        if not node.children:
            return node.depth

        return max(self._compute_tree_depth(child) for child in node.children)

    def _count_nodes(self, node: LLMGuidedMCTSNode) -> int:
        """Count total nodes in the tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _compute_all_mcts_policies(self, node: LLMGuidedMCTSNode) -> None:
        """Recursively compute MCTS policies for all nodes."""
        if node.children:
            node.compute_mcts_policy()
            for child in node.children:
                self._compute_all_mcts_policies(child)

    # ========================================================================
    # LangGraph Integration
    # ========================================================================

    def build_langgraph(self) -> StateGraph:
        """
        Build LangGraph state machine for MCTS orchestration.

        Returns:
            Compiled StateGraph
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph not available. Install with: pip install langgraph")

        workflow = StateGraph(TreeState)

        # Add nodes
        workflow.add_node("select", self._langgraph_select)
        workflow.add_node("expand", self._langgraph_expand)
        workflow.add_node("evaluate", self._langgraph_evaluate)
        workflow.add_node("backpropagate", self._langgraph_backpropagate)

        # Define flow
        workflow.set_entry_point("select")
        workflow.add_edge("select", "expand")
        workflow.add_edge("expand", "evaluate")
        workflow.add_edge("evaluate", "backpropagate")
        workflow.add_conditional_edges(
            "backpropagate",
            self._langgraph_should_continue,
            {"continue": "select", "end": END},
        )

        return workflow

    def _langgraph_select(self, state: TreeState) -> dict:
        """LangGraph node for selection phase."""
        root = state["root"]
        current = self._select(root)
        return {"current_node": current}

    async def _langgraph_expand(self, state: TreeState) -> dict:
        """LangGraph node for expansion phase."""
        node = state["current_node"]
        episode_id = state["episode_id"]

        if node and not node.is_terminal and node.visits > 0:
            children = await self._expand(node, episode_id)
            if children:
                return {
                    "current_node": children[0],
                    "total_expansions": state["total_expansions"] + 1,
                }

        return {}

    async def _langgraph_evaluate(self, state: TreeState) -> dict:
        """LangGraph node for evaluation phase."""
        node = state["current_node"]
        if node is None:
            return {}

        reward, is_solution = await self._evaluate(node)

        updates: dict[str, Any] = {
            "total_evaluations": state["total_evaluations"] + 1,
            "total_llm_calls": state["total_llm_calls"] + 1,
        }

        if is_solution:
            updates["solution_found"] = True
            updates["best_solution"] = node.state.code

        return updates

    def _langgraph_backpropagate(self, state: TreeState) -> dict:
        """LangGraph node for backpropagation phase."""
        node = state["current_node"]
        if node:
            reward = node.llm_value_estimate * C.REWARD_SCALING_FACTOR - C.REWARD_OFFSET
            self._backpropagate(node, reward)

        return {"iteration": state["iteration"] + 1}

    def _langgraph_should_continue(self, state: TreeState) -> str:
        """Determine whether to continue search."""
        if state["solution_found"] and self._config.early_termination_on_solution:
            return "end"
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        return "continue"

    async def search_with_langgraph(
        self,
        problem: str,
        test_cases: list[str],
        initial_code: str = "",
    ) -> MCTSSearchResult:
        """
        Run MCTS search using LangGraph orchestration.

        Args:
            problem: Problem description
            test_cases: List of test case assertions
            initial_code: Optional starting code

        Returns:
            MCTSSearchResult with best code and statistics
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph not available")

        start_time = time.perf_counter()
        episode_id = str(uuid.uuid4())

        # Create root node
        root = create_root_node(
            problem=problem,
            initial_code=initial_code,
            test_cases=test_cases,
            episode_id=episode_id,
            seed=self._config.seed,
        )

        # Build and compile graph (no checkpointing - MCTS tree is in-memory)
        graph = self.build_langgraph()
        app = graph.compile()

        # Initial state
        initial_state: TreeState = {
            "root": root,
            "current_node": None,
            "problem": problem,
            "test_cases": test_cases,
            "iteration": 0,
            "max_iterations": self._config.num_iterations,
            "solution_found": False,
            "best_solution": None,
            "episode_id": episode_id,
            "total_expansions": 0,
            "total_evaluations": 0,
            "total_llm_calls": 0,
            "total_tokens": 0,
            "should_terminate": False,
        }

        # Run graph
        config = {"configurable": {"thread_id": episode_id}}
        final_state = await app.ainvoke(initial_state, config=config)

        # Build result
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        best_node = self._find_best_node(root)

        return MCTSSearchResult(
            solution_found=final_state["solution_found"],
            best_code=final_state.get("best_solution") or (best_node.state.code if best_node else ""),
            best_value=best_node.q_value if best_node else 0.0,
            num_iterations=final_state["iteration"],
            num_expansions=final_state["total_expansions"],
            num_evaluations=final_state["total_evaluations"],
            tree_depth=self._compute_tree_depth(root),
            tree_size=self._count_nodes(root),
            execution_time_ms=execution_time_ms,
            llm_calls=final_state["total_llm_calls"],
            tokens_used=final_state["total_tokens"],
            episode_id=episode_id,
            root_visits=root.visits,
        )

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_searches": self._total_searches,
            "total_solutions": self._total_solutions,
            "success_rate": (self._total_solutions / self._total_searches if self._total_searches > 0 else 0.0),
            "generator_stats": {
                "total_calls": self._generator.total_calls,
                "total_tokens": self._generator.total_tokens,
            },
            "reflector_stats": {
                "total_calls": self._reflector.total_calls,
                "total_tokens": self._reflector.total_tokens,
            },
            "data_collector_stats": (self._data_collector.get_statistics() if self._data_collector else None),
            "config": self._config.to_dict(),
        }


# ============================================================================
# Factory Function
# ============================================================================


def create_llm_mcts_engine(
    llm_client: LLMClientProtocol,
    preset: LLMGuidedMCTSPreset = LLMGuidedMCTSPreset.BALANCED,
    **config_overrides,
) -> LLMGuidedMCTSEngine:
    """
    Create an LLM-Guided MCTS engine with preset configuration.

    Args:
        llm_client: LLM client for API calls
        preset: Configuration preset
        **config_overrides: Optional configuration overrides

    Returns:
        Configured LLMGuidedMCTSEngine
    """
    config = create_llm_mcts_preset(preset)
    if config_overrides:
        config = config.copy(**config_overrides)

    return LLMGuidedMCTSEngine(llm_client, config)
