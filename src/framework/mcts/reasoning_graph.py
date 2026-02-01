"""
Enhanced LangGraph Integration for Reasoning-Enabled MCTS.

Provides LangGraph workflow components for integrating:
- Process Reward Models (PRMs)
- Extended Thinking evaluation
- Hybrid search strategies
- Dual-agent architecture

Based on the LangGraph LATS (Language Agent Tree Search) pattern
with modern reasoning model enhancements.
"""

from __future__ import annotations

import asyncio
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, NotRequired, TypedDict

import numpy as np

# LangGraph imports
try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = None
    END = "END"
    MemorySaver = None
    LANGGRAPH_AVAILABLE = False

from .core import MCTSEngine, MCTSNode, MCTSState
from .extended_thinking import (
    AdaptiveThinkingRouter,
    ClaudeExtendedThinkingEvaluator,
    ExtendedThinkingEvaluator,
    ParallelThinkingEvaluator,
    ThinkingBudget,
    ThinkingEnhancedMCTSEvaluator,
    ThinkingMode,
    ThinkingResult,
)
from .hybrid_search import (
    HybridMCTSSearch,
    HybridSearchConfig,
    HybridSearchResult,
    SearchPhase,
    VerifiedHybridSearch,
)
from .policies import HybridRolloutPolicy, SelectionPolicy
from .process_reward_model import (
    EnsemblePRM,
    LLMProcessRewardModel,
    MonteCarloProcessRewardModel,
    PRMEnhancedMCTSConfig,
    PRMMCTSIntegration,
    PRMTrainingCollector,
    ProcessRewardModel,
    ReasoningStep,
    ReasoningTrajectory,
)
from .reasoning_node import (
    ActorAgent,
    DualAgentMCTSController,
    ReasonerAgent,
    ReasoningMCTSNode,
    ReasoningMetadata,
)


# ============================================================================
# Enhanced State Definitions
# ============================================================================


class EnhancedTreeState(TypedDict):
    """
    Enhanced state for reasoning-enabled MCTS in LangGraph.

    Extends standard AgentState with:
    - PRM model and scores
    - Thinking budget tracking
    - Reasoning traces
    - Hybrid search configuration
    """

    # Core state
    root: NotRequired[Any]  # ReasoningMCTSNode
    input: str

    # Process Reward Model
    prm_model: NotRequired[Any]  # ProcessRewardModel
    prm_scores: NotRequired[list[float]]
    prm_config: NotRequired[dict]

    # Extended Thinking
    thinking_budget: int
    thinking_mode: NotRequired[str]
    thinking_traces: Annotated[list[str], operator.add]
    total_thinking_tokens: NotRequired[int]

    # Hybrid Search
    search_config: NotRequired[dict]
    search_phase: NotRequired[str]
    search_result: NotRequired[dict]

    # Reasoning Trajectory
    current_trajectory: NotRequired[dict]
    trajectory_history: Annotated[list[dict], operator.add]

    # Control Flow
    iteration_count: int
    max_iterations: int
    requires_deeper_thinking: NotRequired[bool]

    # Results
    best_solution: NotRequired[str]
    confidence: NotRequired[float]
    verified: NotRequired[bool]

    # Metadata
    metadata: NotRequired[dict]


class ReflectionState(TypedDict):
    """State for the reflection/evaluation node."""

    node_to_evaluate: Any  # ReasoningMCTSNode
    trajectory_context: str
    depth: int
    visits: int
    ucb_score: float


# ============================================================================
# Reasoning-Enhanced Graph Nodes
# ============================================================================


@dataclass
class ReasoningGraphConfig:
    """Configuration for reasoning-enhanced graph."""

    # MCTS settings
    mcts_iterations: int = 50
    exploration_weight: float = 1.414
    max_tree_depth: int = 10

    # PRM settings
    prm_enabled: bool = True
    prm_selection_weight: float = 0.5
    prm_expansion_threshold: float = 0.3

    # Extended thinking settings
    thinking_enabled: bool = True
    min_thinking_tokens: int = 1024
    max_thinking_tokens: int = 65536
    default_thinking_tokens: int = 8192

    # Hybrid search settings
    hybrid_search_enabled: bool = True
    num_parallel_candidates: int = 8
    prm_top_k: int = 3

    # Verification
    verification_enabled: bool = True

    # Complexity routing
    adaptive_complexity: bool = True
    simple_task_threshold: float = 0.3
    overthinking_threshold: float = 0.6


class ReasoningGraphBuilder:
    """
    Builds LangGraph workflows with reasoning model integration.

    Creates graph with:
    - PRM-enhanced selection and evaluation
    - Extended thinking for critical nodes
    - Hybrid search strategy
    - Dual-agent coordination
    """

    def __init__(
        self,
        model_adapter: Any,
        logger: Any,
        config: ReasoningGraphConfig | None = None,
        prm: ProcessRewardModel | None = None,
        thinking_evaluator: ExtendedThinkingEvaluator | None = None,
    ):
        """
        Initialize reasoning graph builder.

        Args:
            model_adapter: Adapter for LLM calls
            logger: Logger instance
            config: Graph configuration
            prm: Optional pre-configured PRM
            thinking_evaluator: Optional pre-configured thinking evaluator
        """
        self.model_adapter = model_adapter
        self.logger = logger
        self.config = config or ReasoningGraphConfig()

        # MCTS engine
        self.mcts_engine = MCTSEngine(
            seed=42,
            exploration_weight=self.config.exploration_weight,
        )

        # Process Reward Model
        self.prm = prm
        if prm is None and self.config.prm_enabled:
            self.prm = self._create_default_prm()

        self.prm_integration = PRMMCTSIntegration(self.prm) if self.prm else None

        # Extended thinking evaluator
        self.thinking_evaluator = thinking_evaluator
        if thinking_evaluator is None and self.config.thinking_enabled:
            self.thinking_evaluator = self._create_default_thinking_evaluator()

        self.thinking_mcts_evaluator = (
            ThinkingEnhancedMCTSEvaluator(self.thinking_evaluator)
            if self.thinking_evaluator
            else None
        )

        # Hybrid search
        self.hybrid_search = None
        if self.config.hybrid_search_enabled:
            self.hybrid_search = self._create_hybrid_search()

        # Dual-agent controller
        self.dual_agent_controller = None

        # Statistics
        self.total_runs = 0
        self.total_thinking_tokens = 0

    def _create_default_prm(self) -> ProcessRewardModel:
        """Create default LLM-based PRM."""

        async def evaluate_fn(prompt: str) -> dict:
            response = await self.model_adapter.generate(
                prompt=prompt,
                temperature=0.2,
            )
            return {"text": response.text}

        return LLMProcessRewardModel(
            evaluate_fn=evaluate_fn,
            cache_size=1000,
        )

    def _create_default_thinking_evaluator(self) -> ExtendedThinkingEvaluator:
        """Create default thinking evaluator."""
        # Use parallel evaluator for robustness
        base_evaluator = ClaudeExtendedThinkingEvaluator(
            client=self.model_adapter,
            default_budget=ThinkingBudget(
                min_tokens=self.config.min_thinking_tokens,
                max_tokens=self.config.max_thinking_tokens,
                default_tokens=self.config.default_thinking_tokens,
            ),
        )

        parallel_evaluator = ParallelThinkingEvaluator(
            base_evaluator=base_evaluator,
            num_samples=3,
            aggregation="max",
        )

        return AdaptiveThinkingRouter(
            evaluator=base_evaluator,
            parallel_evaluator=parallel_evaluator,
            overthinking_threshold=self.config.overthinking_threshold,
        )

    def _create_hybrid_search(self) -> HybridMCTSSearch:
        """Create hybrid search instance."""
        search_config = HybridSearchConfig(
            num_parallel_candidates=self.config.num_parallel_candidates,
            prm_top_k=self.config.prm_top_k,
        )

        if self.config.verification_enabled:
            return VerifiedHybridSearch(
                mcts_engine=self.mcts_engine,
                prm=self.prm,
                thinking_evaluator=self.thinking_evaluator,
                config=search_config,
            )
        else:
            return HybridMCTSSearch(
                mcts_engine=self.mcts_engine,
                prm=self.prm,
                thinking_evaluator=self.thinking_evaluator,
                config=search_config,
            )

    def build_graph(self) -> StateGraph:
        """
        Build the reasoning-enhanced LangGraph workflow.

        Returns:
            Configured StateGraph
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph not installed")

        workflow = StateGraph(EnhancedTreeState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("prm_select", self._prm_enhanced_select_node)
        workflow.add_node("thinking_expand", self._thinking_expand_node)
        workflow.add_node("parallel_simulate", self._parallel_simulate_node)
        workflow.add_node("prm_evaluate", self._prm_evaluate_node)
        workflow.add_node("hierarchical_backprop", self._hierarchical_backprop_node)
        workflow.add_node("refine", self._extended_thinking_refine_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "prm_select")
        workflow.add_edge("prm_select", "thinking_expand")
        workflow.add_edge("thinking_expand", "parallel_simulate")
        workflow.add_edge("parallel_simulate", "prm_evaluate")

        # Conditional edge for deeper thinking
        workflow.add_conditional_edges(
            "prm_evaluate",
            self._requires_deeper_thinking,
            {
                "refine": "refine",
                "backprop": "hierarchical_backprop",
            },
        )

        workflow.add_edge("refine", "hierarchical_backprop")

        # Conditional edge for iteration
        workflow.add_conditional_edges(
            "hierarchical_backprop",
            self._should_continue,
            {
                "continue": "prm_select",
                "synthesize": "synthesize",
            },
        )

        workflow.add_edge("synthesize", END)

        return workflow

    # ========================================================================
    # Graph Node Implementations
    # ========================================================================

    async def _initialize_node(self, state: EnhancedTreeState) -> dict:
        """Initialize search state."""
        self.logger.info(f"Initializing reasoning search: {state['input'][:100]}")
        self.total_runs += 1

        # Create root state
        root_state = MCTSState(
            state_id="root",
            features={"query": state["input"][:200]},
        )

        # Create root node with reasoning capability
        root = ReasoningMCTSNode(
            state=root_state,
            rng=self.mcts_engine.rng,
        )

        # Initial trajectory
        trajectory = ReasoningTrajectory(query=state["input"])

        return {
            "root": root,
            "iteration_count": 0,
            "thinking_traces": [],
            "total_thinking_tokens": 0,
            "current_trajectory": trajectory.__dict__,
            "metadata": {
                "start_time": time.time(),
                "config": self.config.__dict__,
            },
        }

    async def _prm_enhanced_select_node(self, state: EnhancedTreeState) -> dict:
        """PRM-enhanced node selection."""
        root = state["root"]

        if not isinstance(root, ReasoningMCTSNode):
            # Fallback to standard selection
            selected = self.mcts_engine.select(root)
            return {"metadata": {"selected_node_id": selected.state.state_id}}

        # Use PRM-enhanced selection
        selected = root
        path = [root]

        while selected.children and not selected.terminal:
            if self.mcts_engine.should_expand(selected):
                break

            selected = selected.select_child_with_prm(
                exploration_weight=self.config.exploration_weight,
                prm_weight=self.config.prm_selection_weight,
            )
            path.append(selected)

        return {
            "metadata": {
                "selected_node_id": selected.state.state_id,
                "selection_depth": len(path),
                "prm_selection_used": True,
            },
        }

    async def _thinking_expand_node(self, state: EnhancedTreeState) -> dict:
        """Expand node using extended thinking for action generation."""
        root = state["root"]

        # Find the selected leaf
        leaf = self._find_leaf(root)

        if leaf.terminal:
            return {"metadata": {"expansion": "terminal_node"}}

        # Generate available actions
        actions = self._generate_actions(leaf)
        if not actions:
            leaf.terminal = True
            return {"metadata": {"expansion": "no_actions"}}

        leaf.available_actions = actions

        # Use thinking for strategy generation if dual-agent is available
        if self.dual_agent_controller:
            children = await self.dual_agent_controller.expand_with_reasoning(
                leaf,
                context=state["input"],
                n_strategies=min(5, len(actions)),
            )
            return {
                "metadata": {
                    "expansion": "dual_agent",
                    "children_created": len(children),
                },
            }

        # Standard expansion
        action = leaf.get_unexpanded_action()
        if action:
            new_state = MCTSState(
                state_id=f"{leaf.state.state_id}_{action}",
                features={**leaf.state.features, "action": action},
            )

            if isinstance(leaf, ReasoningMCTSNode):
                step = ReasoningStep(
                    content=action,
                    step_index=0,
                    step_type="action",
                )
                leaf.add_reasoning_child(action, new_state, step)
            else:
                leaf.add_child(action, new_state)

        return {"metadata": {"expansion": "standard", "action": action}}

    async def _parallel_simulate_node(self, state: EnhancedTreeState) -> dict:
        """Run parallel simulations with PRM scoring."""
        root = state["root"]
        leaf = self._find_leaf(root)

        # Create rollout policy
        rollout_policy = HybridRolloutPolicy(
            heuristic_fn=lambda s: 0.5,
            heuristic_weight=0.7,
            random_weight=0.3,
        )

        # Run simulation
        value = await self.mcts_engine.simulate(leaf, rollout_policy, max_depth=10)

        # If using hybrid search, run full hybrid pipeline
        if self.hybrid_search and state.get("iteration_count", 0) % 5 == 0:
            result = await self.hybrid_search.search(
                root=root,
                action_generator=lambda s: self._generate_actions_from_state(s),
                state_transition=lambda s, a: self._transition_state(s, a),
                rollout_policy=rollout_policy,
                query=state["input"],
            )

            return {
                "search_result": {
                    "best_action": result.best_action,
                    "iterations": result.iterations,
                    "phases": [p.value for p in result.phases_completed],
                },
                "metadata": {"simulation": "hybrid_search"},
            }

        return {
            "metadata": {
                "simulation": "standard",
                "value": value,
            },
        }

    async def _prm_evaluate_node(self, state: EnhancedTreeState) -> dict:
        """Evaluate nodes using PRM."""
        root = state["root"]
        leaf = self._find_leaf(root)

        if not isinstance(leaf, ReasoningMCTSNode):
            return {
                "requires_deeper_thinking": False,
                "prm_scores": [],
            }

        # Get trajectory
        trajectory = leaf.get_trajectory()

        # Score with PRM
        prm_scores = []
        if self.prm:
            scores = await self.prm.score_trajectory(trajectory)
            prm_scores = [s.step_score for s in scores]

            # Set the last score on the leaf
            if scores:
                leaf.set_prm_score(scores[-1])

        # Determine if deeper thinking is needed
        avg_score = np.mean(prm_scores) if prm_scores else 0.5
        requires_deeper = (
            avg_score < 0.6  # Low confidence
            and leaf.visits < 3  # Not well explored
            and self.config.thinking_enabled
        )

        return {
            "prm_scores": prm_scores,
            "requires_deeper_thinking": requires_deeper,
            "confidence": avg_score,
            "metadata": {"prm_avg_score": avg_score},
        }

    async def _extended_thinking_refine_node(self, state: EnhancedTreeState) -> dict:
        """Deep analysis using extended thinking."""
        root = state["root"]
        leaf = self._find_leaf(root)

        if not self.thinking_mcts_evaluator:
            return {"metadata": {"refine": "no_evaluator"}}

        # Run thinking evaluation
        result = await self.thinking_mcts_evaluator.evaluate_node(
            state=leaf.state,
            trajectory_context=state["input"],
            depth=leaf.depth,
            visits=leaf.visits,
            ucb_score=leaf.value,
            query=state["input"],
        )

        # Update node with thinking result
        if isinstance(leaf, ReasoningMCTSNode):
            leaf.set_thinking_result(result)

        self.total_thinking_tokens += result.tokens_used

        return {
            "thinking_traces": [result.thinking_trace],
            "total_thinking_tokens": state.get("total_thinking_tokens", 0) + result.tokens_used,
            "confidence": result.score,
            "metadata": {
                "refine": "extended_thinking",
                "tokens_used": result.tokens_used,
                "mode": result.mode.value,
            },
        }

    async def _hierarchical_backprop_node(self, state: EnhancedTreeState) -> dict:
        """Hierarchical backpropagation with PRM signals."""
        root = state["root"]
        leaf = self._find_leaf(root)

        # Get value to backpropagate
        confidence = state.get("confidence", 0.5)

        # Use PRM-weighted backpropagation if available
        if self.prm_integration and isinstance(leaf, ReasoningMCTSNode):
            trajectory = leaf.get_trajectory()
            value = await self.prm_integration.compute_backprop_value(
                trajectory=trajectory,
                outcome_value=confidence,
            )
        else:
            value = confidence

        # Backpropagate
        self.mcts_engine.backpropagate(leaf, value)

        return {
            "iteration_count": state.get("iteration_count", 0) + 1,
            "metadata": {"backprop_value": value},
        }

    async def _synthesize_node(self, state: EnhancedTreeState) -> dict:
        """Synthesize final solution from search results."""
        root = state["root"]

        # Find best action
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            best_action = best_child.action
            confidence = best_child.value
        else:
            best_action = None
            confidence = 0.0

        # Generate final response
        synthesis_prompt = f"""Based on the MCTS search results, synthesize a final response.

Query: {state['input']}
Best Action: {best_action}
Confidence: {confidence:.2f}
Iterations: {state.get('iteration_count', 0)}

Provide a comprehensive final answer."""

        try:
            response = await self.model_adapter.generate(
                prompt=synthesis_prompt,
                temperature=0.3,
            )
            final_response = response.text
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            final_response = f"Recommended action: {best_action}"

        return {
            "best_solution": final_response,
            "confidence": confidence,
            "verified": state.get("verified", False),
            "metadata": {
                "end_time": time.time(),
                "total_iterations": state.get("iteration_count", 0),
                "total_thinking_tokens": self.total_thinking_tokens,
                "best_action": best_action,
            },
        }

    # ========================================================================
    # Conditional Edge Functions
    # ========================================================================

    def _requires_deeper_thinking(self, state: EnhancedTreeState) -> str:
        """Determine if deeper thinking is needed."""
        if state.get("requires_deeper_thinking", False):
            return "refine"
        return "backprop"

    def _should_continue(self, state: EnhancedTreeState) -> str:
        """Determine if search should continue."""
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", self.config.mcts_iterations)
        confidence = state.get("confidence", 0.0)

        # Stop conditions
        if iteration >= max_iterations:
            return "synthesize"
        if confidence > 0.95:  # High confidence early termination
            return "synthesize"

        return "continue"

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _find_leaf(self, root: MCTSNode) -> MCTSNode:
        """Find the current leaf node for expansion."""
        node = root
        while node.children and not node.terminal:
            node = max(node.children, key=lambda c: c.visits if c.visits > 0 else float("inf"))
        return node

    def _generate_actions(self, node: MCTSNode) -> list[str]:
        """Generate available actions for a node."""
        depth = node.depth

        if depth == 0:
            return ["analyze", "decompose", "search", "synthesize", "verify"]
        elif depth < self.config.max_tree_depth:
            return ["continue", "refine", "alternative", "conclude"]
        else:
            return []  # Terminal

    def _generate_actions_from_state(self, state: MCTSState) -> list[str]:
        """Generate actions from state for hybrid search."""
        depth = len(state.state_id.split("_")) - 1
        if depth < self.config.max_tree_depth:
            return ["action_A", "action_B", "action_C", "action_D"]
        return []

    def _transition_state(self, state: MCTSState, action: str) -> MCTSState:
        """Transition state for hybrid search."""
        return MCTSState(
            state_id=f"{state.state_id}_{action}",
            features={**state.features, "last_action": action},
        )


# ============================================================================
# Convenience Function
# ============================================================================


def create_reasoning_graph(
    model_adapter: Any,
    logger: Any,
    config: ReasoningGraphConfig | None = None,
) -> tuple[StateGraph, ReasoningGraphBuilder]:
    """
    Create a reasoning-enhanced MCTS graph.

    Args:
        model_adapter: LLM adapter
        logger: Logger instance
        config: Optional configuration

    Returns:
        Tuple of (compiled graph, graph builder)
    """
    builder = ReasoningGraphBuilder(
        model_adapter=model_adapter,
        logger=logger,
        config=config,
    )

    graph = builder.build_graph()

    return graph, builder


async def run_reasoning_search(
    query: str,
    model_adapter: Any,
    logger: Any,
    config: ReasoningGraphConfig | None = None,
    max_iterations: int = 50,
) -> dict[str, Any]:
    """
    Run a reasoning-enhanced MCTS search.

    Args:
        query: Search query
        model_adapter: LLM adapter
        logger: Logger instance
        config: Optional configuration
        max_iterations: Maximum iterations

    Returns:
        Search results dictionary
    """
    graph, builder = create_reasoning_graph(model_adapter, logger, config)

    compiled = graph.compile()

    initial_state: EnhancedTreeState = {
        "input": query,
        "thinking_budget": config.default_thinking_tokens if config else 8192,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "thinking_traces": [],
        "trajectory_history": [],
    }

    result = await compiled.ainvoke(initial_state)

    return {
        "solution": result.get("best_solution", ""),
        "confidence": result.get("confidence", 0.0),
        "verified": result.get("verified", False),
        "iterations": result.get("iteration_count", 0),
        "thinking_tokens": result.get("total_thinking_tokens", 0),
        "metadata": result.get("metadata", {}),
    }
