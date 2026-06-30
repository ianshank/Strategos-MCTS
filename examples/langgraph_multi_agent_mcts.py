"""
LangGraph Multi-Agent Framework with MCTS Integration.

Implements a state machine architecture combining:
- Enhanced HRM (Hierarchical Reasoning) agent
- Enhanced TRM (Recursive Refinement) agent
- Monte Carlo Tree Search (MCTS) for tactical simulation
- RAG (Retrieval Augmented Generation) with vector stores
- LangGraph for explicit state management and routing

Based on 2025 research in multi-agent systems, MCTS, and LangGraph architecture.
"""

from __future__ import annotations

import asyncio
import operator
import random
import uuid
from typing import Annotated, NotRequired, TypedDict

# Optional third-party deps. Guarded so the module (and its public
# ``LangGraphMultiAgentFramework`` class) can be imported without the heavy
# ``langchain``/``langgraph`` extras installed; they are only required when a
# framework instance is actually constructed and run.
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - exercised only without the optional dep
    OpenAIEmbeddings = None  # type: ignore[assignment,misc]

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - exercised only without the optional dep
    MemorySaver = None  # type: ignore[assignment,misc]
    StateGraph = None  # type: ignore[assignment,misc]
    END = None  # type: ignore[assignment]

# Import core MCTS components
from src.framework.mcts.core import MCTSNode, MCTSState

# ---------------------------------------------------------------------------
# Default tunables for the reference LLM-orchestration agents below. These are
# intentionally named constants (no magic numbers inline) so callers and tests
# can reason about / override them via the ``**kwargs`` accepted by each agent.
# ---------------------------------------------------------------------------
DEFAULT_DECOMPOSITION_QUALITY_SCORE = 0.7  # HRM hierarchical-decomposition confidence
DEFAULT_FINAL_QUALITY_SCORE = 0.7  # TRM post-refinement confidence
DEFAULT_HRM_TEMPERATURE = 0.3  # Lower temp: structured, deterministic decomposition
DEFAULT_TRM_TEMPERATURE = 0.5  # Moderate temp: iterative refinement
DEFAULT_SYNTHESIS_TEMPERATURE = 0.5  # Final-answer synthesis temperature
DEFAULT_AGENT_CONFIDENCE = 0.5  # Neutral confidence when an agent result is absent
MCTS_SIM_BASE_VALUE_RANGE = (0.3, 0.7)  # Random base-value range for the reference MCTS rollout
MCTS_AGENT_CONFIDENCE_BIAS = 0.15  # Weight applied to agent confidence in the rollout value


class _LLMOrchestrationAgent:
    """Base class for the reference LLM-backed orchestration agents.

    These agents realise their roles (hierarchical reasoning / iterative
    refinement) purely through an injected ``model_adapter`` and deliberately do
    NOT use the neural ``nn.Module`` agents in ``src.agents``. The base captures
    the shared construction + prompt/generate plumbing; subclasses supply a
    role-specific system preamble and metadata.
    """

    #: Subclasses override these.
    _ROLE_PREAMBLE: str = ""
    _TEMPERATURE: float = DEFAULT_HRM_TEMPERATURE

    def __init__(self, model_adapter, logger, **kwargs) -> None:
        """Store injected dependencies.

        Args:
            model_adapter: Async LLM adapter exposing ``generate(...)`` that
                returns an object with a ``.text`` (str) attribute.
            logger: Logger-like object (``.info``/``.error``).
            **kwargs: Arbitrary extra configuration (e.g. the framework passes
                ``**(hrm_config or {})``). Unknown keys are accepted and
                ignored so config drift never breaks construction. Recognised:
                ``temperature`` (float) to override the role default.
        """
        self.model_adapter = model_adapter
        self.logger = logger
        self.temperature = float(kwargs.get("temperature", self._TEMPERATURE))
        # Retain any extra config for introspection without failing on it.
        self.config = dict(kwargs)

    def _build_prompt(self, query: str, rag_context: str | None) -> str:
        """Construct the role-specific prompt from query (+ optional context)."""
        parts = [self._ROLE_PREAMBLE.strip(), f"Query:\n{query}"]
        if rag_context:
            parts.append(f"Retrieved context:\n{rag_context}")
        parts.append("Response:")
        return "\n\n".join(p for p in parts if p)

    def _metadata(self) -> dict:
        """Role-specific metadata. Overridden by subclasses."""
        raise NotImplementedError

    async def process(self, query: str, rag_context: str | None = None) -> dict:
        """Run the agent: build prompt, call the LLM, return response+metadata.

        Returns:
            ``{"response": <str>, "metadata": <dict>}``. On LLM failure the
            response degrades to an empty string while metadata is preserved,
            so downstream nodes can still fall back gracefully.
        """
        prompt = self._build_prompt(query, rag_context)
        try:
            response = await self.model_adapter.generate(
                prompt=prompt,
                temperature=self.temperature,
            )
            text = response.text
        except Exception as exc:  # pragma: no cover - exercised via failing-LLM tests
            self.logger.error(f"{type(self).__name__} generation failed: {exc}")
            text = ""
        return {"response": text, "metadata": self._metadata()}


class HRMAgent(_LLMOrchestrationAgent):
    """Hierarchical Reasoning agent (LLM-orchestration reference).

    Decomposes the query into structured sub-goals via the model adapter.
    """

    _ROLE_PREAMBLE = (
        "You are a hierarchical reasoning module. Decompose the query into a "
        "structured hierarchy of sub-goals, then outline how solving them "
        "answers the query."
    )
    _TEMPERATURE = DEFAULT_HRM_TEMPERATURE

    def _metadata(self) -> dict:
        return {
            "agent": "hrm",
            "decomposition_quality_score": float(
                self.config.get("decomposition_quality_score", DEFAULT_DECOMPOSITION_QUALITY_SCORE)
            ),
        }


class TRMAgent(_LLMOrchestrationAgent):
    """Recursive/iterative Refinement agent (LLM-orchestration reference).

    Iteratively refines a candidate answer via the model adapter.
    """

    _ROLE_PREAMBLE = (
        "You are an iterative refinement module. Produce an answer to the "
        "query, then critique and refine it for correctness and completeness."
    )
    _TEMPERATURE = DEFAULT_TRM_TEMPERATURE

    def _metadata(self) -> dict:
        return {
            "agent": "trm",
            "final_quality_score": float(self.config.get("final_quality_score", DEFAULT_FINAL_QUALITY_SCORE)),
        }


class AgentState(TypedDict):
    """Shared state for LangGraph agent framework."""

    # Input
    query: str
    use_mcts: bool
    use_rag: bool

    # RAG context
    rag_context: NotRequired[str]
    retrieved_docs: NotRequired[list[dict]]

    # Agent results
    hrm_results: NotRequired[dict]
    trm_results: NotRequired[dict]
    agent_outputs: Annotated[list[dict], operator.add]

    # MCTS simulation
    mcts_root: NotRequired[MCTSNode]
    mcts_iterations: NotRequired[int]
    mcts_best_action: NotRequired[str]
    mcts_stats: NotRequired[dict]

    # Evaluation
    confidence_scores: NotRequired[dict[str, float]]
    consensus_reached: NotRequired[bool]
    consensus_score: NotRequired[float]

    # Control flow
    iteration: int
    max_iterations: int

    # Output
    final_response: NotRequired[str]
    metadata: NotRequired[dict]


class LangGraphMultiAgentFramework:
    """
    LangGraph-based multi-agent framework with MCTS integration.

    Features:
    - State machine architecture with explicit routing
    - Parallel agent execution (HRM + TRM)
    - MCTS for tactical simulation and planning
    - RAG integration with vector stores
    - Memory/checkpointing for conversation state
    """

    def __init__(
        self,
        model_adapter,
        logger,
        # RAG configuration
        vector_store=None,
        embedding_model=None,
        top_k_retrieval: int = 5,
        # Agent configuration
        hrm_config: dict | None = None,
        trm_config: dict | None = None,
        # MCTS configuration
        mcts_iterations: int = 100,
        mcts_exploration_weight: float = 1.414,
        # Framework configuration
        max_iterations: int = 3,
        consensus_threshold: float = 0.75,
    ):
        """Initialize LangGraph multi-agent framework.

        Raises:
            ImportError: if the optional ``langchain``/``langgraph`` extras are
                not installed. The module imports without them (so the class can
                be referenced), but constructing a framework requires them.
        """
        _missing = [
            name
            for name, sym in (
                ("langchain-openai", OpenAIEmbeddings),
                ("langgraph", StateGraph),
                ("langgraph", MemorySaver),
            )
            if sym is None
        ]
        if _missing:
            raise ImportError(
                "LangGraphMultiAgentFramework requires optional extras that are not installed: "
                f"{', '.join(sorted(set(_missing)))}. Install them, e.g. `pip install langchain-openai langgraph`."
            )

        self.model_adapter = model_adapter
        self.logger = logger
        self.top_k_retrieval = top_k_retrieval
        self.mcts_iterations = mcts_iterations
        self.mcts_exploration_weight = mcts_exploration_weight
        self.max_iterations = max_iterations
        self.consensus_threshold = consensus_threshold

        # Initialize RAG components
        self.vector_store = vector_store
        self.embeddings = embedding_model or OpenAIEmbeddings()

        # Initialize enhanced agents
        self.hrm_agent = HRMAgent(
            model_adapter=model_adapter,
            logger=logger,
            **(hrm_config or {}),
        )

        self.trm_agent = TRMAgent(
            model_adapter=model_adapter,
            logger=logger,
            **(trm_config or {}),
        )

        # Build LangGraph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)

        self.logger.info("LangGraph multi-agent framework initialized")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""

        # Define graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("entry", self.entry_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("route_decision", self.route_decision_node)
        workflow.add_node("hrm_agent", self.hrm_agent_node)
        workflow.add_node("trm_agent", self.trm_agent_node)
        workflow.add_node("mcts_simulator", self.mcts_simulator_node)
        workflow.add_node("aggregate_results", self.aggregate_results_node)
        workflow.add_node("evaluate_consensus", self.evaluate_consensus_node)
        workflow.add_node("synthesize", self.synthesize_node)

        # Define edges
        workflow.set_entry_point("entry")
        workflow.add_edge("entry", "retrieve_context")
        workflow.add_edge("retrieve_context", "route_decision")

        # Conditional routing from route_decision
        workflow.add_conditional_edges(
            "route_decision",
            self.route_to_agents,
            {
                "hrm": "hrm_agent",
                "trm": "trm_agent",
                "mcts": "mcts_simulator",
                "aggregate": "aggregate_results",
            },
        )

        # Agent nodes flow to aggregation
        workflow.add_edge("hrm_agent", "aggregate_results")
        workflow.add_edge("trm_agent", "aggregate_results")
        workflow.add_edge("mcts_simulator", "aggregate_results")

        # Aggregation to evaluation
        workflow.add_edge("aggregate_results", "evaluate_consensus")

        # Conditional: consensus or iterate
        workflow.add_conditional_edges(
            "evaluate_consensus",
            self.check_consensus,
            {
                "synthesize": "synthesize",
                "iterate": "route_decision",
            },
        )

        # Synthesis to end
        workflow.add_edge("synthesize", END)

        return workflow

    def entry_node(self, state: AgentState) -> dict:
        """Initialize state and parse query."""

        self.logger.info(f"Entry node: {state['query'][:100]}")

        return {
            "iteration": 0,
            "agent_outputs": [],
        }

    def retrieve_context_node(self, state: AgentState) -> dict:
        """Retrieve context from vector store using RAG."""

        if not state.get("use_rag", True) or not self.vector_store:
            return {"rag_context": ""}

        query = state["query"]

        # Retrieve documents
        docs = self.vector_store.similarity_search(query, k=self.top_k_retrieval)

        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])

        self.logger.info(f"Retrieved {len(docs)} documents")

        return {
            "rag_context": context,
            "retrieved_docs": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
        }

    def route_decision_node(self, _state: AgentState) -> dict:
        """Analyze query and prepare routing."""

        # This node just marks that routing decision will be made
        # Actual routing happens in route_to_agents()
        return {}

    def route_to_agents(self, state: AgentState) -> str:
        """Route to appropriate agent based on state."""

        iteration = state.get("iteration", 0)

        # First iteration: run HRM and TRM in sequence
        if iteration == 0:
            if "hrm_results" not in state:
                return "hrm"
            elif "trm_results" not in state:
                return "trm"

        # Second iteration: run MCTS if enabled
        if state.get("use_mcts", False) and "mcts_stats" not in state:
            return "mcts"

        # All agents complete, aggregate
        return "aggregate"

    async def hrm_agent_node(self, state: AgentState) -> dict:
        """Execute HRM agent for hierarchical decomposition."""

        self.logger.info("Executing HRM agent")

        result = await self.hrm_agent.process(
            query=state["query"],
            rag_context=state.get("rag_context"),
        )

        return {
            "hrm_results": {
                "response": result["response"],
                "metadata": result["metadata"],
            },
            "agent_outputs": [
                {
                    "agent": "hrm",
                    "response": result["response"],
                    "confidence": result["metadata"].get(
                        "decomposition_quality_score", DEFAULT_DECOMPOSITION_QUALITY_SCORE
                    ),
                }
            ],
        }

    async def trm_agent_node(self, state: AgentState) -> dict:
        """Execute TRM agent for iterative refinement."""

        self.logger.info("Executing TRM agent")

        result = await self.trm_agent.process(
            query=state["query"],
            rag_context=state.get("rag_context"),
        )

        return {
            "trm_results": {
                "response": result["response"],
                "metadata": result["metadata"],
            },
            "agent_outputs": [
                {
                    "agent": "trm",
                    "response": result["response"],
                    "confidence": result["metadata"].get("final_quality_score", DEFAULT_FINAL_QUALITY_SCORE),
                }
            ],
        }

    async def mcts_simulator_node(self, state: AgentState) -> dict:
        """Execute MCTS simulation for tactical planning."""

        self.logger.info("Executing MCTS simulation")

        # Initialize MCTS tree
        root = MCTSNode(state=MCTSState("root_state"))

        # Run MCTS iterations
        for _i in range(self.mcts_iterations):
            # Selection: traverse to leaf
            node = self._mcts_select(root)

            # Expansion: add children
            if not node.terminal and node.visits > 0:
                node = self._mcts_expand(node, state)

            # Simulation: evaluate leaf
            value = await self._mcts_simulate(node, state)

            # Backpropagation: update ancestors
            self._mcts_backpropagate(node, value)

        # Select best action (using max visits - most robust selection)
        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        best_action = best_child.action if best_child else "no_action"

        # Compute statistics
        stats = {
            "iterations": self.mcts_iterations,
            "root_visits": root.visits,
            "root_value": root.value,
            "best_action": best_action,
            "best_action_visits": best_child.visits if best_child else 0,
            "best_action_value": best_child.value if best_child else 0.0,
        }

        self.logger.info(f"MCTS complete: best_action={best_action}")

        return {
            "mcts_root": root,
            "mcts_best_action": best_action,
            "mcts_stats": stats,
            "agent_outputs": [
                {
                    "agent": "mcts",
                    "response": f"Simulated {self.mcts_iterations} scenarios. Recommended action: {best_action}",
                    "confidence": min(
                        best_child.visits / self.mcts_iterations if best_child else DEFAULT_AGENT_CONFIDENCE, 1.0
                    ),
                }
            ],
        }

    def _mcts_select(self, node: MCTSNode) -> MCTSNode:
        """MCTS selection phase: traverse to leaf using UCB1."""
        while node.children and not node.terminal:
            node = node.select_child(self.mcts_exploration_weight)
        return node

    def _mcts_expand(self, node: MCTSNode, state: AgentState) -> MCTSNode:
        """MCTS expansion phase: add child nodes."""

        # Generate possible actions (simplified)
        actions = self._generate_actions(node, state)

        if not actions:
            node.terminal = True
            return node

        # Add one child (expand one action at a time)
        action = random.choice(actions)
        child_id = f"{node.state.state_id}_{action}"
        child = node.add_child(action=action, child_state=MCTSState(child_id))

        return child

    def _generate_actions(self, node: MCTSNode, _state: AgentState) -> list[str]:
        """Generate possible actions from current node."""

        # Simplified action generation
        # In production, this would use domain knowledge or LLM
        if node.state.state_id == "root_state":
            return ["action_A", "action_B", "action_C"]
        elif len(node.state.state_id.split("_")) < 3:  # Depth limit
            return ["continue_A", "continue_B", "fallback"]
        else:
            return []  # Terminal

    async def _mcts_simulate(self, _node: MCTSNode, state: AgentState) -> float:
        """MCTS simulation phase: evaluate node value."""

        # Use HRM/TRM results to evaluate
        # In production, this could run a lightweight simulation

        # Simplified: random evaluation with bias from agents
        base_value = random.uniform(*MCTS_SIM_BASE_VALUE_RANGE)

        # Bias based on agent confidence
        if state.get("hrm_results"):
            hrm_confidence = state["hrm_results"]["metadata"].get(
                "decomposition_quality_score", DEFAULT_AGENT_CONFIDENCE
            )
            base_value += hrm_confidence * MCTS_AGENT_CONFIDENCE_BIAS

        if state.get("trm_results"):
            trm_confidence = state["trm_results"]["metadata"].get("final_quality_score", DEFAULT_AGENT_CONFIDENCE)
            base_value += trm_confidence * MCTS_AGENT_CONFIDENCE_BIAS

        return min(base_value, 1.0)

    def _mcts_backpropagate(self, node: MCTSNode, value: float):
        """MCTS backpropagation phase: update ancestors."""
        while node:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    def aggregate_results_node(self, state: AgentState) -> dict:
        """Aggregate results from all agents."""

        self.logger.info("Aggregating agent results")

        agent_outputs = state.get("agent_outputs", [])

        # Compute confidence scores
        confidence_scores = {output["agent"]: output["confidence"] for output in agent_outputs}

        return {
            "confidence_scores": confidence_scores,
        }

    def evaluate_consensus_node(self, state: AgentState) -> dict:
        """Evaluate consensus among agents."""

        agent_outputs = state.get("agent_outputs", [])

        if len(agent_outputs) < 2:
            return {
                "consensus_reached": True,
                "consensus_score": 1.0,
            }

        # Simplified consensus: average confidence
        avg_confidence = sum(o["confidence"] for o in agent_outputs) / len(agent_outputs)

        consensus_reached = avg_confidence >= self.consensus_threshold

        self.logger.info(f"Consensus: {consensus_reached} (score={avg_confidence:.2f})")

        return {
            "consensus_reached": consensus_reached,
            "consensus_score": avg_confidence,
        }

    def check_consensus(self, state: AgentState) -> str:
        """Check if consensus reached or need more iterations."""

        if state.get("consensus_reached", False):
            return "synthesize"

        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "synthesize"  # Max iterations, synthesize anyway

        return "iterate"

    async def synthesize_node(self, state: AgentState) -> dict:
        """Synthesize final response from agent outputs."""

        self.logger.info("Synthesizing final response")

        agent_outputs = state.get("agent_outputs", [])

        # Weighted synthesis based on confidence
        synthesis_prompt = f"""Query: {state["query"]}

Agent Outputs:
"""

        for output in agent_outputs:
            synthesis_prompt += f"""
{output["agent"].upper()} (confidence={output["confidence"]:.2f}):
{output["response"]}

"""

        synthesis_prompt += """
Synthesize these outputs into a comprehensive final response.
Prioritize higher-confidence outputs. Integrate insights from all agents.

Final Response:"""

        try:
            response = await self.model_adapter.generate(
                prompt=synthesis_prompt,
                temperature=DEFAULT_SYNTHESIS_TEMPERATURE,
            )
            final_response = response.text
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            # Fallback: highest-confidence response, or empty when no agent ran
            # (guard against max() on an empty sequence).
            if agent_outputs:
                final_response = max(agent_outputs, key=lambda o: o["confidence"])["response"]
            else:
                final_response = ""

        # Build metadata
        metadata = {
            "agents_used": [o["agent"] for o in agent_outputs],
            "confidence_scores": state.get("confidence_scores", {}),
            "consensus_score": state.get("consensus_score", 0.0),
            "iterations": state.get("iteration", 0),
        }

        if state.get("mcts_stats"):
            metadata["mcts_stats"] = state["mcts_stats"]

        return {
            "final_response": final_response,
            "metadata": metadata,
        }

    async def process(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict | None = None,
    ) -> dict:
        """Process query through LangGraph.

        Each call is independent: unless an explicit ``config`` (with a
        ``thread_id``) is supplied, a fresh thread id is generated so the
        checkpointer does not replay accumulated state from prior calls
        (``agent_outputs`` uses an additive reducer, so a shared thread would
        otherwise accumulate across calls and never reach a stop condition).
        """

        # Initial state
        initial_state = {
            "query": query,
            "use_rag": use_rag,
            "use_mcts": use_mcts,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "agent_outputs": [],
        }

        # Run graph. Use a unique thread per invocation by default so repeated
        # calls are isolated from one another.
        config = config or {"configurable": {"thread_id": uuid.uuid4().hex}}

        result = await self.app.ainvoke(initial_state, config=config)

        return {
            "response": result.get("final_response", ""),
            "metadata": result.get("metadata", {}),
            "state": result,
        }


# Example usage
async def example_usage():
    """Example usage of LangGraph multi-agent framework."""

    from apps.agents.utils.logging_config import LoggerAdapter
    from apps.agents.utils.model_adapters import UnifiedModelAdapter

    # Initialize framework
    framework = LangGraphMultiAgentFramework(
        model_adapter=UnifiedModelAdapter(),
        logger=LoggerAdapter(),
        mcts_iterations=100,
        max_iterations=2,
    )

    # Process query
    result = await framework.process(
        query="Recommend defensive positions for night attack scenario",
        use_rag=True,
        use_mcts=True,
    )

    print(f"Response: {result['response']}")
    print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    asyncio.run(example_usage())
