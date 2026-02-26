"""
LangGraph Integration Module - Extract graph building with new MCTS core integration.

Provides:
- Graph building extracted from LangGraphMultiAgentFramework
- Integration with new deterministic MCTS core
- Backward compatibility with original process() signature
- Support for parallel HRM/TRM execution
"""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from typing import Annotated, Any, NotRequired, TypedDict

# LangGraph imports (these would be installed dependencies)
try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
except ImportError:
    # Stubs for development without LangGraph installed
    StateGraph = None
    END = "END"
    MemorySaver = None

# Import new MCTS modules
from .mcts.config import ConfigPreset, MCTSConfig, create_preset_config
from .mcts.core import MCTSEngine, MCTSNode, MCTSState
from .mcts.experiments import ExperimentTracker
from .mcts.policies import (
    HybridRolloutPolicy,
)

# Neural Meta-Controller imports (optional)
try:
    from src.agents.meta_controller.base import (
        AbstractMetaController,
        MetaControllerFeatures,
    )
    from src.agents.meta_controller.bert_controller import BERTMetaController
    from src.agents.meta_controller.config_loader import (
        MetaControllerConfig,
        MetaControllerConfigLoader,
    )
    from src.agents.meta_controller.rnn_controller import RNNMetaController

    _META_CONTROLLER_AVAILABLE = True
except ImportError:
    _META_CONTROLLER_AVAILABLE = False
    AbstractMetaController = None  # type: ignore
    MetaControllerFeatures = None  # type: ignore
    RNNMetaController = None  # type: ignore
    BERTMetaController = None  # type: ignore
    MetaControllerConfig = None  # type: ignore
    MetaControllerConfigLoader = None  # type: ignore

# Neuro-Symbolic imports (optional)
try:
    from src.neuro_symbolic import (
        ConstraintSystem,
        NeuroSymbolicConfig,
        NeuroSymbolicMCTSConfig,
        NeuroSymbolicMCTSIntegration,
        SymbolicAgentGraphExtension,
        SymbolicAgentNodeConfig,
        SymbolicReasoningAgent,
    )
    from src.neuro_symbolic.config import ConstraintConfig

    _NEURO_SYMBOLIC_AVAILABLE = True
except ImportError:
    _NEURO_SYMBOLIC_AVAILABLE = False
    NeuroSymbolicConfig = None  # type: ignore
    SymbolicReasoningAgent = None  # type: ignore
    SymbolicAgentGraphExtension = None  # type: ignore
    SymbolicAgentNodeConfig = None  # type: ignore
    NeuroSymbolicMCTSIntegration = None  # type: ignore
    NeuroSymbolicMCTSConfig = None  # type: ignore
    ConstraintSystem = None  # type: ignore
    ConstraintConfig = None  # type: ignore

logger = logging.getLogger(__name__)


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
    adk_results: NotRequired[dict[str, Any]]
    agent_outputs: Annotated[list[dict], operator.add]

    # MCTS simulation (updated for new core)
    mcts_root: NotRequired[Any]  # MCTSNode
    mcts_iterations: NotRequired[int]
    mcts_best_action: NotRequired[str]
    mcts_stats: NotRequired[dict]
    mcts_config: NotRequired[dict]

    # Evaluation
    confidence_scores: NotRequired[dict[str, float]]
    consensus_reached: NotRequired[bool]
    consensus_score: NotRequired[float]

    # Control flow
    iteration: int
    max_iterations: int

    # Neural Meta-Controller (optional)
    routing_history: NotRequired[list[dict]]
    meta_controller_predictions: NotRequired[list[dict]]
    last_routed_agent: NotRequired[str]

    # Neuro-Symbolic Agent (optional)
    symbolic_results: NotRequired[dict]
    symbolic_proof_tree: NotRequired[dict]

    # Output
    final_response: NotRequired[str]
    metadata: NotRequired[dict]


class GraphBuilder:
    """
    Builds and configures the LangGraph state machine for multi-agent orchestration.

    Extracts graph building logic from LangGraphMultiAgentFramework for modularity.
    """

    def __init__(
        self,
        hrm_agent,
        trm_agent,
        model_adapter,
        logger,
        vector_store=None,
        mcts_config: MCTSConfig | None = None,
        top_k_retrieval: int = 5,
        max_iterations: int = 3,
        consensus_threshold: float = 0.75,
        enable_parallel_agents: bool = True,
        meta_controller_config: Any | None = None,
        adk_agents: dict[str, Any] | None = None,
        neuro_symbolic_config: Any | None = None,
    ):
        """
        Initialize graph builder.

        Args:
            hrm_agent: HRM agent instance
            trm_agent: TRM agent instance
            model_adapter: Model adapter for LLM calls
            logger: Logger instance
            vector_store: Optional vector store for RAG
            mcts_config: MCTS configuration (uses balanced preset if None)
            top_k_retrieval: Number of documents for RAG
            max_iterations: Maximum agent iterations
            consensus_threshold: Threshold for consensus
            enable_parallel_agents: Run HRM/TRM in parallel
            meta_controller_config: Optional neural meta-controller configuration
            adk_agents: Dictionary of ADK agent instances
            neuro_symbolic_config: Optional neuro-symbolic configuration
        """
        self.hrm_agent = hrm_agent
        self.trm_agent = trm_agent
        self.model_adapter = model_adapter
        self.logger = logger
        self.vector_store = vector_store
        self.top_k_retrieval = top_k_retrieval
        self.max_iterations = max_iterations
        self.consensus_threshold = consensus_threshold
        self.enable_parallel_agents = enable_parallel_agents
        self.adk_agents = adk_agents or {}

        # MCTS configuration
        self.mcts_config = mcts_config or create_preset_config(ConfigPreset.BALANCED)

        # MCTS engine with deterministic behavior
        self.mcts_engine = MCTSEngine(
            seed=self.mcts_config.seed,
            exploration_weight=self.mcts_config.exploration_weight,
            progressive_widening_k=self.mcts_config.progressive_widening_k,
            progressive_widening_alpha=self.mcts_config.progressive_widening_alpha,
            max_parallel_rollouts=self.mcts_config.max_parallel_rollouts,
            cache_size_limit=self.mcts_config.cache_size_limit,
        )

        # Experiment tracking
        self.experiment_tracker = ExperimentTracker(name="langgraph_mcts")

        # Neural Meta-Controller (optional)
        self.meta_controller: Any | None = None
        self.meta_controller_config = meta_controller_config
        self.use_neural_routing = False

        if meta_controller_config is not None:
            self._init_meta_controller(meta_controller_config)

        # Neuro-Symbolic Agent (optional)
        self.symbolic_agent: Any | None = None
        self.symbolic_extension: Any | None = None
        self.neuro_symbolic_mcts: Any | None = None
        self.use_symbolic_reasoning = False

        if neuro_symbolic_config is not None:
            self._init_neuro_symbolic(neuro_symbolic_config)

        logger.debug(
            "GraphBuilder initialized: max_iterations=%d, consensus_threshold=%.2f, parallel_agents=%s, mcts_seed=%d",
            self.max_iterations,
            self.consensus_threshold,
            self.enable_parallel_agents,
            self.mcts_config.seed,
        )

    def build_graph(self) -> StateGraph:
        """
        Build LangGraph state machine.

        Returns:
            Configured StateGraph
        """
        if StateGraph is None:
            raise ImportError("LangGraph not installed. Install with: pip install langgraph")

        logger.info("Building LangGraph state machine")
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("entry", self._entry_node)
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("route_decision", self._route_decision_node)
        workflow.add_node("parallel_agents", self._parallel_agents_node)
        workflow.add_node("hrm_agent", self._hrm_agent_node)
        workflow.add_node("trm_agent", self._trm_agent_node)
        workflow.add_node("mcts_simulator", self._mcts_simulator_node)

        # Add ADK agent nodes
        for name, agent in self.adk_agents.items():
            node_name = f"adk_{name}"
            workflow.add_node(node_name, self._create_adk_node_handler(name, agent))

        # Add symbolic reasoning agent node if enabled
        if self.use_symbolic_reasoning and self.symbolic_extension:
            workflow.add_node("symbolic_agent", self._symbolic_agent_node)

        workflow.add_node("aggregate_results", self._aggregate_results_node)
        workflow.add_node("evaluate_consensus", self._evaluate_consensus_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("entry")
        workflow.add_edge("entry", "retrieve_context")
        workflow.add_edge("retrieve_context", "route_decision")

        # Conditional routing
        routing_map = {
            "parallel": "parallel_agents",
            "hrm": "hrm_agent",
            "trm": "trm_agent",
            "mcts": "mcts_simulator",
            "aggregate": "aggregate_results",
        }

        # Add symbolic agent routing if enabled
        if self.use_symbolic_reasoning:
            routing_map["symbolic"] = "symbolic_agent"

        # Add ADK routing entries
        for name in self.adk_agents:
            routing_map[f"adk_{name}"] = f"adk_{name}"

        workflow.add_conditional_edges(
            "route_decision",
            self._route_to_agents,
            routing_map,
        )

        # Parallel agents to aggregation
        workflow.add_edge("parallel_agents", "aggregate_results")

        # Sequential agent nodes
        workflow.add_edge("hrm_agent", "aggregate_results")
        workflow.add_edge("trm_agent", "aggregate_results")
        workflow.add_edge("mcts_simulator", "aggregate_results")

        # Symbolic agent to aggregation
        if self.use_symbolic_reasoning:
            workflow.add_edge("symbolic_agent", "aggregate_results")

        # ADK agents to aggregation
        for name in self.adk_agents:
            workflow.add_edge(f"adk_{name}", "aggregate_results")

        # Aggregation to evaluation
        workflow.add_edge("aggregate_results", "evaluate_consensus")

        # Conditional consensus check
        workflow.add_conditional_edges(
            "evaluate_consensus",
            self._check_consensus,
            {
                "synthesize": "synthesize",
                "iterate": "route_decision",
            },
        )

        # Synthesis to end
        workflow.add_edge("synthesize", END)

        return workflow

    def _entry_node(self, state: AgentState) -> dict:
        """Initialize state and parse query with validation."""
        query = state.get("query", "")

        # Validate query input
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty or whitespace only")

        # Log truncated query for privacy
        self.logger.info(f"Entry node: {query[:100]}{'...' if len(query) > 100 else ''}")

        return {
            "iteration": 0,
            "agent_outputs": [],
            "mcts_config": self.mcts_config.to_dict(),
        }

    def _retrieve_context_node(self, state: AgentState) -> dict:
        """Retrieve context from vector store using RAG with error handling."""
        if not state.get("use_rag", True) or not self.vector_store:
            return {"rag_context": "", "retrieved_docs": []}

        query = state.get("query", "")
        if not query:
            self.logger.warning("Empty query in retrieve_context_node")
            return {"rag_context": "", "retrieved_docs": []}

        try:
            # Retrieve documents with error handling
            docs = self.vector_store.similarity_search(query, k=self.top_k_retrieval)

            # Format context
            context = "\n\n".join([doc.page_content for doc in docs])

            self.logger.info(f"Retrieved {len(docs)} documents")

            return {
                "rag_context": context,
                "retrieved_docs": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
            }
        except Exception:
            self.logger.exception("RAG retrieval failed")
            # Graceful degradation - continue without RAG context
            return {"rag_context": "", "retrieved_docs": []}

    def _route_decision_node(self, _state: AgentState) -> dict:
        """Prepare routing decision."""
        return {}

    def _init_meta_controller(self, config: Any) -> None:
        """
        Initialize the neural meta-controller based on configuration.

        Args:
            config: MetaControllerConfig or dict with configuration
        """
        if not _META_CONTROLLER_AVAILABLE:
            self.logger.warning("Meta-controller modules not available. Falling back to rule-based routing.")
            return

        try:
            # Handle both config object and dict
            mc_config = MetaControllerConfigLoader.load_from_dict(config) if isinstance(config, dict) else config

            if not mc_config.enabled:
                self.logger.info("Neural meta-controller disabled in config")
                return

            # Initialize based on type
            if mc_config.type == "rnn":
                self.meta_controller = RNNMetaController(
                    name="GraphBuilder_RNN",
                    seed=mc_config.inference.seed,
                    hidden_dim=mc_config.rnn.hidden_dim,
                    num_layers=mc_config.rnn.num_layers,
                    dropout=mc_config.rnn.dropout,
                    device=mc_config.inference.device,
                )
                # Load trained model if path specified
                if mc_config.rnn.model_path:
                    self.meta_controller.load_model(mc_config.rnn.model_path)
                    self.logger.info(f"Loaded RNN model from {mc_config.rnn.model_path}")

            elif mc_config.type == "bert":
                self.meta_controller = BERTMetaController(
                    name="GraphBuilder_BERT",
                    seed=mc_config.inference.seed,
                    model_name=mc_config.bert.model_name,
                    lora_r=mc_config.bert.lora_r,
                    lora_alpha=mc_config.bert.lora_alpha,
                    lora_dropout=mc_config.bert.lora_dropout,
                    device=mc_config.inference.device,
                    use_lora=mc_config.bert.use_lora,
                )
                # Load trained model if path specified
                if mc_config.bert.model_path:
                    self.meta_controller.load_model(mc_config.bert.model_path)
                    self.logger.info(f"Loaded BERT model from {mc_config.bert.model_path}")
            else:
                raise ValueError(f"Unknown meta-controller type: {mc_config.type}")

            self.use_neural_routing = True
            self.logger.info(f"Initialized {mc_config.type.upper()} neural meta-controller")

        except Exception as e:
            self.logger.error(f"Failed to initialize meta-controller: {e}")
            if hasattr(config, "fallback_to_rule_based") and config.fallback_to_rule_based:
                self.logger.warning("Falling back to rule-based routing")
            else:
                raise

    def _init_neuro_symbolic(self, config: Any) -> None:
        """
        Initialize neuro-symbolic reasoning components.

        Args:
            config: NeuroSymbolicConfig or dict with configuration
        """
        if not _NEURO_SYMBOLIC_AVAILABLE:
            self.logger.warning("Neuro-symbolic modules not available. Skipping initialization.")
            return

        try:
            # Handle both config object and dict
            if isinstance(config, dict):
                ns_config = NeuroSymbolicConfig.from_dict(config)
            else:
                ns_config = config

            # Create symbolic reasoning agent
            self.symbolic_agent = SymbolicReasoningAgent(
                config=ns_config,
                neural_fallback=self._neural_fallback_for_symbolic,
                logger=self.logger,
            )

            # Create graph extension for routing
            self.symbolic_extension = SymbolicAgentGraphExtension(
                reasoning_agent=self.symbolic_agent,
                config=SymbolicAgentNodeConfig(),
                logger=self.logger,
            )

            # Create MCTS integration for constraint pruning
            mcts_ns_config = NeuroSymbolicMCTSConfig(
                neural_weight=ns_config.agent.neural_confidence_weight,
                symbolic_weight=ns_config.agent.symbolic_confidence_weight,
            )
            self.neuro_symbolic_mcts = NeuroSymbolicMCTSIntegration(
                config=mcts_ns_config,
                reasoning_agent=self.symbolic_agent,
                logger=self.logger,
            )

            self.use_symbolic_reasoning = True
            self.logger.info("Initialized neuro-symbolic reasoning components")

        except Exception as e:
            self.logger.error(f"Failed to initialize neuro-symbolic components: {e}")
            self.use_symbolic_reasoning = False

    async def _neural_fallback_for_symbolic(self, query: str, state: Any) -> str:
        """Neural fallback when symbolic reasoning fails."""
        try:
            response = await self.model_adapter.generate(
                prompt=f"Answer this question: {query}",
                temperature=0.5,
            )
            return str(response.text)
        except Exception as e:
            self.logger.error(f"Neural fallback failed: {e}")
            return f"Could not determine answer for: {query}"

    def _extract_meta_controller_features(self, state: AgentState) -> Any:
        """
        Extract features from AgentState for meta-controller prediction.

        Args:
            state: Current agent state

        Returns:
            MetaControllerFeatures instance
        """
        if not _META_CONTROLLER_AVAILABLE or MetaControllerFeatures is None:
            return None

        # Extract HRM confidence
        hrm_conf = 0.0
        if "hrm_results" in state:
            hrm_conf = state["hrm_results"].get("metadata", {}).get("decomposition_quality_score", 0.5)

        # Extract TRM confidence
        trm_conf = 0.0
        if "trm_results" in state:
            trm_conf = state["trm_results"].get("metadata", {}).get("final_quality_score", 0.5)

        # Extract MCTS value
        mcts_val = 0.0
        if "mcts_stats" in state:
            mcts_val = state["mcts_stats"].get("best_action_value", 0.5)

        # Consensus score
        consensus = state.get("consensus_score", 0.0)

        # Last agent used
        last_agent = state.get("last_routed_agent", "none")

        # Iteration
        iteration = state.get("iteration", 0)

        # Query length
        query_length = len(state.get("query", ""))

        # Has RAG context
        has_rag = bool(state.get("rag_context", ""))

        return MetaControllerFeatures(
            hrm_confidence=hrm_conf,
            trm_confidence=trm_conf,
            mcts_value=mcts_val,
            consensus_score=consensus,
            last_agent=last_agent,
            iteration=iteration,
            query_length=query_length,
            has_rag_context=has_rag,
        )

    def _neural_route_decision(self, state: AgentState) -> str:
        """
        Make routing decision using neural meta-controller.

        Args:
            state: Current agent state

        Returns:
            Route decision string ("parallel", "hrm", "trm", "mcts", "aggregate")
        """
        try:
            features = self._extract_meta_controller_features(state)
            if features is None:
                return self._rule_based_route_decision(state)

            assert self.meta_controller is not None, "Meta controller not initialized"
            prediction = self.meta_controller.predict(features)

            # Log prediction for debugging
            self.logger.debug(
                f"Neural routing: agent={prediction.agent}, "
                f"confidence={prediction.confidence:.3f}, "
                f"probs={prediction.probabilities}"
            )

            # Map agent prediction to route
            agent = prediction.agent

            if agent == "hrm":
                if "hrm_results" not in state:
                    return "hrm"
            elif agent == "trm":
                if "trm_results" not in state:
                    return "trm"
            elif agent == "mcts" and state.get("use_mcts", False) and "mcts_stats" not in state:
                return "mcts"

            # If predicted agent already ran or not applicable, use rule-based
            return self._rule_based_route_decision(state)

        except Exception as e:
            self.logger.error(f"Neural routing failed: {e}")
            # Fallback to rule-based routing
            return self._rule_based_route_decision(state)

    def _rule_based_route_decision(self, state: AgentState) -> str:
        """
        Make routing decision using rule-based logic.

        Args:
            state: Current agent state

        Returns:
            Route decision string
        """
        iteration = state.get("iteration", 0)

        # Check for symbolic reasoning triggers first
        if (
            self.use_symbolic_reasoning
            and self.symbolic_extension
            and "symbolic_results" not in state
            and self.symbolic_extension.should_route_to_symbolic(state.get("query", ""), state)
        ):
            return "symbolic"

        # First iteration: run HRM and TRM
        if iteration == 0:
            # Check for ADK triggers
            query_lower = state["query"].lower()

            # Simple keyword matching for demo purposes
            for name in self.adk_agents:
                # Check if we haven't run this ADK agent yet
                adk_results = state.get("adk_results", {})
                if name not in adk_results and (
                    (name == "deep_search" and ("research" in query_lower or "investigate" in query_lower))
                    or (name == "ml_engineering" and ("train" in query_lower or "model" in query_lower))
                    or (name == "data_science" and ("analyze" in query_lower or "data" in query_lower))
                ):
                    return f"adk_{name}"

            if self.enable_parallel_agents:
                if "hrm_results" not in state and "trm_results" not in state:
                    return "parallel"
            else:
                if "hrm_results" not in state:
                    return "hrm"
                elif "trm_results" not in state:
                    return "trm"

        # Run MCTS if enabled and not yet done
        if state.get("use_mcts", False) and "mcts_stats" not in state:
            return "mcts"

        return "aggregate"

    def _route_to_agents(self, state: AgentState) -> str:
        """Route to appropriate agent based on state."""
        # Use neural routing if enabled
        if self.use_neural_routing and self.meta_controller is not None:
            return self._neural_route_decision(state)

        # Fall back to rule-based routing
        return self._rule_based_route_decision(state)

    async def _parallel_agents_node(self, state: AgentState) -> dict:
        """Execute HRM and TRM agents in parallel."""
        self.logger.info("Executing HRM and TRM agents in parallel")

        # Run both agents concurrently
        hrm_task = asyncio.create_task(
            self.hrm_agent.process(
                query=state["query"],
                rag_context=state.get("rag_context"),
            )
        )

        trm_task = asyncio.create_task(
            self.trm_agent.process(
                query=state["query"],
                rag_context=state.get("rag_context"),
            )
        )

        # Await both results
        hrm_result, trm_result = await asyncio.gather(hrm_task, trm_task)

        # Combine outputs
        return {
            "hrm_results": {
                "response": hrm_result["response"],
                "metadata": hrm_result["metadata"],
            },
            "trm_results": {
                "response": trm_result["response"],
                "metadata": trm_result["metadata"],
            },
            "agent_outputs": [
                {
                    "agent": "hrm",
                    "response": hrm_result["response"],
                    "confidence": hrm_result["metadata"].get("decomposition_quality_score", 0.7),
                },
                {
                    "agent": "trm",
                    "response": trm_result["response"],
                    "confidence": trm_result["metadata"].get("final_quality_score", 0.7),
                },
            ],
        }

    async def _hrm_agent_node(self, state: AgentState) -> dict:
        """Execute HRM agent."""
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
                    "confidence": result["metadata"].get("decomposition_quality_score", 0.7),
                }
            ],
        }

    async def _trm_agent_node(self, state: AgentState) -> dict:
        """Execute TRM agent."""
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
                    "confidence": result["metadata"].get("final_quality_score", 0.7),
                }
            ],
        }

    async def _symbolic_agent_node(self, state: AgentState) -> dict:
        """Execute symbolic reasoning agent."""
        self.logger.info("Executing symbolic reasoning agent")

        if not self.symbolic_extension:
            return {
                "agent_outputs": [
                    {
                        "agent": "symbolic",
                        "response": "Symbolic reasoning not available",
                        "confidence": 0.0,
                    }
                ],
            }

        result = await self.symbolic_extension.handle_symbolic_node(state)

        # Store proof tree if available
        proof_tree = None
        if "symbolic_results" in result:
            metadata = result["symbolic_results"].get("metadata", {})
            proof_tree = metadata.get("proof_tree")

        return {
            "symbolic_results": result.get("symbolic_results", {}),
            "symbolic_proof_tree": proof_tree,
            "agent_outputs": result.get("agent_outputs", []),
        }

    async def _mcts_simulator_node(self, state: AgentState) -> dict:
        """Execute MCTS simulation using new deterministic engine."""
        self.logger.info("Executing MCTS simulation with deterministic engine")

        start_time = time.perf_counter()

        # Reset engine for this simulation
        self.mcts_engine.clear_cache()

        # Create root state
        root_state = MCTSState(
            state_id="root",
            features={
                "query": state["query"][:100],  # Truncate for hashing
                "has_hrm": "hrm_results" in state,
                "has_trm": "trm_results" in state,
            },
        )

        root = MCTSNode(
            state=root_state,
            rng=self.mcts_engine.rng,
        )

        # Define action generator based on domain
        def action_generator(mcts_state: MCTSState) -> list[str]:
            """Generate available actions for state."""
            depth = len(mcts_state.state_id.split("_")) - 1

            if depth == 0:
                # Root level actions
                return ["action_A", "action_B", "action_C", "action_D"]
            elif depth < self.mcts_config.max_tree_depth:
                # Subsequent actions
                return ["continue", "refine", "fallback", "escalate"]
            else:
                return []  # Terminal

        # Define state transition
        def state_transition(mcts_state: MCTSState, action: str) -> MCTSState:
            """Compute next state from action."""
            new_id = f"{mcts_state.state_id}_{action}"
            new_features = mcts_state.features.copy()
            new_features["last_action"] = action
            new_features["depth"] = len(new_id.split("_")) - 1
            return MCTSState(state_id=new_id, features=new_features)

        # Create rollout policy using agent results
        def heuristic_fn(mcts_state: MCTSState) -> float:
            """Evaluate state using agent confidence."""
            base = 0.5

            # Bias based on agent confidence
            if state.get("hrm_results"):
                hrm_conf = state["hrm_results"]["metadata"].get("decomposition_quality_score", 0.5)
                base += hrm_conf * 0.2

            if state.get("trm_results"):
                trm_conf = state["trm_results"]["metadata"].get("final_quality_score", 0.5)
                base += trm_conf * 0.2

            return min(base, 1.0)

        rollout_policy = HybridRolloutPolicy(
            heuristic_fn=heuristic_fn,
            heuristic_weight=0.7,
            random_weight=0.3,
        )

        # Run MCTS search
        best_action, stats = await self.mcts_engine.search(
            root=root,
            num_iterations=self.mcts_config.num_iterations,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
            max_rollout_depth=self.mcts_config.max_rollout_depth,
            selection_policy=self.mcts_config.selection_policy,
        )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Compute tree statistics
        tree_depth = self.mcts_engine.get_tree_depth(root)
        tree_node_count = self.mcts_engine.count_nodes(root)

        # Track experiment
        self.experiment_tracker.create_result(
            experiment_id=f"mcts_{int(time.time())}",
            config=self.mcts_config,
            mcts_stats=stats,
            execution_time_ms=execution_time_ms,
            tree_depth=tree_depth,
            tree_node_count=tree_node_count,
            metadata={
                "query": state["query"][:100],
                "has_rag": state.get("use_rag", False),
            },
        )

        self.logger.info(
            f"MCTS complete: best_action={best_action}, "
            f"iterations={stats['iterations']}, "
            f"cache_hit_rate={stats['cache_hit_rate']:.2%}"
        )

        return {
            "mcts_root": root,
            "mcts_best_action": best_action,
            "mcts_stats": stats,
            "agent_outputs": [
                {
                    "agent": "mcts",
                    "response": (
                        f"Simulated {stats['iterations']} scenarios with "
                        f"seed {self.mcts_config.seed}. "
                        f"Recommended action: {best_action} "
                        f"(visits={stats['best_action_visits']}, "
                        f"value={stats['best_action_value']:.3f})"
                    ),
                    "confidence": min(
                        stats["best_action_visits"] / stats["iterations"] if stats["iterations"] > 0 else 0.5,
                        1.0,
                    ),
                }
            ],
        }

    def _create_adk_node_handler(self, name: str, agent: Any):
        """Create a handler function for an ADK agent node."""

        async def handler(state: AgentState) -> dict:
            self.logger.info(f"Executing ADK agent: {name}")

            # Initialize if needed (assuming agent has initialize method)
            if hasattr(agent, "initialize"):
                await agent.initialize()

            # Prepare inputs - ADK agents might expect different signatures
            # We'll assume they implement a standard ADKAgentAdapter interface
            # or we pass the query directly
            try:
                # Check if it's a mock or real agent
                if hasattr(agent, "process_query"):  # Assuming custom method
                    response = await agent.process_query(state["query"])
                elif hasattr(agent, "run"):  # Standard LangChain-like
                    response = await agent.run(state["query"])
                elif hasattr(agent, "process"):  # Framework standard
                    response = await agent.process(state["query"])
                else:
                    # Fallback for demonstration/mock objects
                    response = {"response": f"Processed by {name}", "confidence": 0.8}

                # Extract content based on response type
                if isinstance(response, dict):
                    content = response.get("response", str(response))
                    confidence = response.get("confidence", 0.8)
                    metadata = response.get("metadata", {})
                else:
                    content = str(response)
                    confidence = 0.8
                    metadata = {}

                return {
                    "adk_results": {name: {"response": content, "metadata": metadata}},
                    "agent_outputs": [
                        {
                            "agent": f"adk_{name}",
                            "response": content,
                            "confidence": confidence,
                        }
                    ],
                }
            except Exception as e:
                self.logger.error(f"ADK agent {name} failed: {e}")
                return {
                    "agent_outputs": [
                        {
                            "agent": f"adk_{name}",
                            "response": f"Error executing {name}: {e}",
                            "confidence": 0.0,
                        }
                    ]
                }

        return handler

    def _aggregate_results_node(self, state: AgentState) -> dict:
        """Aggregate results from all agents."""
        self.logger.info("Aggregating agent results")

        agent_outputs = state.get("agent_outputs", [])

        confidence_scores = {output["agent"]: output["confidence"] for output in agent_outputs}

        return {"confidence_scores": confidence_scores}

    def _evaluate_consensus_node(self, state: AgentState) -> dict:
        """Evaluate consensus among agents and increment iteration counter.

        The iteration counter is incremented here to ensure proper loop termination
        when consensus is not reached and max_iterations is checked in _check_consensus.
        """
        agent_outputs = state.get("agent_outputs", [])
        current_iteration = state.get("iteration", 0)
        next_iteration = current_iteration + 1

        if len(agent_outputs) < 2:
            self.logger.debug(f"Single agent output, auto-consensus (iteration={next_iteration})")
            return {
                "consensus_reached": True,
                "consensus_score": 1.0,
                "iteration": next_iteration,
            }

        avg_confidence = sum(o["confidence"] for o in agent_outputs) / len(agent_outputs)
        consensus_reached = avg_confidence >= self.consensus_threshold

        self.logger.info(
            f"Consensus: {consensus_reached} (score={avg_confidence:.2f}, "
            f"iteration={next_iteration}/{state.get('max_iterations', self.max_iterations)})"
        )

        return {
            "consensus_reached": consensus_reached,
            "consensus_score": avg_confidence,
            "iteration": next_iteration,
        }

    def _check_consensus(self, state: AgentState) -> str:
        """Check if consensus reached or need more iterations.

        Returns:
            'synthesize' if consensus reached or max iterations exceeded
            'iterate' if more iterations needed
        """
        current_iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", self.max_iterations)

        if state.get("consensus_reached", False):
            self.logger.info(f"Consensus reached at iteration {current_iteration}")
            return "synthesize"

        if current_iteration >= max_iter:
            self.logger.warning(f"Max iterations ({max_iter}) reached without consensus, proceeding to synthesis")
            return "synthesize"

        self.logger.debug(f"Continuing iteration loop (current={current_iteration}, max={max_iter})")
        return "iterate"

    async def _synthesize_node(self, state: AgentState) -> dict:
        """Synthesize final response from agent outputs."""
        self.logger.info("Synthesizing final response")

        agent_outputs = state.get("agent_outputs", [])

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
                temperature=0.5,
            )
            final_response = response.text
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            best_output = max(agent_outputs, key=lambda o: o["confidence"])
            final_response = best_output["response"]

        metadata = {
            "agents_used": [o["agent"] for o in agent_outputs],
            "confidence_scores": state.get("confidence_scores", {}),
            "consensus_score": state.get("consensus_score", 0.0),
            "iterations": state.get("iteration", 0),
            "mcts_config": state.get("mcts_config", {}),
        }

        if state.get("mcts_stats"):
            metadata["mcts_stats"] = state["mcts_stats"]

        return {
            "final_response": final_response,
            "metadata": metadata,
        }


class IntegratedFramework:
    """
    Integrated multi-agent framework with new MCTS core.

    Maintains backward compatibility with original process() signature.
    """

    def __init__(
        self,
        model_adapter,
        logger,
        vector_store=None,
        _embedding_model=None,
        hrm_config: dict | None = None,
        trm_config: dict | None = None,
        mcts_config: MCTSConfig | None = None,
        top_k_retrieval: int = 5,
        max_iterations: int = 3,
        consensus_threshold: float = 0.75,
        enable_parallel_agents: bool = True,
        adk_agents: dict[str, Any] | None = None,
    ):
        """
        Initialize integrated framework.

        Backward compatible with LangGraphMultiAgentFramework.
        """
        self.model_adapter = model_adapter
        self.logger = logger
        self.vector_store = vector_store
        self.adk_agents = adk_agents or {}

        # Import agents (would be real imports in production)
        try:
            from improved_hrm_agent import HRMAgent
            from improved_trm_agent import TRMAgent

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
        except ImportError:
            self.hrm_agent = None
            self.trm_agent = None
            self.logger.warning("Could not import HRM/TRM agents")

        # Build graph
        self.graph_builder = GraphBuilder(
            hrm_agent=self.hrm_agent,
            trm_agent=self.trm_agent,
            model_adapter=model_adapter,
            logger=logger,
            vector_store=vector_store,
            mcts_config=mcts_config,
            top_k_retrieval=top_k_retrieval,
            max_iterations=max_iterations,
            consensus_threshold=consensus_threshold,
            enable_parallel_agents=enable_parallel_agents,
            adk_agents=self.adk_agents,
        )

        # Compile graph
        if StateGraph is not None:
            self.graph = self.graph_builder.build_graph()
            self.memory = MemorySaver() if MemorySaver else None
            self.app = self.graph.compile(checkpointer=self.memory) if self.memory else self.graph.compile()
        else:
            self.graph = None
            self.app = None

        self.logger.info("Integrated framework initialized with new MCTS core")

    async def process(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict | None = None,
    ) -> dict:
        """
        Process query through LangGraph.

        Backward compatible with original signature.

        Args:
            query: User query to process
            use_rag: Enable RAG context retrieval
            use_mcts: Enable MCTS simulation
            config: Optional LangGraph config

        Returns:
            Dictionary with response, metadata, and state
        """
        if self.app is None:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")

        initial_state = {
            "query": query,
            "use_rag": use_rag,
            "use_mcts": use_mcts,
            "iteration": 0,
            "max_iterations": self.graph_builder.max_iterations,
            "agent_outputs": [],
        }

        config = config or {"configurable": {"thread_id": "default"}}

        result = await self.app.ainvoke(initial_state, config=config)

        return {
            "response": result.get("final_response", ""),
            "metadata": result.get("metadata", {}),
            "state": result,
        }

    async def astream(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict | None = None,
    ):
        """
        Stream node-level state updates through LangGraph.

        Yields state updates as each node completes execution.

        Args:
            query: User query to process
            use_rag: Enable RAG context retrieval
            use_mcts: Enable MCTS simulation
            config: Optional LangGraph config

        Yields:
            Tuple of (node_name, state_update) for each node completion
        """
        if self.app is None:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")

        initial_state = {
            "query": query,
            "use_rag": use_rag,
            "use_mcts": use_mcts,
            "iteration": 0,
            "max_iterations": self.graph_builder.max_iterations,
            "agent_outputs": [],
        }

        config = config or {"configurable": {"thread_id": "default"}}

        self.logger.debug(f"Starting streaming execution for query: {query[:100]}")

        async for event in self.app.astream(initial_state, config=config):
            for node_name, state_update in event.items():
                self.logger.debug(f"Node '{node_name}' completed with keys: {list(state_update.keys())}")
                yield node_name, state_update

    async def astream_events(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict | None = None,
        include_types: list[str] | None = None,
    ):
        """
        Stream detailed events from LangGraph execution.

        Provides fine-grained streaming including token-level events when
        LLM calls support streaming, and node lifecycle events.

        Args:
            query: User query to process
            use_rag: Enable RAG context retrieval
            use_mcts: Enable MCTS simulation
            config: Optional LangGraph config
            include_types: Event types to include (default: all)
                Supported: ['on_llm_start', 'on_llm_stream', 'on_llm_end',
                           'on_chain_start', 'on_chain_end', 'on_tool_start', 'on_tool_end']

        Yields:
            StreamEvent dict with event type, name, data, and metadata
        """
        if self.app is None:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")

        initial_state = {
            "query": query,
            "use_rag": use_rag,
            "use_mcts": use_mcts,
            "iteration": 0,
            "max_iterations": self.graph_builder.max_iterations,
            "agent_outputs": [],
        }

        config = config or {"configurable": {"thread_id": "default"}}

        # Default to all event types if not specified
        include_types = include_types or [
            "on_llm_start",
            "on_llm_stream",
            "on_llm_end",
            "on_chain_start",
            "on_chain_end",
        ]

        self.logger.debug(f"Starting event streaming for query: {query[:100]}, event_types: {include_types}")

        try:
            async for event in self.app.astream_events(
                initial_state,
                config=config,
                version="v2",
            ):
                event_type = event.get("event", "")

                # Filter by event type if specified
                if include_types and event_type not in include_types:
                    continue

                # Construct standardized event format
                stream_event = {
                    "event_type": event_type,
                    "name": event.get("name", ""),
                    "run_id": event.get("run_id", ""),
                    "data": event.get("data", {}),
                    "metadata": event.get("metadata", {}),
                    "tags": event.get("tags", []),
                }

                # For LLM streaming events, extract token content
                if event_type == "on_llm_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    if hasattr(chunk, "content"):
                        stream_event["token"] = chunk.content
                    elif isinstance(chunk, dict):
                        stream_event["token"] = chunk.get("content", "")

                self.logger.debug(f"Event: {event_type}, name: {stream_event['name']}")
                yield stream_event

        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            # Yield error event
            yield {
                "event_type": "on_error",
                "name": "streaming_error",
                "run_id": "",
                "data": {"error": str(e)},
                "metadata": {},
                "tags": [],
            }

    def get_experiment_tracker(self) -> ExperimentTracker:
        """Get the experiment tracker for analysis."""
        return self.graph_builder.experiment_tracker

    def set_mcts_seed(self, seed: int) -> None:
        """Set MCTS seed for deterministic behavior."""
        self.graph_builder.mcts_engine.reset_seed(seed)
        self.graph_builder.mcts_config.seed = seed

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def get_graph(self):
        """
        Get the compiled LangGraph application.

        Returns:
            The compiled LangGraph app object.
        """
        return self.app

    def get_graph_structure(self) -> dict:
        """
        Extract the graph structure for visualization and introspection.

        Returns:
            Dict containing nodes, edges, and conditional routing info.
        """
        # Core node definitions with descriptions
        nodes = [
            {"id": "entry", "label": "Entry", "description": "Input validation", "type": "start"},
            {"id": "retrieve_context", "label": "Retrieve Context", "description": "RAG retrieval", "type": "process"},
            {"id": "route_decision", "label": "Route Decision", "description": "Agent selection", "type": "branch"},
            {
                "id": "parallel_agents",
                "label": "Parallel Agents",
                "description": "HRM + TRM parallel",
                "type": "process",
            },
            {"id": "hrm_agent", "label": "HRM Agent", "description": "Hierarchical reasoning", "type": "agent"},
            {"id": "trm_agent", "label": "TRM Agent", "description": "Task refinement", "type": "agent"},
            {"id": "mcts_simulator", "label": "MCTS Simulator", "description": "Monte Carlo search", "type": "agent"},
            {
                "id": "aggregate_results",
                "label": "Aggregate Results",
                "description": "Combine outputs",
                "type": "process",
            },
            {
                "id": "evaluate_consensus",
                "label": "Evaluate Consensus",
                "description": "Check agreement",
                "type": "branch",
            },
            {"id": "synthesize", "label": "Synthesize", "description": "Final response", "type": "end"},
        ]

        # Add symbolic agent if enabled
        if self.graph_builder.use_symbolic_reasoning:
            nodes.insert(
                7,
                {
                    "id": "symbolic_agent",
                    "label": "Symbolic Agent",
                    "description": "Neuro-symbolic reasoning",
                    "type": "agent",
                },
            )

        # Add ADK agent nodes
        for name in self.graph_builder.adk_agents:
            nodes.append(
                {
                    "id": f"adk_{name}",
                    "label": f"ADK: {name}",
                    "description": f"Google ADK agent: {name}",
                    "type": "agent",
                }
            )

        # Sequential edges
        edges = [
            {"source": "__start__", "target": "entry"},
            {"source": "entry", "target": "retrieve_context"},
            {"source": "retrieve_context", "target": "route_decision"},
            {"source": "parallel_agents", "target": "aggregate_results"},
            {"source": "hrm_agent", "target": "aggregate_results"},
            {"source": "trm_agent", "target": "aggregate_results"},
            {"source": "mcts_simulator", "target": "aggregate_results"},
            {"source": "aggregate_results", "target": "evaluate_consensus"},
            {"source": "synthesize", "target": "__end__"},
        ]

        # Add symbolic agent edge
        if self.graph_builder.use_symbolic_reasoning:
            edges.append({"source": "symbolic_agent", "target": "aggregate_results"})

        # Add ADK agent edges
        for name in self.graph_builder.adk_agents:
            edges.append({"source": f"adk_{name}", "target": "aggregate_results"})

        # Conditional routing
        routing_map = {
            "parallel": "parallel_agents",
            "hrm": "hrm_agent",
            "trm": "trm_agent",
            "mcts": "mcts_simulator",
            "aggregate": "aggregate_results",
        }

        if self.graph_builder.use_symbolic_reasoning:
            routing_map["symbolic"] = "symbolic_agent"

        for name in self.graph_builder.adk_agents:
            routing_map[f"adk_{name}"] = f"adk_{name}"

        conditional_edges = {
            "route_decision": {
                "condition": "_route_to_agents",
                "routes": routing_map,
            },
            "evaluate_consensus": {
                "condition": "_check_consensus",
                "routes": {
                    "synthesize": "synthesize",
                    "iterate": "route_decision",
                },
            },
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "conditional_edges": conditional_edges,
            "entry_point": "entry",
            "terminal_node": "synthesize",
        }

    def get_graph_mermaid(
        self,
        include_descriptions: bool = True,
        theme: str = "default",
    ) -> str:
        """
        Generate a Mermaid flowchart diagram of the graph.

        Args:
            include_descriptions: Include node descriptions in labels.
            theme: Mermaid theme (default, forest, dark, neutral).

        Returns:
            Mermaid diagram source code.
        """
        structure = self.get_graph_structure()
        lines = []

        # Add theme configuration
        if theme != "default":
            lines.append(f"%%{{init: {{'theme': '{theme}'}}}}%%")

        lines.append("flowchart TD")

        # Define node styles by type
        lines.append("    %% Node style definitions")
        lines.append("    classDef startNode fill:#90EE90,stroke:#228B22")
        lines.append("    classDef endNode fill:#FFB6C1,stroke:#DC143C")
        lines.append("    classDef processNode fill:#87CEEB,stroke:#4169E1")
        lines.append("    classDef branchNode fill:#DDA0DD,stroke:#8B008B")
        lines.append("    classDef agentNode fill:#FFD700,stroke:#DAA520")
        lines.append("")

        # Add nodes
        lines.append("    %% Nodes")
        node_type_map = {}
        for node in structure["nodes"]:
            node_id = node["id"]
            label = node["label"]
            node_type = node["type"]
            node_type_map[node_id] = node_type

            if include_descriptions and node.get("description"):
                label = f"{label}<br/><small>{node['description']}</small>"

            # Use different shapes based on node type
            if node_type == "start" or node_type == "end":
                lines.append(f"    {node_id}([{label}])")
            elif node_type == "branch":
                lines.append(f"    {node_id}{{{label}}}")
            elif node_type == "agent":
                lines.append(f"    {node_id}[/{label}/]")
            else:
                lines.append(f"    {node_id}[{label}]")

        lines.append("")

        # Add sequential edges
        lines.append("    %% Sequential edges")
        for edge in structure["edges"]:
            source = edge["source"]
            target = edge["target"]

            # Handle special start/end nodes
            if source == "__start__":
                lines.append(f"    START(( )) --> {target}")
            elif target == "__end__":
                lines.append(f"    {source} --> END(( ))")
            else:
                lines.append(f"    {source} --> {target}")

        lines.append("")

        # Add conditional edges
        lines.append("    %% Conditional edges")
        for source_node, edge_info in structure["conditional_edges"].items():
            routes = edge_info["routes"]
            for label, target in routes.items():
                lines.append(f"    {source_node} -->|{label}| {target}")

        lines.append("")

        # Apply styles
        lines.append("    %% Apply styles")
        for node_id, node_type in node_type_map.items():
            if node_type == "start":
                lines.append(f"    class {node_id} startNode")
            elif node_type == "end":
                lines.append(f"    class {node_id} endNode")
            elif node_type == "process":
                lines.append(f"    class {node_id} processNode")
            elif node_type == "branch":
                lines.append(f"    class {node_id} branchNode")
            elif node_type == "agent":
                lines.append(f"    class {node_id} agentNode")

        return "\n".join(lines)

    def draw_mermaid(
        self,
        output_file: str | None = None,
        format: str = "png",
        include_descriptions: bool = True,
    ) -> str:
        """
        Generate and optionally render a Mermaid diagram of the graph.

        This method generates the Mermaid source code and can optionally
        render it to PNG via the Kroki API.

        Args:
            output_file: Optional file path to save PNG (requires network).
            format: Output format ('png', 'svg').
            include_descriptions: Include node descriptions.

        Returns:
            Mermaid diagram source code.

        Raises:
            RuntimeError: If rendering fails and output_file is specified.
        """
        import base64
        import zlib

        mermaid_code = self.get_graph_mermaid(include_descriptions=include_descriptions)

        # If no output file requested, just return the code
        if not output_file:
            return mermaid_code

        # Render via Kroki API
        try:
            import httpx

            # Compress and encode for Kroki
            compressed = zlib.compress(mermaid_code.encode("utf-8"), 9)
            encoded = base64.urlsafe_b64encode(compressed).decode("ascii")

            # Build Kroki URL
            kroki_url = f"https://kroki.io/mermaid/{format}/{encoded}"

            # Fetch rendered diagram
            with httpx.Client(timeout=30.0) as client:
                response = client.get(kroki_url)
                response.raise_for_status()

            # Save to file
            with open(output_file, "wb") as f:
                f.write(response.content)

            self.logger.info(f"Saved graph diagram to {output_file}")

        except ImportError:
            self.logger.warning("httpx not available for Kroki rendering. Install with: pip install httpx")
        except Exception as e:
            self.logger.error(f"Failed to render diagram: {e}")
            raise RuntimeError(f"Diagram rendering failed: {e}") from e

        return mermaid_code

    def get_execution_trace_mermaid(
        self,
        execution_path: list[str],
        timings: dict[str, float] | None = None,
    ) -> str:
        """
        Generate a Mermaid diagram showing an execution trace.

        Args:
            execution_path: List of node IDs in execution order.
            timings: Optional dict of node_id -> execution_time_ms.

        Returns:
            Mermaid sequence diagram source code.
        """
        lines = ["sequenceDiagram"]
        lines.append("    autonumber")
        lines.append("")

        # Define participants
        unique_nodes = []
        for node in execution_path:
            if node not in unique_nodes:
                unique_nodes.append(node)

        for node in unique_nodes:
            # Format label
            label = node.replace("_", " ").title()
            lines.append(f"    participant {node} as {label}")

        lines.append("")

        # Add execution arrows
        for i, node in enumerate(execution_path):
            if i == 0:
                lines.append(f"    Note over {node}: Start")
            elif i < len(execution_path) - 1:
                prev_node = execution_path[i - 1]
                timing_label = ""
                if timings and prev_node in timings:
                    timing_label = f" [{timings[prev_node]:.1f}ms]"
                lines.append(f"    {prev_node}->>+{node}: Execute{timing_label}")

            if i == len(execution_path) - 1:
                lines.append(f"    Note over {node}: End")

        return "\n".join(lines)
