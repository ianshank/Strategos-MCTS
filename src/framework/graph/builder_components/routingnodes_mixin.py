# mypy: disable-error-code="attr-defined,misc,assignment"
from src.framework.graph.state import AgentState
from src.observability.logging import get_structured_logger

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
except ImportError:
    StateGraph = None
    END = "__end__"
    MemorySaver = None

try:
    from src.agents.meta_controller.base import AbstractMetaController, MetaControllerFeatures
    from src.agents.meta_controller.bert_controller import BERTMetaController
    from src.agents.meta_controller.config_loader import MetaControllerConfig, MetaControllerConfigLoader
    from src.agents.meta_controller.rnn_controller import RNNMetaController

    _META_CONTROLLER_AVAILABLE = True
except ImportError:
    _META_CONTROLLER_AVAILABLE = False
    AbstractMetaController = None
    MetaControllerFeatures = None
    RNNMetaController = None
    BERTMetaController = None
    MetaControllerConfig = None
    MetaControllerConfigLoader = None

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
    NeuroSymbolicConfig = None
    SymbolicReasoningAgent = None
    SymbolicAgentGraphExtension = None
    SymbolicAgentNodeConfig = None
    NeuroSymbolicMCTSIntegration = None
    NeuroSymbolicMCTSConfig = None
    ConstraintSystem = None
    ConstraintConfig = None

logger = get_structured_logger(__name__)


class RoutingNodesMixin:

    def _route_decision_node(self, _state: AgentState) -> dict:
        """Prepare routing decision."""
        return {}

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
            self.logger.debug(
                f"Neural routing: agent={prediction.agent}, confidence={prediction.confidence:.3f}, probs={prediction.probabilities}"
            )
            agent = prediction.agent
            if agent == "hrm":
                if "hrm_results" not in state:
                    return "hrm"
            elif agent == "trm":
                if "trm_results" not in state:
                    return "trm"
            elif agent == "mcts" and state.get("use_mcts", False) and ("mcts_stats" not in state):
                return "mcts"
            return self._rule_based_route_decision(state)
        except Exception as e:
            self.logger.error(f"Neural routing failed: {e}")
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
        if (
            self.use_symbolic_reasoning
            and self.symbolic_extension
            and ("symbolic_results" not in state)
            and self.symbolic_extension.should_route_to_symbolic(state.get("query", ""), state)
        ):
            return "symbolic"
        if iteration == 0:
            query_lower = state["query"].lower()
            for name in self.adk_agents:
                adk_results = state.get("adk_results", {})
                if name not in adk_results and (
                    name == "deep_search"
                    and ("research" in query_lower or "investigate" in query_lower)
                    or (name == "ml_engineering" and ("train" in query_lower or "model" in query_lower))
                    or (name == "data_science" and ("analyze" in query_lower or "data" in query_lower))
                ):
                    return f"adk_{name}"
            if self.enable_parallel_agents:
                if "hrm_results" not in state and "trm_results" not in state:
                    return "parallel"
            elif "hrm_results" not in state:
                return "hrm"
            elif "trm_results" not in state:
                return "trm"
        if state.get("use_mcts", False) and "mcts_stats" not in state:
            return "mcts"
        return "aggregate"

    def _route_to_agents(self, state: AgentState) -> str:
        """Route to appropriate agent based on state."""
        if self.use_neural_routing and self.meta_controller is not None:
            return self._neural_route_decision(state)
        return self._rule_based_route_decision(state)
