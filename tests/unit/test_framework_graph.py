"""
Unit tests for src/framework/graph.py - LangGraph orchestration module.

Tests:
- AgentState TypedDict structure
- GraphBuilder initialization and configuration
- Entry node validation
- Retrieve context node behavior
- Route decision logic (rule-based)
- Aggregate results node
- Evaluate consensus node
- Check consensus logic
- Synthesize node
- ADK node handler creation
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.framework.graph import GraphBuilder
from src.framework.mcts.config import ConfigPreset, create_preset_config


@pytest.fixture
def mock_logger():
    return logging.getLogger("test_graph")


@pytest.fixture
def mock_model_adapter():
    adapter = AsyncMock()
    adapter.generate = AsyncMock(return_value=MagicMock(text="synthesized response"))
    return adapter


@pytest.fixture
def mock_hrm_agent():
    agent = AsyncMock()
    agent.process = AsyncMock(
        return_value={
            "response": "HRM response",
            "metadata": {"decomposition_quality_score": 0.8},
        }
    )
    return agent


@pytest.fixture
def mock_trm_agent():
    agent = AsyncMock()
    agent.process = AsyncMock(
        return_value={
            "response": "TRM response",
            "metadata": {"final_quality_score": 0.9},
        }
    )
    return agent


@pytest.fixture
def mcts_config():
    return create_preset_config(ConfigPreset.FAST)


@pytest.fixture
def graph_builder(mock_hrm_agent, mock_trm_agent, mock_model_adapter, mock_logger, mcts_config):
    return GraphBuilder(
        hrm_agent=mock_hrm_agent,
        trm_agent=mock_trm_agent,
        model_adapter=mock_model_adapter,
        logger=mock_logger,
        vector_store=None,
        mcts_config=mcts_config,
        top_k_retrieval=3,
        max_iterations=2,
        consensus_threshold=0.7,
        enable_parallel_agents=True,
    )


@pytest.mark.unit
class TestGraphBuilderInit:
    """Tests for GraphBuilder initialization."""

    def test_default_initialization(self, graph_builder, mcts_config):
        """Test GraphBuilder initializes with correct defaults."""
        assert graph_builder.max_iterations == 2
        assert graph_builder.consensus_threshold == 0.7
        assert graph_builder.enable_parallel_agents is True
        assert graph_builder.top_k_retrieval == 3
        assert graph_builder.mcts_config is mcts_config
        assert graph_builder.use_neural_routing is False
        assert graph_builder.use_symbolic_reasoning is False
        assert graph_builder.adk_agents == {}

    def test_default_mcts_config_when_none(self, mock_hrm_agent, mock_trm_agent, mock_model_adapter, mock_logger):
        """Test that a balanced preset is used when no MCTS config provided."""
        builder = GraphBuilder(
            hrm_agent=mock_hrm_agent,
            trm_agent=mock_trm_agent,
            model_adapter=mock_model_adapter,
            logger=mock_logger,
            mcts_config=None,
        )
        # Should default to balanced preset
        balanced = create_preset_config(ConfigPreset.BALANCED)
        assert builder.mcts_config.num_iterations == balanced.num_iterations

    def test_adk_agents_stored(self, mock_hrm_agent, mock_trm_agent, mock_model_adapter, mock_logger, mcts_config):
        """Test ADK agents are stored properly."""
        adk = {"deep_search": MagicMock(), "ml_engineering": MagicMock()}
        builder = GraphBuilder(
            hrm_agent=mock_hrm_agent,
            trm_agent=mock_trm_agent,
            model_adapter=mock_model_adapter,
            logger=mock_logger,
            mcts_config=mcts_config,
            adk_agents=adk,
        )
        assert len(builder.adk_agents) == 2
        assert "deep_search" in builder.adk_agents

    def test_mcts_engine_created(self, graph_builder):
        """Test that MCTS engine is created during init."""
        assert graph_builder.mcts_engine is not None

    def test_experiment_tracker_created(self, graph_builder):
        """Test that experiment tracker is created."""
        assert graph_builder.experiment_tracker is not None


@pytest.mark.unit
class TestEntryNode:
    """Tests for _entry_node."""

    def test_valid_query(self, graph_builder):
        """Test entry node with valid query."""
        state = {"query": "What is AI?", "use_mcts": False, "use_rag": False}
        result = graph_builder._entry_node(state)
        assert result["iteration"] == 0
        assert result["agent_outputs"] == []
        assert "mcts_config" in result

    def test_empty_query_raises(self, graph_builder):
        """Test entry node raises on empty query."""
        state = {"query": "", "use_mcts": False, "use_rag": False}
        with pytest.raises(ValueError, match="non-empty string"):
            graph_builder._entry_node(state)

    def test_whitespace_query_raises(self, graph_builder):
        """Test entry node raises on whitespace-only query."""
        state = {"query": "   ", "use_mcts": False, "use_rag": False}
        with pytest.raises(ValueError, match="empty or whitespace"):
            graph_builder._entry_node(state)

    def test_missing_query_raises(self, graph_builder):
        """Test entry node raises when query missing from state."""
        state = {"use_mcts": False, "use_rag": False}
        with pytest.raises(ValueError, match="non-empty string"):
            graph_builder._entry_node(state)

    def test_non_string_query_raises(self, graph_builder):
        """Test entry node raises on non-string query."""
        state = {"query": 123, "use_mcts": False, "use_rag": False}
        with pytest.raises(ValueError, match="non-empty string"):
            graph_builder._entry_node(state)


@pytest.mark.unit
class TestRetrieveContextNode:
    """Tests for _retrieve_context_node."""

    def test_no_rag_returns_empty(self, graph_builder):
        """Test returns empty when use_rag is False."""
        state = {"use_rag": False, "query": "test"}
        result = graph_builder._retrieve_context_node(state)
        assert result["rag_context"] == ""
        assert result["retrieved_docs"] == []

    def test_no_vector_store_returns_empty(self, graph_builder):
        """Test returns empty when no vector store configured."""
        state = {"use_rag": True, "query": "test"}
        result = graph_builder._retrieve_context_node(state)
        assert result["rag_context"] == ""
        assert result["retrieved_docs"] == []

    def test_empty_query_returns_empty(self, graph_builder):
        """Test returns empty with empty query."""
        graph_builder.vector_store = MagicMock()
        state = {"use_rag": True, "query": ""}
        result = graph_builder._retrieve_context_node(state)
        assert result["rag_context"] == ""

    def test_successful_retrieval(self, graph_builder):
        """Test successful document retrieval."""
        mock_doc1 = MagicMock(page_content="Doc 1 content", metadata={"source": "a"})
        mock_doc2 = MagicMock(page_content="Doc 2 content", metadata={"source": "b"})
        graph_builder.vector_store = MagicMock()
        graph_builder.vector_store.similarity_search.return_value = [mock_doc1, mock_doc2]

        state = {"use_rag": True, "query": "test query"}
        result = graph_builder._retrieve_context_node(state)
        assert "Doc 1 content" in result["rag_context"]
        assert "Doc 2 content" in result["rag_context"]
        assert len(result["retrieved_docs"]) == 2

    def test_retrieval_failure_graceful_degradation(self, graph_builder):
        """Test graceful degradation on retrieval failure."""
        graph_builder.vector_store = MagicMock()
        graph_builder.vector_store.similarity_search.side_effect = RuntimeError("DB error")

        state = {"use_rag": True, "query": "test query"}
        result = graph_builder._retrieve_context_node(state)
        assert result["rag_context"] == ""
        assert result["retrieved_docs"] == []


@pytest.mark.unit
class TestRouteDecisionNode:
    """Tests for _route_decision_node."""

    def test_returns_empty_dict(self, graph_builder):
        """Test route decision node returns empty dict."""
        result = graph_builder._route_decision_node({})
        assert result == {}


@pytest.mark.unit
class TestRuleBasedRouteDecision:
    """Tests for _rule_based_route_decision."""

    def test_first_iteration_parallel(self, graph_builder):
        """Test parallel routing on first iteration with parallel enabled."""
        state = {"iteration": 0, "query": "test", "use_mcts": False}
        assert graph_builder._rule_based_route_decision(state) == "parallel"

    def test_first_iteration_sequential_hrm_first(
        self, mock_hrm_agent, mock_trm_agent, mock_model_adapter, mock_logger, mcts_config
    ):
        """Test sequential routing picks HRM first."""
        builder = GraphBuilder(
            hrm_agent=mock_hrm_agent,
            trm_agent=mock_trm_agent,
            model_adapter=mock_model_adapter,
            logger=mock_logger,
            mcts_config=mcts_config,
            enable_parallel_agents=False,
        )
        state = {"iteration": 0, "query": "test", "use_mcts": False}
        assert builder._rule_based_route_decision(state) == "hrm"

    def test_first_iteration_sequential_trm_after_hrm(
        self, mock_hrm_agent, mock_trm_agent, mock_model_adapter, mock_logger, mcts_config
    ):
        """Test sequential routing picks TRM after HRM done."""
        builder = GraphBuilder(
            hrm_agent=mock_hrm_agent,
            trm_agent=mock_trm_agent,
            model_adapter=mock_model_adapter,
            logger=mock_logger,
            mcts_config=mcts_config,
            enable_parallel_agents=False,
        )
        state = {"iteration": 0, "query": "test", "use_mcts": False, "hrm_results": {"response": "done"}}
        assert builder._rule_based_route_decision(state) == "trm"

    def test_mcts_routing_when_enabled(self, graph_builder):
        """Test MCTS routing when enabled and not yet done."""
        state = {
            "iteration": 1,
            "query": "test",
            "use_mcts": True,
            "hrm_results": {},
            "trm_results": {},
        }
        assert graph_builder._rule_based_route_decision(state) == "mcts"

    def test_aggregate_when_all_done(self, graph_builder):
        """Test aggregate routing when all agents have run."""
        state = {
            "iteration": 1,
            "query": "test",
            "use_mcts": False,
            "hrm_results": {},
            "trm_results": {},
        }
        assert graph_builder._rule_based_route_decision(state) == "aggregate"

    def test_adk_deep_search_trigger(
        self, mock_hrm_agent, mock_trm_agent, mock_model_adapter, mock_logger, mcts_config
    ):
        """Test ADK deep_search agent triggered by keyword."""
        builder = GraphBuilder(
            hrm_agent=mock_hrm_agent,
            trm_agent=mock_trm_agent,
            model_adapter=mock_model_adapter,
            logger=mock_logger,
            mcts_config=mcts_config,
            adk_agents={"deep_search": MagicMock()},
        )
        state = {"iteration": 0, "query": "Please research this topic"}
        assert builder._rule_based_route_decision(state) == "adk_deep_search"

    def test_mcts_not_routed_when_already_done(self, graph_builder):
        """Test MCTS not routed when stats already present."""
        state = {
            "iteration": 1,
            "query": "test",
            "use_mcts": True,
            "mcts_stats": {"iterations": 50},
        }
        assert graph_builder._rule_based_route_decision(state) == "aggregate"


@pytest.mark.unit
class TestRouteToAgents:
    """Tests for _route_to_agents."""

    def test_uses_rule_based_when_no_neural(self, graph_builder):
        """Test falls back to rule-based routing."""
        state = {"iteration": 0, "query": "test", "use_mcts": False}
        result = graph_builder._route_to_agents(state)
        assert result == "parallel"


@pytest.mark.unit
class TestAggregateResultsNode:
    """Tests for _aggregate_results_node."""

    def test_aggregates_confidence_scores(self, graph_builder):
        """Test confidence scores are extracted correctly."""
        state = {
            "agent_outputs": [
                {"agent": "hrm", "response": "r1", "confidence": 0.8},
                {"agent": "trm", "response": "r2", "confidence": 0.9},
            ]
        }
        result = graph_builder._aggregate_results_node(state)
        assert result["confidence_scores"]["hrm"] == 0.8
        assert result["confidence_scores"]["trm"] == 0.9

    def test_empty_outputs(self, graph_builder):
        """Test with no agent outputs."""
        state = {"agent_outputs": []}
        result = graph_builder._aggregate_results_node(state)
        assert result["confidence_scores"] == {}


@pytest.mark.unit
class TestEvaluateConsensusNode:
    """Tests for _evaluate_consensus_node."""

    def test_single_output_auto_consensus(self, graph_builder):
        """Test single agent output triggers auto-consensus."""
        state = {
            "agent_outputs": [{"agent": "hrm", "confidence": 0.5}],
            "iteration": 0,
        }
        result = graph_builder._evaluate_consensus_node(state)
        assert result["consensus_reached"] is True
        assert result["consensus_score"] == 1.0
        assert result["iteration"] == 1

    def test_empty_outputs_auto_consensus(self, graph_builder):
        """Test zero agent outputs triggers auto-consensus."""
        state = {"agent_outputs": [], "iteration": 0}
        result = graph_builder._evaluate_consensus_node(state)
        assert result["consensus_reached"] is True

    def test_high_confidence_reaches_consensus(self, graph_builder):
        """Test high average confidence reaches consensus."""
        state = {
            "agent_outputs": [
                {"agent": "hrm", "confidence": 0.8},
                {"agent": "trm", "confidence": 0.9},
            ],
            "iteration": 0,
        }
        result = graph_builder._evaluate_consensus_node(state)
        assert result["consensus_reached"] is True
        assert result["consensus_score"] == pytest.approx(0.85, abs=0.01)
        assert result["iteration"] == 1

    def test_low_confidence_no_consensus(self, graph_builder):
        """Test low average confidence does not reach consensus."""
        state = {
            "agent_outputs": [
                {"agent": "hrm", "confidence": 0.3},
                {"agent": "trm", "confidence": 0.4},
            ],
            "iteration": 0,
        }
        result = graph_builder._evaluate_consensus_node(state)
        assert result["consensus_reached"] is False
        assert result["consensus_score"] == pytest.approx(0.35, abs=0.01)

    def test_iteration_counter_increments(self, graph_builder):
        """Test iteration counter increments."""
        state = {
            "agent_outputs": [{"agent": "a", "confidence": 0.5}, {"agent": "b", "confidence": 0.5}],
            "iteration": 2,
        }
        result = graph_builder._evaluate_consensus_node(state)
        assert result["iteration"] == 3


@pytest.mark.unit
class TestCheckConsensus:
    """Tests for _check_consensus."""

    def test_consensus_reached(self, graph_builder):
        """Test returns synthesize when consensus reached."""
        state = {"consensus_reached": True, "iteration": 1}
        assert graph_builder._check_consensus(state) == "synthesize"

    def test_max_iterations_exceeded(self, graph_builder):
        """Test returns synthesize when max iterations exceeded."""
        state = {"consensus_reached": False, "iteration": 5, "max_iterations": 2}
        assert graph_builder._check_consensus(state) == "synthesize"

    def test_continue_iteration(self, graph_builder):
        """Test returns iterate when neither consensus nor max iterations."""
        state = {"consensus_reached": False, "iteration": 1, "max_iterations": 5}
        assert graph_builder._check_consensus(state) == "iterate"

    def test_uses_builder_max_iterations_as_default(self, graph_builder):
        """Test uses builder's max_iterations when not in state."""
        # builder.max_iterations = 2
        state = {"consensus_reached": False, "iteration": 3}
        assert graph_builder._check_consensus(state) == "synthesize"


@pytest.mark.unit
class TestSynthesizeNode:
    """Tests for _synthesize_node."""

    @pytest.mark.asyncio
    async def test_successful_synthesis(self, graph_builder, mock_model_adapter):
        """Test successful synthesis via LLM."""
        state = {
            "query": "What is AI?",
            "agent_outputs": [
                {"agent": "hrm", "response": "HRM answer", "confidence": 0.8},
                {"agent": "trm", "response": "TRM answer", "confidence": 0.9},
            ],
            "confidence_scores": {"hrm": 0.8, "trm": 0.9},
            "consensus_score": 0.85,
            "iteration": 1,
            "mcts_config": {},
        }
        result = await graph_builder._synthesize_node(state)
        assert result["final_response"] == "synthesized response"
        assert "agents_used" in result["metadata"]
        assert result["metadata"]["iterations"] == 1

    @pytest.mark.asyncio
    async def test_synthesis_fallback_on_error(self, graph_builder, mock_model_adapter):
        """Test fallback to best agent output on LLM failure."""
        mock_model_adapter.generate.side_effect = RuntimeError("LLM failed")
        state = {
            "query": "What is AI?",
            "agent_outputs": [
                {"agent": "hrm", "response": "HRM answer", "confidence": 0.8},
                {"agent": "trm", "response": "TRM answer", "confidence": 0.9},
            ],
            "confidence_scores": {},
            "consensus_score": 0.85,
            "iteration": 1,
            "mcts_config": {},
        }
        result = await graph_builder._synthesize_node(state)
        # Should fall back to highest confidence output
        assert result["final_response"] == "TRM answer"

    @pytest.mark.asyncio
    async def test_synthesis_includes_mcts_stats(self, graph_builder):
        """Test MCTS stats included in metadata when present."""
        state = {
            "query": "test",
            "agent_outputs": [{"agent": "mcts", "response": "mcts out", "confidence": 0.7}],
            "confidence_scores": {},
            "consensus_score": 0.7,
            "iteration": 1,
            "mcts_config": {},
            "mcts_stats": {"iterations": 50, "best_action": "A"},
        }
        result = await graph_builder._synthesize_node(state)
        assert "mcts_stats" in result["metadata"]


@pytest.mark.unit
class TestCreateAdkNodeHandler:
    """Tests for _create_adk_node_handler."""

    @pytest.mark.asyncio
    async def test_adk_handler_with_process_query(self, graph_builder):
        """Test ADK handler calls process_query if available."""
        mock_agent = AsyncMock()
        mock_agent.process_query = AsyncMock(return_value={"response": "adk result", "confidence": 0.85})
        mock_agent.initialize = AsyncMock()

        handler = graph_builder._create_adk_node_handler("test_agent", mock_agent)
        state = {"query": "test query"}
        result = await handler(state)

        assert result["adk_results"]["test_agent"]["response"] == "adk result"
        assert result["agent_outputs"][0]["agent"] == "adk_test_agent"

    @pytest.mark.asyncio
    async def test_adk_handler_fallback(self, graph_builder):
        """Test ADK handler fallback for unknown agent type."""
        mock_agent = MagicMock(spec=[])  # No special methods

        handler = graph_builder._create_adk_node_handler("basic", mock_agent)
        state = {"query": "test"}
        result = await handler(state)

        assert "adk_basic" in result["agent_outputs"][0]["agent"]

    @pytest.mark.asyncio
    async def test_adk_handler_error(self, graph_builder):
        """Test ADK handler error handling."""
        mock_agent = AsyncMock()
        mock_agent.process_query = AsyncMock(side_effect=RuntimeError("agent error"))
        mock_agent.initialize = AsyncMock()

        handler = graph_builder._create_adk_node_handler("failing", mock_agent)
        state = {"query": "test"}
        result = await handler(state)

        assert result["agent_outputs"][0]["confidence"] == 0.0
        assert "Error" in result["agent_outputs"][0]["response"]


@pytest.mark.unit
class TestParallelAgentsNode:
    """Tests for _parallel_agents_node."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self, graph_builder):
        """Test parallel HRM and TRM execution."""
        state = {"query": "What is AI?", "rag_context": "some context"}
        result = await graph_builder._parallel_agents_node(state)

        assert "hrm_results" in result
        assert "trm_results" in result
        assert len(result["agent_outputs"]) == 2
        assert result["hrm_results"]["response"] == "HRM response"
        assert result["trm_results"]["response"] == "TRM response"


@pytest.mark.unit
class TestHrmAgentNode:
    """Tests for _hrm_agent_node."""

    @pytest.mark.asyncio
    async def test_hrm_execution(self, graph_builder):
        """Test HRM agent execution."""
        state = {"query": "test query"}
        result = await graph_builder._hrm_agent_node(state)

        assert result["hrm_results"]["response"] == "HRM response"
        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0]["agent"] == "hrm"


@pytest.mark.unit
class TestTrmAgentNode:
    """Tests for _trm_agent_node."""

    @pytest.mark.asyncio
    async def test_trm_execution(self, graph_builder):
        """Test TRM agent execution."""
        state = {"query": "test query"}
        result = await graph_builder._trm_agent_node(state)

        assert result["trm_results"]["response"] == "TRM response"
        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0]["agent"] == "trm"


@pytest.mark.unit
class TestBuildGraph:
    """Tests for build_graph method."""

    def test_build_graph_without_langgraph_raises(self, graph_builder):
        """Test build_graph raises when LangGraph not installed."""
        with patch("src.framework.graph.StateGraph", None):
            with pytest.raises(ImportError, match="LangGraph not installed"):
                graph_builder.build_graph()
