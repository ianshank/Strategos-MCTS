"""
Extended tests for src/framework/graph.py – covers GraphBuilder meta-controller init,
neuro-symbolic init, neural routing, symbolic agent node, MCTS simulator node,
ADK node handler, IntegratedFramework, streaming, visualization, and draw_mermaid.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.framework.graph import AgentState, GraphBuilder, IntegratedFramework
from src.framework.mcts.config import ConfigPreset, MCTSConfig, create_preset_config

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_builder(**overrides):
    """Create a GraphBuilder with mocked dependencies."""
    defaults = dict(
        hrm_agent=AsyncMock(),
        trm_agent=AsyncMock(),
        model_adapter=AsyncMock(),
        logger=logging.getLogger("test_graph_ext2"),
        vector_store=None,
        mcts_config=create_preset_config(ConfigPreset.BALANCED),
        top_k_retrieval=5,
        max_iterations=3,
        consensus_threshold=0.75,
        enable_parallel_agents=True,
    )
    defaults.update(overrides)
    return GraphBuilder(**defaults)


# ---------------------------------------------------------------------------
# _init_meta_controller
# ---------------------------------------------------------------------------


class TestInitMetaController:
    """Tests for GraphBuilder._init_meta_controller (lines 386-439)."""

    def test_meta_controller_not_available_logs_warning(self):
        """When meta-controller modules are unavailable, falls back to rule-based."""
        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", False):
            builder = _make_builder(meta_controller_config={"enabled": True, "type": "rnn"})
        assert builder.meta_controller is None
        assert builder.use_neural_routing is False

    def test_meta_controller_disabled_in_config(self):
        """Disabled config should skip initialization."""
        mock_config = MagicMock()
        mock_config.enabled = False

        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", True):
            with patch("src.framework.graph.MetaControllerConfigLoader") as mock_loader:
                mock_loader.load_from_dict.return_value = mock_config
                builder = _make_builder(meta_controller_config={"enabled": False})
        assert builder.use_neural_routing is False

    def test_meta_controller_rnn_init(self):
        """RNN meta-controller initialization with model path."""
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.type = "rnn"
        mock_config.rnn.hidden_dim = 64
        mock_config.rnn.num_layers = 2
        mock_config.rnn.dropout = 0.1
        mock_config.rnn.model_path = None
        mock_config.inference.seed = 42
        mock_config.inference.device = "cpu"

        mock_rnn = MagicMock()

        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", True):
            with patch("src.framework.graph.MetaControllerConfigLoader") as mock_loader:
                mock_loader.load_from_dict.return_value = mock_config
                with patch("src.framework.graph.RNNMetaController", return_value=mock_rnn):
                    builder = _make_builder(meta_controller_config={"type": "rnn"})

        assert builder.use_neural_routing is True
        assert builder.meta_controller is mock_rnn

    def test_meta_controller_bert_init(self):
        """BERT meta-controller initialization."""
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.type = "bert"
        mock_config.bert.model_name = "bert-base"
        mock_config.bert.lora_r = 8
        mock_config.bert.lora_alpha = 16
        mock_config.bert.lora_dropout = 0.05
        mock_config.bert.use_lora = False
        mock_config.bert.model_path = "/tmp/model.pt"
        mock_config.inference.seed = 42
        mock_config.inference.device = "cpu"

        mock_bert = MagicMock()

        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", True):
            with patch("src.framework.graph.MetaControllerConfigLoader") as mock_loader:
                mock_loader.load_from_dict.return_value = mock_config
                with patch("src.framework.graph.BERTMetaController", return_value=mock_bert):
                    builder = _make_builder(meta_controller_config={"type": "bert"})

        assert builder.use_neural_routing is True
        mock_bert.load_model.assert_called_once_with("/tmp/model.pt")

    def test_meta_controller_unknown_type_raises(self):
        """Unknown controller type raises ValueError (no fallback)."""
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.type = "unknown_type"
        mock_config.fallback_to_rule_based = False

        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", True):
            with patch("src.framework.graph.MetaControllerConfigLoader") as mock_loader:
                mock_loader.load_from_dict.return_value = mock_config
                with pytest.raises(ValueError, match="Unknown meta-controller type"):
                    _make_builder(meta_controller_config={"type": "unknown"})

    def test_meta_controller_init_failure_with_fallback(self):
        """When init fails and fallback enabled, falls back gracefully."""
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.type = "rnn"
        mock_config.fallback_to_rule_based = True

        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", True):
            with patch("src.framework.graph.MetaControllerConfigLoader") as mock_loader:
                mock_loader.load_from_dict.return_value = mock_config
                with patch("src.framework.graph.RNNMetaController", side_effect=RuntimeError("init fail")):
                    builder = _make_builder(meta_controller_config=mock_config)

        assert builder.use_neural_routing is False


# ---------------------------------------------------------------------------
# _init_neuro_symbolic
# ---------------------------------------------------------------------------


class TestInitNeuroSymbolic:
    """Tests for GraphBuilder._init_neuro_symbolic (lines 448-489)."""

    def test_neuro_symbolic_not_available(self):
        """When neuro-symbolic modules are unavailable, skips init."""
        with patch("src.framework.graph._NEURO_SYMBOLIC_AVAILABLE", False):
            builder = _make_builder(neuro_symbolic_config={"enabled": True})
        assert builder.use_symbolic_reasoning is False

    def test_neuro_symbolic_init_from_dict(self):
        """Initialize neuro-symbolic from dict config."""
        mock_ns_config = MagicMock()
        mock_ns_config.agent.neural_confidence_weight = 0.6
        mock_ns_config.agent.symbolic_confidence_weight = 0.4

        mock_agent = MagicMock()
        mock_ext = MagicMock()
        mock_mcts_int = MagicMock()

        with patch("src.framework.graph._NEURO_SYMBOLIC_AVAILABLE", True):
            with patch("src.framework.graph.NeuroSymbolicConfig") as mock_cls:
                mock_cls.from_dict.return_value = mock_ns_config
                with patch("src.framework.graph.SymbolicReasoningAgent", return_value=mock_agent):
                    with patch("src.framework.graph.SymbolicAgentGraphExtension", return_value=mock_ext):
                        with patch("src.framework.graph.NeuroSymbolicMCTSIntegration", return_value=mock_mcts_int):
                            with patch("src.framework.graph.SymbolicAgentNodeConfig"):
                                with patch("src.framework.graph.NeuroSymbolicMCTSConfig"):
                                    builder = _make_builder(neuro_symbolic_config={"key": "val"})

        assert builder.use_symbolic_reasoning is True
        assert builder.symbolic_agent is mock_agent

    def test_neuro_symbolic_init_failure(self):
        """When init fails, symbolic reasoning disabled gracefully."""
        with patch("src.framework.graph._NEURO_SYMBOLIC_AVAILABLE", True):
            with patch("src.framework.graph.NeuroSymbolicConfig") as mock_cls:
                mock_cls.from_dict.side_effect = RuntimeError("bad config")
                builder = _make_builder(neuro_symbolic_config={"broken": True})

        assert builder.use_symbolic_reasoning is False


# ---------------------------------------------------------------------------
# _neural_fallback_for_symbolic
# ---------------------------------------------------------------------------


class TestNeuralFallback:
    """Tests for _neural_fallback_for_symbolic (lines 493-501)."""

    @pytest.mark.asyncio
    async def test_neural_fallback_success(self):
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = MagicMock(text="fallback answer")
        builder = _make_builder(model_adapter=mock_adapter)
        result = await builder._neural_fallback_for_symbolic("What is X?", {})
        assert result == "fallback answer"

    @pytest.mark.asyncio
    async def test_neural_fallback_failure(self):
        mock_adapter = AsyncMock()
        mock_adapter.generate.side_effect = RuntimeError("LLM down")
        builder = _make_builder(model_adapter=mock_adapter)
        result = await builder._neural_fallback_for_symbolic("What is X?", {})
        assert "Could not determine" in result


# ---------------------------------------------------------------------------
# _extract_meta_controller_features
# ---------------------------------------------------------------------------


class TestExtractMetaControllerFeatures:
    """Tests for _extract_meta_controller_features (lines 513-546)."""

    def test_returns_none_when_unavailable(self):
        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", False):
            builder = _make_builder()
            result = builder._extract_meta_controller_features({"query": "test"})
        assert result is None

    def test_extracts_features_from_state(self):
        mock_features_cls = MagicMock()

        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", True):
            with patch("src.framework.graph.MetaControllerFeatures", mock_features_cls):
                builder = _make_builder()
                state = {
                    "query": "test query",
                    "hrm_results": {"metadata": {"decomposition_quality_score": 0.85}},
                    "trm_results": {"metadata": {"final_quality_score": 0.9}},
                    "mcts_stats": {"best_action_value": 0.7},
                    "consensus_score": 0.5,
                    "last_routed_agent": "hrm",
                    "iteration": 2,
                    "rag_context": "some context",
                }
                result = builder._extract_meta_controller_features(state)

        assert result is not None
        mock_features_cls.assert_called_once()
        call_kwargs = mock_features_cls.call_args[1]
        assert call_kwargs["hrm_confidence"] == 0.85
        assert call_kwargs["trm_confidence"] == 0.9
        assert call_kwargs["mcts_value"] == 0.7
        assert call_kwargs["iteration"] == 2
        assert call_kwargs["has_rag_context"] is True


# ---------------------------------------------------------------------------
# _neural_route_decision
# ---------------------------------------------------------------------------


class TestNeuralRouteDecision:
    """Tests for _neural_route_decision (lines 567-600)."""

    def test_falls_back_when_features_none(self):
        builder = _make_builder()
        builder.use_neural_routing = True
        builder.meta_controller = MagicMock()

        with patch.object(builder, "_extract_meta_controller_features", return_value=None):
            with patch.object(builder, "_rule_based_route_decision", return_value="parallel") as mock_rb:
                result = builder._neural_route_decision({"query": "test", "iteration": 0})
        assert result == "parallel"
        mock_rb.assert_called_once()

    def test_routes_to_hrm_when_predicted(self):
        builder = _make_builder()
        builder.use_neural_routing = True
        mock_mc = MagicMock()
        mock_mc.predict.return_value = MagicMock(agent="hrm", confidence=0.95, probabilities={})
        builder.meta_controller = mock_mc

        mock_features = MagicMock()
        with patch.object(builder, "_extract_meta_controller_features", return_value=mock_features):
            result = builder._neural_route_decision({"query": "test", "iteration": 0})
        assert result == "hrm"

    def test_routes_to_trm_when_predicted(self):
        builder = _make_builder()
        builder.use_neural_routing = True
        mock_mc = MagicMock()
        mock_mc.predict.return_value = MagicMock(agent="trm", confidence=0.9, probabilities={})
        builder.meta_controller = mock_mc

        mock_features = MagicMock()
        with patch.object(builder, "_extract_meta_controller_features", return_value=mock_features):
            result = builder._neural_route_decision({"query": "test", "iteration": 0})
        assert result == "trm"

    def test_routes_to_mcts_when_predicted_and_enabled(self):
        builder = _make_builder()
        builder.use_neural_routing = True
        mock_mc = MagicMock()
        mock_mc.predict.return_value = MagicMock(agent="mcts", confidence=0.85, probabilities={})
        builder.meta_controller = mock_mc

        mock_features = MagicMock()
        with patch.object(builder, "_extract_meta_controller_features", return_value=mock_features):
            result = builder._neural_route_decision(
                {"query": "test", "iteration": 0, "use_mcts": True}
            )
        assert result == "mcts"

    def test_falls_back_when_agent_already_ran(self):
        builder = _make_builder()
        builder.use_neural_routing = True
        mock_mc = MagicMock()
        mock_mc.predict.return_value = MagicMock(agent="hrm", confidence=0.9, probabilities={})
        builder.meta_controller = mock_mc

        mock_features = MagicMock()
        with patch.object(builder, "_extract_meta_controller_features", return_value=mock_features):
            with patch.object(builder, "_rule_based_route_decision", return_value="aggregate") as mock_rb:
                result = builder._neural_route_decision(
                    {"query": "test", "iteration": 0, "hrm_results": {"response": "done"}}
                )
        assert result == "aggregate"

    def test_falls_back_on_exception(self):
        builder = _make_builder()
        builder.use_neural_routing = True
        builder.meta_controller = MagicMock()
        builder.meta_controller.predict.side_effect = RuntimeError("model error")

        mock_features = MagicMock()
        with patch.object(builder, "_extract_meta_controller_features", return_value=mock_features):
            with patch.object(builder, "_rule_based_route_decision", return_value="parallel"):
                result = builder._neural_route_decision({"query": "test", "iteration": 0})
        assert result == "parallel"


# ---------------------------------------------------------------------------
# _rule_based_route_decision - symbolic routing (line 621)
# ---------------------------------------------------------------------------


class TestSymbolicRouting:
    """Test symbolic routing branch in _rule_based_route_decision."""

    def test_routes_to_symbolic_when_enabled(self):
        builder = _make_builder()
        builder.use_symbolic_reasoning = True
        mock_ext = MagicMock()
        mock_ext.should_route_to_symbolic.return_value = True
        builder.symbolic_extension = mock_ext

        state = {"query": "prove theorem", "iteration": 0}
        result = builder._rule_based_route_decision(state)
        assert result == "symbolic"

    def test_skips_symbolic_when_already_done(self):
        builder = _make_builder()
        builder.use_symbolic_reasoning = True
        mock_ext = MagicMock()
        mock_ext.should_route_to_symbolic.return_value = True
        builder.symbolic_extension = mock_ext

        state = {"query": "prove theorem", "iteration": 0, "symbolic_results": {"done": True}}
        result = builder._rule_based_route_decision(state)
        # Should not return "symbolic" since symbolic_results already exist
        assert result != "symbolic"


# ---------------------------------------------------------------------------
# _symbolic_agent_node
# ---------------------------------------------------------------------------


class TestSymbolicAgentNode:
    """Tests for _symbolic_agent_node (lines 757-778)."""

    @pytest.mark.asyncio
    async def test_symbolic_agent_not_available(self):
        builder = _make_builder()
        builder.symbolic_extension = None
        result = await builder._symbolic_agent_node({"query": "test"})
        assert result["agent_outputs"][0]["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_symbolic_agent_success(self):
        builder = _make_builder()
        mock_ext = AsyncMock()
        mock_ext.handle_symbolic_node.return_value = {
            "symbolic_results": {
                "response": "proved",
                "metadata": {"proof_tree": {"nodes": []}},
            },
            "agent_outputs": [{"agent": "symbolic", "response": "proved", "confidence": 0.95}],
        }
        builder.symbolic_extension = mock_ext

        result = await builder._symbolic_agent_node({"query": "prove P"})
        assert result["symbolic_results"]["response"] == "proved"
        assert result["symbolic_proof_tree"] == {"nodes": []}


# ---------------------------------------------------------------------------
# _mcts_simulator_node
# ---------------------------------------------------------------------------


class TestMCTSSimulatorNode:
    """Tests for _mcts_simulator_node (lines 786-891)."""

    @pytest.mark.asyncio
    async def test_mcts_simulator_basic(self):
        builder = _make_builder()

        mock_stats = {
            "iterations": 50,
            "cache_hit_rate": 0.25,
            "best_action_visits": 20,
            "best_action_value": 0.8,
        }
        builder.mcts_engine = MagicMock()
        builder.mcts_engine.search = AsyncMock(return_value=("action_A", mock_stats))
        builder.mcts_engine.get_tree_depth.return_value = 3
        builder.mcts_engine.count_nodes.return_value = 100
        builder.mcts_engine.rng = MagicMock()
        builder.experiment_tracker = MagicMock()

        state = {
            "query": "What is AI?",
            "use_mcts": True,
            "hrm_results": {"metadata": {"decomposition_quality_score": 0.8}},
        }
        result = await builder._mcts_simulator_node(state)

        assert result["mcts_best_action"] == "action_A"
        assert result["mcts_stats"]["iterations"] == 50
        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0]["agent"] == "mcts"
        builder.experiment_tracker.create_result.assert_called_once()


# ---------------------------------------------------------------------------
# ADK node handler
# ---------------------------------------------------------------------------


class TestADKNodeHandler:
    """Tests for _create_adk_node_handler (lines 913-970)."""

    @pytest.mark.asyncio
    async def test_adk_handler_process_query(self):
        builder = _make_builder()
        mock_agent = AsyncMock()
        mock_agent.process_query = AsyncMock(
            return_value={"response": "result", "confidence": 0.9, "metadata": {"k": "v"}}
        )
        mock_agent.initialize = AsyncMock()
        handler = builder._create_adk_node_handler("test_agent", mock_agent)
        result = await handler({"query": "test"})
        assert result["adk_results"]["test_agent"]["response"] == "result"
        assert result["agent_outputs"][0]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_adk_handler_run_method(self):
        builder = _make_builder()
        mock_agent = AsyncMock()
        del mock_agent.process_query  # Remove process_query
        mock_agent.run = AsyncMock(return_value="string response")
        handler = builder._create_adk_node_handler("runner", mock_agent)
        result = await handler({"query": "go"})
        assert result["agent_outputs"][0]["response"] == "string response"
        assert result["agent_outputs"][0]["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_adk_handler_process_method(self):
        builder = _make_builder()
        mock_agent = AsyncMock()
        del mock_agent.process_query
        del mock_agent.run
        mock_agent.process = AsyncMock(return_value={"response": "processed", "confidence": 0.7})
        handler = builder._create_adk_node_handler("proc", mock_agent)
        result = await handler({"query": "go"})
        assert result["adk_results"]["proc"]["response"] == "processed"

    @pytest.mark.asyncio
    async def test_adk_handler_fallback(self):
        builder = _make_builder()
        mock_agent = MagicMock()
        del mock_agent.process_query
        del mock_agent.run
        del mock_agent.process
        del mock_agent.initialize
        handler = builder._create_adk_node_handler("fallback_agent", mock_agent)
        result = await handler({"query": "go"})
        assert "fallback_agent" in result["agent_outputs"][0]["response"]

    @pytest.mark.asyncio
    async def test_adk_handler_exception(self):
        builder = _make_builder()
        mock_agent = AsyncMock()
        mock_agent.process_query = AsyncMock(side_effect=RuntimeError("agent failed"))
        mock_agent.initialize = AsyncMock()
        handler = builder._create_adk_node_handler("bad_agent", mock_agent)
        result = await handler({"query": "go"})
        assert result["agent_outputs"][0]["confidence"] == 0.0
        assert "Error" in result["agent_outputs"][0]["response"]


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    """Tests for build_graph (lines 239-324)."""

    def test_build_graph_no_langgraph_raises(self):
        with patch("src.framework.graph.StateGraph", None):
            builder = _make_builder()
            with pytest.raises(ImportError, match="LangGraph not installed"):
                builder.build_graph()

    def test_build_graph_with_adk_agents(self):
        mock_sg = MagicMock()
        mock_workflow = MagicMock()
        mock_sg.return_value = mock_workflow

        with patch("src.framework.graph.StateGraph", mock_sg):
            with patch("src.framework.graph.END", "END"):
                mock_agent = MagicMock()
                builder = _make_builder(adk_agents={"deep_search": mock_agent})
                result = builder.build_graph()

        # Verify ADK node was added
        add_node_calls = [str(c) for c in mock_workflow.add_node.call_args_list]
        node_names = [c[0][0] for c in mock_workflow.add_node.call_args_list]
        assert "adk_deep_search" in node_names

    def test_build_graph_with_symbolic_reasoning(self):
        mock_sg = MagicMock()
        mock_workflow = MagicMock()
        mock_sg.return_value = mock_workflow

        with patch("src.framework.graph.StateGraph", mock_sg):
            with patch("src.framework.graph.END", "END"):
                builder = _make_builder()
                builder.use_symbolic_reasoning = True
                builder.symbolic_extension = MagicMock()
                result = builder.build_graph()

        node_names = [c[0][0] for c in mock_workflow.add_node.call_args_list]
        assert "symbolic_agent" in node_names


# ---------------------------------------------------------------------------
# IntegratedFramework
# ---------------------------------------------------------------------------


class TestIntegratedFramework:
    """Tests for IntegratedFramework (lines 1114-1163, 1186-1202)."""

    def test_init_without_agents(self):
        """IntegratedFramework initializes even when agent imports fail."""
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test_int")

        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(
                model_adapter=mock_adapter,
                logger=mock_logger,
            )
        assert fw.hrm_agent is None
        assert fw.trm_agent is None
        assert fw.app is None

    @pytest.mark.asyncio
    async def test_process_raises_when_no_langgraph(self):
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test_int")

        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)

        with pytest.raises(RuntimeError, match="LangGraph not available"):
            await fw.process("test query")

    @pytest.mark.asyncio
    async def test_process_success(self):
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test_int")

        mock_app = AsyncMock()
        mock_app.ainvoke.return_value = {
            "final_response": "answer",
            "metadata": {"agents_used": ["hrm"]},
        }

        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)
        fw.app = mock_app

        result = await fw.process("test query", use_rag=False, use_mcts=False)
        assert result["response"] == "answer"
        assert result["metadata"]["agents_used"] == ["hrm"]

    @pytest.mark.asyncio
    async def test_astream_raises_when_no_langgraph(self):
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test_int")

        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)

        with pytest.raises(RuntimeError, match="LangGraph not available"):
            async for _ in fw.astream("test"):
                pass

    @pytest.mark.asyncio
    async def test_astream_events_raises_when_no_langgraph(self):
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test_int")

        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)

        with pytest.raises(RuntimeError, match="LangGraph not available"):
            async for _ in fw.astream_events("test"):
                pass

    @pytest.mark.asyncio
    async def test_astream_events_error_yields_error_event(self):
        """When streaming fails, an error event is yielded."""
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test_int")

        mock_app = MagicMock()

        async def _failing_stream(*args, **kwargs):
            raise RuntimeError("stream broke")
            yield  # make it an async generator

        mock_app.astream_events = _failing_stream

        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)
        fw.app = mock_app
        fw.graph_builder = MagicMock()
        fw.graph_builder.max_iterations = 3

        events = []
        async for event in fw.astream_events("test"):
            events.append(event)

        assert len(events) == 1
        assert events[0]["event_type"] == "on_error"
        assert "stream broke" in events[0]["data"]["error"]

    @pytest.mark.asyncio
    async def test_astream_events_with_llm_stream(self):
        """Test token extraction from on_llm_stream events."""
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test_int")

        chunk_obj = MagicMock()
        chunk_obj.content = "hello"

        async def _mock_stream(*args, **kwargs):
            yield {"event": "on_llm_stream", "name": "generate", "run_id": "r1",
                   "data": {"chunk": chunk_obj}, "metadata": {}, "tags": []}

        mock_app = MagicMock()
        mock_app.astream_events = _mock_stream

        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)
        fw.app = mock_app
        fw.graph_builder = MagicMock()
        fw.graph_builder.max_iterations = 3

        events = []
        async for event in fw.astream_events("test"):
            events.append(event)

        assert len(events) == 1
        assert events[0]["token"] == "hello"

    @pytest.mark.asyncio
    async def test_astream_events_dict_chunk(self):
        """Test token extraction from dict chunk in on_llm_stream."""
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test_int")

        async def _mock_stream(*args, **kwargs):
            yield {"event": "on_llm_stream", "name": "gen", "run_id": "r2",
                   "data": {"chunk": {"content": "world"}}, "metadata": {}, "tags": []}

        mock_app = MagicMock()
        mock_app.astream_events = _mock_stream

        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)
        fw.app = mock_app
        fw.graph_builder = MagicMock()
        fw.graph_builder.max_iterations = 3

        events = []
        async for event in fw.astream_events("test"):
            events.append(event)

        assert events[0]["token"] == "world"


# ---------------------------------------------------------------------------
# Visualization: get_experiment_tracker, set_mcts_seed
# ---------------------------------------------------------------------------


class TestFrameworkHelpers:
    """Tests for get_experiment_tracker, set_mcts_seed (lines 1348, 1352-1353)."""

    def test_get_experiment_tracker(self):
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test")
        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)
        tracker = fw.get_experiment_tracker()
        assert tracker is fw.graph_builder.experiment_tracker

    def test_set_mcts_seed(self):
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test")
        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)
        fw.graph_builder.mcts_engine = MagicMock()
        fw.graph_builder.mcts_config = MagicMock()
        fw.set_mcts_seed(123)
        fw.graph_builder.mcts_engine.reset_seed.assert_called_with(123)
        assert fw.graph_builder.mcts_config.seed == 123


# ---------------------------------------------------------------------------
# Visualization: get_graph_structure, get_graph_mermaid, draw_mermaid
# ---------------------------------------------------------------------------


class TestVisualization:
    """Tests for visualization methods."""

    def _make_fw(self):
        mock_adapter = AsyncMock()
        mock_logger = logging.getLogger("test")
        with patch("src.framework.graph.StateGraph", None):
            fw = IntegratedFramework(model_adapter=mock_adapter, logger=mock_logger)
        return fw

    def test_get_graph_structure_basic(self):
        fw = self._make_fw()
        structure = fw.get_graph_structure()
        assert "nodes" in structure
        assert "edges" in structure
        assert "conditional_edges" in structure
        assert structure["entry_point"] == "entry"
        node_ids = [n["id"] for n in structure["nodes"]]
        assert "entry" in node_ids
        assert "synthesize" in node_ids

    def test_get_graph_structure_with_adk(self):
        fw = self._make_fw()
        fw.graph_builder.adk_agents = {"deep_search": MagicMock()}
        structure = fw.get_graph_structure()
        node_ids = [n["id"] for n in structure["nodes"]]
        assert "adk_deep_search" in node_ids

    def test_get_graph_structure_with_symbolic(self):
        fw = self._make_fw()
        fw.graph_builder.use_symbolic_reasoning = True
        structure = fw.get_graph_structure()
        node_ids = [n["id"] for n in structure["nodes"]]
        assert "symbolic_agent" in node_ids

    def test_get_graph_mermaid(self):
        fw = self._make_fw()
        mermaid = fw.get_graph_mermaid()
        assert "flowchart TD" in mermaid
        assert "entry" in mermaid
        assert "synthesize" in mermaid

    def test_get_graph_mermaid_with_theme(self):
        fw = self._make_fw()
        mermaid = fw.get_graph_mermaid(theme="dark")
        assert "'dark'" in mermaid

    def test_get_graph_mermaid_without_descriptions(self):
        fw = self._make_fw()
        mermaid = fw.get_graph_mermaid(include_descriptions=False)
        assert "flowchart TD" in mermaid
        # descriptions like "Input validation" should NOT appear
        assert "<small>" not in mermaid

    def test_draw_mermaid_no_output_file(self):
        fw = self._make_fw()
        code = fw.draw_mermaid()
        assert "flowchart TD" in code

    def test_draw_mermaid_render_failure(self):
        fw = self._make_fw()
        with patch("src.framework.graph.IntegratedFramework.get_graph_mermaid", return_value="flowchart TD"):
            with pytest.raises(RuntimeError, match="Diagram rendering failed"):
                with patch.dict("sys.modules", {"httpx": MagicMock()}):
                    import importlib
                    mock_httpx = MagicMock()
                    mock_client = MagicMock()
                    mock_client.__enter__ = MagicMock(return_value=mock_client)
                    mock_client.__exit__ = MagicMock(return_value=False)
                    mock_client.get.side_effect = RuntimeError("network error")
                    mock_httpx.Client.return_value = mock_client
                    with patch.dict("sys.modules", {"httpx": mock_httpx}):
                        fw.draw_mermaid(output_file="/tmp/test.png")

    def test_get_execution_trace_mermaid(self):
        fw = self._make_fw()
        trace = fw.get_execution_trace_mermaid(
            execution_path=["entry", "retrieve_context", "parallel_agents", "synthesize"],
            timings={"entry": 5.0, "retrieve_context": 120.5},
        )
        assert "sequenceDiagram" in trace
        assert "autonumber" in trace
        assert "Entry" in trace
        assert "120.5ms" in trace

    def test_get_execution_trace_mermaid_no_timings(self):
        fw = self._make_fw()
        trace = fw.get_execution_trace_mermaid(
            execution_path=["entry", "synthesize"],
        )
        assert "sequenceDiagram" in trace
        assert "Start" in trace
        assert "End" in trace
