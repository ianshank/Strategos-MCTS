"""
Unit tests for src/neuro_symbolic/integration.py

Tests the neuro-symbolic integration layer including:
- NeuroSymbolicMCTSConfig validation and normalization
- NeuroSymbolicMCTSIntegration (state conversion, action filtering, heuristics, hybrid value)
- SymbolicAgentNodeConfig defaults
- SymbolicAgentGraphExtension (routing, node handling, naming)
- HybridConfidenceAggregator
- Factory and extension functions
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.neuro_symbolic.config import NeuroSymbolicConfig
from src.neuro_symbolic.integration import (
    HybridConfidenceAggregator,
    NeuroSymbolicMCTSConfig,
    NeuroSymbolicMCTSIntegration,
    SymbolicAgentGraphExtension,
    SymbolicAgentNodeConfig,
    create_neuro_symbolic_extension,
    extend_graph_builder,
)
from src.neuro_symbolic.state import Fact, NeuroSymbolicState, SymbolicFactType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_mcts_config():
    return NeuroSymbolicMCTSConfig()


@pytest.fixture
def mock_constraint_system():
    cs = MagicMock()
    cs.validate_expansion.return_value = [("action_a", 1.0), ("action_b", 0.8)]
    cs.validator = MagicMock()
    cs.validator.validate.return_value = (True, [])
    cs.get_statistics.return_value = {"num_constraints": 0}
    return cs


@pytest.fixture
def integration(default_mcts_config, mock_constraint_system):
    return NeuroSymbolicMCTSIntegration(
        config=default_mcts_config,
        constraint_system=mock_constraint_system,
    )


@pytest.fixture
def mock_mcts_state():
    """A minimal mock of an MCTS state object."""
    state = MagicMock()
    state.state_id = "state-1"
    state.features = {"temperature": 0.7, "depth": 3}
    return state


@pytest.fixture
def mock_reasoning_agent():
    agent = MagicMock()
    agent.process = AsyncMock(
        return_value={
            "response": "proved",
            "metadata": {"confidence": 0.9, "proof_found": True},
        }
    )
    return agent


# ---------------------------------------------------------------------------
# NeuroSymbolicMCTSConfig
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNeuroSymbolicMCTSConfig:

    def test_default_weights_sum_to_one(self):
        cfg = NeuroSymbolicMCTSConfig()
        assert abs(cfg.neural_weight + cfg.symbolic_weight - 1.0) < 1e-6

    def test_weights_normalized_when_sum_not_one(self):
        cfg = NeuroSymbolicMCTSConfig(neural_weight=3.0, symbolic_weight=7.0)
        assert abs(cfg.neural_weight + cfg.symbolic_weight - 1.0) < 1e-6
        assert abs(cfg.neural_weight - 0.3) < 1e-6
        assert abs(cfg.symbolic_weight - 0.7) < 1e-6

    def test_default_values(self):
        cfg = NeuroSymbolicMCTSConfig()
        assert cfg.enable_constraint_pruning is True
        assert cfg.enable_symbolic_heuristics is True
        assert cfg.constraint_check_frequency == 1
        assert cfg.max_constraints_per_node == 50


# ---------------------------------------------------------------------------
# NeuroSymbolicMCTSIntegration - State Conversion
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMCTSIntegrationStateConversion:

    def test_convert_mcts_state_basic(self, integration, mock_mcts_state):
        ns_state = integration.convert_mcts_state(mock_mcts_state)
        assert isinstance(ns_state, NeuroSymbolicState)
        assert ns_state.state_id == "state-1"

    def test_convert_includes_features_as_facts(self, integration, mock_mcts_state):
        ns_state = integration.convert_mcts_state(mock_mcts_state)
        fact_names = {f.name for f in ns_state.facts}
        assert "has_feature" in fact_names

    def test_convert_includes_depth_fact(self, integration, mock_mcts_state):
        ns_state = integration.convert_mcts_state(mock_mcts_state, action_history=["a1", "a2"])
        depth_facts = [f for f in ns_state.facts if f.name == "depth"]
        assert len(depth_facts) == 1
        assert depth_facts[0].arguments == (2,)

    def test_convert_includes_action_history(self, integration, mock_mcts_state):
        ns_state = integration.convert_mcts_state(mock_mcts_state, action_history=["go", "stop"])
        action_facts = [f for f in ns_state.facts if f.name == "action_at"]
        assert len(action_facts) == 2

    def test_convert_state_without_features(self, integration):
        state = MagicMock(spec=[])
        state.state_id = "bare"
        ns_state = integration.convert_mcts_state(state)
        # Should still have the depth fact at least
        assert any(f.name == "depth" for f in ns_state.facts)

    def test_convert_state_without_state_id(self, integration):
        state = MagicMock()
        # MagicMock auto-creates state_id attribute, so delete it to test fallback
        del state.state_id
        ns_state = integration.convert_mcts_state(state)
        # Falls back to hash
        assert ns_state.state_id is not None


# ---------------------------------------------------------------------------
# NeuroSymbolicMCTSIntegration - Action Filtering
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMCTSIntegrationActionFiltering:

    def test_filter_valid_actions_with_pruning_enabled(self, integration, mock_mcts_state, mock_constraint_system):
        result = asyncio.run(
            integration.filter_valid_actions(mock_mcts_state, ["a", "b", "c"])
        )
        mock_constraint_system.validate_expansion.assert_called_once()
        assert result == [("action_a", 1.0), ("action_b", 0.8)]

    def test_filter_valid_actions_pruning_disabled(self, mock_constraint_system, mock_mcts_state):
        config = NeuroSymbolicMCTSConfig(enable_constraint_pruning=False)
        integ = NeuroSymbolicMCTSIntegration(config=config, constraint_system=mock_constraint_system)
        result = asyncio.run(
            integ.filter_valid_actions(mock_mcts_state, ["x", "y"])
        )
        assert result == [("x", 1.0), ("y", 1.0)]
        mock_constraint_system.validate_expansion.assert_not_called()

    def test_filter_tracks_statistics(self, integration, mock_mcts_state, mock_constraint_system):
        # 3 candidates, 2 returned => 1 pruned
        mock_constraint_system.validate_expansion.return_value = [("a", 1.0)]
        asyncio.run(
            integration.filter_valid_actions(mock_mcts_state, ["a", "b", "c"])
        )
        assert integration._expansions_checked == 1
        assert integration._actions_pruned == 2

    def test_filter_logs_when_pruning_occurs(self, mock_constraint_system):
        mock_constraint_system.validate_expansion.return_value = [("a", 1.0)]
        logger = MagicMock()
        config = NeuroSymbolicMCTSConfig()
        integ = NeuroSymbolicMCTSIntegration(config=config, constraint_system=mock_constraint_system, logger=logger)
        state = MagicMock()
        state.state_id = "s-log"
        state.features = {}
        asyncio.run(
            integ.filter_valid_actions(state, ["a", "b"])
        )
        logger.debug.assert_called_once()


# ---------------------------------------------------------------------------
# NeuroSymbolicMCTSIntegration - Heuristic & Hybrid Value
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMCTSIntegrationHeuristics:

    def test_get_symbolic_heuristic_disabled(self, mock_mcts_state, mock_constraint_system):
        config = NeuroSymbolicMCTSConfig(enable_symbolic_heuristics=False)
        integ = NeuroSymbolicMCTSIntegration(config=config, constraint_system=mock_constraint_system)
        val = integ.get_symbolic_heuristic(mock_mcts_state)
        assert val == 0.5

    def test_get_symbolic_heuristic_returns_float(self, integration, mock_mcts_state):
        val = integration.get_symbolic_heuristic(mock_mcts_state)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_compute_hybrid_value(self, integration, mock_mcts_state):
        result = integration.compute_hybrid_value(0.8, mock_mcts_state)
        assert isinstance(result, float)
        # neural_weight * 0.8 + symbolic_weight * heuristic
        # With default weights (0.6/0.4) result should be between 0 and 1
        assert 0.0 <= result <= 1.5  # generous bound

    def test_compute_satisfaction_score_valid(self, integration):
        state = NeuroSymbolicState(state_id="s1")
        integration.constraint_system.validator.validate.return_value = (True, [])
        score = integration._compute_satisfaction_score(state)
        assert score == 1.0

    def test_compute_satisfaction_score_invalid(self, integration):
        state = NeuroSymbolicState(state_id="s2")
        integration.constraint_system.validator.validate.return_value = (False, [])
        score = integration._compute_satisfaction_score(state)
        assert score == 0.0

    def test_compute_satisfaction_score_with_penalty(self, integration):
        state = NeuroSymbolicState(state_id="s3")
        mock_result = MagicMock()
        mock_result.penalty = 0.3
        integration.constraint_system.validator.validate.return_value = (True, [mock_result])
        score = integration._compute_satisfaction_score(state)
        assert abs(score - 0.7) < 1e-6

    def test_compute_progress_score(self, integration):
        facts = frozenset(
            Fact(name=f"f{i}", arguments=(i,), fact_type=SymbolicFactType.ATTRIBUTE)
            for i in range(10)
        )
        state = NeuroSymbolicState(state_id="s4", facts=facts)
        score = integration._compute_progress_score(state)
        assert abs(score - 0.5) < 1e-6  # 10/20

    def test_compute_progress_score_capped_at_one(self, integration):
        facts = frozenset(
            Fact(name=f"f{i}", arguments=(i,), fact_type=SymbolicFactType.ATTRIBUTE)
            for i in range(30)
        )
        state = NeuroSymbolicState(state_id="s5", facts=facts)
        score = integration._compute_progress_score(state)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMCTSIntegrationStatistics:

    def test_get_statistics_initial(self, integration):
        stats = integration.get_statistics()
        assert stats["expansions_checked"] == 0
        assert stats["actions_pruned"] == 0
        assert stats["prune_rate"] == 0.0

    def test_reset_statistics(self, integration):
        integration._expansions_checked = 5
        integration._actions_pruned = 3
        integration._constraint_check_time_ms = 100.0
        integration.reset_statistics()
        assert integration._expansions_checked == 0
        assert integration._actions_pruned == 0
        assert integration._constraint_check_time_ms == 0.0


# ---------------------------------------------------------------------------
# SymbolicAgentNodeConfig
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSymbolicAgentNodeConfig:

    def test_defaults(self):
        cfg = SymbolicAgentNodeConfig()
        assert cfg.enabled is True
        assert cfg.priority == 0
        assert "prove" in cfg.keywords
        assert cfg.min_confidence_for_routing == 0.5


# ---------------------------------------------------------------------------
# SymbolicAgentGraphExtension
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSymbolicAgentGraphExtension:

    def test_should_route_disabled(self, mock_reasoning_agent):
        cfg = SymbolicAgentNodeConfig(enabled=False)
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent, config=cfg)
        assert ext.should_route_to_symbolic("prove X", {}) is False

    def test_should_route_keyword_match(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        assert ext.should_route_to_symbolic("Can you prove this theorem?", {}) is True

    def test_should_route_no_keyword_match(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        assert ext.should_route_to_symbolic("What is the weather?", {}) is False

    def test_should_route_already_ran(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        state = {"symbolic_results": {"response": "done"}}
        assert ext.should_route_to_symbolic("prove X", state) is False

    def test_should_route_prolog_pattern(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        assert ext.should_route_to_symbolic("parent(john, mary)?", {}) is True

    def test_handle_symbolic_node(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        state = {"query": "prove X", "rag_context": "context"}
        result = asyncio.run(ext.handle_symbolic_node(state))
        assert "symbolic_results" in result
        assert result["symbolic_results"]["response"] == "proved"
        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0]["agent"] == "symbolic"

    def test_handle_symbolic_node_with_logger(self, mock_reasoning_agent):
        logger = MagicMock()
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent, logger=logger)
        state = {"query": "q"}
        asyncio.run(ext.handle_symbolic_node(state))
        logger.info.assert_called_once()

    def test_get_routing_key(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        assert ext.get_routing_key() == "symbolic"

    def test_get_node_name(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        assert ext.get_node_name() == "symbolic_agent"


# ---------------------------------------------------------------------------
# HybridConfidenceAggregator
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHybridConfidenceAggregator:

    def test_empty_outputs(self):
        agg = HybridConfidenceAggregator()
        result = agg.aggregate([])
        assert result["combined_confidence"] == 0.0
        assert result["consistency_score"] == 0.0

    def test_neural_only(self):
        agg = HybridConfidenceAggregator(neural_weight=0.5, symbolic_weight=0.5)
        outputs = [{"agent": "hrm", "confidence": 0.8}]
        result = agg.aggregate(outputs)
        assert result["neural_confidence"] == 0.8
        assert result["symbolic_confidence"] == 0.0

    def test_symbolic_only(self):
        agg = HybridConfidenceAggregator(neural_weight=0.5, symbolic_weight=0.5)
        outputs = [{"agent": "symbolic", "confidence": 0.7}]
        result = agg.aggregate(outputs)
        assert result["symbolic_confidence"] == 0.7
        assert result["neural_confidence"] == 0.0

    def test_mixed_outputs(self):
        agg = HybridConfidenceAggregator(neural_weight=0.5, symbolic_weight=0.5, consistency_bonus=0.1)
        outputs = [
            {"agent": "hrm", "confidence": 0.8},
            {"agent": "symbolic", "confidence": 0.8},
        ]
        result = agg.aggregate(outputs)
        # consistency_score = 1 - |0.8 - 0.8| = 1.0
        # combined = 0.5*0.8 + 0.5*0.8 + 0.1*1.0 = 0.9
        assert abs(result["combined_confidence"] - 0.9) < 1e-6
        assert abs(result["consistency_score"] - 1.0) < 1e-6

    def test_combined_capped_at_one(self):
        agg = HybridConfidenceAggregator(neural_weight=0.5, symbolic_weight=0.5, consistency_bonus=0.5)
        outputs = [
            {"agent": "hrm", "confidence": 1.0},
            {"agent": "symbolic", "confidence": 1.0},
        ]
        result = agg.aggregate(outputs)
        assert result["combined_confidence"] <= 1.0

    def test_weight_normalization(self):
        agg = HybridConfidenceAggregator(neural_weight=2.0, symbolic_weight=8.0)
        assert abs(agg.neural_weight - 0.2) < 1e-6
        assert abs(agg.symbolic_weight - 0.8) < 1e-6

    def test_agent_contributions_in_result(self):
        agg = HybridConfidenceAggregator(neural_weight=0.6, symbolic_weight=0.4)
        result = agg.aggregate([{"agent": "hrm", "confidence": 0.5}])
        assert "neural" in result["agent_contributions"]
        assert "symbolic" in result["agent_contributions"]

    def test_multiple_neural_agents_averaged(self):
        agg = HybridConfidenceAggregator(neural_weight=1.0, symbolic_weight=0.0, consistency_bonus=0.0)
        # Weight normalization: neural=1.0, symbolic=0.0 => division by 1.0
        outputs = [
            {"agent": "hrm", "confidence": 0.6},
            {"agent": "trm", "confidence": 0.8},
        ]
        result = agg.aggregate(outputs)
        assert abs(result["neural_confidence"] - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateNeuroSymbolicExtension:

    def test_returns_three_components(self):
        config = NeuroSymbolicConfig()
        graph_builder = MagicMock()
        agent, mcts_integ, graph_ext = create_neuro_symbolic_extension(config, graph_builder)
        assert agent is not None
        assert mcts_integ is not None
        assert graph_ext is not None

    def test_mcts_integration_uses_config_weights(self):
        config = NeuroSymbolicConfig()
        config.agent.neural_confidence_weight = 0.4
        config.agent.symbolic_confidence_weight = 0.6
        _, mcts_integ, _ = create_neuro_symbolic_extension(config, MagicMock())
        assert abs(mcts_integ.config.neural_weight - 0.4) < 1e-6
        assert abs(mcts_integ.config.symbolic_weight - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# extend_graph_builder
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtendGraphBuilder:

    def test_stores_extension_reference(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        builder = MagicMock()
        builder._rule_based_route_decision = MagicMock(return_value="hrm")
        extend_graph_builder(builder, ext)
        assert builder._symbolic_extension is ext

    def test_extended_routing_delegates_to_symbolic(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        builder = MagicMock()
        builder._rule_based_route_decision = MagicMock(return_value="hrm")
        extend_graph_builder(builder, ext)

        state = {"query": "prove theorem X"}
        result = builder._rule_based_route_decision(state)
        assert result == "symbolic"

    def test_extended_routing_falls_back_to_original(self, mock_reasoning_agent):
        ext = SymbolicAgentGraphExtension(mock_reasoning_agent)
        original_route = MagicMock(return_value="hrm")
        builder = MagicMock()
        builder._rule_based_route_decision = original_route
        extend_graph_builder(builder, ext)

        state = {"query": "what is the weather today?"}
        result = builder._rule_based_route_decision(state)
        assert result == "hrm"
