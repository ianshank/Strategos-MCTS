"""
Integration tests for neuro-symbolic module.

Tests:
- MCTS integration with constraint pruning
- Graph extension with symbolic agent
- Hybrid confidence aggregation
- End-to-end neuro-symbolic reasoning

Best Practices 2025:
- Integration test isolation
- Mock external dependencies
- Realistic scenarios
"""

from unittest.mock import MagicMock

import pytest

from src.neuro_symbolic.config import (
    ConstraintConfig,
    ConstraintEnforcement,
    get_default_config,
)
from src.neuro_symbolic.constraints import ConstraintSystem
from src.neuro_symbolic.integration import (
    HybridConfidenceAggregator,
    NeuroSymbolicMCTSConfig,
    NeuroSymbolicMCTSIntegration,
    SymbolicAgentGraphExtension,
    SymbolicAgentNodeConfig,
    create_neuro_symbolic_extension,
    extend_graph_builder,
)
from src.neuro_symbolic.reasoning import SymbolicReasoningAgent
from src.neuro_symbolic.state import Fact, NeuroSymbolicState


class TestNeuroSymbolicMCTSIntegration:
    """Tests for MCTS integration."""

    @pytest.fixture
    def mcts_integration(self):
        """Create MCTS integration instance."""
        config = NeuroSymbolicMCTSConfig()
        return NeuroSymbolicMCTSIntegration(config)

    def test_convert_mcts_state(self, mcts_integration):
        """Test MCTS state conversion."""
        # Mock MCTS state
        mcts_state = MagicMock()
        mcts_state.state_id = "root"
        mcts_state.features = {"depth": 0, "query": "test"}

        ns_state = mcts_integration.convert_mcts_state(mcts_state)

        assert isinstance(ns_state, NeuroSymbolicState)
        assert ns_state.state_id == "root"
        assert ns_state.has_fact("has_feature", "depth", 0)
        assert ns_state.has_fact("has_feature", "query", "test")

    def test_convert_mcts_state_with_history(self, mcts_integration):
        """Test MCTS state conversion with action history."""
        mcts_state = MagicMock()
        mcts_state.state_id = "node_1"
        mcts_state.features = {}

        action_history = ["init", "explore", "evaluate"]
        ns_state = mcts_integration.convert_mcts_state(mcts_state, action_history)

        # Should have action_at facts
        assert ns_state.has_fact("action_at", 0, "init")
        assert ns_state.has_fact("action_at", 1, "explore")
        assert ns_state.has_fact("action_at", 2, "evaluate")
        assert ns_state.has_fact("depth", 3)

    @pytest.mark.asyncio
    async def test_filter_valid_actions_no_constraints(self, mcts_integration):
        """Test action filtering with no constraints."""
        mcts_state = MagicMock()
        mcts_state.state_id = "root"
        mcts_state.features = {}

        actions = ["action_a", "action_b", "action_c"]
        valid = await mcts_integration.filter_valid_actions(mcts_state, actions)

        # All actions should be valid with no constraints
        assert len(valid) == 3
        for _action, score in valid:
            assert score == 1.0

    @pytest.mark.asyncio
    async def test_filter_valid_actions_with_constraints(self):
        """Test action filtering with constraints."""
        config = NeuroSymbolicMCTSConfig()
        constraint_system = ConstraintSystem(ConstraintConfig())

        # Add constraint requiring "ready" fact
        constraint_system.register_predicate_constraint(
            constraint_id="c1",
            name="require_ready",
            required_facts=[("ready", ())],
            enforcement=ConstraintEnforcement.HARD,
        )

        integration = NeuroSymbolicMCTSIntegration(
            config=config,
            constraint_system=constraint_system,
        )

        # State without "ready" fact
        mcts_state = MagicMock()
        mcts_state.state_id = "root"
        mcts_state.features = {}

        actions = ["action_a", "action_b"]
        valid = await integration.filter_valid_actions(mcts_state, actions)

        # Actions should be pruned due to constraint violation
        assert len(valid) == 0

    @pytest.mark.asyncio
    async def test_filter_valid_actions_disabled(self):
        """Test action filtering when disabled."""
        config = NeuroSymbolicMCTSConfig(enable_constraint_pruning=False)
        integration = NeuroSymbolicMCTSIntegration(config=config)

        mcts_state = MagicMock()
        mcts_state.state_id = "root"
        mcts_state.features = {}

        actions = ["action_a", "action_b"]
        valid = await integration.filter_valid_actions(mcts_state, actions)

        # All actions should pass when pruning disabled
        assert len(valid) == 2

    def test_get_symbolic_heuristic(self, mcts_integration):
        """Test symbolic heuristic computation."""
        mcts_state = MagicMock()
        mcts_state.state_id = "root"
        mcts_state.features = {}

        heuristic = mcts_integration.get_symbolic_heuristic(mcts_state)

        # Should return value in [0, 1]
        assert 0.0 <= heuristic <= 1.0

    def test_compute_hybrid_value(self, mcts_integration):
        """Test hybrid value computation."""
        mcts_state = MagicMock()
        mcts_state.state_id = "root"
        mcts_state.features = {}

        neural_value = 0.8
        hybrid = mcts_integration.compute_hybrid_value(neural_value, mcts_state)

        # Should combine neural and symbolic
        assert 0.0 <= hybrid <= 1.0

    def test_get_statistics(self, mcts_integration):
        """Test statistics retrieval."""
        stats = mcts_integration.get_statistics()

        assert "expansions_checked" in stats
        assert "actions_pruned" in stats
        assert "prune_rate" in stats
        assert "constraint_stats" in stats

    def test_reset_statistics(self, mcts_integration):
        """Test statistics reset."""
        mcts_integration._expansions_checked = 10
        mcts_integration._actions_pruned = 5

        mcts_integration.reset_statistics()

        assert mcts_integration._expansions_checked == 0
        assert mcts_integration._actions_pruned == 0


class TestSymbolicAgentGraphExtension:
    """Tests for graph extension."""

    @pytest.fixture
    def graph_extension(self):
        """Create graph extension."""
        config = get_default_config()
        agent = SymbolicReasoningAgent(config)
        return SymbolicAgentGraphExtension(
            reasoning_agent=agent,
            config=SymbolicAgentNodeConfig(),
        )

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("prove that X is valid", True),
            ("what is the logic behind this?", True),
            ("apply rule A to B", True),
            ("what constraint applies?", True),
            ("why is this true?", True),
            ("parent(john, mary)?", True),  # Prolog-style
            ("simple question", False),
            ("calculate 2+2", False),
        ],
    )
    def test_should_route_to_symbolic(self, graph_extension, query, expected):
        """Test routing decision logic."""
        state = {"query": query}

        result = graph_extension.should_route_to_symbolic(query, state)

        assert result == expected

    def test_should_not_route_if_disabled(self):
        """Test routing respects disabled flag."""
        config = get_default_config()
        agent = SymbolicReasoningAgent(config)
        extension = SymbolicAgentGraphExtension(
            reasoning_agent=agent,
            config=SymbolicAgentNodeConfig(enabled=False),
        )

        result = extension.should_route_to_symbolic("prove X", {})

        assert result is False

    def test_should_not_route_if_already_ran(self, graph_extension):
        """Test routing prevents duplicate runs."""
        state = {"query": "prove X", "symbolic_results": {"response": "done"}}

        result = graph_extension.should_route_to_symbolic("prove X", state)

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_symbolic_node(self, graph_extension):
        """Test symbolic node handler."""
        state = {
            "query": "is socrates a human?",
            "rag_context": "Socrates is a human.",
        }

        result = await graph_extension.handle_symbolic_node(state)

        assert "symbolic_results" in result
        assert "agent_outputs" in result
        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0]["agent"] == "symbolic"

    def test_get_routing_key(self, graph_extension):
        """Test routing key."""
        assert graph_extension.get_routing_key() == "symbolic"

    def test_get_node_name(self, graph_extension):
        """Test node name."""
        assert graph_extension.get_node_name() == "symbolic_agent"


class TestHybridConfidenceAggregator:
    """Tests for hybrid confidence aggregation."""

    def test_aggregator_creation(self):
        """Test aggregator initialization with normalization."""
        aggregator = HybridConfidenceAggregator(
            neural_weight=0.6,
            symbolic_weight=0.8,
        )

        total = aggregator.neural_weight + aggregator.symbolic_weight
        assert abs(total - 1.0) < 1e-6

    def test_aggregate_empty(self):
        """Test aggregation with no outputs."""
        aggregator = HybridConfidenceAggregator()

        result = aggregator.aggregate([])

        assert result["combined_confidence"] == 0.0
        assert result["consistency_score"] == 0.0

    def test_aggregate_neural_only(self):
        """Test aggregation with only neural outputs."""
        aggregator = HybridConfidenceAggregator(
            neural_weight=0.6,
            symbolic_weight=0.4,
        )

        outputs = [
            {"agent": "hrm", "confidence": 0.8},
            {"agent": "trm", "confidence": 0.9},
        ]

        result = aggregator.aggregate(outputs)

        assert abs(result["neural_confidence"] - 0.85) < 1e-6
        assert result["symbolic_confidence"] == 0.0

    def test_aggregate_symbolic_only(self):
        """Test aggregation with only symbolic outputs."""
        aggregator = HybridConfidenceAggregator(
            neural_weight=0.5,
            symbolic_weight=0.5,
        )

        outputs = [
            {"agent": "symbolic", "confidence": 0.9},
        ]

        result = aggregator.aggregate(outputs)

        assert result["symbolic_confidence"] == 0.9
        assert result["neural_confidence"] == 0.0

    def test_aggregate_mixed(self):
        """Test aggregation with mixed outputs."""
        aggregator = HybridConfidenceAggregator(
            neural_weight=0.5,
            symbolic_weight=0.5,
            consistency_bonus=0.1,
        )

        outputs = [
            {"agent": "hrm", "confidence": 0.8},
            {"agent": "symbolic", "confidence": 0.8},
        ]

        result = aggregator.aggregate(outputs)

        # Same confidence = high consistency
        assert result["consistency_score"] == 1.0
        assert result["combined_confidence"] > 0.8

    def test_aggregate_low_consistency(self):
        """Test aggregation with low consistency."""
        aggregator = HybridConfidenceAggregator(
            neural_weight=0.5,
            symbolic_weight=0.5,
            consistency_bonus=0.1,
        )

        outputs = [
            {"agent": "hrm", "confidence": 0.9},
            {"agent": "symbolic", "confidence": 0.3},
        ]

        result = aggregator.aggregate(outputs)

        # Large difference = low consistency
        assert result["consistency_score"] < 0.5


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_neuro_symbolic_extension(self):
        """Test creating all neuro-symbolic components."""
        config = get_default_config()
        mock_graph_builder = MagicMock()
        mock_logger = MagicMock()

        agent, mcts_integration, graph_extension = create_neuro_symbolic_extension(
            config=config,
            graph_builder=mock_graph_builder,
            logger=mock_logger,
        )

        assert isinstance(agent, SymbolicReasoningAgent)
        assert isinstance(mcts_integration, NeuroSymbolicMCTSIntegration)
        assert isinstance(graph_extension, SymbolicAgentGraphExtension)


class TestGraphBuilderExtension:
    """Tests for extending GraphBuilder."""

    def test_extend_graph_builder(self):
        """Test extending graph builder with symbolic agent."""
        config = get_default_config()
        agent = SymbolicReasoningAgent(config)
        extension = SymbolicAgentGraphExtension(
            reasoning_agent=agent,
            config=SymbolicAgentNodeConfig(),
        )

        # Mock graph builder
        mock_builder = MagicMock()
        mock_builder._rule_based_route_decision = MagicMock(return_value="aggregate")

        extend_graph_builder(mock_builder, extension)

        # Extension should be stored
        assert mock_builder._symbolic_extension == extension

        # Test routing with symbolic query
        mock_builder._rule_based_route_decision({"query": "prove X"})


class TestEndToEndScenarios:
    """End-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(self):
        """Test full neuro-symbolic reasoning pipeline."""
        config = get_default_config()
        agent = SymbolicReasoningAgent(config)

        # Add knowledge
        agent.add_knowledge(
            facts=[],
            rules=[
                ("mortality", ("mortal", ("?X",)), [("human", ("?X",))]),
            ],
        )

        # Process query with facts
        state = NeuroSymbolicState(
            state_id="test",
            facts=frozenset([
                Fact(name="human", arguments=("socrates",)),
            ]),
        )

        result = await agent.process(
            query="mortal(socrates)?",
            state=state,
        )

        assert "response" in result
        assert result["metadata"]["agent"] == "symbolic"

    @pytest.mark.asyncio
    async def test_mcts_constraint_integration(self):
        """Test MCTS with constraint integration."""
        config = NeuroSymbolicMCTSConfig()
        constraint_system = ConstraintSystem(ConstraintConfig())

        # Add temporal constraint
        constraint_system.register_temporal_constraint(
            constraint_id="order",
            name="init_first",
            must_precede=[("init", "start")],
            enforcement=ConstraintEnforcement.HARD,
        )

        integration = NeuroSymbolicMCTSIntegration(
            config=config,
            constraint_system=constraint_system,
        )

        # State with init action done
        mcts_state = MagicMock()
        mcts_state.state_id = "node_1"
        mcts_state.features = {}

        action_history = ["init"]
        valid = await integration.filter_valid_actions(
            mcts_state,
            ["start", "stop", "restart"],
            action_history,
        )

        # "start" should be valid since "init" was done
        action_names = [a for a, _ in valid]
        assert "start" in action_names

    @pytest.mark.asyncio
    async def test_graph_extension_with_rag(self):
        """Test graph extension with RAG context."""
        config = get_default_config()
        agent = SymbolicReasoningAgent(config)
        extension = SymbolicAgentGraphExtension(
            reasoning_agent=agent,
            config=SymbolicAgentNodeConfig(),
        )

        state = {
            "query": "is python a programming language?",
            "rag_context": "Python is a programming language. It is widely used.",
        }

        result = await extension.handle_symbolic_node(state)

        assert "symbolic_results" in result
        # Should have extracted facts from RAG context
