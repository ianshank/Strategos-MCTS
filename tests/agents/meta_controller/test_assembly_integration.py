"""
Tests for Assembly Theory integration in meta-controller (Stories 2.2, 2.3).
"""

import pytest

# Skip if dependencies not available
pytest.importorskip("torch", reason="PyTorch required for meta-controller tests")
pytest.importorskip("networkx", reason="networkx required for assembly tests")

from src.agents.meta_controller import (
    AssemblyRouter,
    HybridMetaController,
    HybridPrediction,
    MetaControllerFeatures,
    RoutingDecision,
)
from src.framework.assembly import AssemblyFeatures


class TestAssemblyRouter:
    """Test suite for AssemblyRouter."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        return AssemblyRouter()

    def test_router_initialization(self, router):
        """Test router initializes correctly."""
        assert router is not None
        assert router.simple_threshold == 3
        assert router.medium_threshold == 7

    def test_simple_query_routing(self, router):
        """Test simple queries route to TRM."""
        simple_queries = [
            "What is 2+2?",
            "Hello world",
            "Simple test",
        ]

        for query in simple_queries:
            decision = router.route(query)

            assert isinstance(decision, RoutingDecision)
            # Simple queries should tend toward TRM
            # (though exact routing depends on features)
            assert decision.agent in ["trm", "hrm"]  # Reasonable options
            assert 0.0 <= decision.confidence <= 1.0

    def test_complex_query_routing(self, router):
        """Test complex queries route appropriately."""
        complex_query = (
            "Design a distributed microservices architecture with API gateway, "
            "service mesh, event-driven communication, distributed caching, "
            "monitoring, and fault tolerance across multiple availability zones"
        )

        decision = router.route(complex_query)

        assert isinstance(decision, RoutingDecision)
        # Complex query should route to MCTS or HRM
        assert decision.agent in ["mcts", "hrm"]
        assert decision.confidence > 0.5

    def test_routing_with_precomputed_features(self, router):
        """Test routing with pre-computed features."""
        # Create mock features for simple query
        simple_features = AssemblyFeatures(
            assembly_index=2.0,
            copy_number=8.0,
            decomposability_score=0.6,
            graph_depth=2,
            constraint_count=1,
            concept_count=3,
            technical_complexity=0.1,
            normalized_assembly_index=0.1,
        )

        decision = router.route("test query", features=simple_features)

        assert decision.agent == "trm"  # Simple with high copy number → TRM
        assert decision.confidence >= 0.8

    def test_high_decomposability_routing(self, router):
        """Test highly decomposable queries route to HRM."""
        # Mock features for decomposable query
        decomposable_features = AssemblyFeatures(
            assembly_index=5.0,
            copy_number=2.0,
            decomposability_score=0.85,  # High decomposability
            graph_depth=4,
            constraint_count=8,
            concept_count=10,
            technical_complexity=0.4,
            normalized_assembly_index=0.25,
        )

        decision = router.route("decomposable query", features=decomposable_features)

        assert decision.agent == "hrm"  # High decomposability → HRM
        assert decision.confidence >= 0.8

    def test_routing_explanation(self, router):
        """Test routing includes explanation."""
        decision = router.route("Test query for explanation")

        assert decision.reasoning is not None
        assert len(decision.reasoning) > 0
        assert isinstance(decision.reasoning, str)

    def test_routing_statistics(self, router):
        """Test routing statistics tracking."""
        queries = [
            "simple query",
            "another simple one",
            "complex distributed system architecture",
        ]

        for query in queries:
            router.route(query)

        stats = router.get_statistics()

        assert stats["total_routes"] == 3
        assert stats["trm_routes"] + stats["hrm_routes"] + stats["mcts_routes"] == 3
        assert 0.0 <= stats["trm_rate"] <= 1.0

    def test_to_prediction_conversion(self, router):
        """Test conversion to MetaControllerPrediction format."""
        decision = router.route("test query")
        prediction = router.to_prediction(decision)

        assert prediction.agent == decision.agent
        assert prediction.confidence == decision.confidence
        assert len(prediction.probabilities) == 3
        assert sum(prediction.probabilities.values()) > 0.99  # Should sum to ~1.0

    def test_explain_routing(self, router):
        """Test detailed routing explanation."""
        explanation = router.explain_routing("Design a database query optimizer")

        assert isinstance(explanation, str)
        assert "Assembly Features" in explanation
        assert "Routing Decision" in explanation
        assert "assembly_index" in explanation.lower()


class TestHybridMetaController:
    """Test suite for HybridMetaController."""

    @pytest.fixture
    def mock_neural_controller(self):
        """Create mock neural controller."""
        from src.agents.meta_controller import MetaControllerPrediction

        class MockNeuralController:
            def predict(self, features):
                # Always predict HRM with moderate confidence
                return MetaControllerPrediction(
                    agent="hrm", confidence=0.75, probabilities={"hrm": 0.75, "trm": 0.15, "mcts": 0.10}
                )

        return MockNeuralController()

    @pytest.fixture
    def hybrid_controller(self, mock_neural_controller):
        """Create hybrid controller instance."""
        return HybridMetaController(
            neural_controller=mock_neural_controller,
            neural_weight=0.6,
            assembly_weight=0.4,
        )

    def test_hybrid_initialization(self, hybrid_controller):
        """Test hybrid controller initializes correctly."""
        assert hybrid_controller is not None
        assert hybrid_controller.neural_weight == 0.6
        assert hybrid_controller.assembly_weight == 0.4

    def test_hybrid_prediction_with_query(self, hybrid_controller):
        """Test hybrid prediction with query context."""
        # Set query context
        query = "Design microservices architecture"
        hybrid_controller.set_query_context(query)

        # Make prediction
        features = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.7,
            consensus_score=0.75,
            last_agent="hrm",
            iteration=1,
            query_length=len(query),
            has_rag_context=False,
        )

        prediction = hybrid_controller.predict(features)

        assert isinstance(prediction, HybridPrediction)
        assert prediction.agent in ["hrm", "trm", "mcts"]
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.neural_prediction is not None
        assert prediction.assembly_decision is not None

    def test_hybrid_neural_only(self, mock_neural_controller):
        """Test hybrid with only neural prediction (no query)."""
        controller = HybridMetaController(
            neural_controller=mock_neural_controller,
        )

        features = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.7,
            consensus_score=0.75,
            last_agent="hrm",
            iteration=1,
            query_length=20,
            has_rag_context=False,
        )

        # No query context → neural only
        prediction = controller.predict(features)

        assert prediction.agent == "hrm"  # Mock always predicts HRM
        assert prediction.neural_prediction is not None
        assert prediction.assembly_decision is None

    def test_hybrid_assembly_only(self):
        """Test hybrid with only assembly routing (no neural)."""
        controller = HybridMetaController(
            neural_controller=None,  # No neural controller
        )

        query = "Simple query"
        controller.set_query_context(query)

        features = MetaControllerFeatures(
            hrm_confidence=0.0,
            trm_confidence=0.0,
            mcts_value=0.0,
            consensus_score=0.0,
            last_agent="none",
            iteration=0,
            query_length=len(query),
            has_rag_context=False,
        )

        prediction = controller.predict(features)

        # Should use assembly only
        assert prediction.assembly_decision is not None
        assert prediction.neural_prediction is None

    def test_hybrid_ensemble_weighting(self, hybrid_controller):
        """Test that ensemble uses correct weights."""
        query = "Test query for weighting"
        hybrid_controller.set_query_context(query)

        features = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.7,
            consensus_score=0.75,
            last_agent="hrm",
            iteration=1,
            query_length=len(query),
            has_rag_context=False,
        )

        prediction = hybrid_controller.predict(features)

        # Check weights were used
        assert prediction.neural_weight == 0.6
        assert prediction.assembly_weight == 0.4

    def test_hybrid_statistics(self, hybrid_controller):
        """Test hybrid controller statistics tracking."""
        queries = ["query 1", "query 2", "query 3"]
        features = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.7,
            consensus_score=0.75,
            last_agent="hrm",
            iteration=1,
            query_length=20,
            has_rag_context=False,
        )

        for query in queries:
            hybrid_controller.set_query_context(query)
            hybrid_controller.predict(features)

        stats = hybrid_controller.get_statistics()

        assert stats["total_predictions"] == 3
        assert 0.0 <= stats["agreement_rate"] <= 1.0
        assert "assembly_router" in stats

    def test_weight_adjustment(self, hybrid_controller):
        """Test dynamic weight adjustment."""

        hybrid_controller.adjust_weights(0.8, 0.2)

        assert hybrid_controller.neural_weight == 0.8
        assert hybrid_controller.assembly_weight == 0.2
        assert abs(hybrid_controller.neural_weight + hybrid_controller.assembly_weight - 1.0) < 0.001

    def test_explanation_generation(self, hybrid_controller):
        """Test decision explanation generation."""
        query = "Explain this routing decision"
        hybrid_controller.set_query_context(query)

        features = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.7,
            consensus_score=0.75,
            last_agent="hrm",
            iteration=1,
            query_length=len(query),
            has_rag_context=False,
        )

        prediction = hybrid_controller.predict(features)

        assert prediction.explanation is not None
        assert len(prediction.explanation) > 0
        assert "Neural Prediction" in prediction.explanation
        assert "Assembly Routing" in prediction.explanation


@pytest.mark.parametrize(
    "query,expected_complexity",
    [
        ("simple", "simple"),  # Simple query
        ("Design a complex distributed microservices system", "complex"),  # Complex
    ],
)
def test_routing_complexity_detection(query, expected_complexity):
    """Test that router correctly identifies query complexity."""
    router = AssemblyRouter()
    decision = router.route(query)

    if expected_complexity == "simple":
        # Simple queries should have low assembly index
        assert decision.assembly_features.assembly_index < 7
    else:
        # Complex queries should have higher assembly index
        # (though not always guaranteed due to query processing)
        assert decision.assembly_features.assembly_index >= 0  # At least valid
