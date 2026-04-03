"""Unit tests for AssemblyRouter (assembly-aware routing logic)."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.meta_controller.base import MetaControllerPrediction
from src.framework.assembly.config import AssemblyConfig
from src.framework.assembly.features import AssemblyFeatures


def _make_features(
    assembly_index: float = 5.0,
    copy_number: float = 3.0,
    decomposability_score: float = 0.5,
    graph_depth: int = 2,
    constraint_count: int = 3,
    concept_count: int = 4,
    technical_complexity: float = 0.3,
    normalized_assembly_index: float = 0.25,
) -> AssemblyFeatures:
    """Helper to create AssemblyFeatures with defaults."""
    return AssemblyFeatures(
        assembly_index=assembly_index,
        copy_number=copy_number,
        decomposability_score=decomposability_score,
        graph_depth=graph_depth,
        constraint_count=constraint_count,
        concept_count=concept_count,
        technical_complexity=technical_complexity,
        normalized_assembly_index=normalized_assembly_index,
    )


@pytest.mark.unit
class TestAssemblyRouterInit:
    """Test AssemblyRouter initialization."""

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_default_init(self, mock_extractor_cls):
        """Router initializes with default config."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        assert router.config is not None
        assert router.domain == "general"
        assert router.simple_threshold == 3
        assert router.medium_threshold == 7

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_custom_config(self, mock_extractor_cls):
        """Router uses custom config thresholds."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        config = AssemblyConfig(routing_simple_threshold=5, routing_medium_threshold=10)
        router = AssemblyRouter(config=config, domain="software")
        assert router.simple_threshold == 5
        assert router.medium_threshold == 10
        assert router.domain == "software"

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_initial_statistics(self, mock_extractor_cls):
        """Statistics start at zero."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        stats = router.get_statistics()
        assert stats["total_routes"] == 0
        assert stats["trm_routes"] == 0
        assert stats["hrm_routes"] == 0
        assert stats["mcts_routes"] == 0
        assert stats["trm_rate"] == 0.0
        assert stats["hrm_rate"] == 0.0
        assert stats["mcts_rate"] == 0.0


@pytest.mark.unit
class TestRoutingRules:
    """Test the routing decision rules."""

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_rule1_simple_high_reuse_routes_trm(self, mock_extractor_cls):
        """Simple query with high copy number routes to TRM (rule 1)."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(assembly_index=2.0, copy_number=6.0)
        decision = router.route("simple query", features=features)
        assert decision.agent == "trm"
        assert decision.confidence == 0.9
        assert "Simple query" in decision.reasoning
        assert "reusability" in decision.reasoning

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_rule2_very_high_copy_number_routes_trm(self, mock_extractor_cls):
        """Very high copy number (>10) routes to TRM (rule 2)."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(assembly_index=5.0, copy_number=11.0)
        decision = router.route("medium query", features=features)
        assert decision.agent == "trm"
        assert decision.confidence == 0.85

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_rule3_simple_query_routes_trm(self, mock_extractor_cls):
        """Simple query (low assembly index, moderate copy) routes to TRM (rule 3)."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(assembly_index=2.0, copy_number=3.0)
        decision = router.route("simple", features=features)
        assert decision.agent == "trm"
        assert decision.confidence == 0.8

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_rule4_highly_decomposable_routes_hrm(self, mock_extractor_cls):
        """Highly decomposable query routes to HRM (rule 4)."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(assembly_index=5.0, copy_number=2.0, decomposability_score=0.8, graph_depth=3)
        decision = router.route("decomposable query", features=features)
        assert decision.agent == "hrm"
        assert decision.confidence == 0.9
        assert "decomposable" in decision.reasoning.lower()

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_rule5_medium_complexity_moderate_decomp_routes_hrm(self, mock_extractor_cls):
        """Medium complexity with moderate decomposability routes to HRM (rule 5)."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(assembly_index=5.0, copy_number=2.0, decomposability_score=0.5)
        decision = router.route("medium query", features=features)
        assert decision.agent == "hrm"
        assert decision.confidence == 0.85

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_rule6_technical_structured_routes_hrm(self, mock_extractor_cls):
        """Technical query with structured hierarchy routes to HRM (rule 6)."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(
            assembly_index=5.0,
            copy_number=2.0,
            decomposability_score=0.35,
            technical_complexity=0.6,
            graph_depth=4,
        )
        decision = router.route("technical query", features=features)
        assert decision.agent == "hrm"
        assert decision.confidence == 0.8

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_rule7_low_decomposability_routes_mcts(self, mock_extractor_cls):
        """Low decomposability routes to MCTS (rule 7)."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(
            assembly_index=5.0,
            copy_number=2.0,
            decomposability_score=0.2,
            technical_complexity=0.3,
            graph_depth=2,
        )
        decision = router.route("hard query", features=features)
        assert decision.agent == "mcts"
        assert decision.confidence == 0.85
        assert "decomposability" in decision.reasoning.lower()

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_rule8_high_complexity_routes_mcts(self, mock_extractor_cls):
        """High complexity (assembly index >= medium threshold) routes to MCTS (rule 8)."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(
            assembly_index=8.0,
            copy_number=2.0,
            decomposability_score=0.4,
            technical_complexity=0.3,
            graph_depth=2,
        )
        decision = router.route("complex query", features=features)
        assert decision.agent == "mcts"
        assert decision.confidence == 0.9

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_default_routes_hrm(self, mock_extractor_cls):
        """Default case routes to HRM with lower confidence."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        # Medium assembly index, moderate decomposability (0.3-0.4 range), low tech complexity
        # This falls through rules 1-8 to the default
        features = _make_features(
            assembly_index=5.0,
            copy_number=2.0,
            decomposability_score=0.35,
            technical_complexity=0.3,
            graph_depth=2,
        )
        decision = router.route("ambiguous query", features=features)
        assert decision.agent == "hrm"
        assert decision.confidence == 0.6


@pytest.mark.unit
class TestRouteMethod:
    """Test the route method behavior."""

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_route_extracts_features_when_none(self, mock_extractor_cls):
        """Route calls feature extractor when features not provided."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = _make_features(assembly_index=2.0, copy_number=3.0)
        mock_extractor_cls.return_value = mock_extractor

        router = AssemblyRouter()
        router.route("test query")
        mock_extractor.extract.assert_called_once_with("test query")

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_route_uses_provided_features(self, mock_extractor_cls):
        """Route uses provided features instead of extracting."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        mock_extractor = MagicMock()
        mock_extractor_cls.return_value = mock_extractor

        router = AssemblyRouter()
        features = _make_features(assembly_index=2.0, copy_number=3.0)
        decision = router.route("test query", features=features)
        mock_extractor.extract.assert_not_called()
        assert decision.assembly_features is features

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_route_returns_routing_decision(self, mock_extractor_cls):
        """Route returns a RoutingDecision dataclass."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter, RoutingDecision

        router = AssemblyRouter()
        features = _make_features(assembly_index=2.0, copy_number=3.0)
        decision = router.route("test", features=features)
        assert isinstance(decision, RoutingDecision)
        assert decision.agent in ("hrm", "trm", "mcts")
        assert 0.0 <= decision.confidence <= 1.0
        assert isinstance(decision.reasoning, str)
        assert decision.assembly_features is features


@pytest.mark.unit
class TestStatistics:
    """Test routing statistics tracking."""

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_statistics_increment(self, mock_extractor_cls):
        """Statistics update after each route call."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        # Route to TRM
        router.route("q1", features=_make_features(assembly_index=2.0, copy_number=3.0))
        # Route to MCTS
        router.route("q2", features=_make_features(assembly_index=8.0, copy_number=2.0, decomposability_score=0.4))

        stats = router.get_statistics()
        assert stats["total_routes"] == 2
        assert stats["trm_routes"] == 1
        assert stats["mcts_routes"] == 1
        assert stats["trm_rate"] == 0.5
        assert stats["mcts_rate"] == 0.5

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_reset_statistics(self, mock_extractor_cls):
        """Reset statistics clears all counters."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        router.route("q", features=_make_features(assembly_index=2.0, copy_number=3.0))
        router.reset_statistics()
        stats = router.get_statistics()
        assert stats["total_routes"] == 0
        assert stats["trm_rate"] == 0.0


@pytest.mark.unit
class TestToPrediction:
    """Test conversion to MetaControllerPrediction."""

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_to_prediction_format(self, mock_extractor_cls):
        """to_prediction returns correct MetaControllerPrediction."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(assembly_index=2.0, copy_number=3.0)
        decision = router.route("q", features=features)
        prediction = router.to_prediction(decision)

        assert isinstance(prediction, MetaControllerPrediction)
        assert prediction.agent == decision.agent
        assert prediction.confidence == decision.confidence
        assert set(prediction.probabilities.keys()) == {"hrm", "trm", "mcts"}

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_to_prediction_probability_distribution(self, mock_extractor_cls):
        """Prediction probabilities are valid (selected agent gets highest)."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter()
        features = _make_features(assembly_index=2.0, copy_number=3.0)
        decision = router.route("q", features=features)
        prediction = router.to_prediction(decision)

        # Selected agent should have highest probability
        assert prediction.probabilities[decision.agent] == decision.confidence
        # Other agents should have lower probability
        for agent, prob in prediction.probabilities.items():
            if agent != decision.agent:
                assert prob < decision.confidence

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_to_prediction_high_confidence(self, mock_extractor_cls):
        """High confidence produces sharp distribution."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter, RoutingDecision

        router = AssemblyRouter()
        decision = RoutingDecision(
            agent="trm",
            confidence=0.9,
            reasoning="test",
            assembly_features=_make_features(),
        )
        prediction = router.to_prediction(decision)
        assert prediction.probabilities["trm"] == 0.9
        base = (1.0 - 0.9) / 2.0
        assert prediction.probabilities["hrm"] == pytest.approx(base)
        assert prediction.probabilities["mcts"] == pytest.approx(base)


@pytest.mark.unit
class TestExplainRouting:
    """Test the explain_routing method."""

    @patch("src.agents.meta_controller.assembly_router.AssemblyFeatureExtractor")
    def test_explain_routing_returns_string(self, mock_extractor_cls):
        """explain_routing produces a multi-line explanation."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = _make_features(assembly_index=2.0, copy_number=3.0)
        mock_extractor.explain_features.return_value = "Feature explanation text"
        mock_extractor_cls.return_value = mock_extractor

        router = AssemblyRouter()
        explanation = router.explain_routing("What is 2+2?")

        assert "Assembly-Based Routing Analysis" in explanation
        assert "Assembly Features:" in explanation
        assert "Routing Decision:" in explanation
        assert "Selected Agent:" in explanation
        assert "Confidence:" in explanation
