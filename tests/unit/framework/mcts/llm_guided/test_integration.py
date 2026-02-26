"""Tests for Integration Layer with HRM, TRM, and Meta-Controller."""

import json
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.framework.mcts.llm_guided.integration import (
    AgentType,
    HRMAdapter,
    IntegrationConfig,
    MetaControllerAdapter,
    RefinementResult,
    RoutingDecision,
    SubProblemDecomposition,
    TRMAdapter,
    UnifiedSearchOrchestrator,
    UnifiedSearchResult,
    create_unified_orchestrator,
)


class TestSubProblemDecomposition:
    """Tests for SubProblemDecomposition dataclass."""

    def test_creation(self):
        """Test basic creation."""
        decomp = SubProblemDecomposition(
            original_problem="Write a sorting algorithm",
            subproblems=["Parse input", "Sort elements", "Format output"],
            hierarchy_levels=[0, 0, 0],
            confidences=[0.9, 0.85, 0.9],
        )

        assert decomp.original_problem == "Write a sorting algorithm"
        assert decomp.num_subproblems == 3
        assert len(decomp.confidences) == 3

    def test_get_leaf_problems(self):
        """Test getting leaf problems from hierarchy."""
        decomp = SubProblemDecomposition(
            original_problem="Complex problem",
            subproblems=["Root", "Level1-A", "Level1-B", "Level2-A"],
            hierarchy_levels=[0, 1, 1, 2],
            confidences=[0.9, 0.85, 0.85, 0.8],
        )

        leaves = decomp.get_leaf_problems()
        assert leaves == ["Level2-A"]

    def test_get_leaf_problems_same_level(self):
        """Test leaf problems when all at same level."""
        decomp = SubProblemDecomposition(
            original_problem="Simple problem",
            subproblems=["Part A", "Part B", "Part C"],
            hierarchy_levels=[0, 0, 0],
            confidences=[0.9, 0.85, 0.9],
        )

        leaves = decomp.get_leaf_problems()
        assert leaves == ["Part A", "Part B", "Part C"]

    def test_num_subproblems(self):
        """Test num_subproblems property."""
        decomp = SubProblemDecomposition(
            original_problem="Test",
            subproblems=["A", "B"],
            hierarchy_levels=[0, 1],
            confidences=[1.0, 0.9],
        )

        assert decomp.num_subproblems == 2


class TestRefinementResult:
    """Tests for RefinementResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = RefinementResult(
            original_code="def foo(): pass",
            refined_code="def foo(): return 42",
            num_iterations=3,
            converged=True,
            improvement_score=0.5,
        )

        assert result.original_code == "def foo(): pass"
        assert result.refined_code == "def foo(): return 42"
        assert result.num_iterations == 3
        assert result.converged is True

    def test_with_intermediate_codes(self):
        """Test with intermediate codes."""
        result = RefinementResult(
            original_code="v0",
            refined_code="v3",
            num_iterations=3,
            converged=True,
            improvement_score=0.7,
            intermediate_codes=["v0", "v1", "v2", "v3"],
            residual_norms=[0.3, 0.2, 0.15, 0.05],
        )

        assert len(result.intermediate_codes) == 4
        assert len(result.residual_norms) == 4


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_creation(self):
        """Test basic creation."""
        decision = RoutingDecision(
            selected_agent=AgentType.HRM,
            confidence=0.85,
            probabilities={"hrm": 0.6, "trm": 0.25, "mcts": 0.15},
            reasoning="Complex hierarchical problem",
        )

        assert decision.selected_agent == AgentType.HRM
        assert decision.confidence == 0.85
        assert decision.probabilities["hrm"] == 0.6


class TestIntegrationConfig:
    """Tests for IntegrationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = IntegrationConfig()

        assert config.use_hrm_decomposition is True
        assert config.use_trm_refinement is True
        assert config.use_meta_controller is True
        assert config.decomposition_threshold == 0.7
        assert config.refinement_max_iterations == 16

    def test_custom_values(self):
        """Test custom configuration values."""
        config = IntegrationConfig(
            use_hrm_decomposition=False,
            refinement_max_iterations=32,
            low_confidence_threshold=0.3,
        )

        assert config.use_hrm_decomposition is False
        assert config.refinement_max_iterations == 32
        assert config.low_confidence_threshold == 0.3


class TestHRMAdapter:
    """Tests for HRMAdapter."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        return client

    def test_initialization(self, mock_llm_client):
        """Test adapter initialization."""
        adapter = HRMAdapter(llm_client=mock_llm_client)

        assert adapter.has_neural_agent is False
        assert adapter._llm_client is not None

    def test_has_neural_agent_false(self, mock_llm_client):
        """Test has_neural_agent when no HRM agent provided."""
        adapter = HRMAdapter(llm_client=mock_llm_client)
        assert adapter.has_neural_agent is False

    @pytest.mark.asyncio
    async def test_decompose_with_llm(self, mock_llm_client):
        """Test LLM-based decomposition."""
        mock_llm_client.complete.return_value = json.dumps(
            {
                "subproblems": ["Parse input", "Process data", "Format output"],
                "levels": [0, 0, 0],
                "confidences": [0.9, 0.85, 0.9],
            }
        )

        adapter = HRMAdapter(llm_client=mock_llm_client)
        result = await adapter.decompose("Write a data processor")

        assert result.num_subproblems == 3
        assert "Parse input" in result.subproblems
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_decompose_fallback_on_error(self, mock_llm_client):
        """Test fallback when LLM fails."""
        mock_llm_client.complete.side_effect = Exception("LLM error")

        adapter = HRMAdapter(llm_client=mock_llm_client)
        result = await adapter.decompose("Test problem")

        # Should return original problem as single subproblem
        assert result.num_subproblems == 1
        assert result.subproblems == ["Test problem"]
        assert result.confidences[0] < 1.0  # Lower confidence on fallback

    @pytest.mark.asyncio
    async def test_decompose_no_agents(self):
        """Test decompose when no agents available."""
        adapter = HRMAdapter()  # No LLM client, no HRM agent

        result = await adapter.decompose("Test problem")

        assert result.num_subproblems == 1
        assert result.subproblems == ["Test problem"]


class TestTRMAdapter:
    """Tests for TRMAdapter."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        return client

    def test_initialization(self, mock_llm_client):
        """Test adapter initialization."""
        adapter = TRMAdapter(llm_client=mock_llm_client)

        assert adapter.has_neural_agent is False
        assert adapter._llm_client is not None

    @pytest.mark.asyncio
    async def test_refine_with_llm(self, mock_llm_client):
        """Test LLM-based refinement."""
        mock_llm_client.complete.return_value = json.dumps(
            {
                "refined_code": "def foo(): return 42",
                "improvement_score": 0.1,
                "converged": True,
            }
        )

        adapter = TRMAdapter(llm_client=mock_llm_client)
        result = await adapter.refine(
            code="def foo(): pass",
            problem="Return 42",
            max_iterations=3,
        )

        assert result.refined_code == "def foo(): return 42"
        assert result.converged is True

    @pytest.mark.asyncio
    async def test_refine_multiple_iterations(self, mock_llm_client):
        """Test refinement over multiple iterations."""
        # First iteration: not converged
        # Second iteration: converged
        mock_llm_client.complete.side_effect = [
            json.dumps(
                {
                    "refined_code": "def foo(): return 41",
                    "improvement_score": 0.3,
                    "converged": False,
                }
            ),
            json.dumps(
                {
                    "refined_code": "def foo(): return 42",
                    "improvement_score": 0.1,
                    "converged": True,
                }
            ),
        ]

        adapter = TRMAdapter(llm_client=mock_llm_client)
        result = await adapter.refine(
            code="def foo(): pass",
            problem="Return 42",
            max_iterations=5,
        )

        assert result.num_iterations == 2
        assert result.converged is True
        assert len(result.intermediate_codes) == 3  # original + 2 iterations

    @pytest.mark.asyncio
    async def test_refine_no_agents(self):
        """Test refine when no agents available."""
        adapter = TRMAdapter()  # No LLM client, no TRM agent

        result = await adapter.refine(
            code="def foo(): pass",
            problem="Test",
        )

        # Should return original code unchanged
        assert result.original_code == result.refined_code
        assert result.num_iterations == 0


class TestMetaControllerAdapter:
    """Tests for MetaControllerAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = MetaControllerAdapter()

        assert adapter.has_meta_controller is False

    def test_route_heuristic_complex(self):
        """Test heuristic routing for complex problems."""
        adapter = MetaControllerAdapter()

        decision = adapter.route(
            problem="This is a complex hierarchical optimization problem",
            context={},
        )

        assert decision.selected_agent == AgentType.HRM
        assert "hierarchical" in decision.reasoning.lower() or "complex" in decision.reasoning.lower()

    def test_route_heuristic_simple(self):
        """Test heuristic routing for simple problems."""
        adapter = MetaControllerAdapter()

        decision = adapter.route(
            problem="Simple basic task",
            context={},
        )

        assert decision.selected_agent == AgentType.TRM
        assert decision.confidence > 0

    def test_route_heuristic_default(self):
        """Test heuristic routing default case."""
        adapter = MetaControllerAdapter()

        decision = adapter.route(
            problem="Write a function to compute fibonacci",
            context={},
        )

        assert decision.selected_agent == AgentType.LLM_MCTS
        assert decision.probabilities is not None

    def test_routing_statistics(self):
        """Test routing statistics collection."""
        adapter = MetaControllerAdapter()

        # Make several routing decisions
        adapter.route("Complex problem", {})
        adapter.route("Simple task", {})
        adapter.route("Another problem", {})

        stats = adapter.get_routing_statistics()

        assert stats["total_decisions"] == 3
        assert "agent_distribution" in stats
        assert "average_confidence" in stats


class TestUnifiedSearchOrchestrator:
    """Tests for UnifiedSearchOrchestrator."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        # Default response for generator - returns correct code that passes tests
        client.complete.return_value = json.dumps(
            {
                "variants": [
                    {"code": "def foo(): return 1", "confidence": 0.9},
                ]
            }
        )
        return client

    @pytest.fixture
    def orchestrator(self, mock_llm_client):
        """Create orchestrator with mocked dependencies."""
        from src.framework.mcts.llm_guided.config import LLMGuidedMCTSConfig

        config = LLMGuidedMCTSConfig(
            num_iterations=3,
            early_termination_on_solution=True,
        )
        integration_config = IntegrationConfig(
            use_hrm_decomposition=False,  # Disable for simpler testing
            use_trm_refinement=False,
            use_meta_controller=False,
        )

        return UnifiedSearchOrchestrator(
            llm_client=mock_llm_client,
            mcts_config=config,
            integration_config=integration_config,
        )

    @pytest.mark.asyncio
    async def test_search_basic(self, mock_llm_client):
        """Test basic search functionality by mocking the MCTS engine."""
        from src.framework.mcts.llm_guided.config import LLMGuidedMCTSConfig
        from src.framework.mcts.llm_guided.engine import MCTSSearchResult

        # Create a mock search result
        mock_result = MCTSSearchResult(
            solution_found=True,
            best_code="def foo(): return 1",
            best_value=1.0,
            num_iterations=3,
            num_expansions=3,
            num_evaluations=3,
            tree_depth=2,
            tree_size=4,
            execution_time_ms=100.0,
            llm_calls=3,
            tokens_used=500,
            episode_id="test-episode",
            root_visits=3,
            action_stats={},
            test_results=None,
        )

        config = LLMGuidedMCTSConfig(num_iterations=3)
        integration_config = IntegrationConfig(
            use_hrm_decomposition=False,
            use_trm_refinement=False,
            use_meta_controller=False,
        )

        orchestrator = UnifiedSearchOrchestrator(
            llm_client=mock_llm_client,
            mcts_config=config,
            integration_config=integration_config,
        )

        # Patch the internal MCTS engine's search method
        orchestrator._mcts_engine.search = AsyncMock(return_value=mock_result)

        result = await orchestrator.search(
            problem="Return 1",
            test_cases=["assert foo() == 1"],
        )

        assert isinstance(result, UnifiedSearchResult)
        assert result.solution_found is True
        assert "return 1" in result.best_code

    @pytest.mark.asyncio
    async def test_search_with_routing(self, mock_llm_client):
        """Test search with meta-controller routing."""
        from src.framework.mcts.llm_guided.config import LLMGuidedMCTSConfig
        from src.framework.mcts.llm_guided.engine import MCTSSearchResult

        # Create a mock search result
        mock_result = MCTSSearchResult(
            solution_found=True,
            best_code="def foo(): return 1",
            best_value=1.0,
            num_iterations=3,
            num_expansions=3,
            num_evaluations=3,
            tree_depth=2,
            tree_size=4,
            execution_time_ms=100.0,
            llm_calls=3,
            tokens_used=500,
            episode_id="test-episode",
            root_visits=3,
            action_stats={},
            test_results=None,
        )

        config = LLMGuidedMCTSConfig(num_iterations=3)
        integration_config = IntegrationConfig(
            use_meta_controller=True,
            use_hrm_decomposition=False,
            use_trm_refinement=False,
        )

        orchestrator = UnifiedSearchOrchestrator(
            llm_client=mock_llm_client,
            mcts_config=config,
            integration_config=integration_config,
        )

        # Patch the internal MCTS engine's search method
        orchestrator._mcts_engine.search = AsyncMock(return_value=mock_result)

        result = await orchestrator.search(
            problem="Return 1",
            test_cases=["assert foo() == 1"],
        )

        assert result.routing_decision is not None
        assert result.agent_used in AgentType

    def test_get_statistics(self, orchestrator):
        """Test statistics collection."""
        stats = orchestrator.get_statistics()

        assert "mcts" in stats
        assert "routing" in stats
        assert "hrm_available" in stats
        assert "trm_available" in stats


class TestCreateUnifiedOrchestrator:
    """Tests for create_unified_orchestrator factory function."""

    def test_create_with_fast_preset(self):
        """Test creation with fast preset."""
        mock_client = AsyncMock()
        orchestrator = create_unified_orchestrator(mock_client, preset="fast")

        assert orchestrator is not None
        assert orchestrator._mcts_config.name == "fast"

    def test_create_with_balanced_preset(self):
        """Test creation with balanced preset."""
        mock_client = AsyncMock()
        orchestrator = create_unified_orchestrator(mock_client, preset="balanced")

        assert orchestrator is not None
        assert orchestrator._mcts_config.name == "balanced"

    def test_create_with_overrides(self):
        """Test creation with parameter overrides."""
        mock_client = AsyncMock()
        orchestrator = create_unified_orchestrator(
            mock_client,
            preset="fast",
            num_iterations=50,  # Override
        )

        assert orchestrator._mcts_config.num_iterations == 50


class TestAgentType:
    """Tests for AgentType enum."""

    def test_values(self):
        """Test enum values."""
        assert AgentType.HRM.value == "hrm"
        assert AgentType.TRM.value == "trm"
        assert AgentType.MCTS.value == "mcts"
        assert AgentType.LLM_MCTS.value == "llm_mcts"

    def test_from_string(self):
        """Test creation from string."""
        agent = AgentType("hrm")
        assert agent == AgentType.HRM
