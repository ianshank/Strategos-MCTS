"""
Unit tests for src/framework/mcts/llm_guided/integration.py.

Tests:
- SubProblemDecomposition dataclass
- RefinementResult dataclass
- RoutingDecision dataclass
- IntegrationConfig validation
- HRMAdapter (neural, llm, fallback paths)
- TRMAdapter (neural, llm, fallback paths)
- MetaControllerAdapter (neural, heuristic routing)
- UnifiedSearchOrchestrator
- create_unified_orchestrator factory
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# SubProblemDecomposition tests
# ---------------------------------------------------------------------------


class TestSubProblemDecomposition:
    def test_num_subproblems(self):
        from src.framework.mcts.llm_guided.integration import SubProblemDecomposition

        decomp = SubProblemDecomposition(
            original_problem="problem",
            subproblems=["a", "b", "c"],
            hierarchy_levels=[0, 1, 1],
            confidences=[0.9, 0.8, 0.7],
        )
        assert decomp.num_subproblems == 3

    def test_get_leaf_problems(self):
        from src.framework.mcts.llm_guided.integration import SubProblemDecomposition

        decomp = SubProblemDecomposition(
            original_problem="problem",
            subproblems=["root", "leaf1", "leaf2"],
            hierarchy_levels=[0, 1, 1],
            confidences=[0.9, 0.8, 0.7],
        )
        leaves = decomp.get_leaf_problems()
        assert leaves == ["leaf1", "leaf2"]

    def test_get_leaf_problems_empty_levels(self):
        from src.framework.mcts.llm_guided.integration import SubProblemDecomposition

        decomp = SubProblemDecomposition(
            original_problem="problem",
            subproblems=["a", "b"],
            hierarchy_levels=[],
            confidences=[0.9, 0.8],
        )
        leaves = decomp.get_leaf_problems()
        assert leaves == ["a", "b"]

    def test_get_leaf_problems_single_level(self):
        from src.framework.mcts.llm_guided.integration import SubProblemDecomposition

        decomp = SubProblemDecomposition(
            original_problem="problem",
            subproblems=["a", "b"],
            hierarchy_levels=[0, 0],
            confidences=[0.9, 0.8],
        )
        leaves = decomp.get_leaf_problems()
        assert leaves == ["a", "b"]


# ---------------------------------------------------------------------------
# RefinementResult tests
# ---------------------------------------------------------------------------


class TestRefinementResult:
    def test_defaults(self):
        from src.framework.mcts.llm_guided.integration import RefinementResult

        result = RefinementResult(
            original_code="x=1",
            refined_code="x = 1",
            num_iterations=3,
            converged=True,
            improvement_score=0.5,
        )
        assert result.intermediate_codes == []
        assert result.residual_norms == []
        assert result.converged is True


# ---------------------------------------------------------------------------
# RoutingDecision tests
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    def test_creation(self):
        from src.framework.mcts.llm_guided.integration import AgentType, RoutingDecision

        decision = RoutingDecision(
            selected_agent=AgentType.HRM,
            confidence=0.85,
            probabilities={"hrm": 0.6, "trm": 0.2, "mcts": 0.2},
            reasoning="Complex problem",
        )
        assert decision.selected_agent == AgentType.HRM
        assert decision.confidence == 0.85


# ---------------------------------------------------------------------------
# IntegrationConfig tests
# ---------------------------------------------------------------------------


class TestIntegrationConfig:
    def test_defaults(self):
        from src.framework.mcts.llm_guided.integration import IntegrationConfig

        config = IntegrationConfig()
        assert config.use_hrm_decomposition is True
        assert config.use_trm_refinement is True
        assert config.use_meta_controller is True
        assert config.decomposition_threshold == 0.7
        assert config.refinement_max_iterations == 16
        assert config.refinement_convergence_threshold == 0.01
        assert config.fallback_to_mcts_on_low_confidence is True
        assert config.low_confidence_threshold == 0.5
        assert config.enable_parallel_search is False
        assert config.combine_results_strategy == "best"

    def test_custom_values(self):
        from src.framework.mcts.llm_guided.integration import IntegrationConfig

        config = IntegrationConfig(
            decomposition_threshold=0.5,
            refinement_max_iterations=8,
            enable_parallel_search=True,
        )
        assert config.decomposition_threshold == 0.5
        assert config.refinement_max_iterations == 8
        assert config.enable_parallel_search is True


# ---------------------------------------------------------------------------
# AgentType tests
# ---------------------------------------------------------------------------


class TestAgentType:
    def test_values(self):
        from src.framework.mcts.llm_guided.integration import AgentType

        assert AgentType.HRM.value == "hrm"
        assert AgentType.TRM.value == "trm"
        assert AgentType.MCTS.value == "mcts"
        assert AgentType.LLM_MCTS.value == "llm_mcts"


# ---------------------------------------------------------------------------
# HRMAdapter tests
# ---------------------------------------------------------------------------


class TestHRMAdapter:
    def test_has_neural_agent_false(self):
        from src.framework.mcts.llm_guided.integration import HRMAdapter

        adapter = HRMAdapter()
        assert adapter.has_neural_agent is False

    def test_has_neural_agent_true(self):
        from src.framework.mcts.llm_guided.integration import HRMAdapter

        adapter = HRMAdapter(hrm_agent=MagicMock())
        assert adapter.has_neural_agent is True

    @pytest.mark.asyncio
    async def test_decompose_no_agent_no_llm(self):
        """Fallback when no agent or LLM available."""
        from src.framework.mcts.llm_guided.integration import HRMAdapter

        adapter = HRMAdapter()
        result = await adapter.decompose("solve x + 1 = 2")
        assert result.original_problem == "solve x + 1 = 2"
        assert result.subproblems == ["solve x + 1 = 2"]
        assert result.confidences == [1.0]

    @pytest.mark.asyncio
    async def test_decompose_neural(self):
        """Neural decomposition path."""
        from src.framework.mcts.llm_guided.integration import HRMAdapter

        mock_hrm = MagicMock()
        adapter = HRMAdapter(hrm_agent=mock_hrm)
        result = await adapter.decompose("some problem", context="ctx")

        assert result.original_problem == "some problem"
        assert result.confidences == [0.8]

    @pytest.mark.asyncio
    async def test_decompose_llm_success(self):
        """LLM decomposition path succeeds."""
        from src.framework.mcts.llm_guided.integration import HRMAdapter

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {
                "subproblems": ["parse", "process"],
                "levels": [0, 0],
                "confidences": [0.9, 0.85],
            }
        )
        adapter = HRMAdapter(llm_client=mock_llm)
        result = await adapter.decompose("build a parser")

        assert result.subproblems == ["parse", "process"]
        assert result.confidences == [0.9, 0.85]

    @pytest.mark.asyncio
    async def test_decompose_llm_failure(self):
        """LLM decomposition path fails gracefully."""
        from src.framework.mcts.llm_guided.integration import HRMAdapter

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = RuntimeError("API error")
        adapter = HRMAdapter(llm_client=mock_llm)
        result = await adapter.decompose("build a parser")

        assert result.subproblems == ["build a parser"]
        assert result.confidences == [0.5]

    @pytest.mark.asyncio
    async def test_decompose_llm_with_context(self):
        """LLM decomposition includes context in prompt."""
        from src.framework.mcts.llm_guided.integration import HRMAdapter

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {"subproblems": ["a"], "levels": [0], "confidences": [0.9]}
        )
        adapter = HRMAdapter(llm_client=mock_llm)
        await adapter.decompose("problem", context="extra context")

        call_args = mock_llm.complete.call_args[0][0]
        assert "extra context" in call_args

    def test_build_decomposition_prompt_with_context(self):
        from src.framework.mcts.llm_guided.integration import HRMAdapter

        adapter = HRMAdapter()
        prompt = adapter._build_decomposition_prompt("problem text", context="some context")
        assert "problem text" in prompt
        assert "some context" in prompt

    def test_build_decomposition_prompt_without_context(self):
        from src.framework.mcts.llm_guided.integration import HRMAdapter

        adapter = HRMAdapter()
        prompt = adapter._build_decomposition_prompt("problem text", context=None)
        assert "problem text" in prompt
        assert "Context:" not in prompt


# ---------------------------------------------------------------------------
# TRMAdapter tests
# ---------------------------------------------------------------------------


class TestTRMAdapter:
    def test_has_neural_agent_false(self):
        from src.framework.mcts.llm_guided.integration import TRMAdapter

        adapter = TRMAdapter()
        assert adapter.has_neural_agent is False

    def test_has_neural_agent_true(self):
        from src.framework.mcts.llm_guided.integration import TRMAdapter

        adapter = TRMAdapter(trm_agent=MagicMock())
        assert adapter.has_neural_agent is True

    @pytest.mark.asyncio
    async def test_refine_no_agent_no_llm(self):
        """Fallback when no agent or LLM available."""
        from src.framework.mcts.llm_guided.integration import TRMAdapter

        adapter = TRMAdapter()
        result = await adapter.refine("code", "problem")
        assert result.original_code == "code"
        assert result.refined_code == "code"
        assert result.converged is True
        assert result.num_iterations == 0
        assert result.improvement_score == 0.0

    @pytest.mark.asyncio
    async def test_refine_neural(self):
        """Neural refinement path."""
        from src.framework.mcts.llm_guided.integration import TRMAdapter

        mock_trm = MagicMock()
        adapter = TRMAdapter(trm_agent=mock_trm)
        result = await adapter.refine("def f(): pass", "problem")
        assert result.converged is True
        assert result.num_iterations == 0

    @pytest.mark.asyncio
    async def test_refine_llm_converges(self):
        """LLM refinement converges."""
        from src.framework.mcts.llm_guided.integration import IntegrationConfig, TRMAdapter

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {"refined_code": "def f(): return 42", "improvement_score": 0.005, "converged": True}
        )
        config = IntegrationConfig(refinement_convergence_threshold=0.01)
        adapter = TRMAdapter(llm_client=mock_llm, config=config)
        result = await adapter.refine("def f(): pass", "problem")

        assert result.converged is True
        assert result.refined_code == "def f(): return 42"
        assert result.num_iterations == 1

    @pytest.mark.asyncio
    async def test_refine_llm_max_iterations(self):
        """LLM refinement hits max iterations."""
        from src.framework.mcts.llm_guided.integration import IntegrationConfig, TRMAdapter

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {"refined_code": "improved", "improvement_score": 0.5, "converged": False}
        )
        config = IntegrationConfig(refinement_max_iterations=3)
        adapter = TRMAdapter(llm_client=mock_llm, config=config)
        result = await adapter.refine("code", "problem", max_iterations=2)

        assert result.converged is False
        assert result.num_iterations == 2

    @pytest.mark.asyncio
    async def test_refine_llm_error_breaks(self):
        """LLM refinement error breaks loop."""
        from src.framework.mcts.llm_guided.integration import TRMAdapter

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = RuntimeError("API down")
        adapter = TRMAdapter(llm_client=mock_llm)
        result = await adapter.refine("code", "problem", max_iterations=5)

        assert result.converged is False
        assert result.refined_code == "code"

    @pytest.mark.asyncio
    async def test_refine_uses_custom_max_iterations(self):
        """Custom max_iterations overrides config."""
        from src.framework.mcts.llm_guided.integration import IntegrationConfig, TRMAdapter

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {"refined_code": "better", "improvement_score": 0.5, "converged": False}
        )
        config = IntegrationConfig(refinement_max_iterations=100)
        adapter = TRMAdapter(llm_client=mock_llm, config=config)
        await adapter.refine("code", "problem", max_iterations=2)
        assert mock_llm.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_refine_with_test_cases(self):
        """Test cases appear in the refinement prompt."""
        from src.framework.mcts.llm_guided.integration import TRMAdapter

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {"refined_code": "x", "improvement_score": 0.001, "converged": True}
        )
        adapter = TRMAdapter(llm_client=mock_llm)
        await adapter.refine("code", "problem", test_cases=["assert f(1)==2"])
        call_args = mock_llm.complete.call_args[0][0]
        assert "assert f(1)==2" in call_args

    def test_build_refinement_prompt_with_tests(self):
        from src.framework.mcts.llm_guided.integration import TRMAdapter

        adapter = TRMAdapter()
        prompt = adapter._build_refinement_prompt("code", "problem", ["test1", "test2"], 0)
        assert "test1" in prompt
        assert "test2" in prompt
        assert "iteration 1" in prompt

    def test_build_refinement_prompt_no_tests(self):
        from src.framework.mcts.llm_guided.integration import TRMAdapter

        adapter = TRMAdapter()
        prompt = adapter._build_refinement_prompt("code", "problem", None, 0)
        assert "No test cases provided" in prompt


# ---------------------------------------------------------------------------
# MetaControllerAdapter tests
# ---------------------------------------------------------------------------


class TestMetaControllerAdapter:
    def test_has_meta_controller_false(self):
        from src.framework.mcts.llm_guided.integration import MetaControllerAdapter

        adapter = MetaControllerAdapter()
        assert adapter.has_meta_controller is False

    def test_has_meta_controller_true(self):
        from src.framework.mcts.llm_guided.integration import MetaControllerAdapter

        adapter = MetaControllerAdapter(meta_controller=MagicMock())
        assert adapter.has_meta_controller is True

    def test_route_heuristic_complex(self):
        """Complex problem routes to HRM."""
        from src.framework.mcts.llm_guided.integration import AgentType, MetaControllerAdapter

        adapter = MetaControllerAdapter()
        decision = adapter.route("This is a complex optimization problem", {})
        assert decision.selected_agent == AgentType.HRM
        assert "Complex" in decision.reasoning or "complex" in decision.reasoning.lower()

    def test_route_heuristic_simple(self):
        """Simple problem routes to TRM."""
        from src.framework.mcts.llm_guided.integration import AgentType, MetaControllerAdapter

        adapter = MetaControllerAdapter()
        decision = adapter.route("simple basic task", {})
        assert decision.selected_agent == AgentType.TRM

    def test_route_heuristic_default(self):
        """Default routes to LLM_MCTS."""
        from src.framework.mcts.llm_guided.integration import AgentType, MetaControllerAdapter

        adapter = MetaControllerAdapter()
        decision = adapter.route("Write a function that sums numbers", {})
        assert decision.selected_agent == AgentType.LLM_MCTS

    def test_route_heuristic_long_problem(self):
        """Long problem routes to HRM."""
        from src.framework.mcts.llm_guided.integration import AgentType, MetaControllerAdapter

        adapter = MetaControllerAdapter()
        long_problem = "x " * 300  # > 500 chars
        decision = adapter.route(long_problem, {})
        assert decision.selected_agent == AgentType.HRM

    def test_route_neural(self):
        """Neural routing path."""
        from src.framework.mcts.llm_guided.integration import AgentType, MetaControllerAdapter

        mock_mc = MagicMock()
        mock_mc.extract_features.return_value = "features"
        mock_mc.predict.return_value = MagicMock(
            agent="trm",
            confidence=0.95,
            probabilities={"hrm": 0.02, "trm": 0.95, "mcts": 0.03},
        )
        adapter = MetaControllerAdapter(meta_controller=mock_mc)
        decision = adapter.route("problem", {"key": "val"})
        assert decision.selected_agent == AgentType.TRM
        assert decision.confidence == 0.95
        mock_mc.extract_features.assert_called_once_with({"key": "val"})

    def test_routing_history_tracked(self):
        from src.framework.mcts.llm_guided.integration import MetaControllerAdapter

        adapter = MetaControllerAdapter()
        adapter.route("a problem", {})
        adapter.route("another problem", {})
        assert len(adapter._routing_history) == 2

    def test_get_routing_statistics_empty(self):
        from src.framework.mcts.llm_guided.integration import MetaControllerAdapter

        adapter = MetaControllerAdapter()
        stats = adapter.get_routing_statistics()
        assert stats == {"total_decisions": 0}

    def test_get_routing_statistics_with_history(self):
        from src.framework.mcts.llm_guided.integration import MetaControllerAdapter

        adapter = MetaControllerAdapter()
        adapter.route("complex optimization task", {})
        adapter.route("simple basic task", {})
        adapter.route("general task about code", {})

        stats = adapter.get_routing_statistics()
        assert stats["total_decisions"] == 3
        assert "agent_distribution" in stats
        assert "average_confidence" in stats
        assert stats["average_confidence"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# UnifiedSearchOrchestrator tests
# ---------------------------------------------------------------------------


class TestUnifiedSearchOrchestrator:
    @pytest.mark.asyncio
    async def test_search_default_mcts_route(self):
        """Search with meta-controller disabled defaults to LLM_MCTS."""
        from src.framework.mcts.llm_guided.integration import (
            AgentType,
            IntegrationConfig,
            UnifiedSearchOrchestrator,
        )

        config = IntegrationConfig(
            use_meta_controller=False,
            use_hrm_decomposition=False,
            use_trm_refinement=False,
        )

        mock_mcts_result = MagicMock()
        mock_mcts_result.solution_found = True
        mock_mcts_result.best_code = "solution code"
        mock_mcts_result.best_value = 0.9
        mock_mcts_result.num_iterations = 10
        mock_mcts_result.num_expansions = 5
        mock_mcts_result.llm_calls = 3

        with patch.object(
            UnifiedSearchOrchestrator,
            "__init__",
            lambda self, *a, **kw: None,
        ):
            orch = UnifiedSearchOrchestrator.__new__(UnifiedSearchOrchestrator)
            orch._config = config
            orch._router = MagicMock()
            orch._hrm = MagicMock()
            orch._trm = MagicMock()
            orch._mcts_engine = AsyncMock()
            orch._mcts_engine.search = AsyncMock(return_value=mock_mcts_result)
            orch._logger = MagicMock()

            result = await orch.search("problem", ["test1"])

            assert result.solution_found is True
            assert result.best_code == "solution code"
            assert result.agent_used == AgentType.LLM_MCTS

    @pytest.mark.asyncio
    async def test_search_with_hrm_decomposition(self):
        """Search uses HRM decomposition when routed to HRM."""
        from src.framework.mcts.llm_guided.integration import (
            AgentType,
            IntegrationConfig,
            RoutingDecision,
            SubProblemDecomposition,
            UnifiedSearchOrchestrator,
        )

        config = IntegrationConfig(
            use_meta_controller=True,
            use_hrm_decomposition=True,
            use_trm_refinement=False,
        )

        mock_mcts_result = MagicMock()
        mock_mcts_result.solution_found = True
        mock_mcts_result.best_code = "combined"
        mock_mcts_result.best_value = 0.8
        mock_mcts_result.num_iterations = 5
        mock_mcts_result.num_expansions = 3
        mock_mcts_result.llm_calls = 2

        with patch.object(
            UnifiedSearchOrchestrator,
            "__init__",
            lambda self, *a, **kw: None,
        ):
            orch = UnifiedSearchOrchestrator.__new__(UnifiedSearchOrchestrator)
            orch._config = config
            orch._router = MagicMock()
            orch._router.route.return_value = RoutingDecision(
                selected_agent=AgentType.HRM,
                confidence=0.9,
                probabilities={"hrm": 0.9},
            )
            orch._hrm = AsyncMock()
            orch._hrm.decompose = AsyncMock(
                return_value=SubProblemDecomposition(
                    original_problem="problem",
                    subproblems=["sub1", "sub2"],
                    hierarchy_levels=[1, 1],
                    confidences=[0.9, 0.8],
                )
            )
            orch._trm = AsyncMock()
            orch._mcts_engine = AsyncMock()
            orch._mcts_engine.search = AsyncMock(return_value=mock_mcts_result)
            orch._logger = MagicMock()

            result = await orch.search("problem", ["test1"])

            assert result.decomposition is not None
            assert result.decomposition.num_subproblems == 2
            # MCTS should have been called twice (once per subproblem)
            assert orch._mcts_engine.search.call_count == 2

    @pytest.mark.asyncio
    async def test_search_with_trm_refinement(self):
        """Search uses TRM refinement when applicable."""
        from src.framework.mcts.llm_guided.integration import (
            AgentType,
            IntegrationConfig,
            RefinementResult,
            RoutingDecision,
            UnifiedSearchOrchestrator,
        )

        config = IntegrationConfig(
            use_meta_controller=True,
            use_hrm_decomposition=False,
            use_trm_refinement=True,
        )

        mock_mcts_result = MagicMock()
        mock_mcts_result.solution_found = True
        mock_mcts_result.best_code = "initial"
        mock_mcts_result.best_value = 0.7
        mock_mcts_result.num_iterations = 5
        mock_mcts_result.num_expansions = 3
        mock_mcts_result.llm_calls = 2

        with patch.object(
            UnifiedSearchOrchestrator,
            "__init__",
            lambda self, *a, **kw: None,
        ):
            orch = UnifiedSearchOrchestrator.__new__(UnifiedSearchOrchestrator)
            orch._config = config
            orch._router = MagicMock()
            orch._router.route.return_value = RoutingDecision(
                selected_agent=AgentType.LLM_MCTS,
                confidence=0.9,
                probabilities={"mcts": 0.9},
            )
            orch._hrm = AsyncMock()
            orch._trm = AsyncMock()
            orch._trm.refine = AsyncMock(
                return_value=RefinementResult(
                    original_code="initial",
                    refined_code="refined",
                    num_iterations=2,
                    converged=True,
                    improvement_score=0.3,
                )
            )
            orch._mcts_engine = AsyncMock()
            orch._mcts_engine.search = AsyncMock(return_value=mock_mcts_result)
            orch._logger = MagicMock()

            result = await orch.search("problem", ["test1"])

            assert result.refinement is not None
            assert result.best_code == "refined"

    def test_get_statistics(self):
        from src.framework.mcts.llm_guided.integration import UnifiedSearchOrchestrator

        with patch.object(
            UnifiedSearchOrchestrator,
            "__init__",
            lambda self, *a, **kw: None,
        ):
            orch = UnifiedSearchOrchestrator.__new__(UnifiedSearchOrchestrator)
            orch._mcts_engine = MagicMock()
            orch._mcts_engine.get_statistics.return_value = {"iterations": 10}
            orch._router = MagicMock()
            orch._router.get_routing_statistics.return_value = {"total_decisions": 5}
            orch._router.has_meta_controller = False
            orch._hrm = MagicMock()
            orch._hrm.has_neural_agent = False
            orch._trm = MagicMock()
            orch._trm.has_neural_agent = True

            stats = orch.get_statistics()
            assert stats["mcts"] == {"iterations": 10}
            assert stats["routing"] == {"total_decisions": 5}
            assert stats["hrm_available"] is False
            assert stats["trm_available"] is True
            assert stats["meta_controller_available"] is False


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestProtocols:
    def test_hrm_adapter_is_problem_decomposer(self):
        from src.framework.mcts.llm_guided.integration import HRMAdapter, ProblemDecomposer

        adapter = HRMAdapter()
        assert isinstance(adapter, ProblemDecomposer)

    def test_trm_adapter_is_solution_refiner(self):
        from src.framework.mcts.llm_guided.integration import SolutionRefiner, TRMAdapter

        adapter = TRMAdapter()
        assert isinstance(adapter, SolutionRefiner)

    def test_meta_controller_adapter_is_agent_router(self):
        from src.framework.mcts.llm_guided.integration import AgentRouter, MetaControllerAdapter

        adapter = MetaControllerAdapter()
        assert isinstance(adapter, AgentRouter)


# ---------------------------------------------------------------------------
# create_unified_orchestrator factory tests
# ---------------------------------------------------------------------------


class TestCreateUnifiedOrchestrator:
    @patch("src.framework.mcts.llm_guided.integration.LLMGuidedMCTSEngine")
    def test_create_with_defaults(self, mock_engine_cls):
        from src.framework.mcts.llm_guided.integration import (
            UnifiedSearchOrchestrator,
            create_unified_orchestrator,
        )

        mock_llm = MagicMock()
        orch = create_unified_orchestrator(mock_llm)
        assert isinstance(orch, UnifiedSearchOrchestrator)

    @patch("src.framework.mcts.llm_guided.integration.LLMGuidedMCTSEngine")
    def test_create_with_preset(self, mock_engine_cls):
        from src.framework.mcts.llm_guided.integration import create_unified_orchestrator

        mock_llm = MagicMock()
        orch = create_unified_orchestrator(mock_llm, preset="fast")
        assert orch is not None

    @patch("src.framework.mcts.llm_guided.integration.LLMGuidedMCTSEngine")
    def test_create_with_agents(self, mock_engine_cls):
        from src.framework.mcts.llm_guided.integration import create_unified_orchestrator

        mock_llm = MagicMock()
        orch = create_unified_orchestrator(
            mock_llm,
            hrm_agent=MagicMock(),
            trm_agent=MagicMock(),
            meta_controller=MagicMock(),
        )
        assert orch is not None
