"""Unit tests for EnterpriseGraphBuilder and EnterpriseAgentState.

Tests cover the graph extension module at
src/enterprise/integration/graph_extension.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.enterprise.base.domain_detector import DetectionResult, DomainDetector
from src.enterprise.config.enterprise_settings import EnterpriseDomain
from src.enterprise.integration.graph_extension import (
    EnterpriseAgentState,
    EnterpriseGraphBuilder,
)


@pytest.fixture
def mock_domain_detector() -> MagicMock:
    detector = MagicMock(spec=DomainDetector)
    detector.detect.return_value = DetectionResult(
        domain=EnterpriseDomain.MA_DUE_DILIGENCE,
        confidence=0.8,
        all_scores={EnterpriseDomain.MA_DUE_DILIGENCE: 0.8},
    )
    return detector


@pytest.fixture
def mock_use_case() -> MagicMock:
    use_case = MagicMock()
    use_case.name = "ma_due_diligence"
    use_case.domain = "finance"
    use_case.process = AsyncMock(
        return_value={
            "result": "Analysis complete",
            "confidence": 0.85,
            "domain_state": {"key": "value"},
            "agent_results": {"agent1": "result1"},
            "mcts_stats": {"iterations": 50},
        }
    )
    return use_case


@pytest.fixture
def mock_factory(mock_use_case: MagicMock) -> MagicMock:
    factory = MagicMock()
    factory.create_all_enabled.return_value = {
        EnterpriseDomain.MA_DUE_DILIGENCE: mock_use_case,
    }
    factory.create_from_query.return_value = mock_use_case
    return factory


@pytest.fixture
def builder(
    mock_factory: MagicMock,
    mock_domain_detector: MagicMock,
) -> EnterpriseGraphBuilder:
    return EnterpriseGraphBuilder(
        base_graph_builder=None,
        use_case_factory=mock_factory,
        domain_detector=mock_domain_detector,
    )


@pytest.fixture
def sample_state() -> EnterpriseAgentState:
    return EnterpriseAgentState(
        query="Analyze acquisition target TestCo",
        use_mcts=True,
        use_rag=False,
        hrm_results={"context": "some rag context"},
        trm_results={},
        agent_outputs=[],
        confidence_scores={},
        consensus_reached=False,
        iteration=0,
        max_iterations=5,
    )


@pytest.mark.unit
class TestEnterpriseAgentState:
    """Tests for EnterpriseAgentState TypedDict."""

    def test_standard_fields(self):
        state: EnterpriseAgentState = {
            "query": "test query",
            "use_mcts": True,
            "use_rag": False,
            "hrm_results": {},
            "trm_results": {},
            "agent_outputs": [],
            "confidence_scores": {},
            "consensus_reached": False,
            "iteration": 0,
            "max_iterations": 5,
        }
        assert state["query"] == "test query"
        assert state["use_mcts"] is True

    def test_enterprise_fields(self):
        state: EnterpriseAgentState = {
            "query": "test",
            "use_mcts": False,
            "use_rag": False,
            "hrm_results": {},
            "trm_results": {},
            "agent_outputs": [],
            "confidence_scores": {},
            "consensus_reached": False,
            "iteration": 0,
            "max_iterations": 3,
            "enterprise_domain": "ma_due_diligence",
            "domain_state": {"key": "val"},
            "domain_agents_results": {"agent": "result"},
            "use_case_metadata": {"meta": "data"},
            "enterprise_result": {"result": "done"},
        }
        assert state["enterprise_domain"] == "ma_due_diligence"
        assert state["domain_state"] == {"key": "val"}

    def test_enterprise_fields_are_optional(self):
        """Enterprise-specific fields are NotRequired and can be omitted."""
        state: EnterpriseAgentState = {
            "query": "test",
            "use_mcts": False,
            "use_rag": False,
            "hrm_results": {},
            "trm_results": {},
            "agent_outputs": [],
            "confidence_scores": {},
            "consensus_reached": False,
            "iteration": 0,
            "max_iterations": 3,
        }
        assert "enterprise_domain" not in state
        assert "domain_state" not in state


@pytest.mark.unit
class TestEnterpriseGraphBuilder:
    """Tests for EnterpriseGraphBuilder."""

    def test_init_loads_use_cases(self, builder: EnterpriseGraphBuilder, mock_factory: MagicMock):
        mock_factory.create_all_enabled.assert_called_once()
        assert len(builder.use_cases) == 1
        assert EnterpriseDomain.MA_DUE_DILIGENCE in builder.use_cases

    def test_use_cases_property(self, builder: EnterpriseGraphBuilder, mock_use_case: MagicMock):
        use_cases = builder.use_cases
        assert use_cases[EnterpriseDomain.MA_DUE_DILIGENCE] is mock_use_case

    def test_get_use_case_found(self, builder: EnterpriseGraphBuilder, mock_use_case: MagicMock):
        result = builder.get_use_case(EnterpriseDomain.MA_DUE_DILIGENCE)
        assert result is mock_use_case

    def test_get_use_case_not_found(self, builder: EnterpriseGraphBuilder):
        result = builder.get_use_case(EnterpriseDomain.CLINICAL_TRIAL)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_enterprise_query(
        self,
        builder: EnterpriseGraphBuilder,
        sample_state: EnterpriseAgentState,
        mock_use_case: MagicMock,
    ):
        result = await builder.process_enterprise_query(sample_state)

        assert result["enterprise_domain"] == "finance"
        assert result["domain_state"] == {"key": "value"}
        assert result["domain_agents_results"] == {"agent1": "result1"}
        assert result["enterprise_result"]["result"] == "Analysis complete"
        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0]["agent"] == "enterprise_ma_due_diligence"
        assert result["agent_outputs"][0]["confidence"] == 0.85
        assert result["use_case_metadata"]["use_case"] == "ma_due_diligence"

    @pytest.mark.asyncio
    async def test_process_enterprise_query_passes_context(
        self,
        builder: EnterpriseGraphBuilder,
        sample_state: EnterpriseAgentState,
        mock_use_case: MagicMock,
    ):
        await builder.process_enterprise_query(sample_state)

        mock_use_case.process.assert_called_once()
        call_kwargs = mock_use_case.process.call_args[1]
        assert call_kwargs["query"] == "Analyze acquisition target TestCo"
        assert call_kwargs["use_mcts"] is True
        assert "rag_context" in call_kwargs["context"]
        assert call_kwargs["context"]["rag_context"] == "some rag context"

    @pytest.mark.asyncio
    async def test_process_enterprise_query_no_domain_detected(
        self,
        builder: EnterpriseGraphBuilder,
        sample_state: EnterpriseAgentState,
        mock_factory: MagicMock,
    ):
        mock_factory.create_from_query.return_value = None

        result = await builder.process_enterprise_query(sample_state)

        assert result["enterprise_domain"] is None
        assert result["enterprise_result"] is None

    @pytest.mark.asyncio
    async def test_process_enterprise_query_with_use_case_metadata(
        self,
        builder: EnterpriseGraphBuilder,
        mock_use_case: MagicMock,
    ):
        state: EnterpriseAgentState = {
            "query": "Analyze acquisition",
            "use_mcts": False,
            "use_rag": False,
            "hrm_results": {},
            "trm_results": {},
            "agent_outputs": [],
            "confidence_scores": {},
            "consensus_reached": False,
            "iteration": 0,
            "max_iterations": 3,
            "use_case_metadata": {"extra_key": "extra_value"},
        }

        await builder.process_enterprise_query(state)

        call_kwargs = mock_use_case.process.call_args[1]
        assert call_kwargs["context"]["extra_key"] == "extra_value"

    def test_create_enterprise_node_success(
        self,
        builder: EnterpriseGraphBuilder,
    ):
        node_handler = builder.create_enterprise_node(EnterpriseDomain.MA_DUE_DILIGENCE)
        assert callable(node_handler)

    def test_create_enterprise_node_unknown_domain(
        self,
        builder: EnterpriseGraphBuilder,
    ):
        with pytest.raises(ValueError, match="Use case not loaded for domain"):
            builder.create_enterprise_node(EnterpriseDomain.CLINICAL_TRIAL)

    @pytest.mark.asyncio
    async def test_enterprise_node_handler_execution(
        self,
        builder: EnterpriseGraphBuilder,
        sample_state: EnterpriseAgentState,
        mock_use_case: MagicMock,
    ):
        node_handler = builder.create_enterprise_node(EnterpriseDomain.MA_DUE_DILIGENCE)
        result = await node_handler(sample_state)

        assert result["enterprise_domain"] == "ma_due_diligence"
        assert result["domain_state"] == {"key": "value"}
        assert result["domain_agents_results"] == {"agent1": "result1"}
        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0]["agent"] == "enterprise_ma_due_diligence"

    @pytest.mark.asyncio
    async def test_enterprise_node_handler_passes_mcts_flag(
        self,
        builder: EnterpriseGraphBuilder,
        mock_use_case: MagicMock,
    ):
        state: EnterpriseAgentState = {
            "query": "Analyze acquisition",
            "use_mcts": False,
            "use_rag": False,
            "hrm_results": {},
            "trm_results": {},
            "agent_outputs": [],
            "confidence_scores": {},
            "consensus_reached": False,
            "iteration": 0,
            "max_iterations": 3,
        }
        node_handler = builder.create_enterprise_node(EnterpriseDomain.MA_DUE_DILIGENCE)
        await node_handler(state)

        call_kwargs = mock_use_case.process.call_args[1]
        assert call_kwargs["use_mcts"] is False

    def test_should_route_to_enterprise_true(
        self,
        builder: EnterpriseGraphBuilder,
        sample_state: EnterpriseAgentState,
        mock_domain_detector: MagicMock,
    ):
        mock_domain_detector.detect.return_value = DetectionResult(
            domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            confidence=0.8,
        )
        assert builder.should_route_to_enterprise(sample_state) is True

    def test_should_route_to_enterprise_false(
        self,
        builder: EnterpriseGraphBuilder,
        sample_state: EnterpriseAgentState,
        mock_domain_detector: MagicMock,
    ):
        mock_domain_detector.detect.return_value = DetectionResult(
            domain=None,
            confidence=0.0,
        )
        assert builder.should_route_to_enterprise(sample_state) is False

    def test_should_route_to_enterprise_empty_query(
        self,
        builder: EnterpriseGraphBuilder,
        mock_domain_detector: MagicMock,
    ):
        mock_domain_detector.detect.return_value = DetectionResult(
            domain=None,
            confidence=0.0,
        )
        state: EnterpriseAgentState = {
            "query": "",
            "use_mcts": False,
            "use_rag": False,
            "hrm_results": {},
            "trm_results": {},
            "agent_outputs": [],
            "confidence_scores": {},
            "consensus_reached": False,
            "iteration": 0,
            "max_iterations": 3,
        }
        assert builder.should_route_to_enterprise(state) is False

    def test_get_enterprise_route_enterprise_domain(
        self,
        builder: EnterpriseGraphBuilder,
        sample_state: EnterpriseAgentState,
        mock_domain_detector: MagicMock,
        mock_factory: MagicMock,
        mock_use_case: MagicMock,
    ):
        mock_domain_detector.detect.return_value = DetectionResult(
            domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            confidence=0.8,
        )
        route = builder.get_enterprise_route(sample_state)
        assert route == "enterprise_ma_due_diligence"

    def test_get_enterprise_route_standard_when_no_domain(
        self,
        builder: EnterpriseGraphBuilder,
        sample_state: EnterpriseAgentState,
        mock_domain_detector: MagicMock,
    ):
        mock_domain_detector.detect.return_value = DetectionResult(
            domain=None,
            confidence=0.0,
        )
        route = builder.get_enterprise_route(sample_state)
        assert route == "standard"

    def test_get_enterprise_route_standard_when_no_use_case(
        self,
        builder: EnterpriseGraphBuilder,
        sample_state: EnterpriseAgentState,
        mock_domain_detector: MagicMock,
        mock_factory: MagicMock,
    ):
        mock_domain_detector.detect.return_value = DetectionResult(
            domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            confidence=0.8,
        )
        mock_factory.create_from_query.return_value = None
        route = builder.get_enterprise_route(sample_state)
        assert route == "standard"

    def test_init_with_defaults(self, mock_factory: MagicMock, mock_domain_detector: MagicMock):
        """Builder can be created with minimal arguments."""
        builder = EnterpriseGraphBuilder(
            use_case_factory=mock_factory,
            domain_detector=mock_domain_detector,
        )
        assert builder._base_builder is None

    def test_init_with_base_builder(self, mock_factory: MagicMock, mock_domain_detector: MagicMock):
        mock_base = MagicMock()
        builder = EnterpriseGraphBuilder(
            base_graph_builder=mock_base,
            use_case_factory=mock_factory,
            domain_detector=mock_domain_detector,
        )
        assert builder._base_builder is mock_base
