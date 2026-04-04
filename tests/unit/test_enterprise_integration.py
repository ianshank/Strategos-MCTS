"""
Tests for enterprise integration modules.

Tests EnterpriseGraphBuilder (graph_extension.py) and
EnterpriseMetaControllerAdapter (meta_controller_adapter.py).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.enterprise.config.enterprise_settings import EnterpriseDomain
from src.enterprise.integration.graph_extension import (
    EnterpriseAgentState,
    EnterpriseGraphBuilder,
)
from src.enterprise.integration.meta_controller_adapter import (
    EnterpriseMetaControllerAdapter,
    EnterpriseMetaControllerFeatures,
)

# ---------------------------------------------------------------------------
# EnterpriseAgentState
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnterpriseAgentState:
    """Tests for the EnterpriseAgentState TypedDict."""

    def test_create_minimal_state(self):
        """State can be created with only standard fields."""
        state: EnterpriseAgentState = {"query": "test query"}
        assert state["query"] == "test query"

    def test_create_full_state(self):
        """State can contain both standard and enterprise fields."""
        state: EnterpriseAgentState = {
            "query": "Analyze acquisition",
            "use_mcts": True,
            "use_rag": False,
            "enterprise_domain": "ma_due_diligence",
            "domain_state": {"phase": "initial"},
            "domain_agents_results": {},
            "use_case_metadata": {"use_case": "ma"},
            "enterprise_result": {"result": "ok"},
        }
        assert state["enterprise_domain"] == "ma_due_diligence"
        assert state["use_mcts"] is True


# ---------------------------------------------------------------------------
# EnterpriseGraphBuilder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnterpriseGraphBuilder:
    """Tests for EnterpriseGraphBuilder."""

    @pytest.fixture
    def mock_factory(self):
        factory = MagicMock()
        factory.create_all_enabled.return_value = {}
        factory.create_from_query.return_value = None
        return factory

    @pytest.fixture
    def mock_detector(self):
        detector = MagicMock()
        result = MagicMock()
        result.is_detected = False
        detector.detect.return_value = result
        return detector

    @pytest.fixture
    def builder(self, mock_factory, mock_detector):
        return EnterpriseGraphBuilder(
            base_graph_builder=None,
            use_case_factory=mock_factory,
            domain_detector=mock_detector,
            logger=logging.getLogger("test"),
        )

    # -- init / load --

    def test_init_defaults(self, mock_factory, mock_detector):
        """Builder initializes with injected dependencies."""
        builder = EnterpriseGraphBuilder(
            use_case_factory=mock_factory,
            domain_detector=mock_detector,
        )
        assert builder.use_cases == {}
        mock_factory.create_all_enabled.assert_called_once()

    def test_load_use_cases_populates_dict(self, mock_factory, mock_detector):
        """_load_use_cases fills use_cases from factory."""
        mock_uc = MagicMock()
        mock_factory.create_all_enabled.return_value = {
            EnterpriseDomain.MA_DUE_DILIGENCE: mock_uc,
        }
        builder = EnterpriseGraphBuilder(
            use_case_factory=mock_factory,
            domain_detector=mock_detector,
        )
        assert builder.use_cases[EnterpriseDomain.MA_DUE_DILIGENCE] is mock_uc

    # -- get_use_case --

    def test_get_use_case_returns_none_for_missing(self, builder):
        assert builder.get_use_case(EnterpriseDomain.CLINICAL_TRIAL) is None

    def test_get_use_case_returns_loaded(self, mock_factory, mock_detector):
        mock_uc = MagicMock()
        mock_factory.create_all_enabled.return_value = {
            EnterpriseDomain.CLINICAL_TRIAL: mock_uc,
        }
        builder = EnterpriseGraphBuilder(
            use_case_factory=mock_factory,
            domain_detector=mock_detector,
        )
        assert builder.get_use_case(EnterpriseDomain.CLINICAL_TRIAL) is mock_uc

    # -- process_enterprise_query --

    @pytest.mark.asyncio
    async def test_process_enterprise_query_no_domain(self, builder):
        """Returns None values when no domain is detected."""
        state: EnterpriseAgentState = {"query": "generic question"}
        result = await builder.process_enterprise_query(state)
        assert result["enterprise_domain"] is None
        assert result["enterprise_result"] is None

    @pytest.mark.asyncio
    async def test_process_enterprise_query_with_domain(self, mock_factory, mock_detector):
        """Returns populated result when domain is detected."""
        mock_uc = AsyncMock()
        mock_uc.name = "ma_due_diligence"
        mock_uc.domain = "finance"
        mock_uc.process.return_value = {
            "result": "Analysis complete",
            "confidence": 0.85,
            "domain_state": {"phase": "final"},
            "agent_results": {"doc_agent": {}},
            "mcts_stats": {"iterations": 10},
        }
        mock_factory.create_from_query.return_value = mock_uc
        mock_factory.create_all_enabled.return_value = {}

        builder = EnterpriseGraphBuilder(
            use_case_factory=mock_factory,
            domain_detector=mock_detector,
        )

        state: EnterpriseAgentState = {
            "query": "Analyze acquisition of TargetCo",
            "use_mcts": True,
            "hrm_results": {"context": "some context"},
        }
        result = await builder.process_enterprise_query(state)

        assert result["enterprise_domain"] == "finance"
        assert result["enterprise_result"]["confidence"] == 0.85
        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0]["agent"] == "enterprise_ma_due_diligence"

    @pytest.mark.asyncio
    async def test_process_enterprise_query_uses_mcts_default(self, mock_factory, mock_detector):
        """use_mcts defaults to True when not in state."""
        mock_uc = AsyncMock()
        mock_uc.name = "test"
        mock_uc.domain = "test"
        mock_uc.process.return_value = {"result": "", "confidence": 0.5}
        mock_factory.create_from_query.return_value = mock_uc
        mock_factory.create_all_enabled.return_value = {}

        builder = EnterpriseGraphBuilder(use_case_factory=mock_factory, domain_detector=mock_detector)
        state: EnterpriseAgentState = {"query": "test"}
        await builder.process_enterprise_query(state)

        _, kwargs = mock_uc.process.call_args
        assert kwargs["use_mcts"] is True

    # -- create_enterprise_node --

    def test_create_enterprise_node_raises_for_missing_domain(self, builder):
        """Raises ValueError if domain not loaded."""
        with pytest.raises(ValueError, match="Use case not loaded"):
            builder.create_enterprise_node(EnterpriseDomain.MA_DUE_DILIGENCE)

    @pytest.mark.asyncio
    async def test_create_enterprise_node_returns_handler(self, mock_factory, mock_detector):
        """Created node handler processes state correctly."""
        mock_uc = AsyncMock()
        mock_uc.name = "ma_dd"
        mock_uc.process.return_value = {
            "result": "done",
            "confidence": 0.9,
            "domain_state": {},
            "agent_results": {},
            "mcts_stats": {},
        }
        mock_factory.create_all_enabled.return_value = {
            EnterpriseDomain.MA_DUE_DILIGENCE: mock_uc,
        }

        builder = EnterpriseGraphBuilder(use_case_factory=mock_factory, domain_detector=mock_detector)
        handler = builder.create_enterprise_node(EnterpriseDomain.MA_DUE_DILIGENCE)

        state: EnterpriseAgentState = {"query": "Analyze deal", "use_mcts": False}
        result = await handler(state)

        assert result["enterprise_domain"] == "ma_due_diligence"
        assert len(result["agent_outputs"]) == 1

    # -- should_route_to_enterprise --

    def test_should_route_to_enterprise_false(self, builder, mock_detector):
        state: EnterpriseAgentState = {"query": "Hello world"}
        assert builder.should_route_to_enterprise(state) is False
        mock_detector.detect.assert_called_once_with("Hello world")

    def test_should_route_to_enterprise_true(self, mock_factory, mock_detector):
        result = MagicMock()
        result.is_detected = True
        mock_detector.detect.return_value = result
        mock_factory.create_all_enabled.return_value = {}

        builder = EnterpriseGraphBuilder(use_case_factory=mock_factory, domain_detector=mock_detector)
        state: EnterpriseAgentState = {"query": "Analyze acquisition"}
        assert builder.should_route_to_enterprise(state) is True

    # -- get_enterprise_route --

    def test_get_enterprise_route_standard(self, builder, mock_factory):
        """Returns 'standard' when no enterprise domain detected."""
        state: EnterpriseAgentState = {"query": "generic"}
        assert builder.get_enterprise_route(state) == "standard"

    def test_get_enterprise_route_enterprise(self, mock_factory, mock_detector):
        """Returns enterprise route when domain detected."""
        det_result = MagicMock()
        det_result.is_detected = True
        mock_detector.detect.return_value = det_result

        mock_uc = MagicMock()
        mock_uc.name = "ma_due_diligence"
        mock_factory.create_from_query.return_value = mock_uc
        mock_factory.create_all_enabled.return_value = {}

        builder = EnterpriseGraphBuilder(use_case_factory=mock_factory, domain_detector=mock_detector)
        state: EnterpriseAgentState = {"query": "Analyze acquisition"}
        assert builder.get_enterprise_route(state) == "enterprise_ma_due_diligence"

    def test_get_enterprise_route_fallback_standard(self, mock_factory, mock_detector):
        """Falls back to standard when detected but no use case from query."""
        det_result = MagicMock()
        det_result.is_detected = True
        mock_detector.detect.return_value = det_result
        mock_factory.create_from_query.return_value = None
        mock_factory.create_all_enabled.return_value = {}

        builder = EnterpriseGraphBuilder(use_case_factory=mock_factory, domain_detector=mock_detector)
        state: EnterpriseAgentState = {"query": "something"}
        assert builder.get_enterprise_route(state) == "standard"


# ---------------------------------------------------------------------------
# EnterpriseMetaControllerFeatures
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnterpriseMetaControllerFeatures:
    """Tests for the features dataclass."""

    def test_defaults(self):
        features = EnterpriseMetaControllerFeatures()
        assert features.hrm_confidence == 0.0
        assert features.detected_domain is None
        assert features.regulatory_jurisdictions == []
        assert features.is_time_sensitive is False

    def test_custom_values(self):
        features = EnterpriseMetaControllerFeatures(
            hrm_confidence=0.8,
            detected_domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            domain_confidence=0.9,
            regulatory_jurisdictions=["US", "EU"],
        )
        assert features.domain_confidence == 0.9
        assert len(features.regulatory_jurisdictions) == 2


# ---------------------------------------------------------------------------
# EnterpriseMetaControllerAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnterpriseMetaControllerAdapter:
    """Tests for EnterpriseMetaControllerAdapter."""

    @pytest.fixture
    def mock_detector(self):
        detector = MagicMock()
        detector.detection_threshold = 0.05
        result = MagicMock()
        result.domain = None
        result.confidence = 0.0
        detector.detect.return_value = result
        detector.requires_compliance.return_value = False
        detector.estimate_complexity.return_value = 0.3
        detector.extract_jurisdictions.return_value = ["US"]
        return detector

    @pytest.fixture
    def adapter(self, mock_detector):
        return EnterpriseMetaControllerAdapter(
            domain_detector=mock_detector,
            logger=logging.getLogger("test"),
        )

    # -- init --

    def test_init_sets_threshold(self, mock_detector):
        EnterpriseMetaControllerAdapter(
            domain_detection_threshold=0.1,
            domain_detector=mock_detector,
        )
        assert mock_detector.detection_threshold == 0.1

    def test_enterprise_agents_list(self):
        assert "enterprise_ma" in EnterpriseMetaControllerAdapter.ENTERPRISE_AGENTS
        assert "hrm" in EnterpriseMetaControllerAdapter.ENTERPRISE_AGENTS

    # -- detect_domain --

    def test_detect_domain_none(self, adapter, mock_detector):
        domain, confidence = adapter.detect_domain("hello world")
        assert domain is None
        assert confidence == 0.0

    def test_detect_domain_found(self, mock_detector):
        result = MagicMock()
        result.domain = EnterpriseDomain.CLINICAL_TRIAL
        result.confidence = 0.7
        mock_detector.detect.return_value = result

        adapter = EnterpriseMetaControllerAdapter(domain_detector=mock_detector)
        domain, confidence = adapter.detect_domain("clinical trial phase 2")
        assert domain == EnterpriseDomain.CLINICAL_TRIAL
        assert confidence == 0.7

    # -- extract_enterprise_features --

    def test_extract_enterprise_features_basic(self, adapter, mock_detector):
        state = {"confidence_scores": {"hrm": 0.6, "trm": 0.4}, "iteration": 2}
        features = adapter.extract_enterprise_features(state, "simple query")
        assert features.hrm_confidence == 0.6
        assert features.trm_confidence == 0.4
        assert features.iteration == 2
        assert features.query_length == len("simple query")

    def test_extract_enterprise_features_empty_state(self, adapter):
        features = adapter.extract_enterprise_features({}, "test")
        assert features.hrm_confidence == 0.0
        assert features.last_agent == "none"
        assert features.has_rag_context is False

    def test_extract_enterprise_features_with_rag(self, adapter):
        state = {"rag_context": "some context"}
        features = adapter.extract_enterprise_features(state, "query")
        assert features.has_rag_context is True

    # -- route_to_enterprise --

    def test_route_to_enterprise_ma(self, adapter, mock_detector):
        mock_detector.detection_threshold = 0.05
        features = EnterpriseMetaControllerFeatures(
            detected_domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            domain_confidence=0.8,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_ma"

    def test_route_to_enterprise_clinical(self, adapter, mock_detector):
        mock_detector.detection_threshold = 0.05
        features = EnterpriseMetaControllerFeatures(
            detected_domain=EnterpriseDomain.CLINICAL_TRIAL,
            domain_confidence=0.6,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_clinical"

    def test_route_to_enterprise_regulatory(self, adapter, mock_detector):
        mock_detector.detection_threshold = 0.05
        features = EnterpriseMetaControllerFeatures(
            detected_domain=EnterpriseDomain.REGULATORY_COMPLIANCE,
            domain_confidence=0.5,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_regulatory"

    def test_route_below_threshold_falls_to_hrm(self, adapter, mock_detector):
        mock_detector.detection_threshold = 0.9
        features = EnterpriseMetaControllerFeatures(
            detected_domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            domain_confidence=0.1,
        )
        assert adapter.route_to_enterprise(features) == "hrm"

    def test_route_compliance_required(self, adapter, mock_detector):
        mock_detector.detection_threshold = 0.9
        features = EnterpriseMetaControllerFeatures(
            detected_domain=None,
            domain_confidence=0.0,
            requires_compliance_check=True,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_regulatory"

    def test_route_high_complexity_to_mcts(self, adapter, mock_detector):
        mock_detector.detection_threshold = 0.9
        features = EnterpriseMetaControllerFeatures(
            detected_domain=None,
            domain_confidence=0.0,
            requires_compliance_check=False,
            estimated_complexity=0.8,
        )
        assert adapter.route_to_enterprise(features) == "mcts"

    def test_route_default_hrm(self, adapter, mock_detector):
        mock_detector.detection_threshold = 0.9
        features = EnterpriseMetaControllerFeatures(
            detected_domain=None,
            domain_confidence=0.0,
            requires_compliance_check=False,
            estimated_complexity=0.3,
        )
        assert adapter.route_to_enterprise(features) == "hrm"

    # -- _is_time_sensitive --

    def test_is_time_sensitive_true(self, adapter):
        assert adapter._is_time_sensitive("This is urgent") is True
        assert adapter._is_time_sensitive("deadline approaching") is True
        assert adapter._is_time_sensitive("ASAP please") is True
        assert adapter._is_time_sensitive("this is critical") is True

    def test_is_time_sensitive_false(self, adapter):
        assert adapter._is_time_sensitive("normal request") is False

    # -- _requires_expert --

    def test_requires_expert_low_confidence(self, adapter):
        state = {"confidence_scores": {"max": 0.3}}
        assert adapter._requires_expert("query", state) is True

    def test_requires_expert_keyword(self, adapter):
        state = {"confidence_scores": {"max": 0.9}}
        assert adapter._requires_expert("need a legal opinion", state) is True
        assert adapter._requires_expert("specialist needed", state) is True

    def test_requires_expert_false(self, adapter):
        state = {"confidence_scores": {"max": 0.9}}
        assert adapter._requires_expert("simple query", state) is False
