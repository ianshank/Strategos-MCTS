"""Unit tests for EnterpriseMetaControllerAdapter and EnterpriseMetaControllerFeatures.

Tests cover the meta controller adapter module at
src/enterprise/integration/meta_controller_adapter.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.enterprise.base.domain_detector import DetectionResult, DomainDetector
from src.enterprise.config.enterprise_settings import EnterpriseDomain
from src.enterprise.integration.meta_controller_adapter import (
    EnterpriseMetaControllerAdapter,
    EnterpriseMetaControllerFeatures,
)


@pytest.fixture
def mock_domain_detector() -> MagicMock:
    detector = MagicMock(spec=DomainDetector)
    detector.detect.return_value = DetectionResult(
        domain=EnterpriseDomain.MA_DUE_DILIGENCE,
        confidence=0.75,
        all_scores={EnterpriseDomain.MA_DUE_DILIGENCE: 0.75},
    )
    detector.detection_threshold = 0.05
    detector.requires_compliance.return_value = False
    detector.estimate_complexity.return_value = 0.5
    detector.extract_jurisdictions.return_value = ["US"]
    return detector


@pytest.fixture
def adapter(mock_domain_detector: MagicMock) -> EnterpriseMetaControllerAdapter:
    return EnterpriseMetaControllerAdapter(
        base_controller=None,
        domain_detection_threshold=0.05,
        domain_detector=mock_domain_detector,
    )


@pytest.fixture
def sample_state() -> dict:
    return {
        "confidence_scores": {"hrm": 0.8, "trm": 0.6, "max": 0.8},
        "mcts_best_value": 0.7,
        "consensus_score": 0.65,
        "last_agent": "hrm",
        "iteration": 2,
        "rag_context": "some context",
    }


@pytest.mark.unit
class TestEnterpriseMetaControllerFeatures:
    """Tests for EnterpriseMetaControllerFeatures dataclass."""

    def test_defaults(self):
        features = EnterpriseMetaControllerFeatures()
        assert features.hrm_confidence == 0.0
        assert features.trm_confidence == 0.0
        assert features.mcts_value == 0.0
        assert features.consensus_score == 0.0
        assert features.last_agent == "none"
        assert features.iteration == 0
        assert features.query_length == 0
        assert features.has_rag_context is False
        assert features.detected_domain is None
        assert features.domain_confidence == 0.0
        assert features.requires_compliance_check is False
        assert features.estimated_complexity == 0.5
        assert features.regulatory_jurisdictions == []
        assert features.is_time_sensitive is False
        assert features.requires_expert_review is False

    def test_custom_values(self):
        features = EnterpriseMetaControllerFeatures(
            hrm_confidence=0.9,
            trm_confidence=0.7,
            mcts_value=0.85,
            consensus_score=0.8,
            last_agent="trm",
            iteration=3,
            query_length=150,
            has_rag_context=True,
            detected_domain=EnterpriseDomain.CLINICAL_TRIAL,
            domain_confidence=0.6,
            requires_compliance_check=True,
            estimated_complexity=0.8,
            regulatory_jurisdictions=["US", "EU"],
            is_time_sensitive=True,
            requires_expert_review=True,
        )
        assert features.hrm_confidence == 0.9
        assert features.detected_domain == EnterpriseDomain.CLINICAL_TRIAL
        assert features.regulatory_jurisdictions == ["US", "EU"]
        assert features.is_time_sensitive is True


@pytest.mark.unit
class TestEnterpriseMetaControllerAdapter:
    """Tests for EnterpriseMetaControllerAdapter."""

    def test_enterprise_agents_class_attribute(self):
        expected = [
            "hrm",
            "trm",
            "mcts",
            "enterprise_ma",
            "enterprise_clinical",
            "enterprise_regulatory",
        ]
        assert expected == EnterpriseMetaControllerAdapter.ENTERPRISE_AGENTS

    def test_init_sets_detection_threshold(self, mock_domain_detector: MagicMock):
        EnterpriseMetaControllerAdapter(
            domain_detection_threshold=0.3,
            domain_detector=mock_domain_detector,
        )
        assert mock_domain_detector.detection_threshold == 0.3

    def test_init_default_base_controller_is_none(self, adapter: EnterpriseMetaControllerAdapter):
        assert adapter._base_controller is None

    # --- detect_domain ---

    def test_detect_domain_returns_domain_and_confidence(
        self,
        adapter: EnterpriseMetaControllerAdapter,
        mock_domain_detector: MagicMock,
    ):
        domain, confidence = adapter.detect_domain("Analyze the acquisition of TargetCo")
        assert domain == EnterpriseDomain.MA_DUE_DILIGENCE
        assert confidence == 0.75
        mock_domain_detector.detect.assert_called_once_with("Analyze the acquisition of TargetCo")

    def test_detect_domain_none_when_no_match(
        self,
        adapter: EnterpriseMetaControllerAdapter,
        mock_domain_detector: MagicMock,
    ):
        mock_domain_detector.detect.return_value = DetectionResult(
            domain=None,
            confidence=0.0,
        )
        domain, confidence = adapter.detect_domain("What is the weather today?")
        assert domain is None
        assert confidence == 0.0

    # --- extract_enterprise_features ---

    def test_extract_enterprise_features_populates_all_fields(
        self,
        adapter: EnterpriseMetaControllerAdapter,
        sample_state: dict,
        mock_domain_detector: MagicMock,
    ):
        query = "Analyze the acquisition target"
        features = adapter.extract_enterprise_features(sample_state, query)

        assert features.hrm_confidence == 0.8
        assert features.trm_confidence == 0.6
        assert features.mcts_value == 0.7
        assert features.consensus_score == 0.65
        assert features.last_agent == "hrm"
        assert features.iteration == 2
        assert features.query_length == len(query)
        assert features.has_rag_context is True
        assert features.detected_domain == EnterpriseDomain.MA_DUE_DILIGENCE
        assert features.domain_confidence == 0.75

    def test_extract_enterprise_features_empty_state(
        self,
        adapter: EnterpriseMetaControllerAdapter,
    ):
        features = adapter.extract_enterprise_features({}, "test query")

        assert features.hrm_confidence == 0.0
        assert features.trm_confidence == 0.0
        assert features.mcts_value == 0.0
        assert features.consensus_score == 0.0
        assert features.last_agent == "none"
        assert features.iteration == 0
        assert features.has_rag_context is False

    def test_extract_enterprise_features_calls_detector_methods(
        self,
        adapter: EnterpriseMetaControllerAdapter,
        mock_domain_detector: MagicMock,
    ):
        state = {"confidence_scores": {}}
        query = "Check GDPR compliance for EU operations"
        adapter.extract_enterprise_features(state, query)

        mock_domain_detector.detect.assert_called_once_with(query)
        mock_domain_detector.requires_compliance.assert_called_once_with(query)
        mock_domain_detector.estimate_complexity.assert_called_once_with(query, state)
        mock_domain_detector.extract_jurisdictions.assert_called_once_with(query)

    # --- route_to_enterprise ---

    def test_route_to_enterprise_ma(self, adapter: EnterpriseMetaControllerAdapter):
        features = EnterpriseMetaControllerFeatures(
            detected_domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            domain_confidence=0.8,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_ma"

    def test_route_to_enterprise_clinical(self, adapter: EnterpriseMetaControllerAdapter):
        features = EnterpriseMetaControllerFeatures(
            detected_domain=EnterpriseDomain.CLINICAL_TRIAL,
            domain_confidence=0.7,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_clinical"

    def test_route_to_enterprise_regulatory(self, adapter: EnterpriseMetaControllerAdapter):
        features = EnterpriseMetaControllerFeatures(
            detected_domain=EnterpriseDomain.REGULATORY_COMPLIANCE,
            domain_confidence=0.6,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_regulatory"

    def test_route_to_enterprise_below_threshold(self, adapter: EnterpriseMetaControllerAdapter):
        """When domain confidence is below threshold, fall back to other logic."""
        features = EnterpriseMetaControllerFeatures(
            detected_domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            domain_confidence=0.01,  # Below the 0.05 threshold
        )
        # Should not route to enterprise_ma; falls through to default
        route = adapter.route_to_enterprise(features)
        assert route == "hrm"

    def test_route_to_enterprise_no_domain_detected(self, adapter: EnterpriseMetaControllerAdapter):
        features = EnterpriseMetaControllerFeatures(
            detected_domain=None,
            domain_confidence=0.0,
        )
        assert adapter.route_to_enterprise(features) == "hrm"

    def test_route_to_enterprise_compliance_fallback(self, adapter: EnterpriseMetaControllerAdapter):
        features = EnterpriseMetaControllerFeatures(
            detected_domain=None,
            domain_confidence=0.0,
            requires_compliance_check=True,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_regulatory"

    def test_route_to_enterprise_high_complexity_uses_mcts(self, adapter: EnterpriseMetaControllerAdapter):
        features = EnterpriseMetaControllerFeatures(
            detected_domain=None,
            domain_confidence=0.0,
            requires_compliance_check=False,
            estimated_complexity=0.9,
        )
        assert adapter.route_to_enterprise(features) == "mcts"

    def test_route_to_enterprise_medium_complexity_uses_hrm(self, adapter: EnterpriseMetaControllerAdapter):
        features = EnterpriseMetaControllerFeatures(
            detected_domain=None,
            domain_confidence=0.0,
            requires_compliance_check=False,
            estimated_complexity=0.5,
        )
        assert adapter.route_to_enterprise(features) == "hrm"

    def test_route_priority_domain_over_compliance(self, adapter: EnterpriseMetaControllerAdapter):
        """Domain detection takes priority over compliance fallback."""
        features = EnterpriseMetaControllerFeatures(
            detected_domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            domain_confidence=0.8,
            requires_compliance_check=True,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_ma"

    def test_route_priority_compliance_over_complexity(self, adapter: EnterpriseMetaControllerAdapter):
        """Compliance check takes priority over complexity routing."""
        features = EnterpriseMetaControllerFeatures(
            detected_domain=None,
            domain_confidence=0.0,
            requires_compliance_check=True,
            estimated_complexity=0.9,
        )
        assert adapter.route_to_enterprise(features) == "enterprise_regulatory"

    # --- _is_time_sensitive ---

    @pytest.mark.parametrize(
        "query",
        [
            "This is an urgent compliance review",
            "We have a deadline for this audit",
            "ASAP regulatory submission needed",
            "Handle immediately the data breach",
            "This is a time-sensitive matter",
            "Critical safety issue found",
            "High priority acquisition review",
        ],
    )
    def test_is_time_sensitive_true(self, adapter: EnterpriseMetaControllerAdapter, query: str):
        assert adapter._is_time_sensitive(query) is True

    def test_is_time_sensitive_false(self, adapter: EnterpriseMetaControllerAdapter):
        assert adapter._is_time_sensitive("Analyze the quarterly report") is False

    def test_is_time_sensitive_case_insensitive(self, adapter: EnterpriseMetaControllerAdapter):
        assert adapter._is_time_sensitive("URGENT compliance review") is True

    # --- _requires_expert ---

    def test_requires_expert_low_confidence(self, adapter: EnterpriseMetaControllerAdapter):
        state = {"confidence_scores": {"max": 0.3}}
        assert adapter._requires_expert("simple query", state) is True

    def test_requires_expert_high_confidence_no_keywords(self, adapter: EnterpriseMetaControllerAdapter):
        state = {"confidence_scores": {"max": 0.9}}
        assert adapter._requires_expert("simple query", state) is False

    @pytest.mark.parametrize(
        "query",
        [
            "We need a legal opinion on this matter",
            "Request expert review of the findings",
            "Consult a specialist about this case",
            "Need regulatory interpretation for this rule",
            "This is a complex transaction requiring review",
        ],
    )
    def test_requires_expert_keyword_match(self, adapter: EnterpriseMetaControllerAdapter, query: str):
        state = {"confidence_scores": {"max": 0.9}}
        assert adapter._requires_expert(query, state) is True

    def test_requires_expert_empty_confidence_scores(self, adapter: EnterpriseMetaControllerAdapter):
        """When max confidence is missing, defaults to 1.0 so no expert needed."""
        state = {"confidence_scores": {}}
        assert adapter._requires_expert("simple query", state) is False

    def test_requires_expert_no_state_scores(self, adapter: EnterpriseMetaControllerAdapter):
        """When confidence_scores key is missing entirely."""
        assert adapter._requires_expert("simple query", {}) is False
