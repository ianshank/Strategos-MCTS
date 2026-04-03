"""
Tests for enterprise domain detector module.

Tests DomainPattern, DetectionResult, DomainDetector class methods,
and the singleton get_domain_detector function.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.enterprise.base.domain_detector import (
    DetectionResult,
    DomainDetector,
    DomainPattern,
)
from src.enterprise.config.enterprise_settings import (
    DomainDetectorConfig,
    EnterpriseDomain,
)


@pytest.mark.unit
class TestDomainPattern:
    """Tests for DomainPattern dataclass."""

    def test_default_weight(self):
        """Test default weight is 1.0."""
        pattern = DomainPattern(keywords=["test"])
        assert pattern.weight == 1.0

    def test_custom_weight(self):
        """Test custom weight assignment."""
        pattern = DomainPattern(keywords=["a", "b"], weight=0.5)
        assert pattern.weight == 0.5
        assert pattern.keywords == ["a", "b"]

    def test_empty_keywords(self):
        """Test pattern with empty keywords list."""
        pattern = DomainPattern(keywords=[])
        assert pattern.keywords == []


@pytest.mark.unit
class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_is_detected_true(self):
        """Test is_detected returns True when domain is set."""
        result = DetectionResult(
            domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            confidence=0.8,
        )
        assert result.is_detected is True

    def test_is_detected_false(self):
        """Test is_detected returns False when domain is None."""
        result = DetectionResult(domain=None, confidence=0.0)
        assert result.is_detected is False

    def test_all_scores_default(self):
        """Test all_scores defaults to empty dict."""
        result = DetectionResult(domain=None, confidence=0.0)
        assert result.all_scores == {}

    def test_all_scores_populated(self):
        """Test all_scores stores domain score mapping."""
        scores = {
            EnterpriseDomain.MA_DUE_DILIGENCE: 0.5,
            EnterpriseDomain.CLINICAL_TRIAL: 0.2,
        }
        result = DetectionResult(
            domain=EnterpriseDomain.MA_DUE_DILIGENCE,
            confidence=0.5,
            all_scores=scores,
        )
        assert len(result.all_scores) == 2


@pytest.mark.unit
class TestDomainDetector:
    """Tests for DomainDetector class."""

    @pytest.fixture
    def config(self):
        """Create a DomainDetectorConfig for testing."""
        return DomainDetectorConfig(
            detection_threshold=0.05,
            complexity_length_divisor=1000.0,
            complexity_length_max=0.3,
            complexity_tech_multiplier=0.1,
            complexity_tech_max=0.3,
            complexity_state_factor=0.1,
            high_complexity_threshold=0.7,
        )

    @pytest.fixture
    def detector(self, config):
        """Create a DomainDetector with test config."""
        return DomainDetector(config=config)

    def test_init_default_patterns(self, detector):
        """Test detector initializes with default patterns."""
        assert EnterpriseDomain.MA_DUE_DILIGENCE in detector._patterns
        assert EnterpriseDomain.CLINICAL_TRIAL in detector._patterns
        assert EnterpriseDomain.REGULATORY_COMPLIANCE in detector._patterns

    def test_init_custom_threshold(self, config):
        """Test detector with custom detection threshold."""
        detector = DomainDetector(detection_threshold=0.5, config=config)
        assert detector.detection_threshold == 0.5

    def test_init_custom_patterns(self, config):
        """Test detector with custom patterns."""
        custom = {
            EnterpriseDomain.MA_DUE_DILIGENCE: DomainPattern(
                keywords=["deal"], weight=2.0
            )
        }
        detector = DomainDetector(patterns=custom, config=config)
        assert len(detector._patterns) == 1
        assert detector._patterns[EnterpriseDomain.MA_DUE_DILIGENCE].weight == 2.0

    def test_detection_threshold_property(self, detector):
        """Test detection_threshold getter and setter."""
        detector.detection_threshold = 0.3
        assert detector.detection_threshold == 0.3

    def test_detection_threshold_invalid(self, detector):
        """Test detection_threshold rejects out-of-range values."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            detector.detection_threshold = 1.5

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            detector.detection_threshold = -0.1

    # --- detect() ---

    def test_detect_ma_domain(self, detector):
        """Test detection of M&A due diligence domain."""
        result = detector.detect("Analyze acquisition target and merger synergy")
        assert result.domain == EnterpriseDomain.MA_DUE_DILIGENCE
        assert result.is_detected is True
        assert result.confidence > 0

    def test_detect_clinical_trial_domain(self, detector):
        """Test detection of clinical trial domain."""
        result = detector.detect("Design a clinical trial with FDA endpoint and cohort")
        assert result.domain == EnterpriseDomain.CLINICAL_TRIAL
        assert result.is_detected is True

    def test_detect_regulatory_compliance_domain(self, detector):
        """Test detection of regulatory compliance domain."""
        result = detector.detect("Run a compliance audit for GDPR regulation")
        assert result.domain == EnterpriseDomain.REGULATORY_COMPLIANCE
        assert result.is_detected is True

    def test_detect_no_match(self, config):
        """Test detection returns None when below threshold."""
        detector = DomainDetector(detection_threshold=1.0, config=config)
        result = detector.detect("Tell me about the weather today")
        assert result.domain is None
        assert result.is_detected is False

    def test_detect_case_insensitive(self, detector):
        """Test detection is case insensitive."""
        result = detector.detect("ACQUISITION MERGER DUE DILIGENCE")
        assert result.domain == EnterpriseDomain.MA_DUE_DILIGENCE

    def test_detect_all_scores_populated(self, detector):
        """Test that all_scores contains all domains."""
        result = detector.detect("acquisition target")
        assert len(result.all_scores) == 3

    def test_detect_empty_patterns(self, config):
        """Test detection with empty patterns dict."""
        detector = DomainDetector(patterns={}, config=config)
        result = detector.detect("anything")
        assert result.domain is None
        assert result.confidence == 0.0

    # --- requires_compliance() ---

    def test_requires_compliance_true(self, detector):
        """Test compliance detection with matching keywords."""
        assert detector.requires_compliance("Check GDPR compliance") is True
        assert detector.requires_compliance("Run an audit") is True
        assert detector.requires_compliance("HIPAA regulation") is True

    def test_requires_compliance_false(self, detector):
        """Test compliance detection with non-matching query."""
        assert detector.requires_compliance("Tell me about cats") is False

    def test_requires_compliance_case_insensitive(self, detector):
        """Test compliance detection is case insensitive."""
        assert detector.requires_compliance("COMPLIANCE CHECK") is True

    # --- estimate_complexity() ---

    def test_estimate_complexity_short_query(self, detector):
        """Test low complexity for short simple query."""
        score = detector.estimate_complexity("hello")
        assert 0.0 <= score <= 1.0
        assert score < 0.3  # Short, no technical keywords

    def test_estimate_complexity_long_technical_query(self, detector):
        """Test higher complexity for long technical query."""
        query = "Analyze and evaluate the implementation, then compare and assess optimization " * 5
        score = detector.estimate_complexity(query)
        assert score > 0.1

    def test_estimate_complexity_with_state(self, detector):
        """Test complexity increases with state context."""
        query = "simple question"
        base = detector.estimate_complexity(query)
        with_rag = detector.estimate_complexity(query, {"rag_context": "some context"})
        with_both = detector.estimate_complexity(
            query, {"rag_context": "ctx", "previous_responses": ["r1"]}
        )
        assert with_rag >= base
        assert with_both >= with_rag

    def test_estimate_complexity_capped_at_one(self, detector):
        """Test complexity is capped at 1.0."""
        query = " ".join(DomainDetector.TECHNICAL_KEYWORDS) * 10
        score = detector.estimate_complexity(
            query, {"rag_context": "x", "previous_responses": ["y"]}
        )
        assert score <= 1.0

    # --- extract_jurisdictions() ---

    def test_extract_jurisdictions_us(self, detector):
        """Test US jurisdiction extraction."""
        result = detector.extract_jurisdictions("Check SEC rules in the United States")
        assert "US" in result

    def test_extract_jurisdictions_eu(self, detector):
        """Test EU jurisdiction extraction."""
        result = detector.extract_jurisdictions("GDPR European Union rules")
        assert "EU" in result

    def test_extract_jurisdictions_uk(self, detector):
        """Test UK jurisdiction extraction."""
        result = detector.extract_jurisdictions("FCA rules in United Kingdom")
        assert "UK" in result

    def test_extract_jurisdictions_apac(self, detector):
        """Test APAC jurisdiction extraction."""
        result = detector.extract_jurisdictions("Regulations in Singapore Asia")
        assert "APAC" in result

    def test_extract_jurisdictions_default_us(self, detector):
        """Test default to US when no jurisdiction detected."""
        result = detector.extract_jurisdictions("Some random text with no region")
        assert result == ["US"]

    def test_extract_jurisdictions_multiple(self, detector):
        """Test multiple jurisdictions detected."""
        result = detector.extract_jurisdictions("GDPR in EU and SEC in United States")
        assert "US" in result
        assert "EU" in result

    # --- get_enterprise_route() ---

    def test_route_ma(self, detector):
        """Test routing to M&A enterprise route."""
        route = detector.get_enterprise_route("Analyze acquisition merger due diligence target company")
        assert route == "enterprise_ma"

    def test_route_clinical(self, detector):
        """Test routing to clinical enterprise route."""
        route = detector.get_enterprise_route("Design clinical trial with FDA endpoint cohort")
        assert route == "enterprise_clinical"

    def test_route_regulatory(self, detector):
        """Test routing to regulatory enterprise route."""
        route = detector.get_enterprise_route("Compliance audit for GDPR regulation enforcement")
        assert route == "enterprise_regulatory"

    def test_route_compliance_fallback(self, config):
        """Test fallback to regulatory route via compliance keywords."""
        detector = DomainDetector(detection_threshold=1.0, config=config)
        route = detector.get_enterprise_route("Check compliance requirements")
        assert route == "enterprise_regulatory"

    def test_route_hrm_fallback(self, config):
        """Test fallback to hrm for simple queries."""
        detector = DomainDetector(detection_threshold=1.0, config=config)
        route = detector.get_enterprise_route("hello")
        assert route == "hrm"

    # --- add_pattern / get_pattern ---

    def test_add_and_get_pattern(self, detector):
        """Test adding and retrieving a pattern."""
        pattern = DomainPattern(keywords=["custom"], weight=0.5)
        detector.add_pattern(EnterpriseDomain.CLINICAL_TRIAL, pattern)
        retrieved = detector.get_pattern(EnterpriseDomain.CLINICAL_TRIAL)
        assert retrieved is pattern

    def test_get_pattern_missing(self, config):
        """Test getting a pattern for unregistered domain returns None."""
        custom = {EnterpriseDomain.CLINICAL_TRIAL: DomainPattern(keywords=["trial"])}
        detector = DomainDetector(patterns=custom, config=config)
        assert detector.get_pattern(EnterpriseDomain.MA_DUE_DILIGENCE) is None


@pytest.mark.unit
class TestGetDomainDetector:
    """Tests for get_domain_detector singleton function."""

    def test_singleton_returns_instance(self):
        """Test singleton returns a DomainDetector."""
        import src.enterprise.base.domain_detector as mod

        mod._default_detector = None
        with patch.object(mod, "DomainDetector") as mock_cls:
            mock_cls.return_value = MagicMock(spec=DomainDetector)
            result = mod.get_domain_detector()
            assert result is not None
            mock_cls.assert_called_once()

    def test_singleton_returns_same_instance(self):
        """Test singleton returns the same instance on subsequent calls."""
        import src.enterprise.base.domain_detector as mod

        sentinel = MagicMock(spec=DomainDetector)
        mod._default_detector = sentinel
        try:
            assert mod.get_domain_detector() is sentinel
            assert mod.get_domain_detector() is sentinel
        finally:
            mod._default_detector = None
