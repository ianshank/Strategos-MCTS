"""
Tests for enterprise settings configuration module.

Tests EnterpriseDomain enum, all config classes (AgentConfig, RolloutPolicyConfig,
DomainDetectorConfig, BaseUseCaseConfig, MADueDiligenceConfig, ClinicalTrialConfig,
RegulatoryComplianceConfig), EnterpriseSettings, and utility functions.
"""

import pytest
from pydantic import ValidationError

from src.enterprise.config.enterprise_settings import (
    AgentConfig,
    BaseUseCaseConfig,
    ClinicalTrialConfig,
    DomainDetectorConfig,
    EnterpriseDomain,
    EnterpriseSettings,
    MADueDiligenceConfig,
    RegulatoryComplianceConfig,
    RolloutPolicyConfig,
    get_enterprise_settings,
    reset_enterprise_settings,
)


@pytest.mark.unit
class TestEnterpriseDomain:
    """Tests for EnterpriseDomain enum."""

    def test_values(self):
        """Test all enum values exist."""
        assert EnterpriseDomain.MA_DUE_DILIGENCE.value == "ma_due_diligence"
        assert EnterpriseDomain.CLINICAL_TRIAL.value == "clinical_trial"
        assert EnterpriseDomain.REGULATORY_COMPLIANCE.value == "regulatory_compliance"

    def test_is_str_enum(self):
        """Test EnterpriseDomain is a string enum."""
        assert isinstance(EnterpriseDomain.MA_DUE_DILIGENCE, str)


@pytest.mark.unit
class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_defaults(self):
        """Test default values."""
        config = AgentConfig()
        assert config.confidence_threshold == 0.7
        assert config.max_retries == 3
        assert config.timeout_seconds == 30.0

    def test_custom_values(self):
        """Test custom values."""
        config = AgentConfig(confidence_threshold=0.9, max_retries=5, timeout_seconds=60.0)
        assert config.confidence_threshold == 0.9
        assert config.max_retries == 5
        assert config.timeout_seconds == 60.0

    def test_validation_confidence_range(self):
        """Test confidence_threshold validation."""
        with pytest.raises(ValidationError):
            AgentConfig(confidence_threshold=1.5)
        with pytest.raises(ValidationError):
            AgentConfig(confidence_threshold=-0.1)

    def test_validation_max_retries_range(self):
        """Test max_retries validation."""
        with pytest.raises(ValidationError):
            AgentConfig(max_retries=0)
        with pytest.raises(ValidationError):
            AgentConfig(max_retries=11)

    def test_validation_timeout_range(self):
        """Test timeout_seconds validation."""
        with pytest.raises(ValidationError):
            AgentConfig(timeout_seconds=0)
        with pytest.raises(ValidationError):
            AgentConfig(timeout_seconds=601)


@pytest.mark.unit
class TestRolloutPolicyConfig:
    """Tests for RolloutPolicyConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RolloutPolicyConfig()
        assert config.heuristic_weight == 0.7
        assert config.random_weight == 0.3
        assert config.depth_bonus_divisor == 20.0
        assert config.max_depth_bonus == 0.2

    def test_validation(self):
        """Test field validation."""
        with pytest.raises(ValidationError):
            RolloutPolicyConfig(heuristic_weight=1.5)
        with pytest.raises(ValidationError):
            RolloutPolicyConfig(depth_bonus_divisor=0)


@pytest.mark.unit
class TestDomainDetectorConfig:
    """Tests for DomainDetectorConfig."""

    def test_defaults(self):
        """Test default values."""
        config = DomainDetectorConfig()
        assert config.detection_threshold == 0.05
        assert config.complexity_length_divisor == 1000.0
        assert config.high_complexity_threshold == 0.7

    def test_validation(self):
        """Test field validation."""
        with pytest.raises(ValidationError):
            DomainDetectorConfig(detection_threshold=2.0)
        with pytest.raises(ValidationError):
            DomainDetectorConfig(complexity_length_divisor=0)


@pytest.mark.unit
class TestBaseUseCaseConfig:
    """Tests for BaseUseCaseConfig."""

    def test_defaults(self):
        """Test default values."""
        config = BaseUseCaseConfig()
        assert config.enabled is True
        assert config.max_mcts_iterations == 100
        assert config.mcts_exploration_weight == 1.414
        assert config.mcts_seed == 42
        assert config.confidence_threshold == 0.7
        assert config.max_analysis_depth == 10
        assert config.parallel_agents == 4
        assert config.timeout_seconds == 300.0

    def test_nested_configs(self):
        """Test nested AgentConfig and RolloutPolicyConfig are created."""
        config = BaseUseCaseConfig()
        assert isinstance(config.agent_config, AgentConfig)
        assert isinstance(config.rollout_policy, RolloutPolicyConfig)

    def test_validate_iterations_positive(self):
        """Test iterations must be positive."""
        with pytest.raises(ValidationError):
            BaseUseCaseConfig(max_mcts_iterations=0)

    def test_validate_iterations_max(self):
        """Test iterations cannot exceed max."""
        with pytest.raises(ValidationError):
            BaseUseCaseConfig(max_mcts_iterations=10001)


@pytest.mark.unit
class TestMADueDiligenceConfig:
    """Tests for MADueDiligenceConfig."""

    def test_defaults(self):
        """Test default values."""
        config = MADueDiligenceConfig()
        assert config.risk_threshold == 0.6
        assert config.critical_risk_threshold == 0.85
        assert config.max_documents_per_analysis == 100
        assert config.jurisdictions == ["US", "EU"]
        assert config.document_storage_bucket is None

    def test_action_weights_default(self):
        """Test default action weights."""
        config = MADueDiligenceConfig()
        assert "financial_analysis" in config.action_weights
        assert abs(sum(config.action_weights.values()) - 1.0) < 1e-6

    def test_reward_weights_default(self):
        """Test default reward weights."""
        config = MADueDiligenceConfig()
        assert "information_gain" in config.reward_weights
        assert abs(sum(config.reward_weights.values()) - 1.0) < 1e-6

    def test_threshold_validation(self):
        """Test critical threshold must be >= risk threshold."""
        with pytest.raises(ValidationError, match="critical_risk_threshold"):
            MADueDiligenceConfig(risk_threshold=0.9, critical_risk_threshold=0.5)

    def test_threshold_validation_equal(self):
        """Test equal thresholds are valid."""
        config = MADueDiligenceConfig(risk_threshold=0.7, critical_risk_threshold=0.7)
        assert config.risk_threshold == 0.7

    def test_custom_jurisdictions(self):
        """Test custom jurisdictions list."""
        config = MADueDiligenceConfig(jurisdictions=["US", "UK", "APAC"])
        assert "UK" in config.jurisdictions


@pytest.mark.unit
class TestClinicalTrialConfig:
    """Tests for ClinicalTrialConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ClinicalTrialConfig()
        assert config.regulatory_guideline_version == "FDA_2024"
        assert config.min_statistical_power == 0.8
        assert config.alpha_level == 0.05
        assert config.max_trial_duration_months == 36
        assert config.max_sample_size == 10000
        assert config.min_sample_size == 10
        assert config.budget_constraint_usd is None

    def test_supported_agencies(self):
        """Test default supported agencies."""
        config = ClinicalTrialConfig()
        assert "FDA" in config.supported_agencies
        assert "EMA" in config.supported_agencies

    def test_phase_weights(self):
        """Test default phase weights."""
        config = ClinicalTrialConfig()
        assert "phase_1" in config.phase_weights
        assert abs(sum(config.phase_weights.values()) - 1.0) < 1e-6

    def test_validation_statistical_power(self):
        """Test statistical power validation."""
        with pytest.raises(ValidationError):
            ClinicalTrialConfig(min_statistical_power=0.3)

    def test_validation_alpha_level(self):
        """Test alpha level validation."""
        with pytest.raises(ValidationError):
            ClinicalTrialConfig(alpha_level=0.5)


@pytest.mark.unit
class TestRegulatoryComplianceConfig:
    """Tests for RegulatoryComplianceConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RegulatoryComplianceConfig()
        assert config.jurisdictions == ["US", "EU", "UK", "APAC"]
        assert config.risk_tolerance == "moderate"
        assert config.remediation_budget is None
        assert config.max_remediation_timeline_days == 90
        assert config.enforcement_horizon_months == 12

    def test_gap_severity_weights(self):
        """Test default gap severity weights."""
        config = RegulatoryComplianceConfig()
        assert config.gap_severity_weights["critical"] == 1.0
        assert config.gap_severity_weights["low"] == 0.25

    def test_risk_tolerance_validation_valid(self):
        """Test valid risk tolerance values."""
        for val in ["conservative", "moderate", "aggressive"]:
            config = RegulatoryComplianceConfig(risk_tolerance=val)
            assert config.risk_tolerance == val

    def test_risk_tolerance_validation_case_insensitive(self):
        """Test risk tolerance is lowered."""
        config = RegulatoryComplianceConfig(risk_tolerance="CONSERVATIVE")
        assert config.risk_tolerance == "conservative"

    def test_risk_tolerance_validation_invalid(self):
        """Test invalid risk tolerance raises error."""
        with pytest.raises(ValidationError, match="risk_tolerance"):
            RegulatoryComplianceConfig(risk_tolerance="reckless")


@pytest.mark.unit
class TestEnterpriseSettings:
    """Tests for EnterpriseSettings master config."""

    def test_defaults(self):
        """Test default enterprise settings."""
        settings = EnterpriseSettings()
        assert settings.enterprise_enabled is True
        assert settings.default_domain == EnterpriseDomain.MA_DUE_DILIGENCE

    def test_lazy_properties(self):
        """Test lazy-loaded sub-configs."""
        settings = EnterpriseSettings()
        assert isinstance(settings.ma_due_diligence, MADueDiligenceConfig)
        assert isinstance(settings.clinical_trial, ClinicalTrialConfig)
        assert isinstance(settings.regulatory_compliance, RegulatoryComplianceConfig)
        assert isinstance(settings.domain_detector, DomainDetectorConfig)

    def test_get_use_case_config(self):
        """Test getting config by domain enum."""
        settings = EnterpriseSettings()
        ma_config = settings.get_use_case_config(EnterpriseDomain.MA_DUE_DILIGENCE)
        assert isinstance(ma_config, MADueDiligenceConfig)

        ct_config = settings.get_use_case_config(EnterpriseDomain.CLINICAL_TRIAL)
        assert isinstance(ct_config, ClinicalTrialConfig)

        rc_config = settings.get_use_case_config(EnterpriseDomain.REGULATORY_COMPLIANCE)
        assert isinstance(rc_config, RegulatoryComplianceConfig)

    def test_safe_dict(self):
        """Test safe_dict returns expected structure without secrets."""
        settings = EnterpriseSettings()
        d = settings.safe_dict()
        assert "enterprise_enabled" in d
        assert "default_domain" in d
        assert "use_cases" in d
        assert "ma_due_diligence" in d["use_cases"]
        assert "clinical_trial" in d["use_cases"]
        assert "regulatory_compliance" in d["use_cases"]


@pytest.mark.unit
class TestGetEnterpriseSettings:
    """Tests for get/reset enterprise settings functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_enterprise_settings()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_enterprise_settings()

    def test_get_returns_instance(self):
        """Test get_enterprise_settings returns EnterpriseSettings."""
        settings = get_enterprise_settings()
        assert isinstance(settings, EnterpriseSettings)

    def test_singleton_same_instance(self):
        """Test singleton returns same instance."""
        s1 = get_enterprise_settings()
        s2 = get_enterprise_settings()
        assert s1 is s2

    def test_reset_clears_singleton(self):
        """Test reset creates new instance on next call."""
        s1 = get_enterprise_settings()
        reset_enterprise_settings()
        s2 = get_enterprise_settings()
        assert s1 is not s2
