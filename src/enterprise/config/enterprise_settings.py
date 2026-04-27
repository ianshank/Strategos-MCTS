"""
Enterprise Use Case Configuration using Pydantic Settings.

All configuration values are loaded from environment variables,
following the no-hardcoded-values principle from CLAUDE.md.

Environment Variable Prefixes:
- ENTERPRISE_*: Global enterprise settings
- MA_DD_*: M&A Due Diligence settings
- CLINICAL_TRIAL_*: Clinical Trial Design settings
- REG_COMPLIANCE_*: Regulatory Compliance settings

Example:
    export ENTERPRISE_ENABLED=true
    export MA_DD_RISK_THRESHOLD=0.6
    export CLINICAL_TRIAL_MIN_STATISTICAL_POWER=0.8
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnterpriseDomain(str, Enum):
    """Supported enterprise domains."""

    MA_DUE_DILIGENCE = "ma_due_diligence"
    CLINICAL_TRIAL = "clinical_trial"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class AgentConfig(BaseSettings):
    """
    Base configuration for domain agents.

    Environment variables use AGENT_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        extra="ignore",
    )

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for agent results",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for agent operations",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=600,
        description="Timeout for individual agent operations",
    )


class RolloutPolicyConfig(BaseSettings):
    """
    Configuration for MCTS rollout policy.

    Environment variables use ROLLOUT_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ROLLOUT_",
        extra="ignore",
    )

    heuristic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for heuristic in hybrid rollout policy",
    )
    random_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for random selection in hybrid rollout policy",
    )
    depth_bonus_divisor: float = Field(
        default=20.0,
        gt=0,
        description="Divisor for calculating depth bonus in heuristic",
    )
    max_depth_bonus: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum depth bonus in heuristic",
    )


class DomainDetectorConfig(BaseSettings):
    """
    Configuration for domain detection.

    Environment variables use DOMAIN_DETECTOR_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOMAIN_DETECTOR_",
        extra="ignore",
    )

    detection_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to detect a domain",
    )
    complexity_length_divisor: float = Field(
        default=1000.0,
        gt=0,
        description="Divisor for length factor in complexity estimation",
    )
    complexity_length_max: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum length factor contribution to complexity",
    )
    complexity_tech_multiplier: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Multiplier per technical keyword in complexity estimation",
    )
    complexity_tech_max: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum technical factor contribution to complexity",
    )
    complexity_state_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Factor added for each state complexity indicator",
    )
    high_complexity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for routing to MCTS for complex queries",
    )


class BaseUseCaseConfig(BaseSettings):
    """
    Base configuration for all enterprise use cases.

    Subclasses add domain-specific configuration.
    All values are configurable via environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="ENTERPRISE_",
        extra="ignore",
        validate_default=True,
    )

    # Common settings
    enabled: bool = Field(default=True, description="Enable this use case")
    max_mcts_iterations: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum MCTS iterations",
    )
    mcts_exploration_weight: float = Field(
        default=1.414,
        ge=0.0,
        le=10.0,
        description="UCB1 exploration constant",
    )
    mcts_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for MCTS reproducibility",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for results",
    )
    max_analysis_depth: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum analysis depth",
    )
    parallel_agents: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel agent executions",
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Operation timeout in seconds",
    )

    # Nested configurations
    agent_config: AgentConfig = Field(default_factory=AgentConfig)
    rollout_policy: RolloutPolicyConfig = Field(default_factory=RolloutPolicyConfig)

    @field_validator("max_mcts_iterations")
    @classmethod
    def validate_iterations(cls, v: int) -> int:
        """Validate MCTS iterations."""
        if v < 1:
            raise ValueError("max_mcts_iterations must be positive")
        return v


class MADueDiligenceConfig(BaseUseCaseConfig):
    """
    M&A Due Diligence specific configuration.

    Environment variables use MA_DD_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="MA_DD_",
        extra="ignore",
    )

    # Document analysis
    document_storage_bucket: str | None = Field(
        default=None,
        description="S3 bucket for deal documents",
    )
    max_documents_per_analysis: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum documents to analyze",
    )
    max_docs_per_batch: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum documents per analysis batch",
    )
    extraction_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for document extraction",
    )

    # Risk assessment
    risk_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Risk threshold for flagging",
    )
    critical_risk_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Critical risk threshold",
    )
    max_refinement_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum refinement rounds for risk identification",
    )

    # Synergy exploration
    synergy_exploration_depth: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Depth of synergy exploration",
    )
    min_synergy_value: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum synergy value to consider",
    )

    # Compliance checking
    jurisdictions: list[str] = Field(
        default_factory=lambda: ["US", "EU"],
        description="Jurisdictions to check",
    )

    # Regulatory thresholds (in USD) - these are statutory values that may change
    hsr_filing_threshold: float = Field(
        default=101_000_000.0,
        gt=0,
        description="Hart-Scott-Rodino Act filing threshold (USD)",
    )
    eu_merger_threshold: float = Field(
        default=150_000_000.0,
        gt=0,
        description="EU Merger Regulation notification threshold (USD)",
    )

    # MCTS specific
    action_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "financial_analysis": 0.3,
            "legal_review": 0.25,
            "operational_assessment": 0.25,
            "tech_evaluation": 0.2,
        },
        description="Weights for different action categories",
    )

    # Reward function weights
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "information_gain": 0.4,
            "risk_discovery": 0.35,
            "timeline_efficiency": 0.25,
        },
        description="Weights for reward function components",
    )

    # Information-gain base reward per action type (0..1).
    # Unknown actions fall back to action_base_gain_default.
    action_base_gain: dict[str, float] = Field(
        default_factory=lambda: {
            "deep_dive_revenue": 0.8,
            "analyze_cost_structure": 0.75,
            "review_contracts": 0.75,
            "check_litigation_history": 0.85,
            "verify_ip_ownership": 0.7,
            "assess_regulatory_compliance": 0.8,
            "evaluate_operations": 0.65,
            "assess_supply_chain": 0.6,
            "review_hr_structure": 0.55,
            "analyze_it_infrastructure": 0.7,
            "assess_tech_stack": 0.75,
            "review_security_posture": 0.8,
            "evaluate_scalability": 0.65,
            "check_tech_debt": 0.7,
            "identify_revenue_synergies": 0.75,
            "identify_cost_synergies": 0.7,
            "escalate_to_expert": 0.6,
            "request_additional_docs": 0.5,
        },
        description="Base information-gain reward per action type",
    )
    action_base_gain_default: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fallback information-gain reward for unknown actions",
    )

    # Numeric weight per risk severity level (keyed by RiskLevel.value).
    risk_level_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "critical": 1.0,
        },
        description="Numeric weight for each risk severity level",
    )

    # Reward function parameters
    reward_decay_factor: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Decay factor for diminishing returns on repeated actions",
    )
    risk_low_count_multiplier: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Risk reward multiplier when few risks found (<3)",
    )
    risk_medium_count_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Risk reward multiplier when moderate risks found (3-5)",
    )
    risk_high_count_multiplier: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Risk reward multiplier when many risks found (>5)",
    )
    risk_count_low_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Threshold for 'few risks' category",
    )
    risk_count_high_threshold: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Threshold for 'many risks' category",
    )

    # Timeline efficiency parameters
    timeline_early_penalty: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Timeline penalty factor for early stage (< 60% depth)",
    )
    timeline_mid_penalty: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Timeline penalty factor for mid stage (60-80% depth)",
    )
    timeline_late_penalty: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Timeline penalty factor for late stage (> 80% depth)",
    )
    timeline_depth_early_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for early timeline stage",
    )
    timeline_depth_late_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for late timeline stage",
    )
    min_phase_coverage_actions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum actions for adequate phase coverage before progression",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> MADueDiligenceConfig:
        """Ensure critical threshold is higher than risk threshold."""
        if self.critical_risk_threshold < self.risk_threshold:
            raise ValueError("critical_risk_threshold must be >= risk_threshold")
        return self


class ClinicalTrialConfig(BaseUseCaseConfig):
    """
    Clinical Trial Design specific configuration.

    Environment variables use CLINICAL_TRIAL_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLINICAL_TRIAL_",
        extra="ignore",
    )

    # Regulatory settings
    regulatory_guideline_version: str = Field(
        default="FDA_2024",
        description="Regulatory guideline version",
    )
    supported_agencies: list[str] = Field(
        default_factory=lambda: ["FDA", "EMA", "PMDA"],
        description="Supported regulatory agencies",
    )

    # Statistical requirements
    min_statistical_power: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Minimum statistical power",
    )
    alpha_level: float = Field(
        default=0.05,
        ge=0.001,
        le=0.1,
        description="Alpha level for significance",
    )

    # Trial constraints
    max_trial_duration_months: int = Field(
        default=36,
        ge=6,
        le=120,
        description="Maximum trial duration in months",
    )
    max_sample_size: int = Field(
        default=10000,
        ge=10,
        le=100000,
        description="Maximum sample size",
    )
    min_sample_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Minimum sample size",
    )
    min_trial_duration_months: int = Field(
        default=3,
        ge=1,
        le=24,
        description="Minimum trial duration in months",
    )
    budget_constraint_usd: float | None = Field(
        default=None,
        ge=0,
        description="Budget constraint in USD",
    )

    # Sample size adjustment factors
    sample_size_increase_factor: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Multiplier for increasing sample size",
    )
    sample_size_decrease_factor: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Multiplier for decreasing sample size",
    )

    # Duration adjustment factors
    duration_adjustment_months: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Months to add/remove when adjusting duration",
    )

    # Phase-specific settings
    phase_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "phase_1": 0.15,
            "phase_2": 0.35,
            "phase_3": 0.50,
        },
        description="Weights for different trial phases",
    )

    # Endpoint configuration
    max_secondary_endpoints: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Maximum secondary endpoints",
    )

    # Action history settings
    recent_action_window: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of recent actions to consider for deduplication",
    )


class RegulatoryComplianceConfig(BaseUseCaseConfig):
    """
    Regulatory Compliance specific configuration.

    Environment variables use REG_COMPLIANCE_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="REG_COMPLIANCE_",
        extra="ignore",
    )

    # Jurisdiction settings
    jurisdictions: list[str] = Field(
        default_factory=lambda: ["US", "EU", "UK", "APAC"],
        description="Applicable jurisdictions",
    )
    regulation_update_frequency_hours: int = Field(
        default=24,
        ge=1,
        description="How often to check for regulation updates",
    )

    # Risk tolerance
    risk_tolerance: str = Field(
        default="moderate",
        description="Risk tolerance level: conservative, moderate, aggressive",
    )

    # Remediation settings
    remediation_budget: float | None = Field(
        default=None,
        ge=0,
        description="Remediation budget in base currency",
    )
    max_remediation_timeline_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Maximum remediation timeline",
    )

    # Gap analysis
    gap_severity_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "critical": 1.0,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25,
        },
        description="Weights for gap severity levels",
    )

    # Enforcement prediction
    enforcement_horizon_months: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Enforcement prediction horizon",
    )

    @field_validator("risk_tolerance")
    @classmethod
    def validate_risk_tolerance(cls, v: str) -> str:
        """Validate risk tolerance value."""
        valid = ["conservative", "moderate", "aggressive"]
        if v.lower() not in valid:
            raise ValueError(f"risk_tolerance must be one of {valid}")
        return v.lower()


class EnterpriseSettings(BaseSettings):
    """
    Master configuration for enterprise use cases.

    Centralizes all enterprise configurations with validation.
    Loaded from .env file and environment variables.

    Example:
        >>> settings = get_enterprise_settings()
        >>> settings.ma_due_diligence.risk_threshold
        0.6
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Global enterprise settings
    enterprise_enabled: bool = Field(
        default=True,
        alias="ENTERPRISE_ENABLED",
        description="Enable enterprise features",
    )
    default_domain: EnterpriseDomain = Field(
        default=EnterpriseDomain.MA_DUE_DILIGENCE,
        alias="ENTERPRISE_DEFAULT_DOMAIN",
        description="Default enterprise domain",
    )

    # Use case configurations (created lazily)
    _ma_due_diligence: MADueDiligenceConfig | None = None
    _clinical_trial: ClinicalTrialConfig | None = None
    _regulatory_compliance: RegulatoryComplianceConfig | None = None
    _domain_detector: DomainDetectorConfig | None = None

    @property
    def domain_detector(self) -> DomainDetectorConfig:
        """Get Domain Detector configuration."""
        if self._domain_detector is None:
            self._domain_detector = DomainDetectorConfig()
        return self._domain_detector

    @property
    def ma_due_diligence(self) -> MADueDiligenceConfig:
        """Get M&A Due Diligence configuration."""
        if self._ma_due_diligence is None:
            self._ma_due_diligence = MADueDiligenceConfig()
        return self._ma_due_diligence

    @property
    def clinical_trial(self) -> ClinicalTrialConfig:
        """Get Clinical Trial configuration."""
        if self._clinical_trial is None:
            self._clinical_trial = ClinicalTrialConfig()
        return self._clinical_trial

    @property
    def regulatory_compliance(self) -> RegulatoryComplianceConfig:
        """Get Regulatory Compliance configuration."""
        if self._regulatory_compliance is None:
            self._regulatory_compliance = RegulatoryComplianceConfig()
        return self._regulatory_compliance

    def get_use_case_config(self, domain: EnterpriseDomain) -> BaseUseCaseConfig:
        """
        Get configuration for a specific use case.

        Args:
            domain: Enterprise domain to get config for

        Returns:
            Configuration for the specified domain
        """
        mapping = {
            EnterpriseDomain.MA_DUE_DILIGENCE: self.ma_due_diligence,
            EnterpriseDomain.CLINICAL_TRIAL: self.clinical_trial,
            EnterpriseDomain.REGULATORY_COMPLIANCE: self.regulatory_compliance,
        }
        return mapping[domain]

    def safe_dict(self) -> dict[str, Any]:
        """
        Return configuration with secrets masked.

        Safe for logging and debugging.
        """
        return {
            "enterprise_enabled": self.enterprise_enabled,
            "default_domain": self.default_domain.value,
            "use_cases": {
                "ma_due_diligence": {
                    "enabled": self.ma_due_diligence.enabled,
                    "risk_threshold": self.ma_due_diligence.risk_threshold,
                    "jurisdictions": self.ma_due_diligence.jurisdictions,
                },
                "clinical_trial": {
                    "enabled": self.clinical_trial.enabled,
                    "min_statistical_power": self.clinical_trial.min_statistical_power,
                },
                "regulatory_compliance": {
                    "enabled": self.regulatory_compliance.enabled,
                    "risk_tolerance": self.regulatory_compliance.risk_tolerance,
                    "jurisdictions": self.regulatory_compliance.jurisdictions,
                },
            },
        }


# Global settings instance (singleton pattern)
_enterprise_settings: EnterpriseSettings | None = None
_logger = logging.getLogger(__name__)


def get_enterprise_settings() -> EnterpriseSettings:
    """
    Get global enterprise settings instance.

    Uses singleton pattern for efficiency.

    Returns:
        EnterpriseSettings instance
    """
    global _enterprise_settings
    if _enterprise_settings is None:
        _enterprise_settings = EnterpriseSettings()
        _logger.info("Enterprise settings loaded", extra=_enterprise_settings.safe_dict())
    return _enterprise_settings


def reset_enterprise_settings() -> None:
    """
    Reset the global settings instance.

    Useful for testing and configuration reloading.
    """
    global _enterprise_settings
    _enterprise_settings = None
    _logger.debug("Enterprise settings reset")
