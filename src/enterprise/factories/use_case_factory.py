"""
Factory for creating enterprise use case instances.

Follows the factory pattern established in src/framework/factories.py.
Supports dynamic registration of use case types and configuration-driven
instantiation.

Example:
    >>> factory = EnterpriseUseCaseFactory()
    >>> use_case = factory.create(EnterpriseDomain.MA_DUE_DILIGENCE)
    >>> result = await use_case.process("Analyze target company")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.adapters.llm.base import LLMClient
    from src.config.settings import Settings

from ..base.use_case import BaseUseCase, UseCaseProtocol
from ..config.enterprise_settings import (
    EnterpriseDomain,
    EnterpriseSettings,
    get_enterprise_settings,
)


class UseCaseFactory(Protocol):
    """Protocol for use case factories."""

    def create(self, **kwargs: Any) -> UseCaseProtocol:
        """Create a use case instance."""
        ...


class EnterpriseUseCaseFactory:
    """
    Factory for creating enterprise use case instances.

    Supports dynamic registration of use case types and
    configuration-driven instantiation.

    Features:
    - Registry pattern for extensibility
    - Lazy loading of use case implementations
    - Configuration override support
    - Dependency injection for LLM clients

    Example:
        >>> factory = EnterpriseUseCaseFactory()
        >>> factory.register(EnterpriseDomain.MA_DUE_DILIGENCE, MADueDiligence)
        >>> use_case = factory.create(EnterpriseDomain.MA_DUE_DILIGENCE)
    """

    # Class-level registry of use case implementations
    _registry: dict[EnterpriseDomain, type[BaseUseCase]] = {}

    def __init__(
        self,
        settings: Settings | None = None,
        enterprise_settings: EnterpriseSettings | None = None,
        llm_client: LLMClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the factory.

        Args:
            settings: Optional application settings
            enterprise_settings: Optional enterprise settings
            llm_client: Optional LLM client for agent operations
            logger: Optional logger instance
        """
        self._settings = settings
        self._enterprise_settings = enterprise_settings or get_enterprise_settings()
        self._llm_client = llm_client
        self._logger = logger or logging.getLogger(__name__)

        # Auto-register built-in use cases
        self._auto_register()

    @property
    def settings(self) -> Settings:
        """Get application settings, loading if needed."""
        if self._settings is None:
            from src.config.settings import get_settings

            self._settings = get_settings()
        return self._settings

    @property
    def enterprise_settings(self) -> EnterpriseSettings:
        """Get enterprise settings."""
        return self._enterprise_settings

    @property
    def llm_client(self) -> LLMClient | None:
        """Get LLM client."""
        return self._llm_client

    @classmethod
    def register(
        cls,
        domain: EnterpriseDomain,
        use_case_class: type[BaseUseCase],
    ) -> None:
        """
        Register a use case implementation.

        Args:
            domain: Enterprise domain to register
            use_case_class: Use case class to register
        """
        cls._registry[domain] = use_case_class
        logging.getLogger(__name__).info(f"Registered use case: {domain.value} -> {use_case_class.__name__}")

    @classmethod
    def unregister(cls, domain: EnterpriseDomain) -> None:
        """
        Unregister a use case implementation.

        Args:
            domain: Enterprise domain to unregister
        """
        if domain in cls._registry:
            del cls._registry[domain]
            logging.getLogger(__name__).info(f"Unregistered use case: {domain.value}")

    @classmethod
    def list_registered(cls) -> list[EnterpriseDomain]:
        """
        List all registered use case domains.

        Returns:
            List of registered domains
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, domain: EnterpriseDomain) -> bool:
        """
        Check if a domain is registered.

        Args:
            domain: Domain to check

        Returns:
            True if registered, False otherwise
        """
        return domain in cls._registry

    def _auto_register(self) -> None:
        """Auto-register built-in use cases."""
        # Import and register use cases lazily to avoid circular imports
        try:
            from ..use_cases.ma_due_diligence import MADueDiligence

            self.register(EnterpriseDomain.MA_DUE_DILIGENCE, MADueDiligence)
        except ImportError:
            self._logger.debug("MA Due Diligence use case not available")

        try:
            from ..use_cases.clinical_trial import ClinicalTrialDesign

            self.register(EnterpriseDomain.CLINICAL_TRIAL, ClinicalTrialDesign)
        except ImportError:
            self._logger.debug("Clinical Trial use case not available")

        try:
            from ..use_cases.regulatory_compliance import RegulatoryCompliance

            self.register(EnterpriseDomain.REGULATORY_COMPLIANCE, RegulatoryCompliance)
        except ImportError:
            self._logger.debug("Regulatory Compliance use case not available")

    def create(
        self,
        domain: EnterpriseDomain,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> UseCaseProtocol:
        """
        Create a use case instance for the specified domain.

        Args:
            domain: Enterprise domain to create
            config_overrides: Optional config overrides
            **kwargs: Additional factory arguments

        Returns:
            Configured use case instance

        Raises:
            ValueError: If domain is not registered
        """
        if domain not in self._registry:
            available = [d.value for d in self._registry]
            raise ValueError(f"Unknown domain: {domain.value}. Registered domains: {available}")

        # Get configuration
        config = self._enterprise_settings.get_use_case_config(domain)

        # Apply overrides if provided
        if config_overrides:
            config_dict = config.model_dump()
            config_dict.update(config_overrides)
            config = type(config)(**config_dict)

        # Get use case class
        use_case_class = self._registry[domain]

        self._logger.info(
            f"Creating use case: domain={domain.value}, class={use_case_class.__name__}, enabled={config.enabled}",
            extra={"domain": domain.value},
        )

        return use_case_class(
            config=config,
            llm_client=self._llm_client,
            logger=self._logger,
            **kwargs,
        )

    def create_all_enabled(self) -> dict[EnterpriseDomain, UseCaseProtocol]:
        """
        Create all enabled use cases.

        Returns:
            Dictionary mapping domains to use case instances
        """
        use_cases = {}
        for domain in self._registry:
            config = self._enterprise_settings.get_use_case_config(domain)
            if config.enabled:
                try:
                    use_cases[domain] = self.create(domain)
                except Exception as e:
                    self._logger.error(f"Failed to create {domain.value} use case: {e}")
        return use_cases

    def create_from_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> UseCaseProtocol | None:
        """
        Auto-detect and create use case from query content.

        Uses keyword matching to detect the appropriate domain.

        Args:
            query: User query
            context: Optional context

        Returns:
            Detected use case or None if no match
        """
        query_lower = query.lower()

        # Domain detection keywords
        domain_keywords = {
            EnterpriseDomain.MA_DUE_DILIGENCE: [
                "acquisition",
                "merger",
                "due diligence",
                "m&a",
                "target company",
                "synergy",
                "valuation",
                "deal structure",
            ],
            EnterpriseDomain.CLINICAL_TRIAL: [
                "clinical trial",
                "fda",
                "ema",
                "regulatory approval",
                "phase",
                "endpoint",
                "cohort",
                "sample size",
                "statistical power",
            ],
            EnterpriseDomain.REGULATORY_COMPLIANCE: [
                "compliance",
                "regulation",
                "audit",
                "enforcement",
                "jurisdiction",
                "gdpr",
                "sox",
                "hipaa",
                "gap analysis",
            ],
        }

        # Score each domain
        domain_scores: dict[EnterpriseDomain, int] = {}
        for domain, keywords in domain_keywords.items():
            if domain not in self._registry:
                continue
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            self._logger.info("No domain detected from query")
            return None

        # Return use case for highest scoring domain
        best_domain = max(domain_scores, key=lambda k: domain_scores[k])
        self._logger.info(f"Detected domain from query: {best_domain.value} (score={domain_scores[best_domain]})")
        return self.create(best_domain)
