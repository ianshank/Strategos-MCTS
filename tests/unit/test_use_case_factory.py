"""
Tests for enterprise use case factory module.

Tests EnterpriseUseCaseFactory: registration, creation, query-based detection,
and create_all_enabled functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.enterprise.base.use_case import BaseDomainState, BaseUseCase
from src.enterprise.config.enterprise_settings import (
    BaseUseCaseConfig,
    EnterpriseDomain,
    EnterpriseSettings,
)
from src.enterprise.factories.use_case_factory import EnterpriseUseCaseFactory


# Concrete test subclass of BaseUseCase
class _StubUseCase(BaseUseCase):
    """Minimal concrete use case for testing."""

    @property
    def name(self) -> str:
        return "stub_use_case"

    @property
    def domain(self) -> str:
        return "test"

    def get_initial_state(self, query, context):
        return BaseDomainState(state_id="s1", domain="test")

    def get_available_actions(self, state):
        return ["action_a"]

    def apply_action(self, state, action):
        return state


class _AnotherStubUseCase(BaseUseCase):
    """Another stub for testing multiple registrations."""

    @property
    def name(self) -> str:
        return "another_stub"

    @property
    def domain(self) -> str:
        return "test2"

    def get_initial_state(self, query, context):
        return BaseDomainState(state_id="s2", domain="test2")

    def get_available_actions(self, state):
        return []

    def apply_action(self, state, action):
        return state


@pytest.mark.unit
class TestEnterpriseUseCaseFactory:
    """Tests for EnterpriseUseCaseFactory."""

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        """Clear registry before and after each test."""
        saved = dict(EnterpriseUseCaseFactory._registry)
        EnterpriseUseCaseFactory._registry.clear()
        yield
        EnterpriseUseCaseFactory._registry.clear()
        EnterpriseUseCaseFactory._registry.update(saved)

    @pytest.fixture
    def enterprise_settings(self):
        """Create EnterpriseSettings for testing."""
        return EnterpriseSettings()

    @pytest.fixture
    def factory(self, enterprise_settings):
        """Create factory with auto-registration disabled."""
        with patch.object(EnterpriseUseCaseFactory, "_auto_register"):
            return EnterpriseUseCaseFactory(enterprise_settings=enterprise_settings)

    # --- Registration ---

    def test_register(self, factory):
        """Test registering a use case class."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.MA_DUE_DILIGENCE, _StubUseCase)
        assert EnterpriseUseCaseFactory.is_registered(EnterpriseDomain.MA_DUE_DILIGENCE)

    def test_unregister(self, factory):
        """Test unregistering a use case class."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.MA_DUE_DILIGENCE, _StubUseCase)
        EnterpriseUseCaseFactory.unregister(EnterpriseDomain.MA_DUE_DILIGENCE)
        assert not EnterpriseUseCaseFactory.is_registered(EnterpriseDomain.MA_DUE_DILIGENCE)

    def test_unregister_missing(self, factory):
        """Test unregistering a non-existent domain does not raise."""
        EnterpriseUseCaseFactory.unregister(EnterpriseDomain.CLINICAL_TRIAL)
        # No error expected

    def test_list_registered(self, factory):
        """Test listing registered domains."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.MA_DUE_DILIGENCE, _StubUseCase)
        EnterpriseUseCaseFactory.register(EnterpriseDomain.CLINICAL_TRIAL, _AnotherStubUseCase)
        registered = EnterpriseUseCaseFactory.list_registered()
        assert EnterpriseDomain.MA_DUE_DILIGENCE in registered
        assert EnterpriseDomain.CLINICAL_TRIAL in registered

    def test_is_registered_false(self, factory):
        """Test is_registered returns False for unknown domain."""
        assert not EnterpriseUseCaseFactory.is_registered(EnterpriseDomain.REGULATORY_COMPLIANCE)

    # --- Initialization ---

    def test_init_properties(self, enterprise_settings):
        """Test factory properties after init."""
        mock_llm = MagicMock()
        with patch.object(EnterpriseUseCaseFactory, "_auto_register"):
            f = EnterpriseUseCaseFactory(
                enterprise_settings=enterprise_settings,
                llm_client=mock_llm,
            )
        assert f.enterprise_settings is enterprise_settings
        assert f.llm_client is mock_llm

    def test_settings_lazy_load(self, enterprise_settings):
        """Test settings property lazy loads if None."""
        with patch.object(EnterpriseUseCaseFactory, "_auto_register"):
            factory = EnterpriseUseCaseFactory(
                settings=None,
                enterprise_settings=enterprise_settings,
            )
        with patch("src.config.settings.get_settings") as mock_get:
            mock_get.return_value = MagicMock()
            # Access settings property triggers lazy load
            _ = factory.settings
            mock_get.assert_called_once()

    # --- create() ---

    def test_create_success(self, factory):
        """Test creating a use case instance."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.MA_DUE_DILIGENCE, _StubUseCase)
        use_case = factory.create(EnterpriseDomain.MA_DUE_DILIGENCE)
        assert use_case.name == "stub_use_case"

    def test_create_unknown_domain(self, factory):
        """Test creating with unregistered domain raises ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            factory.create(EnterpriseDomain.CLINICAL_TRIAL)

    def test_create_with_config_overrides(self, factory):
        """Test creating with config overrides."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.MA_DUE_DILIGENCE, _StubUseCase)
        use_case = factory.create(
            EnterpriseDomain.MA_DUE_DILIGENCE,
            config_overrides={"enabled": False},
        )
        assert use_case.config.enabled is False

    # --- create_all_enabled() ---

    def test_create_all_enabled(self, factory):
        """Test creating all enabled use cases."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.MA_DUE_DILIGENCE, _StubUseCase)
        EnterpriseUseCaseFactory.register(EnterpriseDomain.CLINICAL_TRIAL, _AnotherStubUseCase)
        result = factory.create_all_enabled()
        assert EnterpriseDomain.MA_DUE_DILIGENCE in result
        assert EnterpriseDomain.CLINICAL_TRIAL in result

    def test_create_all_enabled_skips_disabled(self, factory, enterprise_settings):
        """Test disabled use cases are skipped."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.MA_DUE_DILIGENCE, _StubUseCase)
        # Disable MA use case
        with patch.object(
            type(enterprise_settings),
            "get_use_case_config",
            return_value=BaseUseCaseConfig(enabled=False),
        ):
            result = factory.create_all_enabled()
        assert EnterpriseDomain.MA_DUE_DILIGENCE not in result

    def test_create_all_enabled_handles_errors(self, factory):
        """Test create_all_enabled handles creation errors gracefully."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.MA_DUE_DILIGENCE, _StubUseCase)
        with patch.object(factory, "create", side_effect=RuntimeError("boom")):
            result = factory.create_all_enabled()
        assert len(result) == 0

    # --- create_from_query() ---

    def test_create_from_query_ma(self, factory):
        """Test auto-detection of M&A domain from query."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.MA_DUE_DILIGENCE, _StubUseCase)
        use_case = factory.create_from_query("Analyze acquisition target for merger")
        assert use_case is not None
        assert use_case.name == "stub_use_case"

    def test_create_from_query_clinical(self, factory):
        """Test auto-detection of clinical trial domain from query."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.CLINICAL_TRIAL, _AnotherStubUseCase)
        use_case = factory.create_from_query("Design a clinical trial with FDA endpoint")
        assert use_case is not None

    def test_create_from_query_regulatory(self, factory):
        """Test auto-detection of regulatory compliance domain from query."""
        EnterpriseUseCaseFactory.register(EnterpriseDomain.REGULATORY_COMPLIANCE, _StubUseCase)
        use_case = factory.create_from_query("Check compliance with GDPR regulation")
        assert use_case is not None

    def test_create_from_query_no_match(self, factory):
        """Test returns None when no domain matches."""
        result = factory.create_from_query("Tell me about the weather")
        assert result is None

    def test_create_from_query_unregistered_domain_ignored(self, factory):
        """Test unregistered domains are ignored even if keywords match."""
        # Don't register any domains
        result = factory.create_from_query("Analyze acquisition target")
        assert result is None

    # --- auto_register ---

    def test_auto_register_imports(self, enterprise_settings):
        """Test _auto_register attempts to import use cases."""
        EnterpriseUseCaseFactory._registry.clear()
        # Just verify it does not raise even if imports fail
        f = EnterpriseUseCaseFactory(enterprise_settings=enterprise_settings)
        # It should have tried to register whatever is available
        # (may or may not have registered depending on whether use_cases exist)
        assert isinstance(f, EnterpriseUseCaseFactory)
