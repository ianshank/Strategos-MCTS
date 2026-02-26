"""
Tests for framework factory patterns.

Tests the factory classes that enable modular component creation
with dependency injection and configuration management.
"""

from unittest.mock import Mock, patch

import pytest

from src.config.settings import Settings
from src.framework.factories import (
    ComponentFactory,
    LLMClientFactory,
)


class TestLLMClientFactory:
    """Tests for LLMClientFactory."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.LLM_PROVIDER = "openai"
        settings.LLM_MODEL = "gpt-4"
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.OPENAI_API_KEY = "test-key"
        settings.HTTP_TIMEOUT_SECONDS = 60.0
        settings.HTTP_MAX_RETRIES = 3
        settings.get_api_key.return_value = "test-key"
        return settings

    @pytest.fixture
    def factory(self, mock_settings):
        """Create factory instance for testing."""
        return LLMClientFactory(settings=mock_settings)

    def test_factory_initialization(self, mock_settings):
        """Test factory can be initialized with settings."""
        factory = LLMClientFactory(settings=mock_settings)
        assert factory.settings == mock_settings
        assert factory.logger is not None

    def test_factory_initialization_without_settings(self):
        """Test factory uses default settings when none provided."""
        with patch("src.framework.factories.get_settings") as mock_get_settings:
            mock_settings = Mock(spec=Settings)
            mock_get_settings.return_value = mock_settings

            factory = LLMClientFactory()
            assert factory.settings == mock_settings

    @patch("src.adapters.llm.create_client")
    def test_create_with_explicit_provider(self, mock_create_client, factory):
        """Test creating client with explicit provider."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        client = factory.create(provider="openai", model="gpt-4", timeout=30.0, max_retries=5)

        assert client == mock_client
        mock_create_client.assert_called_once()
        call_kwargs = mock_create_client.call_args.kwargs
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["timeout"] == 30.0
        assert call_kwargs["max_retries"] == 5

    @patch("src.adapters.llm.create_client")
    def test_create_uses_settings_defaults(self, mock_create_client, factory):
        """Test factory uses settings when parameters not provided."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        client = factory.create()

        assert client == mock_client
        call_kwargs = mock_create_client.call_args.kwargs
        # Should use provider from settings
        assert call_kwargs["provider"] == "openai"

    @patch("src.adapters.llm.create_client")
    def test_create_applies_default_timeout(self, mock_create_client, factory):
        """Test default timeout is applied when not specified."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        factory.create(provider="openai")

        call_kwargs = mock_create_client.call_args.kwargs
        assert call_kwargs["timeout"] == 60.0  # Default timeout
        assert call_kwargs["max_retries"] == 3  # Default retries

    @patch("src.adapters.llm.create_client")
    def test_create_passes_additional_kwargs(self, mock_create_client, factory):
        """Test additional kwargs are passed to create_client."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        factory.create(provider="anthropic", custom_param="custom_value", another_param=123)

        call_kwargs = mock_create_client.call_args.kwargs
        assert call_kwargs["custom_param"] == "custom_value"
        assert call_kwargs["another_param"] == 123

    @patch("src.adapters.llm.create_client")
    def test_create_from_settings(self, mock_create_client, factory):
        """Test creating client entirely from settings."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        client = factory.create_from_settings()

        assert client == mock_client
        # Should use all defaults from settings
        call_kwargs = mock_create_client.call_args.kwargs
        assert "provider" in call_kwargs

    @patch("src.adapters.llm.create_client")
    def test_create_logs_client_creation(self, mock_create_client, factory):
        """Test that client creation is logged."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        with patch.object(factory.logger, "info") as mock_log:
            factory.create(provider="openai", model="gpt-4")

            mock_log.assert_called_once()
            log_message = mock_log.call_args[0][0]
            assert "openai" in log_message.lower()
            assert "gpt-4" in log_message.lower()

    def test_component_factory_protocol(self):
        """Test that factories implement ComponentFactory protocol."""
        # This is a static check, but we can verify the protocol exists
        assert hasattr(ComponentFactory, "create")


class TestFactoryIntegration:
    """Integration tests for factory patterns."""

    @pytest.fixture
    def mock_settings_integration(self):
        """Create mock settings for integration tests."""
        settings = Mock(spec=Settings)
        settings.LLM_PROVIDER = "openai"
        settings.LLM_MODEL = "gpt-4"
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.OPENAI_API_KEY = "test-key"
        settings.HTTP_TIMEOUT_SECONDS = 60.0
        settings.HTTP_MAX_RETRIES = 3
        settings.get_api_key.return_value = "test-key"
        return settings

    @pytest.mark.integration
    @patch("src.adapters.llm.create_client")
    def test_factory_creates_different_providers(self, mock_create_client, mock_settings_integration):
        """Test factory can create clients for different providers."""
        factory = LLMClientFactory(settings=mock_settings_integration)

        providers = ["openai", "anthropic", "lmstudio"]
        for provider in providers:
            mock_create_client.reset_mock()
            mock_create_client.return_value = Mock()

            client = factory.create(provider=provider)

            assert client is not None
            call_kwargs = mock_create_client.call_args.kwargs
            assert call_kwargs["provider"] == provider

    @pytest.mark.integration
    def test_factory_error_handling(self, mock_settings_integration):
        """Test factory handles invalid configurations gracefully."""
        factory = LLMClientFactory(settings=mock_settings_integration)

        with patch("src.adapters.llm.create_client") as mock_create:
            mock_create.side_effect = ValueError("Invalid provider")

            with pytest.raises(ValueError, match="Invalid provider"):
                factory.create(provider="invalid_provider")


@pytest.mark.component
class TestFactoryComponentLevel:
    """Component-level tests for factory usage patterns."""

    def test_factory_dependency_injection(self):
        """Test factory enables dependency injection pattern."""
        # Create a mock settings object
        custom_settings = Mock(spec=Settings)
        custom_settings.LLM_PROVIDER = "custom_provider"

        # Inject custom settings
        factory = LLMClientFactory(settings=custom_settings)

        assert factory.settings == custom_settings
        assert factory.settings.LLM_PROVIDER == "custom_provider"

    @patch("src.adapters.llm.create_client")
    def test_factory_configuration_override(self, mock_create_client):
        """Test factory allows configuration override."""
        # Settings say one thing
        settings = Mock(spec=Settings)
        settings.LLM_PROVIDER = "openai"
        settings.LLM_MODEL = "gpt-3.5-turbo"
        settings.HTTP_TIMEOUT_SECONDS = 60.0
        settings.HTTP_MAX_RETRIES = 3
        settings.get_api_key.return_value = "test-key"

        factory = LLMClientFactory(settings=settings)
        mock_create_client.return_value = Mock()

        # But we can override at creation time
        factory.create(provider="anthropic", model="claude-3")

        call_kwargs = mock_create_client.call_args.kwargs
        assert call_kwargs["provider"] == "anthropic"
        assert call_kwargs["model"] == "claude-3"
