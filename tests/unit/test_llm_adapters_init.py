"""
Unit tests for LLM adapters __init__ module.

Tests provider registry, factory functions, and lazy loading.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestProviderRegistry:
    """Tests for the provider registry and registration functions."""

    def test_list_providers_returns_defaults(self) -> None:
        """Default providers include openai, anthropic, lmstudio, local."""
        from src.adapters.llm import list_providers

        providers = list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "lmstudio" in providers
        assert "local" in providers

    def test_register_provider_new(self) -> None:
        """Can register a new provider."""
        from src.adapters.llm import _PROVIDER_REGISTRY, register_provider

        try:
            register_provider("test_provider_xyz", "some.module", "SomeClass")
            assert "test_provider_xyz" in _PROVIDER_REGISTRY
            assert _PROVIDER_REGISTRY["test_provider_xyz"] == ("some.module", "SomeClass")
        finally:
            # Cleanup
            _PROVIDER_REGISTRY.pop("test_provider_xyz", None)

    def test_register_provider_duplicate_raises(self) -> None:
        """Registering duplicate provider without override raises ValueError."""
        from src.adapters.llm import _PROVIDER_REGISTRY, register_provider

        try:
            register_provider("test_dup_provider", "mod.a", "ClassA")
            with pytest.raises(ValueError, match="already registered"):
                register_provider("test_dup_provider", "mod.b", "ClassB")
        finally:
            _PROVIDER_REGISTRY.pop("test_dup_provider", None)

    def test_register_provider_override(self) -> None:
        """Can override an existing provider with override=True."""
        from src.adapters.llm import _CLIENT_CACHE, _PROVIDER_REGISTRY, register_provider

        try:
            register_provider("test_override_prov", "mod.a", "ClassA")
            # Put something in cache
            _CLIENT_CACHE["test_override_prov"] = MagicMock()
            register_provider("test_override_prov", "mod.b", "ClassB", override=True)
            assert _PROVIDER_REGISTRY["test_override_prov"] == ("mod.b", "ClassB")
            # Cache should be cleared on override
            assert "test_override_prov" not in _CLIENT_CACHE
        finally:
            _PROVIDER_REGISTRY.pop("test_override_prov", None)
            _CLIENT_CACHE.pop("test_override_prov", None)


@pytest.mark.unit
class TestGetProviderClass:
    """Tests for get_provider_class with lazy loading."""

    def test_unknown_provider_raises_value_error(self) -> None:
        """Unknown provider raises ValueError with available list."""
        from src.adapters.llm import get_provider_class

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider_class("nonexistent_provider_abc")

    def test_cached_class_returned(self) -> None:
        """Cached class is returned without re-importing."""
        from src.adapters.llm import _CLIENT_CACHE, _PROVIDER_REGISTRY, get_provider_class

        mock_class = MagicMock()
        try:
            _PROVIDER_REGISTRY["test_cached_prov"] = ("some.mod", "SomeCls")
            _CLIENT_CACHE["test_cached_prov"] = mock_class

            result = get_provider_class("test_cached_prov")
            assert result is mock_class
        finally:
            _PROVIDER_REGISTRY.pop("test_cached_prov", None)
            _CLIENT_CACHE.pop("test_cached_prov", None)

    def test_import_error_propagated(self) -> None:
        """ImportError is raised when module doesn't exist."""
        from src.adapters.llm import _CLIENT_CACHE, _PROVIDER_REGISTRY, get_provider_class

        try:
            _PROVIDER_REGISTRY["test_bad_import"] = (
                "nonexistent.module.path.xyz",
                "SomeClass",
            )
            with pytest.raises(ImportError, match="Failed to load provider"):
                get_provider_class("test_bad_import")
        finally:
            _PROVIDER_REGISTRY.pop("test_bad_import", None)
            _CLIENT_CACHE.pop("test_bad_import", None)

    def test_attribute_error_becomes_import_error(self) -> None:
        """Missing class in valid module raises ImportError."""
        from src.adapters.llm import _CLIENT_CACHE, _PROVIDER_REGISTRY, get_provider_class

        try:
            # 'json' is a valid module but has no class named 'FakeClassName'
            _PROVIDER_REGISTRY["test_bad_attr"] = ("json", "FakeClassName")
            with pytest.raises(ImportError, match="not found in module"):
                get_provider_class("test_bad_attr")
        finally:
            _PROVIDER_REGISTRY.pop("test_bad_attr", None)
            _CLIENT_CACHE.pop("test_bad_attr", None)


@pytest.mark.unit
class TestCreateClient:
    """Tests for create_client factory function."""

    def test_create_client_calls_provider_class(self) -> None:
        """create_client instantiates the correct provider class."""
        from src.adapters.llm import create_client

        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        with patch("src.adapters.llm.get_provider_class", return_value=mock_class):
            client = create_client("openai", api_key="sk-test", model="gpt-4")

        mock_class.assert_called_once_with(api_key="sk-test", model="gpt-4")
        assert client is mock_instance

    def test_create_client_none_params_excluded(self) -> None:
        """None parameters are not passed to the class constructor."""
        from src.adapters.llm import create_client

        mock_class = MagicMock()

        with patch("src.adapters.llm.get_provider_class", return_value=mock_class):
            create_client("openai")

        # Only kwargs that were explicitly provided should be passed
        mock_class.assert_called_once_with()

    def test_create_client_all_params(self) -> None:
        """All non-None params are forwarded to the class."""
        from src.adapters.llm import create_client

        mock_class = MagicMock()

        with patch("src.adapters.llm.get_provider_class", return_value=mock_class):
            create_client(
                "openai",
                api_key="key",
                model="gpt-4",
                base_url="http://localhost",
                timeout=30.0,
                max_retries=5,
                extra_param="value",
            )

        mock_class.assert_called_once_with(
            api_key="key",
            model="gpt-4",
            base_url="http://localhost",
            timeout=30.0,
            max_retries=5,
            extra_param="value",
        )

    def test_create_client_unknown_provider(self) -> None:
        """create_client with unknown provider raises ValueError."""
        from src.adapters.llm import create_client

        with pytest.raises((ValueError, ImportError)):
            create_client("nonexistent_provider_123")


@pytest.mark.unit
class TestCreateClientFromConfig:
    """Tests for create_client_from_config."""

    def test_from_config_extracts_provider(self) -> None:
        """Provider is extracted from config dict."""
        from src.adapters.llm import create_client_from_config

        mock_class = MagicMock()
        with patch("src.adapters.llm.get_provider_class", return_value=mock_class):
            create_client_from_config({"provider": "openai", "model": "gpt-4"})

        mock_class.assert_called_once_with(model="gpt-4")

    def test_from_config_default_provider(self) -> None:
        """Default provider is openai when not specified."""
        from src.adapters.llm import create_client_from_config

        mock_class = MagicMock()
        with patch("src.adapters.llm.get_provider_class", return_value=mock_class) as mock_get:
            create_client_from_config({"model": "gpt-4"})

        mock_get.assert_called_once_with("openai")

    def test_from_config_does_not_mutate_input(self) -> None:
        """Original config dict is not modified."""
        from src.adapters.llm import create_client_from_config

        config = {"provider": "openai", "model": "gpt-4"}
        original = config.copy()

        mock_class = MagicMock()
        with patch("src.adapters.llm.get_provider_class", return_value=mock_class):
            create_client_from_config(config)

        assert config == original


@pytest.mark.unit
class TestConvenienceAliases:
    """Tests for convenience alias functions."""

    def test_create_openai_client(self) -> None:
        """create_openai_client delegates to create_client with 'openai'."""
        from src.adapters.llm import create_openai_client

        with patch("src.adapters.llm.create_client") as mock_create:
            create_openai_client(model="gpt-4")
        mock_create.assert_called_once_with("openai", model="gpt-4")

    def test_create_anthropic_client(self) -> None:
        """create_anthropic_client delegates to create_client with 'anthropic'."""
        from src.adapters.llm import create_anthropic_client

        with patch("src.adapters.llm.create_client") as mock_create:
            create_anthropic_client(model="sonnet")
        mock_create.assert_called_once_with("anthropic", model="sonnet")

    def test_create_local_client(self) -> None:
        """create_local_client delegates to create_client with 'lmstudio'."""
        from src.adapters.llm import create_local_client

        with patch("src.adapters.llm.create_client") as mock_create:
            create_local_client(base_url="http://localhost:1234")
        mock_create.assert_called_once_with("lmstudio", base_url="http://localhost:1234")


@pytest.mark.unit
class TestAllExports:
    """Tests for __all__ exports."""

    def test_all_exports_importable(self) -> None:
        """All items in __all__ can be imported."""
        import src.adapters.llm as llm_module

        for name in llm_module.__all__:
            assert hasattr(llm_module, name), f"{name} listed in __all__ but not accessible"

    def test_key_exports_present(self) -> None:
        """Key types and functions are exported."""
        from src.adapters.llm import (
            LLMResponse,
            create_client,
        )

        # Just verify they imported without error
        assert LLMResponse is not None
        assert create_client is not None
