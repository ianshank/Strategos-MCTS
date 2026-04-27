"""
LLM Client Factory and Provider Registry.

This module provides a factory function to instantiate the correct LLM client
based on provider settings, with lazy loading of adapters.
"""

import importlib
from typing import Any

from src.observability.logging import get_logger

from .base import BaseLLMClient, LLMClient, LLMResponse, LLMToolResponse, ToolCall
from .exceptions import (
    CircuitBreakerOpenError,
    LLMAuthenticationError,
    LLMClientError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMInvalidRequestError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
    LLMStreamError,
    LLMTimeoutError,
)

logger = get_logger(__name__)

# Provider registry with lazy loading
# Maps provider name to (module_path, class_name)
_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "openai": ("src.adapters.llm.openai_client", "OpenAIClient"),
    "anthropic": ("src.adapters.llm.anthropic_client", "AnthropicClient"),
    "lmstudio": ("src.adapters.llm.lmstudio_client", "LMStudioClient"),
    "local": ("src.adapters.llm.lmstudio_client", "LMStudioClient"),  # Alias
}

# Cache for loaded client classes
_CLIENT_CACHE: dict[str, type[BaseLLMClient]] = {}


def register_provider(name: str, module_path: str, class_name: str, override: bool = False) -> None:
    """
    Register a new LLM provider.

    Args:
        name: Provider identifier (e.g., "azure", "bedrock")
        module_path: Full module path (e.g., "src.adapters.llm.azure_client")
        class_name: Class name in the module (e.g., "AzureOpenAIClient")
        override: If True, allow overriding existing provider
    """
    if name in _PROVIDER_REGISTRY and not override:
        raise ValueError(f"Provider '{name}' already registered. Use override=True to replace.")

    _PROVIDER_REGISTRY[name] = (module_path, class_name)
    # Clear cache if overriding
    if name in _CLIENT_CACHE:
        del _CLIENT_CACHE[name]

    logger.info(f"Registered LLM provider: {name} -> {module_path}.{class_name}")


def list_providers() -> list[str]:
    """
    List all registered provider names.

    Returns:
        List of provider identifiers
    """
    return list(_PROVIDER_REGISTRY.keys())


def get_provider_class(provider: str) -> type[BaseLLMClient]:
    """
    Get the client class for a provider (with lazy loading).

    Args:
        provider: Provider identifier

    Returns:
        Client class (not instantiated)

    Raises:
        ValueError: If provider not registered
        ImportError: If module cannot be loaded
    """
    if provider not in _PROVIDER_REGISTRY:
        available = ", ".join(list_providers())
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

    # Check cache first
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    # Lazy load the module
    module_path, class_name = _PROVIDER_REGISTRY[provider]

    try:
        module = importlib.import_module(module_path)
        client_class: type[BaseLLMClient] = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Failed to load provider '{provider}': {e}") from e
    except AttributeError as e:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'") from e

    # Cache for future use
    _CLIENT_CACHE[provider] = client_class
    return client_class


def create_client(
    provider: str = "openai",
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Create an LLM client instance.

    This is the main factory function for creating provider clients.

    Args:
        provider: Provider name ("openai", "anthropic", "lmstudio", etc.)
        api_key: API key (may be optional for some providers)
        model: Model identifier
        base_url: Base URL for API
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        **kwargs: Provider-specific parameters

    Returns:
        Configured LLMClient instance

    Examples:
        # OpenAI client
        client = create_client("openai", model="gpt-4-turbo-preview")

        # Anthropic client
        client = create_client("anthropic", model="sonnet")

        # Local LM Studio
        client = create_client("lmstudio", base_url="http://localhost:1234/v1")

        # With custom settings (use environment variables for keys)
        client = create_client(
            "openai",
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=120.0,
            max_retries=5,
        )
    """
    client_class = get_provider_class(provider)

    # Build kwargs for client initialization
    init_kwargs = {**kwargs}

    if api_key is not None:
        init_kwargs["api_key"] = api_key
    if model is not None:
        init_kwargs["model"] = model
    if base_url is not None:
        init_kwargs["base_url"] = base_url
    if timeout is not None:
        init_kwargs["timeout"] = timeout
    if max_retries is not None:
        init_kwargs["max_retries"] = max_retries

    logger.info(f"Creating {provider} client with model={model or 'default'}")

    return client_class(**init_kwargs)


def create_client_from_config(config: dict) -> BaseLLMClient:
    """
    Create an LLM client from a configuration dictionary.

    Useful for loading settings from YAML/JSON config files.

    Args:
        config: Configuration dictionary with keys:
            - provider: Required provider name
            - Other keys passed to create_client

    Returns:
        Configured LLMClient instance

    Example:
        config = {
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
            "timeout": 60.0,
            "max_retries": 3
        }
        client = create_client_from_config(config)
    """
    config = config.copy()
    provider = config.pop("provider", "openai")
    return create_client(provider, **config)


# Convenience aliases for common use cases
def create_openai_client(**kwargs) -> BaseLLMClient:
    """Create an OpenAI client."""
    return create_client("openai", **kwargs)


def create_anthropic_client(**kwargs) -> BaseLLMClient:
    """Create an Anthropic Claude client."""
    return create_client("anthropic", **kwargs)


def create_local_client(**kwargs) -> BaseLLMClient:
    """Create a local LM Studio client."""
    return create_client("lmstudio", **kwargs)


__all__ = [
    # Base types
    "LLMClient",
    "LLMResponse",
    "LLMToolResponse",
    "ToolCall",
    "BaseLLMClient",
    # Exceptions
    "LLMClientError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMQuotaExceededError",
    "LLMModelNotFoundError",
    "LLMContextLengthError",
    "LLMInvalidRequestError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMServerError",
    "LLMResponseParseError",
    "LLMStreamError",
    "LLMContentFilterError",
    "CircuitBreakerOpenError",
    # Factory functions
    "create_client",
    "create_client_from_config",
    "create_openai_client",
    "create_anthropic_client",
    "create_local_client",
    # Registry functions
    "register_provider",
    "list_providers",
    "get_provider_class",
]
