"""
Unit tests for network configuration settings.

Tests the new settings for:
- LLM provider timeouts
- CORS configuration
- Rate limit configuration
- Circuit breaker settings
"""

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")


class TestLLMTimeoutSettings:
    """Test LLM provider-specific timeout settings."""

    def test_default_openai_timeout(self):
        """Test default OpenAI timeout value."""
        from src.config.settings import Settings

        # Create settings without API key requirement
        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        assert settings.OPENAI_TIMEOUT == 60.0

    def test_default_anthropic_timeout(self):
        """Test default Anthropic timeout value."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        assert settings.ANTHROPIC_TIMEOUT == 120.0

    def test_default_lmstudio_timeout(self):
        """Test default LMStudio timeout value."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        assert settings.LMSTUDIO_TIMEOUT == 300.0

    def test_timeout_bounds_validation(self):
        """Test timeout values have valid bounds."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        # All timeouts should be positive and within bounds
        assert 1.0 <= settings.OPENAI_TIMEOUT <= 600.0
        assert 1.0 <= settings.ANTHROPIC_TIMEOUT <= 600.0
        assert 1.0 <= settings.LMSTUDIO_TIMEOUT <= 600.0

    def test_custom_timeout_from_env(self, monkeypatch):
        """Test custom timeout from environment variable."""
        from src.config.settings import Settings, reset_settings

        reset_settings()
        monkeypatch.setenv("OPENAI_TIMEOUT", "90.5")
        monkeypatch.setenv("LLM_PROVIDER", "lmstudio")

        settings = Settings(_env_file=None)
        assert settings.OPENAI_TIMEOUT == 90.5


class TestCORSSettings:
    """Test CORS configuration settings."""

    def test_default_cors_origins_empty(self):
        """Test default CORS origins is empty list."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        assert settings.CORS_ALLOWED_ORIGINS == []

    def test_default_cors_allow_credentials(self):
        """Test default CORS allow credentials is True."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        assert settings.CORS_ALLOW_CREDENTIALS is True

    def test_cors_origins_from_env(self, monkeypatch):
        """Test CORS origins from environment variable."""
        from src.config.settings import Settings, reset_settings

        reset_settings()
        # Pydantic parses JSON for list fields
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", '["https://example.com", "https://app.example.com"]')
        monkeypatch.setenv("LLM_PROVIDER", "lmstudio")

        settings = Settings(_env_file=None)
        assert "https://example.com" in settings.CORS_ALLOWED_ORIGINS
        assert "https://app.example.com" in settings.CORS_ALLOWED_ORIGINS


class TestCircuitBreakerSettings:
    """Test circuit breaker configuration."""

    def test_default_circuit_breaker_reset(self):
        """Test default circuit breaker reset time."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        assert settings.CIRCUIT_BREAKER_RESET_SECONDS == 60.0

    def test_circuit_breaker_bounds(self):
        """Test circuit breaker reset time bounds."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        assert 10.0 <= settings.CIRCUIT_BREAKER_RESET_SECONDS <= 600.0


class TestRateLimitSettings:
    """Test rate limit configuration settings."""

    def test_default_retry_after_seconds(self):
        """Test default retry-after value."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        assert settings.RATE_LIMIT_RETRY_AFTER_SECONDS == 60

    def test_retry_after_bounds(self):
        """Test retry-after bounds are valid."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        assert 1 <= settings.RATE_LIMIT_RETRY_AFTER_SECONDS <= 3600


class TestDatetimeBackwardsCompatibility:
    """Test datetime.utcnow() replacement for backwards compatibility."""

    def test_llm_response_created_at_is_utc(self):
        """Test LLMResponse created_at uses UTC timezone."""
        from src.adapters.llm.base import LLMResponse

        response = LLMResponse(text="test")
        # The created_at should be timezone-aware UTC
        assert response.created_at.tzinfo is not None
        assert response.created_at.tzinfo.utcoffset(None).total_seconds() == 0

    def test_client_info_timestamps_are_utc(self):
        """Test ClientInfo timestamps use UTC timezone."""
        from src.api.auth import ClientInfo

        client = ClientInfo(client_id="test-client")
        # Both timestamps should be timezone-aware UTC
        assert client.created_at.tzinfo is not None
        assert client.last_access.tzinfo is not None
        assert client.created_at.tzinfo.utcoffset(None).total_seconds() == 0
        assert client.last_access.tzinfo.utcoffset(None).total_seconds() == 0


class TestSettingsIntegration:
    """Integration tests for settings with other components."""

    def test_settings_safe_dict_includes_new_fields(self):
        """Test that safe_dict includes new configuration fields."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        safe = settings.safe_dict()

        # New fields should be present
        assert "OPENAI_TIMEOUT" in safe
        assert "ANTHROPIC_TIMEOUT" in safe
        assert "LMSTUDIO_TIMEOUT" in safe
        assert "CORS_ALLOWED_ORIGINS" in safe
        assert "CORS_ALLOW_CREDENTIALS" in safe
        assert "CIRCUIT_BREAKER_RESET_SECONDS" in safe
        assert "RATE_LIMIT_RETRY_AFTER_SECONDS" in safe

    def test_settings_repr_works(self):
        """Test settings repr doesn't expose secrets."""
        from src.config.settings import Settings

        settings = Settings(
            LLM_PROVIDER="lmstudio",
            _env_file=None,
        )
        repr_str = repr(settings)

        # Should not contain sensitive values
        assert "sk-" not in repr_str
        assert "api_key" not in repr_str.lower() or "***" in repr_str
