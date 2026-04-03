"""Tests for config/settings.py validators and API key getters."""

import os
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestSettingsValidators:
    """Test API key validators and other field validators."""

    def _make_settings(self, **overrides):
        """Create Settings with valid base config and overrides."""
        env = {
            "OPENAI_API_KEY": "sk-" + "a" * 50,
            "LLM_PROVIDER": "openai",
        }
        env.update(overrides)
        with patch.dict(os.environ, env, clear=False):
            from src.config.settings import Settings
            # Clear cache
            return Settings()

    def test_anthropic_key_placeholder_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="placeholder"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "ANTHROPIC_API_KEY": "your-api-key-here"}):
                Settings()

    def test_anthropic_key_too_short_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="too short"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "ANTHROPIC_API_KEY": "short"}):
                Settings()

    def test_braintrust_key_placeholder_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="placeholder"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "BRAINTRUST_API_KEY": "REPLACE_ME"}):
                Settings()

    def test_braintrust_key_too_short_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="too short"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "BRAINTRUST_API_KEY": "short"}):
                Settings()

    def test_pinecone_key_placeholder_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="placeholder"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "PINECONE_API_KEY": "your-api-key-here"}):
                Settings()

    def test_pinecone_key_too_short_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="too short"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "PINECONE_API_KEY": "x"}):
                Settings()

    def test_langsmith_key_placeholder_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="placeholder"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "LANGSMITH_API_KEY": ""}):
                Settings()

    def test_langsmith_key_too_short_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="too short"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "LANGSMITH_API_KEY": "short"}):
                Settings()

    def test_wandb_key_placeholder_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="placeholder"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "WANDB_API_KEY": "your-api-key-here"}):
                Settings()

    def test_wandb_key_too_short_rejected(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="too short"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "WANDB_API_KEY": "ab"}):
                Settings()

    def test_lmstudio_url_must_be_http(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="http"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "LMSTUDIO_BASE_URL": "ftp://localhost:1234"}):
                Settings()

    def test_s3_bucket_too_short(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="3-63"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "S3_BUCKET": "ab"}):
                Settings()

    def test_s3_bucket_hyphen_start(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="hyphen"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "S3_BUCKET": "-mybucket"}):
                Settings()


@pytest.mark.unit
class TestSettingsAPIKeyGetters:
    """Test get_*_api_key methods."""

    def test_get_braintrust_api_key_none(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai"}, clear=False):
            from src.config.settings import Settings
            s = Settings()
            if s.BRAINTRUST_API_KEY is None:
                assert s.get_braintrust_api_key() is None

    def test_get_braintrust_api_key_present(self):
        key = "bt-" + "x" * 50
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "BRAINTRUST_API_KEY": key}):
            from src.config.settings import Settings
            s = Settings()
            assert s.get_braintrust_api_key() == key

    def test_get_pinecone_api_key_none(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai"}, clear=False):
            from src.config.settings import Settings
            s = Settings()
            if s.PINECONE_API_KEY is None:
                assert s.get_pinecone_api_key() is None

    def test_get_pinecone_api_key_present(self):
        key = "pc-" + "x" * 50
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "PINECONE_API_KEY": key}):
            from src.config.settings import Settings
            s = Settings()
            assert s.get_pinecone_api_key() == key

    def test_get_langsmith_api_key_none(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai"}, clear=False):
            from src.config.settings import Settings
            s = Settings()
            if s.LANGSMITH_API_KEY is None:
                assert s.get_langsmith_api_key() is None

    def test_get_langsmith_api_key_present(self):
        key = "ls-" + "x" * 50
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "LANGSMITH_API_KEY": key}):
            from src.config.settings import Settings
            s = Settings()
            assert s.get_langsmith_api_key() == key

    def test_get_wandb_api_key_none(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai"}, clear=False):
            from src.config.settings import Settings
            s = Settings()
            if s.WANDB_API_KEY is None:
                assert s.get_wandb_api_key() is None

    def test_get_wandb_api_key_present(self):
        key = "wb-" + "x" * 50
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 50, "LLM_PROVIDER": "openai", "WANDB_API_KEY": key}):
            from src.config.settings import Settings
            s = Settings()
            assert s.get_wandb_api_key() == key

    def test_anthropic_provider_requires_key(self):
        from src.config.settings import Settings
        with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
            with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
                Settings()

    def test_lmstudio_provider_has_default_url(self):
        """LM Studio provider works with default URL."""
        from src.config.settings import Settings
        with patch.dict(os.environ, {"LLM_PROVIDER": "lmstudio"}, clear=True):
            s = Settings()
            assert s.LLM_PROVIDER.value == "lmstudio"
