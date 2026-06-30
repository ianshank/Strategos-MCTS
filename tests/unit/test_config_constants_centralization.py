"""Tests locking in centralization of previously-hardcoded values into constants.

These guard against regressions where a literal model name / magic number is
re-introduced inline instead of referencing src.config.constants.
"""

from src.config import constants


def test_new_provider_and_diagram_constants_exist():
    assert constants.DEFAULT_LMSTUDIO_MODEL == "local-model"
    assert constants.DEFAULT_GOOGLE_GEMINI_MODEL == "gemini-2.0-flash-001"
    assert constants.DEFAULT_KROKI_BASE_URL.startswith("https://")
    assert constants.DEFAULT_KROKI_TIMEOUT_SECONDS > 0
    assert 0.0 <= constants.CHESS_ROUTING_CONFIDENCE_BOOST <= 1.0


def test_factories_use_constant_defaults():
    """LLMClientFactory default models must come from constants, not inline literals."""
    from src.framework.factories import LLMClientFactory

    factory = LLMClientFactory.__new__(LLMClientFactory)
    assert factory._get_default_model("anthropic") == constants.DEFAULT_ANTHROPIC_MODEL
    assert factory._get_default_model("lmstudio") == constants.DEFAULT_LMSTUDIO_MODEL
    assert factory._get_default_model("openai") == constants.DEFAULT_OPENAI_MODEL


def test_adk_config_default_model_uses_constant():
    from src.integrations.google_adk.base import ADKConfig

    assert ADKConfig().model_name == constants.DEFAULT_GOOGLE_GEMINI_MODEL
