"""Tests for Phase 6 config centralization (specs/phase_6_config_centralization.SPEC.md).

Covers the two previously-unmet acceptance criteria:
1. LLM temperature defaults are defined once in ``src/config/constants.py`` and
   referenced by the agent modules (no divergent local literals).
2. The import-time ``os.getenv()`` reads in ``llm_guided/constants.py`` are replaced
   by lazy ``Settings`` accessors backed by new ``MCTS_*`` fields; an override propagates.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config import constants as core_constants
from src.config.settings import Settings
from src.framework.mcts.llm_guided import constants as C


def _settings(**overrides: object) -> Settings:
    """Build a valid Settings without requiring an API key (LMStudio needs none)."""
    return Settings(LLM_PROVIDER="lmstudio", **overrides)  # type: ignore[arg-type]


class TestTemperatureCentralization:
    """Criterion #1 — each agent module sources its temperature from constants.py."""

    def test_constants_define_the_temperatures_once(self) -> None:
        assert core_constants.DEFAULT_LLM_TEMPERATURE == 0.7
        assert core_constants.DEFAULT_HRM_TEMPERATURE == 0.5
        assert core_constants.DEFAULT_TRM_TEMPERATURE == 0.5
        assert core_constants.DEFAULT_CHESS_LLM_TEMPERATURE == 0.3

    def test_llm_mcts_reads_the_constant(self) -> None:
        from src.framework.mcts import llm_mcts

        assert llm_mcts.DEFAULT_TEMPERATURE == core_constants.DEFAULT_LLM_TEMPERATURE

    def test_hrm_reads_the_constant(self) -> None:
        from src.framework.agents import llm_hrm

        assert llm_hrm.DEFAULT_HRM_TEMPERATURE == core_constants.DEFAULT_HRM_TEMPERATURE

    def test_trm_reads_the_constant(self) -> None:
        from src.framework.agents import llm_trm

        assert llm_trm.DEFAULT_TRM_TEMPERATURE == core_constants.DEFAULT_TRM_TEMPERATURE

    def test_chess_engine_reads_the_constant(self) -> None:
        from src.games.chess import llm_chess_engine

        assert llm_chess_engine.DEFAULT_CHESS_TEMPERATURE == core_constants.DEFAULT_CHESS_LLM_TEMPERATURE


class TestSettingsFieldsExist:
    """Criterion #2 — the four named Settings fields exist with documented defaults."""

    def test_defaults(self) -> None:
        s = _settings()
        assert s.MCTS_GENERATOR_MODEL == "gpt-4o"
        assert s.MCTS_REFLECTOR_MODEL == "gpt-4o"
        assert s.MCTS_EXECUTION_TIMEOUT == 5.0
        assert s.MCTS_MAX_MEMORY_MB == 256


class TestLazyAccessors:
    """Criterion #2 — accessors resolve from Settings and an override propagates."""

    def test_no_import_time_getenv_for_migrated_values(self) -> None:
        # The module-level defaults are plain literals now, not env-derived.
        assert C.DEFAULT_GENERATOR_MODEL == "gpt-4o"
        assert C.DEFAULT_REFLECTOR_MODEL == "gpt-4o"
        assert C.DEFAULT_EXECUTION_TIMEOUT == 5.0
        assert C.DEFAULT_MAX_MEMORY_MB == 256

    def test_override_propagates(self) -> None:
        s = _settings(
            MCTS_GENERATOR_MODEL="llama-3",
            MCTS_REFLECTOR_MODEL="mixtral",
            MCTS_EXECUTION_TIMEOUT=12.5,
            MCTS_MAX_MEMORY_MB=512,
        )
        assert C.get_generator_model(s) == "llama-3"
        assert C.get_reflector_model(s) == "mixtral"
        assert C.get_execution_timeout(s) == 12.5
        assert C.get_max_memory_mb(s) == 512

    def test_defaults_when_settings_unset(self) -> None:
        s = _settings()
        assert C.get_generator_model(s) == C.DEFAULT_GENERATOR_MODEL
        assert C.get_reflector_model(s) == C.DEFAULT_REFLECTOR_MODEL
        assert C.get_execution_timeout(s) == C.DEFAULT_EXECUTION_TIMEOUT
        assert C.get_max_memory_mb(s) == C.DEFAULT_MAX_MEMORY_MB

    def test_missing_attribute_falls_back_to_default(self) -> None:
        bare = MagicMock(spec=[])  # no MCTS_* attributes
        assert C.get_generator_model(bare) == C.DEFAULT_GENERATOR_MODEL
        assert C.get_max_memory_mb(bare) == C.DEFAULT_MAX_MEMORY_MB

    def test_settings_load_failure_falls_back_to_default(self) -> None:
        with patch("src.config.settings.get_settings", side_effect=RuntimeError("boom")):
            assert C.get_reflector_model() == C.DEFAULT_REFLECTOR_MODEL
            assert C.get_execution_timeout() == C.DEFAULT_EXECUTION_TIMEOUT


class TestConfigDefaultFactoryPropagation:
    """The llm_guided config dataclasses resolve models/limits lazily via the accessors."""

    def test_generator_config_uses_accessor(self) -> None:
        from src.framework.mcts.llm_guided.config import GeneratorConfig

        with patch.object(C, "get_generator_model", return_value="patched-model"):
            assert GeneratorConfig().model == "patched-model"

    def test_execution_limits_use_accessors(self) -> None:
        from src.framework.mcts.llm_guided.config import LLMGuidedMCTSConfig

        with (
            patch.object(C, "get_execution_timeout", return_value=42.0),
            patch.object(C, "get_max_memory_mb", return_value=128),
        ):
            cfg = LLMGuidedMCTSConfig()
            assert cfg.execution_timeout_seconds == 42.0
            assert cfg.max_memory_mb == 128
