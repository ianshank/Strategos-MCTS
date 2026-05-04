"""
Tests for ``src.adapters.llm.model_presets``.

Each test that mutates the module-level registry restores the built-in
state in a fixture teardown so that tests remain independent regardless of
execution order.
"""

from __future__ import annotations

import importlib

import pytest

from src.adapters.llm import model_presets as mp


@pytest.fixture
def fresh_registry():
    """
    Yield a clean registry for the duration of the test, then restore
    built-ins by reloading the module.

    Tests that need to register, override, or clear presets should use
    this fixture so they do not pollute state for other tests.
    """
    # Snapshot existing registry so we can fully restore even if reload
    # were to fail for some reason.
    snapshot = dict(mp._PRESET_REGISTRY)
    mp.clear_registry()
    try:
        yield mp
    finally:
        # Reload the module to restore built-in registrations from source.
        importlib.reload(mp)
        # Defensive: if some test imported names directly from this module
        # (e.g. ``PHI4_REASONING``), keep the registry consistent with the
        # snapshot in case reload somehow lost entries.
        for name, preset in snapshot.items():
            if name not in mp._PRESET_REGISTRY:
                mp._PRESET_REGISTRY[name] = preset


@pytest.mark.unit
class TestModelPresetMatching:
    """Pattern matching semantics."""

    def test_register_and_lookup_by_pattern(self, fresh_registry):
        preset = mp.ModelPreset(name="foo-preset", name_pattern=r"foo")
        mp.register_preset(preset)

        result = mp.get_preset("foo-1.0")
        assert result is preset

    def test_pattern_is_case_insensitive(self, fresh_registry):
        preset = mp.ModelPreset(name="phi-test", name_pattern=r"PHI")
        mp.register_preset(preset)

        # Lower-case model name should still match an upper-case pattern.
        assert mp.get_preset("phi-4") is preset
        assert mp.get_preset("PHI-4") is preset

    def test_matches_method_handles_none_and_empty(self):
        preset = mp.ModelPreset(name="x", name_pattern=r"x")
        assert preset.matches(None) is False
        assert preset.matches("") is False
        assert preset.matches("xyz") is True

    def test_first_match_wins_on_overlapping_patterns(self, fresh_registry):
        first = mp.ModelPreset(name="alpha", name_pattern=r"foo")
        second = mp.ModelPreset(name="beta", name_pattern=r"foo")
        mp.register_preset(first)
        mp.register_preset(second)

        assert mp.get_preset("foobar") is first


@pytest.mark.unit
class TestGetPresetEdgeCases:
    """Edge cases for ``get_preset`` / ``get_preset_by_name``."""

    def test_get_preset_with_none_returns_none(self, fresh_registry):
        preset = mp.ModelPreset(name="phi", name_pattern=r"phi")
        mp.register_preset(preset)

        assert mp.get_preset(None) is None

    def test_get_preset_with_empty_string_returns_none(self, fresh_registry):
        preset = mp.ModelPreset(name="phi", name_pattern=r"phi")
        mp.register_preset(preset)

        assert mp.get_preset("") is None

    def test_unknown_model_returns_none(self, fresh_registry):
        # Registry intentionally empty (fresh_registry cleared it).
        assert mp.get_preset("gpt-4") is None

    def test_get_preset_by_name_finds_phi4(self):
        # Default registry contains the built-in PHI4 preset.
        preset = mp.get_preset_by_name("phi4-reasoning")
        assert preset is not None
        assert preset is mp.PHI4_REASONING

    def test_get_preset_by_name_unknown_returns_none(self):
        assert mp.get_preset_by_name("does-not-exist") is None

    def test_get_preset_by_name_empty_returns_none(self):
        assert mp.get_preset_by_name("") is None
        assert mp.get_preset_by_name(None) is None


@pytest.mark.unit
class TestRegisterPreset:
    """Registration semantics, including duplicate handling."""

    def test_register_duplicate_raises_without_override(self, fresh_registry):
        preset = mp.ModelPreset(name="dup", name_pattern=r"dup")
        mp.register_preset(preset)

        with pytest.raises(ValueError, match="already registered"):
            mp.register_preset(mp.ModelPreset(name="dup", name_pattern=r"other"))

    def test_register_duplicate_succeeds_with_override(self, fresh_registry):
        first = mp.ModelPreset(name="dup", name_pattern=r"first")
        replacement = mp.ModelPreset(name="dup", name_pattern=r"second")
        mp.register_preset(first)

        mp.register_preset(replacement, override=True)

        assert mp.get_preset_by_name("dup") is replacement
        # The old pattern should no longer match.
        assert mp.get_preset("first") is None
        assert mp.get_preset("second") is replacement


@pytest.mark.unit
class TestListPresets:
    """`list_presets` returns names in insertion order."""

    def test_list_presets_includes_phi4_by_default(self):
        names = mp.list_presets()
        assert "phi4-reasoning" in names

    def test_list_presets_returns_empty_after_clear(self, fresh_registry):
        # `fresh_registry` has already cleared the registry.
        assert mp.list_presets() == []


@pytest.mark.unit
class TestClearRegistry:
    """Clear registry behavior (test-only utility)."""

    def test_clear_registry_empties_it(self, fresh_registry):
        mp.register_preset(mp.ModelPreset(name="x", name_pattern=r"x"))
        assert mp.list_presets() == ["x"]

        mp.clear_registry()
        assert mp.list_presets() == []

    def test_reload_restores_builtin_phi4(self):
        # Clear and reload should restore the built-in.
        mp.clear_registry()
        assert "phi4-reasoning" not in mp.list_presets()
        importlib.reload(mp)
        assert "phi4-reasoning" in mp.list_presets()


@pytest.mark.unit
class TestPhi4PresetDefaults:
    """Sanity-check the built-in PHI4_REASONING preset."""

    def test_phi4_preset_attributes(self):
        preset = mp.PHI4_REASONING
        assert preset.name == "phi4-reasoning"
        assert preset.reasoning is True
        assert "<|im_end|>" in preset.stop_tokens
        assert preset.default_temperature == pytest.approx(0.2)

    def test_phi4_preset_pattern_matches_variants(self):
        # The built-in pattern should be lenient about separators.
        for variant in ("phi-4", "phi_4", "phi 4", "phi4", "Phi-4-Reasoning"):
            assert mp.PHI4_REASONING.matches(variant), variant

    def test_phi4_preset_does_not_match_unrelated(self):
        for variant in ("gpt-4", "phi-3", "claude-3", ""):
            assert not mp.PHI4_REASONING.matches(variant), variant
