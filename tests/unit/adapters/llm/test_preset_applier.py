"""
Tests for ``src.adapters.llm.preset_applier.apply_preset``.

These tests verify that ``apply_preset`` is a pure transformation: it never
mutates its input, never overwrites caller-supplied values, and applies
each preset field according to the documented merge rules.
"""

from __future__ import annotations

import pytest

from src.adapters.llm.model_presets import ModelPreset
from src.adapters.llm.preset_applier import apply_preset


@pytest.mark.unit
class TestApplyPresetPassthrough:
    """When preset is None, apply_preset must return a shallow copy."""

    def test_apply_returns_shallow_copy_when_preset_none(self):
        original = {"model": "any", "messages": [{"role": "user", "content": "hi"}]}
        result = apply_preset(original, None)

        assert result == original
        # New dict, not the same object.
        assert result is not original

    def test_apply_does_not_mutate_input_when_preset_provided(self):
        preset = ModelPreset(name="t", name_pattern=r"t", stop_tokens=("<X>",))
        original = {"messages": [], "stop": ["A"]}
        original_snapshot = {"messages": [], "stop": ["A"]}

        apply_preset(original, preset)

        assert original == original_snapshot


@pytest.mark.unit
class TestApplyPresetStops:
    """Stop-token union behavior."""

    def test_apply_adds_stop_tokens(self):
        preset = ModelPreset(name="t", name_pattern=r"t", stop_tokens=("<|x|>",))
        result = apply_preset({}, preset)
        assert result["stop"] == ["<|x|>"]

    def test_apply_unions_with_existing_stops(self):
        preset = ModelPreset(name="t", name_pattern=r"t", stop_tokens=("B",))
        result = apply_preset({"stop": ["A"]}, preset)
        assert "A" in result["stop"]
        assert "B" in result["stop"]
        # No duplicates.
        assert len(result["stop"]) == 2

    def test_apply_dedupes_overlap_between_payload_and_preset(self):
        preset = ModelPreset(name="t", name_pattern=r"t", stop_tokens=("A", "B"))
        result = apply_preset({"stop": ["A", "C"]}, preset)
        # Order: preset entries first (A, B), then payload entries that
        # weren't already added (C).
        assert result["stop"] == ["A", "B", "C"]

    def test_apply_handles_string_stop_value(self):
        preset = ModelPreset(name="t", name_pattern=r"t", stop_tokens=("END",))
        result = apply_preset({"stop": "STOP"}, preset)
        assert "END" in result["stop"]
        assert "STOP" in result["stop"]

    def test_apply_does_not_set_stop_when_no_preset_stops(self):
        preset = ModelPreset(name="t", name_pattern=r"t")  # no stop_tokens
        result = apply_preset({}, preset)
        assert "stop" not in result


@pytest.mark.unit
class TestApplyPresetTemperature:
    """Temperature merge rules."""

    def test_apply_does_not_overwrite_explicit_user_temperature(self):
        preset = ModelPreset(name="t", name_pattern=r"t", default_temperature=0.2)
        # User passed 0.9 explicitly; preset must NOT override it.
        result = apply_preset({"temperature": 0.9}, preset, user_temperature=0.9)
        assert result["temperature"] == 0.9

    def test_apply_uses_preset_temperature_when_user_none(self):
        preset = ModelPreset(name="t", name_pattern=r"t", default_temperature=0.2)
        # Empty payload, no user temperature.
        result = apply_preset({}, preset, user_temperature=None)
        assert result["temperature"] == pytest.approx(0.2)

    def test_apply_uses_preset_temperature_when_payload_temp_is_none(self):
        preset = ModelPreset(name="t", name_pattern=r"t", default_temperature=0.3)
        # Payload carries an explicit None placeholder (e.g. from the LM
        # Studio client building its base payload from a None default).
        result = apply_preset({"temperature": None}, preset, user_temperature=None)
        assert result["temperature"] == pytest.approx(0.3)

    def test_apply_does_not_set_temperature_when_preset_default_none(self):
        preset = ModelPreset(name="t", name_pattern=r"t", default_temperature=None)
        result = apply_preset({}, preset, user_temperature=None)
        assert "temperature" not in result or result["temperature"] is None

    def test_apply_does_not_overwrite_existing_payload_temperature(self):
        # Payload already has a temperature, but caller didn't surface it
        # through user_temperature. We still must not clobber it.
        preset = ModelPreset(name="t", name_pattern=r"t", default_temperature=0.2)
        result = apply_preset({"temperature": 0.55}, preset, user_temperature=None)
        assert result["temperature"] == pytest.approx(0.55)


@pytest.mark.unit
class TestApplyPresetReasoning:
    """Reasoning forwarding rules."""

    def test_apply_injects_reasoning_when_preset_reasoning_true(self):
        preset = ModelPreset(name="t", name_pattern=r"t", reasoning=True)
        result = apply_preset({}, preset, reasoning_effort="high")
        assert result["reasoning"] == {"effort": "high"}

    def test_apply_does_not_inject_reasoning_when_preset_reasoning_false(self):
        preset = ModelPreset(name="t", name_pattern=r"t", reasoning=False)
        result = apply_preset({}, preset, reasoning_effort="high")
        assert "reasoning" not in result

    def test_apply_does_not_inject_reasoning_when_effort_none(self):
        preset = ModelPreset(name="t", name_pattern=r"t", reasoning=True)
        result = apply_preset({}, preset, reasoning_effort=None)
        assert "reasoning" not in result

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_apply_supports_all_effort_levels(self, effort):
        preset = ModelPreset(name="t", name_pattern=r"t", reasoning=True)
        result = apply_preset({}, preset, reasoning_effort=effort)
        assert result["reasoning"] == {"effort": effort}


@pytest.mark.unit
class TestApplyPresetExtraParams:
    """`extra_params` merge precedence."""

    def test_apply_merges_extra_params_without_overwriting_existing(self):
        preset = ModelPreset(
            name="t",
            name_pattern=r"t",
            extra_params={"top_p": 0.5, "min_p": 0.05},
        )
        # Payload already sets top_p — preset must NOT clobber it.
        result = apply_preset({"top_p": 0.95}, preset)
        assert result["top_p"] == 0.95
        assert result["min_p"] == 0.05

    def test_apply_adds_new_extra_params_when_payload_lacks_them(self):
        preset = ModelPreset(
            name="t",
            name_pattern=r"t",
            extra_params={"foo": "bar", "baz": 42},
        )
        result = apply_preset({}, preset)
        assert result["foo"] == "bar"
        assert result["baz"] == 42


@pytest.mark.unit
class TestApplyPresetFullExample:
    """Combined behavior in a realistic scenario."""

    def test_phi4_like_preset_applies_all_fields(self):
        preset = ModelPreset(
            name="t",
            name_pattern=r"t",
            stop_tokens=("<|im_end|>",),
            reasoning=True,
            default_temperature=0.2,
            extra_params={"top_p": 0.9},
        )
        result = apply_preset(
            {"model": "x", "messages": []},
            preset,
            reasoning_effort="medium",
            user_temperature=None,
        )

        assert result["stop"] == ["<|im_end|>"]
        assert result["temperature"] == pytest.approx(0.2)
        assert result["reasoning"] == {"effort": "medium"}
        assert result["top_p"] == 0.9
        # Untouched.
        assert result["model"] == "x"
        assert result["messages"] == []


# ---------------------------------------------------------------------------
# Logging contract: apply_preset emits DEBUG records describing the merge.
# ---------------------------------------------------------------------------


def test_apply_preset_logs_debug_when_no_preset(caplog: pytest.LogCaptureFixture) -> None:
    import logging as _logging

    with caplog.at_level(_logging.DEBUG, logger="src.adapters.llm.preset_applier"):
        apply_preset({"model": "x"}, None)
    messages = [r.getMessage() for r in caplog.records if r.name == "src.adapters.llm.preset_applier"]
    assert any("no preset supplied" in m for m in messages)


def test_apply_preset_logs_what_was_applied(caplog: pytest.LogCaptureFixture) -> None:
    import logging as _logging

    preset = ModelPreset(
        name="t",
        name_pattern=r"t",
        stop_tokens=("<|end|>",),
        reasoning=True,
        default_temperature=0.3,
    )
    with caplog.at_level(_logging.DEBUG, logger="src.adapters.llm.preset_applier"):
        apply_preset({"model": "t"}, preset, reasoning_effort="medium")
    messages = [r.getMessage() for r in caplog.records if r.name == "src.adapters.llm.preset_applier"]
    text = " ".join(messages)
    assert "preset=t" in text
    assert "stop" in text
    assert "temperature" in text
    assert "reasoning" in text
