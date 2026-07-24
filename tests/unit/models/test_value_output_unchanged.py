"""Guards that coarse_dynamics did not disturb ValueNetwork/ValueOutput (AC-3).

value_network.py hard-imports torch, so this test skips when torch is absent (it runs in CI's
neural job). The coarse-dynamics MDN is a standalone module; ValueOutput must be untouched.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

pytest.importorskip("torch")

from src.models.value_network import ValueOutput  # noqa: E402  (after importorskip)


def test_value_output_fields_and_order_unchanged():  # AC-3
    assert [f.name for f in fields(ValueOutput)] == ["value", "features", "uncertainty"]


def test_value_output_optional_fields_default_none():  # AC-3
    field_defaults = {f.name: f.default for f in fields(ValueOutput)}
    assert field_defaults["features"] is None
    assert field_defaults["uncertainty"] is None
