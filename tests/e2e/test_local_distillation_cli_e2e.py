"""Real ``python -m scripts.local_distillation`` entry point (not in-process main)."""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.smoke]


def test_sidecar_module_prints_c4_architecture(run_module) -> None:
    result = run_module("scripts.local_distillation", ["sidecar"])
    assert result.ok, result.describe()
    payload = json.loads(result.stdout)
    assert payload["network"]["board_rows"] == 6
    assert payload["network"]["board_cols"] == 7
    assert payload["network"]["action_size"] == 7
    assert payload["recurrent_enabled"] is False


def test_promote_module_rejects_with_exit_one(run_module) -> None:
    result = run_module(
        "scripts.local_distillation",
        ["promote", "--candidate", "0.4", "--incumbent", "0.55"],
    )
    assert result.returncode == 1, result.describe()
    payload = json.loads(result.stdout)
    assert payload["promote"] is False
    assert "rejected" in payload["reason"]


def test_compare_arms_module_is_toy_not_c4_lift(run_module) -> None:
    result = run_module("scripts.local_distillation", ["compare-arms", "--simulations", "2"])
    assert result.ok, result.describe()
    payload = json.loads(result.stdout)
    assert payload["domain"] == "toy_two_ply"
    assert payload["note"] == "not a Connect Four lift"
    assert payload["primary_endpoint"] == "search_minus_no_search"
