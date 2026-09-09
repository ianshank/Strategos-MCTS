"""CLI smoke for python -m scripts.local_distillation (not a Connect Four lift)."""

from __future__ import annotations

import json

import pytest

from scripts.local_distillation.__main__ import main

pytestmark = [pytest.mark.unit]


def test_cli_sidecar_prints_c4_network(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sidecar"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["network"]["board_rows"] == 6
    assert payload["network"]["board_cols"] == 7
    assert payload["network"]["action_size"] == 7
    assert payload["recurrent_enabled"] is False
    assert payload["primary_endpoint"] == "search_minus_no_search"
    assert payload["committed_results_path"] == "benchmarks/results/local_distillation_c4.json"


def test_cli_promote_accept(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["promote", "--candidate", "0.7", "--incumbent", "0.5", "--min-delta", "0.05"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["promote"] is True
    assert "accepted" in payload["reason"]


def test_cli_promote_reject_exits_one(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["promote", "--candidate", "0.4", "--incumbent", "0.55"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["promote"] is False
    assert "rejected" in payload["reason"]


def test_cli_compare_arms_is_toy_not_c4_lift(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare-arms", "--simulations", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["domain"] == "toy_two_ply"
    assert payload["note"] == "not a Connect Four lift"
    assert payload["primary_endpoint"] == "search_minus_no_search"
    assert payload["search_expansions"] == 2
    assert payload["no_search_expansions"] == 0


def test_cli_promote_missing_required_args() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["promote"])
    assert exc.value.code != 0


def test_cli_unknown_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["not-a-command"])
    assert exc.value.code != 0


def test_cli_compare_arms_rejects_non_int_simulations() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compare-arms", "--simulations", "nope"])
    assert exc.value.code != 0
