"""Unit tests for ``demos.producer_reviewer_phi4_demo``.

These tests exercise the CLI-shaped helpers (parser construction, output
formatting, top-level error paths) without ever touching a real LLM.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from demos import producer_reviewer_phi4_demo as demo
from src.framework.harness.settings import HarnessSettings, reset_harness_settings
from src.framework.harness.topology.base import AgentOutcome

# ─────────────────────────────── fixtures ────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop any cached harness/global settings between tests."""
    # Make sure the test only sees env vars it sets explicitly.
    monkeypatch.delenv("HARNESS_BENCHMARK_TASK_ID", raising=False)
    reset_harness_settings()
    yield  # type: ignore[misc]
    reset_harness_settings()


# ──────────────────────────── parser tests ───────────────────────────────


def test_build_parser_accepts_all_flags() -> None:
    """Every documented flag should be parseable."""
    parser = demo.build_parser()
    args = parser.parse_args(
        [
            "--task",
            "A1",
            "--rounds",
            "2",
            "--producer-max-tokens",
            "256",
            "--reviewer-max-tokens",
            "128",
            "--temperature",
            "0.3",
            "--json",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.task == "A1"
    assert args.rounds == 2
    assert args.producer_max_tokens == 256
    assert args.reviewer_max_tokens == 128
    assert args.temperature == pytest.approx(0.3)
    assert args.json_only is True
    assert args.log_level == "DEBUG"


def test_build_parser_task_default_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When --task is omitted, the env var becomes the default."""
    monkeypatch.setenv("HARNESS_BENCHMARK_TASK_ID", "A2")
    reset_harness_settings()
    parser = demo.build_parser()
    args = parser.parse_args([])
    assert args.task == "A2"


def test_build_parser_rounds_default_from_settings() -> None:
    """``--rounds`` defaults to ``HarnessSettings.PRODUCER_REVIEWER_ROUNDS``."""
    parser = demo.build_parser()
    args = parser.parse_args([])
    assert args.rounds == HarnessSettings().PRODUCER_REVIEWER_ROUNDS


def test_build_parser_max_tokens_defaults_from_settings() -> None:
    """Max-token defaults trace back to the harness settings."""
    parser = demo.build_parser()
    args = parser.parse_args([])
    hs = HarnessSettings()
    assert args.producer_max_tokens == hs.PRODUCER_MAX_TOKENS
    assert args.reviewer_max_tokens == hs.REVIEWER_MAX_TOKENS


# ──────────────────────────── formatting ─────────────────────────────────


def _make_outcome(
    *,
    success: bool = True,
    response: str = "ok",
    metadata: dict[str, Any] | None = None,
) -> AgentOutcome:
    return AgentOutcome(
        agent_name="producer",
        response=response,
        confidence=0.42,
        success=success,
        error=None if success else "boom",
        metadata=metadata or {},
    )


def test_format_outcome_with_minimal_outcome() -> None:
    """A minimal outcome formats without raising and includes the response."""
    out = _make_outcome()
    rendered = demo.format_outcome(out)
    assert "SUCCESS" in rendered
    assert "producer" in rendered
    assert "ok" in rendered


def test_format_outcome_handles_failure_outcome() -> None:
    """A failure outcome is rendered with the FAILURE marker and error string."""
    out = _make_outcome(success=False, response="")
    rendered = demo.format_outcome(out)
    assert "FAILURE" in rendered
    assert "boom" in rendered


def test_format_outcome_json_only_returns_pure_json() -> None:
    """``json_only=True`` returns valid JSON parseable as a dict."""
    out = _make_outcome()
    rendered = demo.format_outcome(out, json_only=True)
    parsed = json.loads(rendered)
    assert parsed["success"] is True
    assert parsed["agent_name"] == "producer"
    assert parsed["response"] == "ok"


# ─────────────────────────── main() error paths ──────────────────────────


def test_main_handles_keyerror_from_unknown_task_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown task id should yield exit code 2 (usage error)."""

    class _FakeLLM:
        async def generate(self, **_: Any) -> Any:  # pragma: no cover - never called
            raise AssertionError("LLM must not be called for unknown task ids")

    def _fake_create_llm(self: Any) -> Any:
        return _FakeLLM()

    monkeypatch.setattr(
        "src.framework.harness.factories.HarnessFactory.create_llm",
        _fake_create_llm,
    )

    rc = demo.main(["--task", "ZZ-not-a-real-task"])
    captured = capsys.readouterr()
    assert rc == demo.EXIT_USAGE_ERROR
    assert "unknown benchmark task id" in captured.err.lower()


def test_main_handles_llm_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When ``create_llm`` raises, main reports the error and exits 1."""

    def _boom(self: Any) -> Any:
        raise RuntimeError("llm config invalid")

    monkeypatch.setattr(
        "src.framework.harness.factories.HarnessFactory.create_llm",
        _boom,
    )
    rc = demo.main(["--task", "A1"])
    captured = capsys.readouterr()
    assert rc == demo.EXIT_PIPELINE_FAILURE
    assert "failed to construct llm client" in captured.err.lower()
