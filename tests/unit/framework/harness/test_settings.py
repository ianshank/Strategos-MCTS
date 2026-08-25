"""Unit tests for ``HarnessSettings``.

Verifies env-var round-trip, validators, helpers, and isolation between
settings instances. Each test uses ``monkeypatch`` to scope env mutations and
``reset_harness_settings`` to clear the lazy cache, so tests don't leak state
into one another.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from src.framework.harness import (
    AggregationPolicy,
    HarnessSettings,
    TopologyName,
    get_harness_settings,
    reset_harness_settings,
)
from src.framework.harness.settings import HarnessPermissions

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_harness_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any pre-existing ``HARNESS_*`` env vars to keep tests deterministic."""
    for key in list(os_environ_keys()):
        if key.startswith("HARNESS_"):
            monkeypatch.delenv(key, raising=False)
    reset_harness_settings()
    yield
    reset_harness_settings()


def os_environ_keys() -> list[str]:
    """Lazy import shim so ``monkeypatch`` can mutate ``os.environ`` cleanly."""
    import os

    return list(os.environ.keys())


def test_defaults_are_sensible() -> None:
    """Construct without env overrides — defaults should validate."""
    s = HarnessSettings()
    assert s.ENABLED is True
    assert s.MAX_ITERATIONS >= 1
    assert s.ITERATION_TIMEOUT_SECONDS <= s.TOTAL_BUDGET_SECONDS
    assert s.TOPOLOGY is TopologyName.PRODUCER_REVIEWER
    assert s.AGGREGATION_POLICY is AggregationPolicy.VERIFIER_RANKED
    assert isinstance(s.MEMORY_ROOT, Path)
    assert s.episodic_dir() == s.MEMORY_ROOT / s.MEMORY_EVENT_LOG_DIR
    assert s.index_path() == s.MEMORY_ROOT / s.MEMORY_INDEX_FILENAME


@pytest.mark.parametrize(
    ("env_key", "env_val", "attr", "expected"),
    [
        ("HARNESS_MAX_ITERATIONS", "42", "MAX_ITERATIONS", 42),
        ("HARNESS_TOPOLOGY", "fan_out_in", "TOPOLOGY", TopologyName.FAN_OUT_IN),
        ("HARNESS_AGGREGATION_POLICY", "first_success", "AGGREGATION_POLICY", AggregationPolicy.FIRST_SUCCESS),
        ("HARNESS_PLANNER_ENABLED", "false", "PLANNER_ENABLED", False),
        ("HARNESS_RALPH_COMPLETION_MARKER", "<!-- DONE -->", "RALPH_COMPLETION_MARKER", "<!-- DONE -->"),
        ("HARNESS_TOOL_OUTPUT_HEAD_CHARS", "500", "TOOL_OUTPUT_HEAD_CHARS", 500),
    ],
)
def test_env_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str,
    env_val: str,
    attr: str,
    expected: object,
) -> None:
    """Every public field must round-trip from a ``HARNESS_*`` env var."""
    monkeypatch.setenv(env_key, env_val)
    s = HarnessSettings()
    assert getattr(s, attr) == expected


def test_path_coercion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """String env vars for path fields must coerce to ``Path``."""
    monkeypatch.setenv("HARNESS_MEMORY_ROOT", str(tmp_path / "mem"))
    monkeypatch.setenv("HARNESS_OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("HARNESS_TOOL_OUTPUT_SPILLOVER_DIR", str(tmp_path / "spill"))
    s = HarnessSettings()
    assert tmp_path / "mem" == s.MEMORY_ROOT
    assert tmp_path / "runs" == s.OUTPUT_DIR
    assert tmp_path / "spill" == s.TOOL_OUTPUT_SPILLOVER_DIR


def test_optional_path_blank_means_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty string for optional paths must coerce to ``None``."""
    monkeypatch.setenv("HARNESS_RECORD_DIR", "")
    monkeypatch.setenv("HARNESS_REPLAY_DIR", "")
    s = HarnessSettings()
    assert s.RECORD_DIR is None
    assert s.REPLAY_DIR is None


def test_iteration_timeout_must_not_exceed_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-field validator rejects nonsensical budget combinations."""
    monkeypatch.setenv("HARNESS_TOTAL_BUDGET_SECONDS", "10")
    monkeypatch.setenv("HARNESS_ITERATION_TIMEOUT_SECONDS", "60")
    with pytest.raises(ValidationError):
        HarnessSettings()


def test_record_and_replay_dirs_must_differ(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Recording and replaying into the same directory is a configuration bug."""
    same = str(tmp_path / "shared")
    monkeypatch.setenv("HARNESS_RECORD_DIR", same)
    monkeypatch.setenv("HARNESS_REPLAY_DIR", same)
    with pytest.raises(ValidationError):
        HarnessSettings()


def test_invalid_topology_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown topology names should fail validation."""
    monkeypatch.setenv("HARNESS_TOPOLOGY", "not_a_real_topology")
    with pytest.raises(ValidationError):
        HarnessSettings()


def test_get_harness_settings_caches() -> None:
    """``get_harness_settings`` should return the same instance until reset."""
    s1 = get_harness_settings()
    s2 = get_harness_settings()
    assert s1 is s2
    reset_harness_settings()
    s3 = get_harness_settings()
    assert s3 is not s1


def test_safe_dict_is_serializable() -> None:
    """``safe_dict`` should be JSON-serialisable for structured logging."""
    import json

    s = HarnessSettings()
    payload = s.safe_dict()
    assert json.dumps(payload)  # would raise if not serialisable


def test_permissions_independent_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permissions are loaded from ``HARNESS_PERM_*`` and don't leak from ``HARNESS_*``."""
    monkeypatch.setenv("HARNESS_PERM_SHELL", "true")
    monkeypatch.setenv("HARNESS_PERM_NETWORK", "false")
    perms = HarnessPermissions()
    assert perms.SHELL is True
    assert perms.NETWORK is False
    assert perms.READ is True  # default preserved
