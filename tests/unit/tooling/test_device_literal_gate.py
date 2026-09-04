"""Invariants for the device-literal PreToolUse gate.

The gate exists because ``tests/README.md``'s rule — *never write a device literal in a test,
take the fixture* — was violated inside the very change that introduced it. A literal
``device="cpu"`` makes a test pass identically on every host while proving nothing about the
accelerator path, which is the "green but not checked" failure the e2e suite was built to
remove.

A gate is only worth having if it fires on the real thing and stays quiet on everything else,
so both halves are asserted here. The false-positive cases matter as much as the true ones:
a hook that flags skip-reason prose or ``src/`` availability probes gets ignored within a day,
and an ignored gate costs attention while enforcing nothing.

Driven through the module's pure functions and through the script's stdin/stdout contract, so
the JSON shape the harness actually consumes is covered too.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
HOOK = REPO_ROOT / ".claude" / "hooks" / "device_literal_gate.py"
SETTINGS = REPO_ROOT / ".claude" / "settings.json"


def _load_module():
    """Import the hook by path; it lives outside any package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("device_literal_gate", HOOK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_module()


# ------------------------------------------------------------------ what it catches


@pytest.mark.parametrize(
    "line",
    [
        'agent = create_hrm_agent(cfg, device="cpu")',
        "engine = NeuralMCTS(net, cfg, device='cuda')",
        'x = torch.randn(2, device = "mps")',
        'trainer = SelfPlayTrainer(net, device="cuda:1")',
        'DEVICE="cpu"',  # case-insensitive on the key
    ],
)
def test_a_device_assignment_is_caught(line: str) -> None:
    assert gate.offending_lines(line) == [line.strip()]


# --------------------------------------------------------------- what it must ignore


@pytest.mark.parametrize(
    "line",
    [
        # Skip reasons and messages name devices as prose, not as a selection. The e2e suite
        # is full of these by design — every skip must name its device.
        'reason = "cuda is not available on this host"',
        'pytest.skip(f"{name} unavailable")',
        'assert case.name == "cuda"',  # a comparison, not an assignment
        'ids = ["cpu", "cuda", "mps"]',  # the matrix itself
        "device = resolve_device(settings.TORCH_DEVICE_OVERRIDE)",  # the correct pattern
        "device=device_case.name",  # taking the fixture
        'log.info("running on %s", "cuda")',
        "",
    ],
)
def test_prose_and_correct_patterns_are_not_flagged(line: str) -> None:
    assert gate.offending_lines(line) == []


def test_the_written_exception_silences_a_line() -> None:
    """A genuine need has an escape hatch, mirroring the `No-Spec:` trailer convention."""
    line = 'net = build(device="cpu")  # device-literal: the gate always loads on CPU by contract'
    assert gate.offending_lines(line) == []


# ------------------------------------------------------------------------- scoping


@pytest.mark.parametrize(
    ("path", "gated"),
    [
        ("tests/e2e/test_thing_e2e.py", True),
        ("/abs/repo/tests/e2e/nested/test_thing.py", True),
        # src/ holds ~40 legitimate device literals (availability ladders, field defaults).
        # Gating it would be almost all false positives.
        ("src/framework/mcts/neural_mcts.py", False),
        ("tests/unit/test_component_factory.py", False),
        # The matrix helpers are where availability is probed; they must name devices.
        ("tests/utils/device_matrix.py", False),
        ("tests/utils/e2e_process.py", False),
    ],
)
def test_only_the_e2e_tree_is_gated(path: str, gated: bool) -> None:
    assert gate.is_gated(path) is gated


# --------------------------------------------------------- the stdin/stdout contract


def _run(payload: dict, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin"}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_a_violation_warns_by_default() -> None:
    proc = _run({"tool_input": {"file_path": "tests/e2e/test_x.py", "content": 'f(device="cpu")'}})
    assert proc.returncode == 0
    emitted = json.loads(proc.stdout)["hookSpecificOutput"]
    assert emitted["hookEventName"] == "PreToolUse"
    assert "additionalContext" in emitted, "warn mode must advise, never deny"
    assert "device-literal" in emitted["additionalContext"]


def test_block_mode_denies() -> None:
    proc = _run(
        {"tool_input": {"file_path": "tests/e2e/test_x.py", "content": 'f(device="cpu")'}},
        {"DEVICE_LITERAL_GATE_MODE": "block"},
    )
    emitted = json.loads(proc.stdout)["hookSpecificOutput"]
    assert emitted["permissionDecision"] == "deny"


def test_bypass_silences_it() -> None:
    proc = _run(
        {"tool_input": {"file_path": "tests/e2e/test_x.py", "content": 'f(device="cpu")'}},
        {"DEVICE_LITERAL_GATE_BYPASS": "1"},
    )
    assert proc.returncode == 0 and proc.stdout == ""


def test_an_edit_payload_is_inspected_not_just_a_write() -> None:
    """Edit and MultiEdit carry the new text under different keys than Write."""
    proc = _run({"tool_input": {"file_path": "tests/e2e/test_x.py", "new_string": 'f(device="mps")'}})
    assert "device-literal" in json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]

    multi = _run({"tool_input": {"file_path": "tests/e2e/test_x.py", "edits": [{"new_string": 'f(device="cuda")'}]}})
    assert "device-literal" in json.loads(multi.stdout)["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"tool_input": {}}, {"tool_input": {"file_path": ""}}, {"tool_input": {"file_path": 7}}],
)
def test_it_fails_open_on_a_malformed_payload(payload: dict) -> None:
    """A broken hook must never wedge an edit."""
    proc = _run(payload)
    assert proc.returncode == 0 and proc.stdout == ""


def test_malformed_stdin_fails_open() -> None:
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input="not json",
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert proc.returncode == 0 and proc.stdout == ""


# --------------------------------------------------------------------- registration


def test_the_hook_is_registered_in_settings() -> None:
    """An unwired hook file is a file, not a gate."""
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    commands = [hook.get("command", "") for entry in settings["hooks"]["PreToolUse"] for hook in entry.get("hooks", [])]
    assert any(
        "device_literal_gate.py" in command for command in commands
    ), "device_literal_gate.py exists but is not registered in .claude/settings.json"


def test_the_e2e_tree_currently_has_no_device_literals() -> None:
    """The rule holds today, so the gate starts from a clean tree rather than a warning storm."""
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "tests" / "e2e").rglob("*.py")):
        for line in gate.offending_lines(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {line}")
    assert not offenders, "device literals in tests/e2e (take the `device` fixture instead):\n" + "\n".join(offenders)
