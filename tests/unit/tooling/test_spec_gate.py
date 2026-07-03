"""Tests for the PreToolUse spec gate (``.claude/hooks/spec_gate.py``).

The hook is deliberately outside ``src/`` (self-contained for Phase 3 plugin
extraction), so it is exercised as a subprocess fed stdin JSON, plus
in-process parity tests that pin its standalone frontmatter reader and id
grammar to the harness implementations.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[3]
HOOK = REPO_ROOT / ".claude" / "hooks" / "spec_gate.py"


def _load_hook_module():
    spec = importlib.util.spec_from_file_location("spec_gate", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("SPEC_GATE_BYPASS", raising=False)
    monkeypatch.delenv("SPEC_GATE_MODE", raising=False)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _make_repo(tmp_path: Path, spec_status: str | None = "approved", branch: str = "spec/demo_gate_spec") -> Path:
    repo = tmp_path / "repo"
    (repo / "specs").mkdir(parents=True)
    (repo / "src").mkdir()
    if spec_status is not None:
        (repo / "specs" / "demo_gate_spec.SPEC.md").write_text(
            f"---\nid: demo_gate_spec\ngoal: g\nmodule: src/\nstatus: {spec_status}\n---\n\n"
            "# Goal\ng\n\n# Acceptance Criteria\n- AC-1: works\n",
            encoding="utf-8",
        )
    (repo / "src" / "x.py").write_text("X = 1\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    if branch != "main":
        _git(repo, "switch", "-q", "-c", branch)
    return repo


def _run_gate(
    payload: dict | str,
    repo: Path | None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if repo is not None:
        env["CLAUDE_PROJECT_DIR"] = str(repo)
    env.update(extra_env or {})
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run([sys.executable, str(HOOK)], input=stdin, capture_output=True, text=True, env=env)


def _payload(repo: Path, rel_path: str, tool: str = "Write", path_key: str = "file_path") -> dict:
    return {
        "session_id": "t",
        "cwd": str(repo),
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": {path_key: str(repo / rel_path)},
    }


def _context(proc: subprocess.CompletedProcess[str]) -> str:
    output = json.loads(proc.stdout)
    return output["hookSpecificOutput"].get("additionalContext", "")


def test_non_src_path_passes_silently(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, branch="main")
    proc = _run_gate(_payload(repo, "docs/x.md"), repo)
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_src_write_on_non_spec_branch_warns(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, branch="main")
    proc = _run_gate(_payload(repo, "src/x.py"), repo)
    assert proc.returncode == 0
    assert "/spec-implement" in _context(proc)
    assert "SPEC_GATE_BYPASS" in _context(proc)


def test_block_mode_denies(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, branch="main")
    proc = _run_gate(_payload(repo, "src/x.py"), repo, {"SPEC_GATE_MODE": "block"})
    output = json.loads(proc.stdout)["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "spec" in output["permissionDecisionReason"]


@pytest.mark.parametrize("status", ["approved", "implemented"])
def test_spec_branch_with_gated_status_passes(tmp_path: Path, status: str) -> None:
    repo = _make_repo(tmp_path, spec_status=status)
    proc = _run_gate(_payload(repo, "src/x.py"), repo)
    assert proc.returncode == 0
    assert proc.stdout == ""


@pytest.mark.parametrize("status", ["draft", "verified", None])
def test_spec_branch_without_gated_status_warns(tmp_path: Path, status: str | None) -> None:
    repo = _make_repo(tmp_path, spec_status=status)
    proc = _run_gate(_payload(repo, "src/x.py"), repo)
    assert proc.returncode == 0
    assert "additionalContext" in proc.stdout


def test_bypass_env_passes_silently(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, branch="main")
    proc = _run_gate(_payload(repo, "src/x.py"), repo, {"SPEC_GATE_BYPASS": "1"})
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_malformed_stdin_fails_open(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, branch="main")
    proc = _run_gate("this is not json", repo)
    assert proc.returncode == 1  # non-blocking error: the tool proceeds
    assert "fail-open" in proc.stderr


def test_notebook_edit_notebook_path_is_gated(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, branch="main")
    proc = _run_gate(_payload(repo, "src/nb.ipynb", tool="NotebookEdit", path_key="notebook_path"), repo)
    assert "additionalContext" in proc.stdout


def test_write_into_new_src_subdir_still_warns(tmp_path: Path) -> None:
    """Nonexistent dirname (scaffolding writes) must anchor at the nearest ancestor, not fail open."""
    repo = _make_repo(tmp_path, branch="main")
    proc = _run_gate(_payload(repo, "src/new_pkg/deep/new_file.py"), repo)
    assert "additionalContext" in proc.stdout


def test_unborn_head_repo_passes_silently(tmp_path: Path) -> None:
    repo = tmp_path / "unborn"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True)
    proc = _run_gate(_payload(repo, "src/x.py"), repo)
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_outside_any_repo_passes_silently(tmp_path: Path) -> None:
    outside = tmp_path / "not_a_repo"
    (outside / "src").mkdir(parents=True)
    payload = _payload(outside, "src/x.py")
    proc = _run_gate(payload, outside)
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_warn_stdout_is_pure_json(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, branch="main")
    proc = _run_gate(_payload(repo, "src/x.py"), repo)
    json.loads(proc.stdout)  # raises if anything but the JSON object was printed


# ---------------------------------------------------------------------------
# Parity pins: the self-contained hook must track the harness implementations
# ---------------------------------------------------------------------------


def test_spec_id_pattern_literal_parity() -> None:
    from src.framework.harness.intent.spec_scaffold import SPEC_ID_PATTERN

    gate = _load_hook_module()
    assert gate._SPEC_BRANCH_RE.pattern == rf"^spec/({SPEC_ID_PATTERN})$"


@pytest.mark.parametrize(
    "text",
    [
        "---\nstatus: approved\n---\n# Goal\ng\n",
        "---\nstatus: approved\r\n---\r\n# Goal\ng\n",  # CRLF: no \n---\n match
        "﻿---\nstatus: approved\n---\n# Goal\ng\n",  # BOM breaks the leading ---\n
        "---\n# a comment\nstatus: approved\n---\nbody\n",
        '---\nstatus: "approved"\n---\nbody\n',  # values are not dequoted
        "---\nstatus: draft\nstatus: approved\n---\nbody\n",  # last occurrence wins
        "---\nstatus: approved\n---",  # closing --- at EOF without trailing newline
        "---\nstatus: APPROVED\n---\nbody\n",  # case preserved (validator is case-sensitive)
        "no frontmatter at all\n",
        "---\nunterminated frontmatter\n",
    ],
)
def test_frontmatter_reader_parity_with_spec_loader(text: str) -> None:
    """The hook's standalone reader must return exactly SpecLoader's status."""
    from src.framework.harness.intent import SpecLoader

    gate = _load_hook_module()
    assert gate.read_frontmatter_status(text) == SpecLoader().parse(text).status


def test_tooling_fixtures_never_pair_real_spec_ids_with_ac_tokens() -> None:
    """Guards spec-trace rule (d): tooling/harness test fixtures must not create
    accidental same-line spec-id + AC-n mappings for real specs."""
    real_ids = [p.name.removesuffix(".SPEC.md") for p in (REPO_ROOT / "specs").glob("*.SPEC.md")]
    ac_token = re.compile(r"\bAC-\d+\b")
    guarded_dirs = [
        REPO_ROOT / "tests" / "unit" / "tooling",
        REPO_ROOT / "tests" / "unit" / "framework" / "harness",
    ]
    offenders = []
    for guarded in guarded_dirs:
        for test_file in guarded.rglob("*.py"):
            for line_no, line in enumerate(test_file.read_text(encoding="utf-8").splitlines(), start=1):
                if ac_token.search(line) and any(real_id in line for real_id in real_ids):
                    offenders.append(f"{test_file}:{line_no}")
    assert offenders == [], f"real spec id + AC-n on one line would satisfy spec-trace rule (d): {offenders}"
