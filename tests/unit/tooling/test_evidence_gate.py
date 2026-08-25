"""Tests for the PostToolUse evidence gate (``.claude/hooks/evidence_gate.py``).

The hook lives outside ``src/`` so the Phase 3 plugin extraction can carry it
unchanged, which means it is exercised two ways here:

* **In-process** — its pure functions (``ledger_grades``, ``findings``) are
  loaded by path and driven directly, because that is where the interesting
  edge cases live.
* **As a subprocess** — fed stdin JSON exactly as the hook runner does, to pin
  the contract that stdout carries only a JSON object and that every failure
  mode is fail-open.

The tests are written adversarially: each one tries to get the gate to either
miss an over-claim or fire on legitimate prose, since both failures end with
the hook being ignored.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[3]
HOOK = REPO_ROOT / ".claude" / "hooks" / "evidence_gate.py"

# A minimal ledger covering one row per grade, so a test can pick the grade it
# needs without depending on the live document's row numbering.
LEDGER = """
| Id | Claim | Source | Grade | Verify | Evidence | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| CL-1 | proven thing | README.md | PROVEN | `pytest` | tests/x.py | - |
| CL-2 | partial thing | README.md | PARTIAL | `pytest` | - | missing link |
| CL-3 | unproven thing | README.md | UNPROVEN | - | - | needs a run |
| CL-4 | false thing | README.md | FALSE | - | - | contradicted at a:1 |
"""


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evidence_gate", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hook() -> ModuleType:
    return _load_hook()


@pytest.fixture(autouse=True)
def _no_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVIDENCE_GATE_BYPASS", raising=False)


def _run(payload: dict, project_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={"CLAUDE_PROJECT_DIR": str(project_dir), "PATH": "/usr/bin:/bin"},
    )


def _make_tree(tmp_path: Path, relative: str, body: str, ledger: str = LEDGER) -> Path:
    """A synthetic project dir with one claim-surface document and a ledger."""
    root = tmp_path / "proj"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "CLAIM_LEDGER.md").write_text(ledger, encoding="utf-8")
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return root


# ----------------------------------------------------------------- ledger parsing


def test_ledger_grades_parses_one_row_per_id(hook: ModuleType) -> None:
    assert hook.ledger_grades(LEDGER) == {
        "CL-1": "PROVEN",
        "CL-2": "PARTIAL",
        "CL-3": "UNPROVEN",
        "CL-4": "FALSE",
    }


def test_ledger_grades_ignores_the_header_and_separator(hook: ModuleType) -> None:
    """The header row says 'Grade', not a grade; it must not become an entry."""
    assert "CL-Id" not in hook.ledger_grades(LEDGER)
    assert len(hook.ledger_grades(LEDGER)) == 4


def test_ledger_grades_skips_rows_with_an_unknown_grade(hook: ModuleType) -> None:
    """A typo'd grade is the validator's problem; the hook must not invent one.

    Silently mapping ``PROVEEN`` to ``PROVEN`` would let a typo suppress a real
    finding, so the row is dropped and the citing line is then treated as
    uncited-or-weak rather than satisfied.
    """
    grades = hook.ledger_grades("| CL-9 | x | README.md | PROVEEN | `c` | e | - |")
    assert grades == {}


def test_ledger_grades_tolerates_a_malformed_ledger(hook: ModuleType) -> None:
    assert hook.ledger_grades("not a table at all\n\n| broken |") == {}


# ------------------------------------------------------------------- findings


@pytest.mark.parametrize(
    "phrase",
    [
        "proven",
        "fully validated",
        "empirically validated",
        "production-ready",
        "production ready",
        "battle-tested",
        "state-of-the-art",
        "state of the art",
    ],
)
def test_promotion_phrase_without_a_citation_is_reported(hook: ModuleType, phrase: str) -> None:
    result = hook.findings(f"The engine is {phrase} across domains.\n", hook.ledger_grades(LEDGER))
    assert len(result) == 1
    assert "cites no CL-<n> row" in result[0]


def test_outperforms_without_a_citation_is_reported(hook: ModuleType) -> None:
    result = hook.findings("The engine outperforms the incumbent.\n", hook.ledger_grades(LEDGER))
    assert len(result) == 1


@pytest.mark.parametrize(
    "line",
    [
        "The `X-API-Key` header is validated against the authenticator.",
        "CI-verified on Python 3.11 and 3.12.",
        "Specs are validated by the harness.",
        "The schema is verified at load time.",
        "Timeouts cannot pre-guarantee the result.",
        "Sanitisation guarantees secrets are masked.",
    ],
)
def test_mechanical_uses_are_not_flagged(hook: ModuleType, line: str) -> None:
    """Every string here is real prose from the live tree, or a near variant.

    Bare "validated"/"verified"/"guarantees" describe mechanisms far more often
    than capabilities in this repo; flagging them produced seven false positives
    on README.md, CHARTER.md and docs/STATUS.md and zero true ones, which is how
    an advisory hook earns the right to be ignored.
    """
    assert hook.findings(line + "\n", hook.ledger_grades(LEDGER)) == []


@pytest.mark.parametrize(
    "line",
    [
        'the difference between "built" and "proven" is mechanical',
        "a `proven` grade is derived, never asserted",
        "the “proven” label is reserved",
    ],
)
def test_a_quoted_or_backticked_term_is_the_document_defining_itself(hook: ModuleType, line: str) -> None:
    assert hook.findings(line + "\n", hook.ledger_grades(LEDGER)) == []


def test_a_quoted_term_does_not_mask_a_real_claim_later_on_the_line(hook: ModuleType) -> None:
    """Quote-stripping must not become a laundering channel of its own."""
    result = hook.findings('we define "evidence" and the engine is proven\n', hook.ledger_grades(LEDGER))
    assert len(result) == 1


def test_promotion_word_backed_by_a_proven_row_is_accepted(hook: ModuleType) -> None:
    assert hook.findings("The gate is proven (CL-1).\n", hook.ledger_grades(LEDGER)) == []


@pytest.mark.parametrize("row,grade", [("CL-2", "PARTIAL"), ("CL-3", "UNPROVEN"), ("CL-4", "FALSE")])
def test_promotion_word_backed_by_a_weak_row_is_reported_with_its_grade(hook: ModuleType, row: str, grade: str) -> None:
    result = hook.findings(f"The engine is proven ({row}).\n", hook.ledger_grades(LEDGER))
    assert len(result) == 1
    assert row in result[0] and grade in result[0]


def test_the_weakest_cited_row_decides(hook: ModuleType) -> None:
    """Citing a strong row alongside a weak one must not launder the claim.

    This is the shape a motivated author reaches for first: append a PROVEN row
    id to a sentence whose substance rests on a PARTIAL one.
    """
    result = hook.findings("Proven end to end (CL-1, CL-2).\n", hook.ledger_grades(LEDGER))
    assert len(result) == 1
    assert "PARTIAL" in result[0]


def test_a_citation_to_a_nonexistent_row_does_not_satisfy_the_gate(hook: ModuleType) -> None:
    result = hook.findings("Proven (CL-99).\n", hook.ledger_grades(LEDGER))
    assert len(result) == 1
    assert "UNPROVEN" in result[0], "an unknown row must be treated as unsupported, not as absent"


def test_findings_are_reported_per_line_with_line_numbers(hook: ModuleType) -> None:
    document = "intro\nis proven\nmiddle\nis production-ready\n"
    result = hook.findings(document, hook.ledger_grades(LEDGER))
    assert len(result) == 2
    assert result[0].startswith("line 2:")
    assert result[1].startswith("line 4:")


def test_only_one_finding_per_line(hook: ModuleType) -> None:
    """A sentence with three promotion phrases is one problem, not three."""
    result = hook.findings("It is proven, battle-tested and production-ready.\n", hook.ledger_grades(LEDGER))
    assert len(result) == 1


def test_the_most_specific_phrase_wins(hook: ModuleType) -> None:
    result = hook.findings("It is fully validated.\n", hook.ledger_grades(LEDGER))
    assert "fully validated" in result[0]


def test_proven_matches_only_on_a_word_boundary(hook: ModuleType) -> None:
    """ "Unproven" and "disproven" are the opposite of a promotion claim."""
    assert hook.findings("The capability is unproven.\n", hook.ledger_grades(LEDGER)) == []
    assert hook.findings("The hypothesis was disproven.\n", hook.ledger_grades(LEDGER)) == []


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_fenced_code_is_not_scanned(hook: ModuleType, fence: str) -> None:
    """A shell transcript printing PROVEN is output, not a claim."""
    document = f"intro\n{fence}bash\nclaim-ledger  # prints PROVEN for a proven row\n{fence}\nouttro\n"
    assert hook.findings(document, hook.ledger_grades(LEDGER)) == []


def test_an_unterminated_fence_swallows_the_rest_of_the_file(hook: ModuleType) -> None:
    """Documents an accepted trade-off rather than asserting ideal behaviour.

    With no closing fence the scanner stays inside the block to end of file, so
    a malformed document under-reports. Under-reporting is the safe direction
    for an advisory hook: the alternative is firing on every code sample in a
    file whose fences are merely uneven.
    """
    assert hook.findings("```\nis proven\n", hook.ledger_grades(LEDGER)) == []


def test_clean_prose_produces_nothing(hook: ModuleType) -> None:
    document = "The engine is implemented and unit-tested. Coverage is 89%.\n"
    assert hook.findings(document, hook.ledger_grades(LEDGER)) == []


def test_matching_is_case_insensitive(hook: ModuleType) -> None:
    assert len(hook.findings("PROVEN across the board.\n", hook.ledger_grades(LEDGER))) == 1


def test_the_surface_list_and_vocabulary_are_data_not_branches(hook: ModuleType) -> None:
    """Adding a surface or a phrase must stay a one-line, reviewable change."""
    assert hook._CLAIM_SURFACES == ("README.md", "CHARTER.md", "docs/STATUS.md")
    required = {grade for _, grade in hook._PROMOTION_WORDS}
    assert required == {"PROVEN"}, "a phrase requiring less than PROVEN needs its own justification"


def test_findings_on_an_empty_document(hook: ModuleType) -> None:
    assert hook.findings("", hook.ledger_grades(LEDGER)) == []


# --------------------------------------------------------------- process contract


def test_hook_warns_on_a_claim_surface(tmp_path: Path) -> None:
    root = _make_tree(tmp_path, "README.md", "The engine is proven.\n")
    proc = _run({"tool_input": {"file_path": str(root / "README.md")}}, root)
    assert proc.returncode == 0
    emitted = json.loads(proc.stdout)
    output = emitted["hookSpecificOutput"]
    assert output["hookEventName"] == "PostToolUse"
    assert "README.md" in output["additionalContext"]
    assert "validate-claims" in output["additionalContext"], "the message must route to the fix"


@pytest.mark.parametrize("relative", ["README.md", "CHARTER.md", "docs/STATUS.md"])
def test_every_claim_surface_is_scanned(tmp_path: Path, relative: str) -> None:
    root = _make_tree(tmp_path, relative, "The engine is proven.\n")
    proc = _run({"tool_input": {"file_path": str(root / relative)}}, root)
    assert proc.stdout, f"{relative} is a claim surface and must be scanned"


def test_the_ledger_itself_is_exempt(tmp_path: Path) -> None:
    """The ledger grades claims; scanning it would fire on every row."""
    relative = "docs/CLAIM_LEDGER.md"
    root = _make_tree(tmp_path, relative, "CL-1 is proven.\n")
    proc = _run({"tool_input": {"file_path": str(root / relative)}}, root)
    assert proc.stdout == ""


@pytest.mark.parametrize(
    "relative",
    [
        "src/module.py",
        "tests/unit/test_x.py",
        "notes.md",
        "CHANGELOG.md",
        "docs/archive/reports/old_report.md",
        "docs/training/MODULE_5.md",
        "docs/plans/OLD_PLAN.md",
    ],
)
def test_non_claim_surfaces_are_ignored(tmp_path: Path, relative: str) -> None:
    root = _make_tree(tmp_path, relative, "# proven, production-ready, battle-tested\n")
    proc = _run({"tool_input": {"file_path": str(root / relative)}}, root)
    assert proc.stdout == "", (
        f"{relative} is a historical or non-claim document; flagging it would ask an author to "
        f"rewrite a record of what was true when written"
    )


def test_clean_document_emits_nothing(tmp_path: Path) -> None:
    root = _make_tree(tmp_path, "README.md", "The engine is implemented and unit-tested.\n")
    proc = _run({"tool_input": {"file_path": str(root / "README.md")}}, root)
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_bypass_env_var_silences_the_gate(tmp_path: Path) -> None:
    root = _make_tree(tmp_path, "README.md", "The engine is proven.\n")
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"tool_input": {"file_path": str(root / "README.md")}}),
        capture_output=True,
        text=True,
        env={"CLAUDE_PROJECT_DIR": str(root), "PATH": "/usr/bin:/bin", "EVIDENCE_GATE_BYPASS": "1"},
    )
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_missing_ledger_fails_open(tmp_path: Path) -> None:
    root = _make_tree(tmp_path, "README.md", "The engine is proven.\n")
    (root / "docs" / "CLAIM_LEDGER.md").unlink()
    proc = _run({"tool_input": {"file_path": str(root / "README.md")}}, root)
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_payload_without_a_file_path_fails_open(tmp_path: Path) -> None:
    root = _make_tree(tmp_path, "README.md", "The engine is proven.\n")
    for payload in ({}, {"tool_input": {}}, {"tool_input": {"file_path": ""}}, {"tool_input": {"file_path": 7}}):
        proc = _run(payload, root)
        assert proc.returncode == 0, payload
        assert proc.stdout == "", payload


def test_path_outside_the_project_is_ignored(tmp_path: Path) -> None:
    root = _make_tree(tmp_path, "README.md", "x\n")
    outside = tmp_path / "elsewhere" / "README.md"
    outside.parent.mkdir(parents=True)
    outside.write_text("The engine is proven.\n", encoding="utf-8")
    proc = _run({"tool_input": {"file_path": str(outside)}}, root)
    assert proc.stdout == ""


def test_malformed_stdin_is_non_blocking(tmp_path: Path) -> None:
    """Fail-open by contract: exit non-zero, emit no stdout, explain on stderr."""
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input="{not json",
        capture_output=True,
        text=True,
        env={"CLAUDE_PROJECT_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "fail-open" in proc.stderr


def test_stdout_is_only_the_json_object(tmp_path: Path) -> None:
    """The hook runner parses stdout; a stray print would corrupt the session."""
    root = _make_tree(tmp_path, "README.md", "The engine is proven.\n")
    proc = _run({"tool_input": {"file_path": str(root / "README.md")}}, root)
    json.loads(proc.stdout)  # raises if anything else was written
    assert proc.stdout.startswith("{") and proc.stdout.rstrip().endswith("}")


# -------------------------------------------------------------------- wiring


def test_the_hook_is_registered_in_settings() -> None:
    """An unregistered hook is decoration. Pins the event and the matcher."""
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    entries = settings["hooks"]["PostToolUse"]
    commands = [hook["command"] for entry in entries for hook in entry["hooks"]]
    assert any("evidence_gate.py" in command for command in commands)
    matchers = [entry["matcher"] for entry in entries]
    assert any("Write" in matcher and "Edit" in matcher for matcher in matchers)


@pytest.mark.parametrize("relative", ["README.md", "CHARTER.md", "docs/STATUS.md"])
def test_the_live_claim_surfaces_are_clean(relative: str) -> None:
    """Regression against the real documents: the gate must start at zero.

    A hook that ships with a standing backlog of findings trains its reader to
    dismiss it. If this fails, either the new prose over-claims or the gate has
    a false positive; both are bugs, and neither may be waived by widening the
    exempt list.
    """
    module = _load_hook()
    grades = module.ledger_grades((REPO_ROOT / "docs" / "CLAIM_LEDGER.md").read_text(encoding="utf-8"))
    document = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert module.findings(document, grades) == []


def test_the_live_ledger_parses_and_is_fully_graded() -> None:
    """Regression against the real document, not a fixture.

    If the live ledger stops parsing, the hook degrades to silence — the exact
    failure mode that makes a check worse than none.
    """
    module = _load_hook()
    text = (REPO_ROOT / "docs" / "CLAIM_LEDGER.md").read_text(encoding="utf-8")
    grades = module.ledger_grades(text)
    row_ids = {line.split("|")[1].strip() for line in text.splitlines() if line.startswith("| CL-")}
    assert grades, "the live ledger must parse"
    assert set(grades) == row_ids, "every live ledger row must carry a recognised grade"
