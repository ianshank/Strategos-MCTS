"""Tests for the capability claim ledger validator.

Spec: ``specs/evidence_claim_ledger.SPEC.md`` (AC-1, AC-2, AC-10).

The point of this module is adversarial: most tests here *falsify* a ledger and assert the validator
rejects it. A gate that has only ever been shown to pass is an assumption. The promotion rule
(``PROVEN`` requires resolvable evidence) is the property the whole evidence chain rests on, so it is
tested by construction — a hand-edited ``PROVEN`` row with no artefact must fail, and no flag may
make it pass.

The live ledger is also validated, so the tree cannot drift away from its own rules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.config.constants import (
    CLAIM_CELL_EMPTY_SENTINEL,
    CLAIM_GRADE_FALSE,
    CLAIM_GRADE_PARTIAL,
    CLAIM_GRADE_PROVEN,
    CLAIM_GRADE_UNPROVEN,
    CLAIM_GRADES,
    CLAIM_LEDGER_COLUMNS,
    CLAIM_LEDGER_RELATIVE_PATH,
    CLAIM_SURFACE_BASELINE_RELATIVE_PATH,
    CLAIM_SURFACE_BASELINE_SCHEMA_VERSION,
    CLAIM_SURFACE_KEY_BULLETS,
    CLAIM_SURFACE_KEY_PATH,
    CLAIM_SURFACE_KEY_ROWS,
    CLAIM_SURFACE_KEY_SECTION,
    CLAIM_SURFACE_KEY_SURFACES,
    CLAIM_SURFACE_KEY_SURPLUS,
)
from src.tools.claim_ledger import (
    ClaimLedgerValidator,
    ClaimRow,
    LedgerFailure,
    SurfaceCoverage,
    main,
    validate,
)
from tests.fixtures.claim_ledger_fixtures import write_claim_surface_baseline as _surface_baseline

REPO_ROOT = Path(__file__).resolve().parents[3]

HEADER = "| " + " | ".join(CLAIM_LEDGER_COLUMNS) + " |"
SEPARATOR = "|" + "|".join(["---"] * len(CLAIM_LEDGER_COLUMNS)) + "|"


def _row(
    claim_id: str = "CL-1",
    claim: str = "A capability the project claims.",
    source: str = "README.md",
    grade: str = CLAIM_GRADE_PROVEN,
    verify: str = "`pytest -q`",
    evidence: str = "README.md",
    notes: str = "`src/pkg/mod.py:10-20`",
) -> str:
    return f"| {claim_id} | {claim} | {source} | {grade} | {verify} | {evidence} | {notes} |"


def _ledger(*rows: str, section: str = "Claims") -> str:
    """Build a syntactically valid ledger document around the given rows."""
    return "\n".join(["# Claim Ledger", "", f"## {section}", "", HEADER, SEPARATOR, *rows, ""])


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A minimal repo-shaped tree: a README to cite as a Source, and a docs dir for the ledger."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    _surface_baseline(tmp_path)
    return tmp_path


def _write(tree: Path, content: str) -> Path:
    target = tree / CLAIM_LEDGER_RELATIVE_PATH
    target.write_text(content, encoding="utf-8")
    return target


def _categories(failures: tuple[LedgerFailure, ...]) -> set[str]:
    return {failure.category for failure in failures}


# --------------------------------------------------------------- the live ledger


@pytest.mark.unit
def test_repository_ledger_is_valid() -> None:
    """The committed ledger obeys its own documented rules.

    This is the regression gate: an author who adds a claim without a grade, cites a path that has
    moved, or promotes a row to PROVEN without an artefact fails here rather than in review.
    """
    report = validate(REPO_ROOT)
    assert report.ok, "docs/CLAIM_LEDGER.md is invalid:\n" + "\n".join(f"  - {f}" for f in report.failures)
    assert report.rows, "a ledger with no claims proves nothing"


@pytest.mark.unit
def test_repository_ledger_grade_counts_are_total() -> None:
    """Every row lands in exactly one grade bucket, so the artifact's counts cannot silently drop rows."""
    report = validate(REPO_ROOT)
    assert sum(report.grade_counts.values()) == len(report.rows)
    assert set(report.grade_counts) == set(CLAIM_GRADES)


@pytest.mark.unit
def test_repository_ledger_records_the_engine_disagreement_as_false() -> None:
    """The charter's engine-agreement claim must stay graded FALSE until milestone E2 lands.

    Pinned deliberately. This is the defect that motivated the whole evidence chain, and a silent
    promotion of this row — without the value-semantics fix — is exactly the failure mode the ledger
    exists to prevent.
    """
    report = validate(REPO_ROOT)
    engine_rows = [row for row in report.rows if "negamax" in row.claim.lower()]
    assert engine_rows, "expected a ledger row covering negamax sign handling"
    assert all(row.grade == CLAIM_GRADE_FALSE for row in engine_rows), (
        "the negamax-agreement claim is graded above FALSE. If specs/hygiene_mcts_value_semantics "
        "has landed, update this test in the same change; otherwise the ledger is now wrong."
    )


# --------------------------------------------------------------- promotion rule (AC-2)


@pytest.mark.unit
def test_proven_without_evidence_is_rejected(tree: Path) -> None:
    """The central rule: PROVEN requires a resolvable artefact."""
    _write(tree, _ledger(_row(grade=CLAIM_GRADE_PROVEN, evidence=CLAIM_CELL_EMPTY_SENTINEL)))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert "promotion" in _categories(report.failures)
    assert any("requires an Evidence artefact" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_proven_with_unresolvable_evidence_is_rejected(tree: Path) -> None:
    """An artefact path that does not exist is not evidence, however confidently it is cited."""
    _write(tree, _ledger(_row(evidence="artifacts/does_not_exist.json")))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert "path" in _categories(report.failures)


@pytest.mark.unit
def test_proven_without_verify_command_is_rejected(tree: Path) -> None:
    """Evidence with no reproduction command is a file, not a proof."""
    _write(tree, _ledger(_row(verify=CLAIM_CELL_EMPTY_SENTINEL)))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("requires a Verify command" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_proven_without_source_is_rejected(tree: Path) -> None:
    """A PROVEN grade must trace to the prose that makes the claim."""
    _write(tree, _ledger(_row(source=CLAIM_CELL_EMPTY_SENTINEL)))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("requires a Source document" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_no_flag_relaxes_the_promotion_rule(tree: Path) -> None:
    """There is deliberately no CLI switch that accepts an unsubstantiated PROVEN row.

    Asserted by exhaustion over the parser's own options rather than by inspection, so adding a
    permissive flag in future breaks this test instead of quietly weakening the gate.
    """
    _write(tree, _ledger(_row(evidence=CLAIM_CELL_EMPTY_SENTINEL)))
    for extra in ([], ["--json"], ["--debug"], ["-v"]):
        assert main(["--repo-root", str(tree), *extra]) == 1, f"flags {extra} weakened the promotion rule"


@pytest.mark.unit
def test_valid_proven_row_passes(tree: Path) -> None:
    """The positive control: a well-formed PROVEN row is accepted."""
    _write(tree, _ledger(_row()))
    report = ClaimLedgerValidator(tree).validate()
    assert report.ok, report.failures
    assert report.grade_counts[CLAIM_GRADE_PROVEN] == 1


# --------------------------------------------------------------- schema (AC-1)


@pytest.mark.unit
@pytest.mark.parametrize("bad_id", ["CL1", "CL-", "cl-1", "XCL-1", "CL-1a", ""])
def test_malformed_claim_id_is_rejected(tree: Path, bad_id: str) -> None:
    _write(tree, _ledger(_row(claim_id=bad_id)))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("malformed claim id" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_duplicate_claim_id_is_rejected(tree: Path) -> None:
    """Two rows with one id makes every downstream count ambiguous."""
    _write(tree, _ledger(_row(claim_id="CL-1"), _row(claim_id="CL-1")))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("duplicate claim id" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_unknown_grade_is_rejected(tree: Path) -> None:
    _write(tree, _ledger(_row(grade="MOSTLY_TRUE")))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("unknown grade" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_empty_claim_text_is_rejected(tree: Path) -> None:
    _write(tree, _ledger(_row(claim="   ")))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("empty claim text" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_unresolvable_source_is_rejected(tree: Path) -> None:
    _write(tree, _ledger(_row(source="docs/GONE.md")))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("Source path does not resolve" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_wrong_cell_count_is_reported_with_a_line_number(tree: Path) -> None:
    """A truncated row fails loudly and points at the line, rather than being parsed into garbage."""
    _write(tree, _ledger("| CL-1 | claim | README.md |"))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    structural = [failure for failure in report.failures if failure.category == "structure"]
    assert any(failure.line > 0 and "expected 7" in failure.message for failure in structural), structural


@pytest.mark.unit
def test_missing_ledger_is_reported(tmp_path: Path) -> None:
    report = ClaimLedgerValidator(tmp_path).validate()
    assert not report.ok
    assert any("ledger not found" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_ledger_with_no_rows_is_rejected(tree: Path) -> None:
    _write(tree, "# Claim Ledger\n\nNo table here.\n")
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("no claim rows" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_non_ledger_tables_are_ignored(tree: Path) -> None:
    """An explanatory table (e.g. the grade legend) must not be parsed as claims.

    Only a table whose header matches the column contract exactly is a ledger table; otherwise the
    document could not explain its own rules without tripping its own validator.
    """
    legend = "\n".join(["## Grades", "", "| Grade | Meaning |", "|---|---|", "| PROVEN | proven |", ""])
    _write(tree, legend + "\n" + _ledger(_row()))
    report = ClaimLedgerValidator(tree).validate()
    assert report.ok, report.failures
    assert len(report.rows) == 1


# --------------------------------------------------------------- notes discipline (AC-1)


@pytest.mark.unit
@pytest.mark.parametrize("grade", [CLAIM_GRADE_PARTIAL, CLAIM_GRADE_UNPROVEN, CLAIM_GRADE_FALSE])
def test_grades_asserting_absence_require_notes(tree: Path, grade: str) -> None:
    """ "Partially works" with no explanation is an adjective, not a finding."""
    _write(tree, _ledger(_row(grade=grade, evidence=CLAIM_CELL_EMPTY_SENTINEL, notes=CLAIM_CELL_EMPTY_SENTINEL)))
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert "notes" in _categories(report.failures)


@pytest.mark.unit
def test_false_grade_requires_a_cited_location(tree: Path) -> None:
    """FALSE is the strongest statement in the ledger, so it must point at the contradiction."""
    _write(
        tree,
        _ledger(
            _row(
                grade=CLAIM_GRADE_FALSE,
                evidence=CLAIM_CELL_EMPTY_SENTINEL,
                notes="This is simply not true, as anyone can see.",
            )
        ),
    )
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("citing the contradicting location" in failure.message for failure in report.failures)


@pytest.mark.unit
@pytest.mark.parametrize(
    "notes",
    [
        "Contradicted at `src/framework/mcts/core.py:377-393`.",
        "See `src/config/settings.py:421`.",
        "See `settings.py:421` in config.",
        "Workflow at `.github/workflows/ci.yml` disagrees.",
    ],
)
def test_false_grade_accepts_real_citations(tree: Path, notes: str) -> None:
    _write(tree, _ledger(_row(grade=CLAIM_GRADE_FALSE, evidence=CLAIM_CELL_EMPTY_SENTINEL, notes=notes)))
    report = ClaimLedgerValidator(tree).validate()
    assert report.ok, report.failures


@pytest.mark.unit
def test_unproven_row_may_omit_verify_and_evidence(tree: Path) -> None:
    """UNPROVEN means no command demonstrates the outcome; requiring one would be incoherent."""
    _write(
        tree,
        _ledger(
            _row(
                grade=CLAIM_GRADE_UNPROVEN,
                verify=CLAIM_CELL_EMPTY_SENTINEL,
                evidence=CLAIM_CELL_EMPTY_SENTINEL,
                notes="No driver exists yet; see `docs/plans/EVIDENCE_FIRST_PROGRAM.md`.",
            )
        ),
    )
    report = ClaimLedgerValidator(tree).validate()
    assert report.ok, report.failures


# --------------------------------------------------------------- parsing details


@pytest.mark.unit
def test_backticked_cells_are_unwrapped(tree: Path) -> None:
    """Authors write paths and commands as inline code; the parser must see the value, not the markup."""
    _write(tree, _ledger(_row(source="`README.md`", evidence="`README.md`")))
    report = ClaimLedgerValidator(tree).validate()
    assert report.ok, report.failures
    assert report.rows[0].source == "README.md"


@pytest.mark.unit
def test_section_headings_are_recorded_per_row(tree: Path) -> None:
    """Section provenance is what lets the status artifact roll claims up without a second source."""
    document = "\n".join(
        [
            "# Claim Ledger",
            "",
            "## Charter bullets",
            "",
            HEADER,
            SEPARATOR,
            _row(claim_id="CL-1"),
            "",
            "## README bullets",
            "",
            HEADER,
            SEPARATOR,
            _row(claim_id="CL-2"),
            "",
        ]
    )
    _write(tree, document)
    report = ClaimLedgerValidator(tree).validate()
    assert report.ok, report.failures
    assert [row.section for row in report.rows] == ["Charter bullets", "README bullets"]


@pytest.mark.unit
def test_rows_are_sorted_numerically_not_lexically(tree: Path) -> None:
    """CL-10 must sort after CL-9, or the JSON output reorders itself as the ledger grows."""
    _write(tree, _ledger(_row(claim_id="CL-10"), _row(claim_id="CL-2"), _row(claim_id="CL-9")))
    report = ClaimLedgerValidator(tree).validate()
    assert [row.claim_id for row in report.rows] == ["CL-2", "CL-9", "CL-10"]


@pytest.mark.unit
def test_alignment_colons_in_the_separator_are_tolerated(tree: Path) -> None:
    """Markdown formatters rewrite separators to `|:---|`; that must not silently drop every row."""
    document = "\n".join(
        ["# Claim Ledger", "", "## Claims", "", HEADER, "|:---|:---:|---:|---|---|---|---|", _row(), ""]
    )
    _write(tree, document)
    report = ClaimLedgerValidator(tree).validate()
    assert report.ok, report.failures
    assert len(report.rows) == 1


@pytest.mark.unit
def test_claim_row_ordinal_of_malformed_id_is_negative() -> None:
    """The sort key degrades predictably rather than raising on a row that failed validation."""
    row = ClaimRow("nonsense", "c", "s", CLAIM_GRADE_PROVEN, "v", "e", "n", "sect", 1)
    assert row.ordinal == -1


# --------------------------------------------------------------- reporting and CLI (AC-10)


@pytest.mark.unit
def test_failure_str_names_the_file_and_line() -> None:
    failure = LedgerFailure("CL-3", "promotion", "boom", 42)
    rendered = str(failure)
    assert CLAIM_LEDGER_RELATIVE_PATH in rendered and ":42" in rendered and "CL-3" in rendered


@pytest.mark.unit
def test_failure_str_without_a_line_omits_the_anchor() -> None:
    assert ":0" not in str(LedgerFailure("", "structure", "boom"))


@pytest.mark.unit
def test_json_output_is_machine_readable_and_stable(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two runs over one tree emit identical bytes — the artifact generator depends on this."""
    _write(tree, _ledger(_row(claim_id="CL-2"), _row(claim_id="CL-1")))
    assert main(["--repo-root", str(tree), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["--repo-root", str(tree), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second

    payload = json.loads(first)
    assert payload["ok"] is True
    assert payload["claim_count"] == 2
    assert [claim["id"] for claim in payload["claims"]] == ["CL-1", "CL-2"]
    assert set(payload["grade_counts"]) == set(CLAIM_GRADES)


@pytest.mark.unit
def test_cli_reports_failures_on_stderr(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tree, _ledger(_row(evidence=CLAIM_CELL_EMPTY_SENTINEL)))
    assert main(["--repo-root", str(tree)]) == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.err
    assert captured.out == ""


@pytest.mark.unit
def test_cli_success_summarises_every_grade(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tree, _ledger(_row()))
    assert main(["--repo-root", str(tree)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out
    for grade in CLAIM_GRADES:
        assert grade in out


@pytest.mark.unit
def test_explicit_ledger_path_overrides_the_default(tmp_path: Path) -> None:
    """The path is injected, not discovered, so a caller can validate a candidate ledger in place."""
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    _surface_baseline(tmp_path)
    elsewhere = tmp_path / "candidate.md"
    elsewhere.write_text(_ledger(_row()), encoding="utf-8")
    report = ClaimLedgerValidator(tmp_path, elsewhere).validate()
    assert report.ok, report.failures


@pytest.mark.unit
def test_debug_logging_traces_each_row(tree: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Per-row DEBUG output is a hard requirement: a silent validator is unfixable in CI."""
    _write(tree, _ledger(_row(claim_id="CL-7")))
    with caplog.at_level(logging.DEBUG, logger="src.tools.claim_ledger"):
        ClaimLedgerValidator(tree).validate()
    assert any("parsed CL-7" in record.getMessage() for record in caplog.records)


@pytest.mark.unit
def test_info_logging_states_the_verdict(tree: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(tree, _ledger(_row(evidence=CLAIM_CELL_EMPTY_SENTINEL)))
    with caplog.at_level(logging.INFO, logger="src.tools.claim_ledger"):
        ClaimLedgerValidator(tree).validate()
    assert any("INVALID" in record.getMessage() for record in caplog.records)


# ------------------------------------------------- claim-surface coverage ratchet
# Spec invariant: "every claim in README.md and CHARTER.md section 2 has exactly one ledger row;
# the row count is asserted against the extracted claim count so a new claim cannot be added
# without a grade." Enforced as a per-surface ungraded-surplus ratchet.


@pytest.mark.unit
def test_live_surfaces_are_within_the_committed_ratchet() -> None:
    """The real tree's README and CHARTER section 2 are measured, not assumed."""
    report = validate(REPO_ROOT)
    assert report.ok, report.failures
    assert {coverage.path for coverage in report.surfaces} == {"CHARTER.md", "README.md"}


@pytest.mark.unit
def test_charter_mission_bullets_are_graded_one_for_one() -> None:
    """CHARTER section 2 is the surface where the invariant holds exactly: zero ungraded claims."""
    charter = next(item for item in validate(REPO_ROOT).surfaces if item.path == "CHARTER.md")
    assert charter.bullets == charter.rows
    assert charter.surplus == 0


@pytest.mark.unit
def test_a_new_ungraded_claim_bullet_fails_the_ratchet(tree: Path) -> None:
    """The regression the ratchet exists to catch: a capability bullet with no ledger row."""
    _write(tree, _ledger(_row(source="README.md")))
    _surface_baseline(tree, path="README.md", section="Features", surplus=0, bullets=1, rows=1)
    (tree / "README.md").write_text(
        "# readme\n\n## Features\n\n- **Graded capability**: already in the ledger.\n",
        encoding="utf-8",
    )
    assert ClaimLedgerValidator(tree).validate().ok

    (tree / "README.md").write_text(
        "# readme\n\n## Features\n"
        "\n- **Graded capability**: already in the ledger.\n"
        "\n- **Brand new capability**: nobody graded this.\n",
        encoding="utf-8",
    )
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert "surface" in _categories(report.failures)
    assert any("no ledger row" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_grading_the_new_claim_restores_green(tree: Path) -> None:
    """Adding a bullet is allowed when a ledger row lands with it."""
    (tree / "README.md").write_text(
        "# readme\n\n## Features\n"
        "\n- **Graded capability**: already in the ledger.\n"
        "\n- **Brand new capability**: graded in the same change.\n",
        encoding="utf-8",
    )
    _write(tree, _ledger(_row(claim_id="CL-1"), _row(claim_id="CL-2")))
    _surface_baseline(tree, path="README.md", section="Features", surplus=0, bullets=2, rows=2)
    assert ClaimLedgerValidator(tree).validate().ok


@pytest.mark.unit
def test_baseline_slack_is_a_violation(tree: Path) -> None:
    """A surplus below the baseline means the baseline no longer describes the tree."""
    _write(tree, _ledger(_row()))
    _surface_baseline(tree, path="README.md", section="", surplus=3)
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("stale" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_missing_baseline_fails_rather_than_scoring_zero(tree: Path) -> None:
    """Fail-closed: deleting the baseline must not disable the gate."""
    _write(tree, _ledger(_row()))
    (tree / CLAIM_SURFACE_BASELINE_RELATIVE_PATH).unlink()
    report = ClaimLedgerValidator(tree).validate()
    assert not report.ok
    assert any("is missing" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_unreadable_baseline_is_reported(tree: Path) -> None:
    _write(tree, _ledger(_row()))
    (tree / CLAIM_SURFACE_BASELINE_RELATIVE_PATH).write_text("{not json", encoding="utf-8")
    report = ClaimLedgerValidator(tree).validate()
    assert any("unreadable" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_unsupported_baseline_schema_version_is_rejected(tree: Path) -> None:
    _write(tree, _ledger(_row()))
    _surface_baseline(tree, schema_version=CLAIM_SURFACE_BASELINE_SCHEMA_VERSION + 1)
    report = ClaimLedgerValidator(tree).validate()
    assert any("schema_version" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_baseline_with_no_surfaces_is_rejected(tree: Path) -> None:
    _write(tree, _ledger(_row()))
    (tree / CLAIM_SURFACE_BASELINE_RELATIVE_PATH).write_text(
        json.dumps({"schema_version": CLAIM_SURFACE_BASELINE_SCHEMA_VERSION, CLAIM_SURFACE_KEY_SURFACES: []}),
        encoding="utf-8",
    )
    report = ClaimLedgerValidator(tree).validate()
    assert any("no surfaces" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_non_integer_surplus_is_rejected(tree: Path) -> None:
    _write(tree, _ledger(_row()))
    (tree / CLAIM_SURFACE_BASELINE_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "schema_version": CLAIM_SURFACE_BASELINE_SCHEMA_VERSION,
                CLAIM_SURFACE_KEY_SURFACES: [{CLAIM_SURFACE_KEY_PATH: "README.md", CLAIM_SURFACE_KEY_SURPLUS: "0"}],
            }
        ),
        encoding="utf-8",
    )
    report = ClaimLedgerValidator(tree).validate()
    assert any(CLAIM_SURFACE_KEY_SURPLUS in failure.message for failure in report.failures)


@pytest.mark.unit
def test_declared_surface_that_does_not_exist_is_rejected(tree: Path) -> None:
    _write(tree, _ledger(_row()))
    _surface_baseline(tree, path="docs/GHOST.md")
    report = ClaimLedgerValidator(tree).validate()
    assert any("does not resolve" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_declared_section_that_does_not_exist_is_rejected(tree: Path) -> None:
    _write(tree, _ledger(_row()))
    _surface_baseline(tree, section="Nonexistent Section")
    report = ClaimLedgerValidator(tree).validate()
    assert any("no heading starts with" in failure.message for failure in report.failures)


@pytest.mark.unit
def test_section_extraction_stops_at_the_next_peer_heading(tree: Path) -> None:
    """A bullet in a later section is not this section's claim."""
    (tree / "README.md").write_text(
        "# readme\n\n## Features\n\n- **Inside**: counted.\n\n## Other\n\n- **Outside**: not counted.\n",
        encoding="utf-8",
    )
    _write(tree, _ledger(_row()))
    _surface_baseline(tree, section="Features", surplus=0, bullets=1, rows=1)
    coverage = next(item for item in ClaimLedgerValidator(tree).validate().surfaces if item.path == "README.md")
    assert coverage.bullets == 1


@pytest.mark.unit
def test_section_extraction_includes_nested_subsections(tree: Path) -> None:
    """The real README groups capabilities under `###` subheadings; those bullets still count."""
    (tree / "README.md").write_text(
        "# readme\n\n## Features\n\n### Core\n\n- **One**: a.\n\n### Extras\n\n- **Two**: b.\n\n## Install\n\n- **Three**: c.\n",
        encoding="utf-8",
    )
    _write(tree, _ledger(_row(claim_id="CL-1"), _row(claim_id="CL-2")))
    _surface_baseline(tree, section="Features", surplus=0, bullets=2, rows=2)
    coverage = next(item for item in ClaimLedgerValidator(tree).validate().surfaces if item.path == "README.md")
    assert coverage.bullets == 2


@pytest.mark.unit
def test_plain_bullets_are_not_claims(tree: Path) -> None:
    """Only a bold-first-token bullet is claim-shaped, so prose asides do not inflate the count."""
    (tree / "README.md").write_text(
        "# readme\n\n## Features\n\n- a plain aside.\n- **A capability**: graded.\n",
        encoding="utf-8",
    )
    _write(tree, _ledger(_row()))
    _surface_baseline(tree, section="Features", surplus=0, bullets=1, rows=1)
    coverage = next(item for item in ClaimLedgerValidator(tree).validate().surfaces if item.path == "README.md")
    assert coverage.bullets == 1


@pytest.mark.unit
def test_surface_coverage_surplus_never_goes_negative() -> None:
    """More ledger rows than bullets is over-grading, not credit against a future claim."""
    assert SurfaceCoverage(path="README.md", section="", bullets=1, rows=4).surplus == 0


@pytest.mark.unit
def test_write_surface_baseline_round_trips(tree: Path) -> None:
    """Regeneration refreshes counts for the declared surfaces and leaves the tree green."""
    (tree / "README.md").write_text(
        "# readme\n\n## Features\n\n- **One**: a.\n- **Two**: b.\n- **Three**: c.\n",
        encoding="utf-8",
    )
    _write(tree, _ledger(_row()))
    _surface_baseline(tree, section="Features", surplus=99)

    written = ClaimLedgerValidator(tree).write_surface_baseline()
    assert [item.as_dict()[CLAIM_SURFACE_KEY_SURPLUS] for item in written] == [2]
    assert ClaimLedgerValidator(tree).validate().ok


@pytest.mark.unit
def test_cli_write_surface_baseline_exits_zero_and_reports(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tree, _ledger(_row()))
    _surface_baseline(tree, surplus=7)
    assert main(["--repo-root", str(tree), "--write-surface-baseline"]) == 0
    assert "surplus 0" in capsys.readouterr().out


@pytest.mark.unit
def test_json_output_carries_the_surfaces(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tree, _ledger(_row()))
    main(["--repo-root", str(tree), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["surfaces"] == [
        {
            CLAIM_SURFACE_KEY_PATH: "README.md",
            CLAIM_SURFACE_KEY_SECTION: "",
            CLAIM_SURFACE_KEY_BULLETS: 0,
            CLAIM_SURFACE_KEY_ROWS: 1,
            CLAIM_SURFACE_KEY_SURPLUS: 0,
        }
    ]


@pytest.mark.unit
def test_surface_debug_logging_traces_each_measurement(tree: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(tree, _ledger(_row()))
    with caplog.at_level(logging.DEBUG, logger="src.tools.claim_ledger"):
        ClaimLedgerValidator(tree).validate()
    assert any("claim bullet(s)" in record.getMessage() for record in caplog.records)
