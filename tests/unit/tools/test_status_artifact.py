"""Tests for the provenance-stamped status artifact generator.

Spec: ``specs/evidence_claim_ledger.SPEC.md`` (AC-3, AC-4, AC-5, AC-10).

Three properties carry the weight here, and each is tested by attempting to violate it:

- **Provenance is mandatory** (AC-4). A result entry cannot be constructed without saying how it was
  produced. This is the difference between an artifact and a rumour.
- **Output is byte-stable** (AC-5). With an injected clock, two runs at one commit are identical, so
  a diff in the artifact means a real change in what the tree can prove.
- **The maturity map cannot outrun its own claims** (AC-3). Declaring a capability "benchmarked"
  while its supporting ledger row is graded FALSE is rejected, so the matrix is falsifiable rather
  than aspirational.
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
import json
import logging
from pathlib import Path

import pytest

from src.config.constants import (
    CAPABILITY_MATURITY_RELATIVE_PATH,
    CAPABILITY_MATURITY_STAGES,
    CLAIM_CELL_EMPTY_SENTINEL,
    CLAIM_GRADE_FALSE,
    CLAIM_GRADE_PROVEN,
    CLAIM_GRADES,
    CLAIM_LEDGER_COLUMNS,
    CLAIM_LEDGER_RELATIVE_PATH,
    EVIDENCE_PROVENANCE_MOCK,
    EVIDENCE_PROVENANCE_TRAINED_WEIGHTS,
    EVIDENCE_PROVENANCES,
    STATUS_ARTIFACT_RELATIVE_PATH,
    STATUS_ARTIFACT_SCHEMA_VERSION,
)
from src.tools.status_artifact import (
    ProvenanceError,
    ResultEntry,
    StatusArtifactBuilder,
    build,
    main,
)
from tests.fixtures.claim_ledger_fixtures import write_claim_surface_baseline

REPO_ROOT = Path(__file__).resolve().parents[3]

FIXED_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

HEADER = "| " + " | ".join(CLAIM_LEDGER_COLUMNS) + " |"
SEPARATOR = "|" + "|".join(["---"] * len(CLAIM_LEDGER_COLUMNS)) + "|"


def _ledger_row(claim_id: str, grade: str) -> str:
    evidence = "README.md" if grade == CLAIM_GRADE_PROVEN else CLAIM_CELL_EMPTY_SENTINEL
    verify = "`pytest -q`"
    notes = "Contradicted at `src/pkg/mod.py:10`." if grade != CLAIM_GRADE_PROVEN else "Reproduced."
    return f"| {claim_id} | A claim. | README.md | {grade} | {verify} | {evidence} | {notes} |"


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A minimal repo-shaped tree with a valid two-row ledger and a coverage gate in pyproject."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / CLAIM_LEDGER_RELATIVE_PATH).write_text(
        "\n".join(
            [
                "# Claim Ledger",
                "",
                "## Claims",
                "",
                HEADER,
                SEPARATOR,
                _ledger_row("CL-1", CLAIM_GRADE_PROVEN),
                _ledger_row("CL-2", CLAIM_GRADE_FALSE),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(["[tool.coverage.report]", "fail_under = 91.5", ""]), encoding="utf-8"
    )
    _write_maturity(tmp_path, [{"name": "cap", "stage": "tested", "claims": ["CL-1"], "notes": "n"}])
    # The ledger the artifact summarises is only "valid" when the claim-surface ratchet is satisfied
    # too, so a synthetic tree needs a baseline for the artifact to report ok.
    write_claim_surface_baseline(tmp_path)
    return tmp_path


def _write_maturity(tree: Path, capabilities: list[dict[str, object]]) -> None:
    (tree / CAPABILITY_MATURITY_RELATIVE_PATH).write_text(
        json.dumps({"schema_version": 1, "capabilities": capabilities}, indent=2), encoding="utf-8"
    )


def _builder(tree: Path, **kwargs: object) -> StatusArtifactBuilder:
    return StatusArtifactBuilder(tree, now=FIXED_NOW, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------- provenance (AC-4)


@pytest.mark.unit
def test_result_entry_requires_a_provenance() -> None:
    """An entry with an empty provenance cannot exist. This is the artifact's whole reason to be."""
    with pytest.raises(ProvenanceError, match="omits provenance"):
        ResultEntry(name="elo", value=1500, provenance="")


@pytest.mark.unit
def test_result_entry_rejects_an_unknown_provenance() -> None:
    with pytest.raises(ProvenanceError, match="unknown provenance"):
        ResultEntry(name="elo", value=1500, provenance="vibes")


@pytest.mark.unit
def test_result_entry_requires_a_name() -> None:
    with pytest.raises(ProvenanceError, match="requires a name"):
        ResultEntry(name="  ", value=1, provenance=EVIDENCE_PROVENANCE_MOCK)


@pytest.mark.unit
@pytest.mark.parametrize("provenance", EVIDENCE_PROVENANCES)
def test_every_declared_provenance_is_accepted(provenance: str) -> None:
    """The closed vocabulary and the validation must agree, or a legal value becomes unusable."""
    assert ResultEntry(name="metric", value=1.0, provenance=provenance).provenance == provenance


@pytest.mark.unit
def test_from_mapping_rejects_an_entry_with_no_provenance_key() -> None:
    """The CLAIM_LEDGER path is not the only ingress; JSON from a benchmark run is checked too."""
    with pytest.raises(ProvenanceError, match="omits provenance"):
        ResultEntry.from_mapping({"name": "lift", "value": 0.12})


@pytest.mark.unit
def test_from_mapping_round_trips_a_full_entry() -> None:
    payload = {
        "name": "policy_lift",
        "value": 0.12,
        "provenance": EVIDENCE_PROVENANCE_TRAINED_WEIGHTS,
        "command": "policy-lift --domain c4",
        "artifact": "benchmarks/results/lift.json",
        "notes": "seeded",
    }
    assert ResultEntry.from_mapping(payload).as_dict() == payload


@pytest.mark.unit
def test_results_reach_the_artifact_with_provenance(tree: Path) -> None:
    entry = ResultEntry(name="lift", value=0.12, provenance=EVIDENCE_PROVENANCE_TRAINED_WEIGHTS)
    payload = _builder(tree).build([entry])
    assert payload["results"] == [entry.as_dict()]
    assert payload["results"][0]["provenance"] == EVIDENCE_PROVENANCE_TRAINED_WEIGHTS


@pytest.mark.unit
def test_cli_rejects_a_results_file_missing_provenance(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = tree / "results.json"
    results.write_text(json.dumps([{"name": "lift", "value": 0.1}]), encoding="utf-8")
    assert main(["--repo-root", str(tree), "--stdout", "--results", str(results)]) == 2
    assert "omits provenance" in capsys.readouterr().err


# --------------------------------------------------------------- determinism (AC-5)


@pytest.mark.unit
def test_serialised_output_is_byte_identical_across_runs(tree: Path) -> None:
    """With the clock injected, nothing in the payload varies run to run."""
    builder = _builder(tree)
    first = builder.serialise(builder.build())
    second = builder.serialise(builder.build())
    assert first == second


@pytest.mark.unit
def test_injected_clock_is_the_only_timestamp(tree: Path) -> None:
    payload = _builder(tree).build()
    assert payload["generated_at"] == FIXED_NOW.isoformat()


@pytest.mark.unit
def test_naive_and_aware_clocks_normalise_to_utc(tree: Path) -> None:
    """A caller passing a non-UTC stamp must not produce an artifact whose time zone drifts."""
    from datetime import timedelta

    eastern = timezone(timedelta(hours=-5))
    payload = StatusArtifactBuilder(tree, now=FIXED_NOW.astimezone(eastern)).build()
    assert payload["generated_at"] == FIXED_NOW.isoformat()


@pytest.mark.unit
def test_serialisation_ends_with_a_newline(tree: Path) -> None:
    builder = _builder(tree)
    assert builder.serialise(builder.build()).endswith("\n")


@pytest.mark.unit
def test_cli_rejects_a_malformed_now(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--repo-root", str(tree), "--stdout", "--now", "yesterday"]) == 2
    assert "ISO-8601" in capsys.readouterr().err


# --------------------------------------------------------------- content (AC-3)


@pytest.mark.unit
def test_coverage_gate_is_read_from_pyproject_not_hardcoded(tree: Path) -> None:
    """The artifact must report the gate the tree enforces, whatever that value is."""
    payload = _builder(tree).build()
    assert payload["coverage"]["gate"] == 91.5


@pytest.mark.unit
def test_coverage_gate_is_unknown_when_pyproject_is_absent(tmp_path: Path) -> None:
    """Absent is reported as unknown, never as zero: a fabricated 0.0 would read as a real figure."""
    (tmp_path / "docs").mkdir()
    payload = StatusArtifactBuilder(tmp_path, now=FIXED_NOW).build()
    assert payload["coverage"]["gate"] == "unknown"


@pytest.mark.unit
def test_coverage_gate_is_unknown_when_the_key_is_missing(tree: Path) -> None:
    (tree / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert _builder(tree).build()["coverage"]["gate"] == "unknown"


@pytest.mark.unit
def test_coverage_gate_is_unknown_when_pyproject_is_malformed(tree: Path) -> None:
    (tree / "pyproject.toml").write_text("this is not = [toml\n", encoding="utf-8")
    assert _builder(tree).build()["coverage"]["gate"] == "unknown"


@pytest.mark.unit
def test_measured_coverage_is_read_from_a_coverage_report(tree: Path) -> None:
    report = tree / "coverage.json"
    report.write_text(json.dumps({"totals": {"percent_covered": 89.71}}), encoding="utf-8")
    payload = _builder(tree, coverage_json=report).build()
    assert payload["coverage"]["measured"] == 89.71
    assert payload["coverage"]["source"] == str(report)


@pytest.mark.unit
def test_measured_coverage_is_unknown_without_a_report(tree: Path) -> None:
    assert _builder(tree).build()["coverage"]["measured"] == "unknown"


@pytest.mark.unit
@pytest.mark.parametrize("body", ["{}", '{"totals": {}}', "not json at all"])
def test_measured_coverage_degrades_on_a_malformed_report(tree: Path, body: str) -> None:
    report = tree / "coverage.json"
    report.write_text(body, encoding="utf-8")
    assert _builder(tree, coverage_json=report).build()["coverage"]["measured"] == "unknown"


@pytest.mark.unit
def test_missing_coverage_report_is_warned_not_fatal(tree: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="src.tools.status_artifact"):
        payload = _builder(tree, coverage_json=tree / "absent.json").build()
    assert payload["coverage"]["measured"] == "unknown"
    assert any("coverage report not found" in record.getMessage() for record in caplog.records)


@pytest.mark.unit
def test_claim_counts_are_derived_from_the_ledger(tree: Path) -> None:
    claims = _builder(tree).build()["claims"]
    assert claims["total"] == 2
    assert claims["by_grade"][CLAIM_GRADE_PROVEN] == 1
    assert claims["by_grade"][CLAIM_GRADE_FALSE] == 1
    assert set(claims["by_grade"]) == set(CLAIM_GRADES)
    assert claims["valid"] is True


@pytest.mark.unit
def test_an_invalid_ledger_makes_the_artifact_not_ok(tree: Path) -> None:
    """The artifact reports the failure rather than omitting the section — silence would read as green."""
    (tree / CLAIM_LEDGER_RELATIVE_PATH).write_text(
        "\n".join(["## Claims", "", HEADER, SEPARATOR, _ledger_row("CL-1", "MOSTLY"), ""]), encoding="utf-8"
    )
    payload = _builder(tree).build()
    assert payload["ok"] is False
    assert payload["claims"]["valid"] is False
    assert payload["claims"]["failures"]


@pytest.mark.unit
def test_a_claim_with_an_unrecognised_grade_cannot_support_a_stage(tree: Path) -> None:
    """Regression: an unknown grade once crashed the builder, which is worse than a red gate.

    Failing closed matters here. If an unrankable grade were treated as the strongest value, a typo
    in the ledger would silently *promote* a capability.
    """
    (tree / CLAIM_LEDGER_RELATIVE_PATH).write_text(
        "\n".join(["## Claims", "", HEADER, SEPARATOR, _ledger_row("CL-1", "MOSTLY"), ""]), encoding="utf-8"
    )
    payload = _builder(tree).build()
    assert payload["ok"] is False
    assert any("not recognised" in problem for problem in payload["capability_maturity_problems"])
    assert payload["capability_maturity"] == []


@pytest.mark.unit
def test_environment_records_the_interpreter_and_extras(tree: Path) -> None:
    environment = _builder(tree).build()["environment"]
    assert environment["python"].count(".") == 2
    assert isinstance(environment["extras"], list)
    assert environment["extras"] == sorted(environment["extras"])


@pytest.mark.unit
def test_environment_degrades_outside_a_git_checkout(tree: Path) -> None:
    """A tarball export has no .git. That must yield `unknown`, not a crash or a fabricated sha."""
    environment = _builder(tree).build()["environment"]
    assert environment["commit"] == "unknown"
    assert environment["dirty"] == "unknown"
    assert environment["branch"] == "unknown"


@pytest.mark.unit
def test_schema_version_is_stamped(tree: Path) -> None:
    assert _builder(tree).build()["schema_version"] == STATUS_ARTIFACT_SCHEMA_VERSION


# --------------------------------------------------------------- maturity map (AC-3)


@pytest.mark.unit
def test_maturity_stage_above_its_claims_ceiling_is_rejected(tree: Path) -> None:
    """The core check: a capability cannot advertise a stage its own ledger rows contradict."""
    _write_maturity(tree, [{"name": "cap", "stage": "gated", "claims": ["CL-2"], "notes": ""}])
    payload = _builder(tree).build()
    assert payload["ok"] is False
    problems = payload["capability_maturity_problems"]
    assert any("caps the stage at" in problem for problem in problems), problems


@pytest.mark.unit
def test_maturity_stage_at_the_ceiling_is_accepted(tree: Path) -> None:
    _write_maturity(tree, [{"name": "cap", "stage": "tested", "claims": ["CL-2"], "notes": ""}])
    payload = _builder(tree).build()
    assert payload["ok"] is True, payload["capability_maturity_problems"]
    assert payload["capability_maturity"][0]["ceiling"] == "tested"


@pytest.mark.unit
def test_maturity_ceiling_is_the_weakest_supporting_claim(tree: Path) -> None:
    """One FALSE row caps the capability, even when every other supporting row is PROVEN."""
    _write_maturity(tree, [{"name": "cap", "stage": "integrated", "claims": ["CL-1", "CL-2"], "notes": ""}])
    problems = _builder(tree).build()["capability_maturity_problems"]
    assert any("CL-2" in problem for problem in problems), problems


@pytest.mark.unit
def test_maturity_rejects_an_unknown_stage(tree: Path) -> None:
    _write_maturity(tree, [{"name": "cap", "stage": "amazing", "claims": ["CL-1"], "notes": ""}])
    assert any("unknown stage" in problem for problem in _builder(tree).build()["capability_maturity_problems"])


@pytest.mark.unit
def test_maturity_rejects_a_claim_absent_from_the_ledger(tree: Path) -> None:
    """A dangling claim reference is how a matrix quietly stops meaning anything."""
    _write_maturity(tree, [{"name": "cap", "stage": "tested", "claims": ["CL-99"], "notes": ""}])
    assert any(
        "absent from the ledger" in problem for problem in _builder(tree).build()["capability_maturity_problems"]
    )


@pytest.mark.unit
def test_maturity_rejects_a_capability_with_no_claims(tree: Path) -> None:
    _write_maturity(tree, [{"name": "cap", "stage": "tested", "claims": [], "notes": ""}])
    assert any("no supporting claims" in problem for problem in _builder(tree).build()["capability_maturity_problems"])


@pytest.mark.unit
def test_maturity_reports_every_problem_not_just_the_first(tree: Path) -> None:
    """One run must surface all disagreements; fixing them one CI round-trip at a time is a tax."""
    _write_maturity(
        tree,
        [
            {"name": "a", "stage": "amazing", "claims": ["CL-1"], "notes": ""},
            {"name": "b", "stage": "tested", "claims": ["CL-99"], "notes": ""},
        ],
    )
    assert len(_builder(tree).build()["capability_maturity_problems"]) == 2


@pytest.mark.unit
def test_missing_maturity_map_is_reported(tree: Path) -> None:
    (tree / CAPABILITY_MATURITY_RELATIVE_PATH).unlink()
    payload = _builder(tree).build()
    assert payload["ok"] is False
    assert any("not found" in problem for problem in payload["capability_maturity_problems"])


@pytest.mark.unit
def test_malformed_maturity_map_is_reported(tree: Path) -> None:
    (tree / CAPABILITY_MATURITY_RELATIVE_PATH).write_text("{ not json", encoding="utf-8")
    assert any("could not parse" in problem for problem in _builder(tree).build()["capability_maturity_problems"])


@pytest.mark.unit
def test_maturity_rows_are_sorted_weakest_first(tree: Path) -> None:
    """Ordering is part of the contract: the weakest capability is the one a reader must see."""
    _write_maturity(
        tree,
        [
            {"name": "strong", "stage": "tested", "claims": ["CL-1"], "notes": ""},
            {"name": "weak", "stage": "imports", "claims": ["CL-2"], "notes": ""},
        ],
    )
    rows = _builder(tree).build()["capability_maturity"]
    assert [row["name"] for row in rows] == ["weak", "strong"]
    assert [row["stage_index"] for row in rows] == sorted(row["stage_index"] for row in rows)


# --------------------------------------------------------------- the live tree


@pytest.mark.unit
def test_repository_maturity_map_agrees_with_the_repository_ledger() -> None:
    """The committed map and ledger must not contradict each other. This is the regression gate."""
    payload = build(REPO_ROOT, now=FIXED_NOW)
    assert payload["capability_maturity_problems"] == [], payload["capability_maturity_problems"]
    assert payload["ok"] is True


@pytest.mark.unit
def test_repository_maturity_map_covers_every_stage_it_declares() -> None:
    """Every stage cited by the committed map is a member of the declared ladder."""
    payload = build(REPO_ROOT, now=FIXED_NOW)
    stages = {row["stage"] for row in payload["capability_maturity"]}
    assert stages <= set(CAPABILITY_MATURITY_STAGES)


@pytest.mark.unit
def test_repository_promotion_capability_is_not_claimed_as_working() -> None:
    """Checkpoint promotion must stay at the bottom of the ladder until milestone E5 lands.

    Pinned because this is the claim the evidence-first program exists to make true. Advancing it
    without a promotion gate in the tree should fail here.
    """
    payload = build(REPO_ROOT, now=FIXED_NOW)
    rows = {row["name"]: row for row in payload["capability_maturity"]}
    assert "checkpoint-promotion" in rows, "expected a capability row for checkpoint promotion"
    assert rows["checkpoint-promotion"]["stage"] == CAPABILITY_MATURITY_STAGES[0]


# --------------------------------------------------------------- CLI (AC-6, AC-10)


@pytest.mark.unit
def test_cli_writes_the_artifact_to_the_default_path(tree: Path) -> None:
    assert main(["--repo-root", str(tree), "--now", FIXED_NOW.isoformat()]) == 0
    written = tree / STATUS_ARTIFACT_RELATIVE_PATH
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8"))["schema_version"] == STATUS_ARTIFACT_SCHEMA_VERSION


@pytest.mark.unit
def test_cli_stdout_mode_writes_no_file(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--repo-root", str(tree), "--stdout", "--now", FIXED_NOW.isoformat()]) == 0
    assert not (tree / STATUS_ARTIFACT_RELATIVE_PATH).exists()
    emitted = capsys.readouterr().out
    assert json.loads(emitted)["ok"] is True


@pytest.mark.unit
def test_cli_strict_fails_when_the_maturity_map_disagrees(tree: Path) -> None:
    """--strict is what makes CI red. Without it the artifact is informational only."""
    _write_maturity(tree, [{"name": "cap", "stage": "gated", "claims": ["CL-2"], "notes": ""}])
    assert main(["--repo-root", str(tree), "--stdout", "--strict"]) == 1
    assert main(["--repo-root", str(tree), "--stdout"]) == 0


@pytest.mark.unit
def test_cli_out_flag_relocates_the_artifact(tree: Path) -> None:
    target = tree / "nested" / "elsewhere.json"
    assert main(["--repo-root", str(tree), "--out", str(target)]) == 0
    assert target.is_file()


@pytest.mark.unit
def test_write_creates_the_parent_directory(tree: Path) -> None:
    """artifacts/ is git-ignored and therefore absent on a fresh clone."""
    builder = _builder(tree)
    written = builder.write(builder.build())
    assert written.parent.is_dir()


@pytest.mark.unit
def test_info_logging_summarises_the_assembly(tree: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="src.tools.status_artifact"):
        _builder(tree).build()
    assert any("status artifact assembled" in record.getMessage() for record in caplog.records)
