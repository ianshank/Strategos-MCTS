"""Tests for the GitHub Actions commit-SHA pin ratchet.

Spec: ``specs/evidence_claim_ledger.SPEC.md`` AC-8.

The ratchet's value depends entirely on it being un-gameable, so most tests here try to game it:
raise the baseline, introduce a new unpinned action, pin something and leave the baseline slack in
place. Each attempt must fail.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.config.constants import (
    ACTION_PIN_BASELINE_RELATIVE_PATH,
    ACTION_PIN_BASELINE_SCHEMA_VERSION,
    ACTION_PIN_SHA_LENGTH,
)
from src.tools.action_pins import ActionPinAuditor, ActionUse, PinReport, audit, main

REPO_ROOT = Path(__file__).resolve().parents[3]

PINNED_SHA = "a" * ACTION_PIN_SHA_LENGTH


def _workflow(*lines: str) -> str:
    return "\n".join(["name: wf", "on:", "  push:", "jobs:", "  job:", "    steps:", *lines, ""])


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A synthetic repository with one workflow using two tag-pinned actions and a matching baseline."""
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        _workflow("      - uses: actions/checkout@v4", "      - uses: actions/setup-python@v5"),
        encoding="utf-8",
    )
    _write_baseline(tmp_path, {"actions/checkout": 1, "actions/setup-python": 1})
    return tmp_path


def _write_baseline(tree: Path, counts: dict[str, int], **overrides: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": ACTION_PIN_BASELINE_SCHEMA_VERSION,
        "total_unpinned_uses": sum(counts.values()),
        "unpinned_uses_by_action": counts,
    }
    payload.update(overrides)
    target = tree / ACTION_PIN_BASELINE_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _set_workflow(tree: Path, *lines: str) -> None:
    (tree / ".github" / "workflows" / "ci.yml").write_text(_workflow(*lines), encoding="utf-8")


# --------------------------------------------------------------- scanning


@pytest.mark.unit
def test_scan_finds_every_reference_with_a_location(tree: Path) -> None:
    uses = ActionPinAuditor(tree).scan()
    assert [use.action for use in uses] == ["actions/checkout", "actions/setup-python"]
    assert all(use.line > 0 for use in uses)
    assert all(use.workflow == "ci.yml" for use in uses)


@pytest.mark.unit
def test_a_forty_character_sha_counts_as_pinned(tree: Path) -> None:
    _set_workflow(tree, f"      - uses: actions/checkout@{PINNED_SHA}")
    report = ActionPinAuditor(tree).audit()
    assert report.uses[0].pinned is True
    assert report.unpinned == ()


@pytest.mark.unit
@pytest.mark.parametrize("ref", ["v4", "main", "a" * 7, "a" * 39, "A" * 40, "refs/tags/v4"])
def test_anything_but_a_full_sha_counts_as_unpinned(tree: Path, ref: str) -> None:
    """Short SHAs are ambiguous and branch names are mutable; neither is an immutable reference."""
    _set_workflow(tree, f"      - uses: actions/checkout@{ref}")
    assert ActionPinAuditor(tree).scan()[0].pinned is False


@pytest.mark.unit
@pytest.mark.parametrize("ref", ["./.github/actions/setup", "../shared/action", "docker://alpine:3"])
def test_local_and_container_references_are_out_of_scope(tree: Path, ref: str) -> None:
    """A composite action in this repository is reviewed here; the tag-mutability risk does not apply."""
    _set_workflow(tree, f"      - uses: {ref}")
    assert ActionPinAuditor(tree).scan() == ()


@pytest.mark.unit
def test_reference_without_a_version_is_skipped(tree: Path) -> None:
    """``uses: actions/checkout`` is invalid to Actions; reporting it here would be noise."""
    _set_workflow(tree, "      - uses: actions/checkout")
    assert ActionPinAuditor(tree).scan() == ()


@pytest.mark.unit
@pytest.mark.parametrize("line", ["      - uses: 'actions/checkout@v4'", '      - uses: "actions/checkout@v4"'])
def test_quoted_references_are_parsed(tree: Path, line: str) -> None:
    _set_workflow(tree, line)
    assert ActionPinAuditor(tree).scan()[0].ref == "v4"


@pytest.mark.unit
def test_subpath_actions_are_named_in_full(tree: Path) -> None:
    """``github/codeql-action/upload-sarif`` and its siblings are separately versioned."""
    _set_workflow(tree, "      - uses: github/codeql-action/upload-sarif@v4")
    assert ActionPinAuditor(tree).scan()[0].action == "github/codeql-action/upload-sarif"


@pytest.mark.unit
def test_both_yaml_extensions_are_discovered(tree: Path) -> None:
    (tree / ".github" / "workflows" / "other.yaml").write_text(
        _workflow("      - uses: actions/cache@v4"), encoding="utf-8"
    )
    assert {p.name for p in ActionPinAuditor(tree).workflow_files()} == {"ci.yml", "other.yaml"}


@pytest.mark.unit
def test_a_tree_with_no_workflows_scans_cleanly(tmp_path: Path) -> None:
    """A consumer vendoring this module must not need a .github directory."""
    assert ActionPinAuditor(tmp_path).scan() == ()


@pytest.mark.unit
def test_use_str_points_at_the_line(tree: Path) -> None:
    use = ActionPinAuditor(tree).scan()[0]
    assert ".github/workflows/ci.yml:" in str(use) and "actions/checkout@v4" in str(use)


# --------------------------------------------------------------- the ratchet


@pytest.mark.unit
def test_matching_baseline_passes(tree: Path) -> None:
    report = ActionPinAuditor(tree).audit()
    assert report.ok, report.violations


@pytest.mark.unit
def test_an_additional_unpinned_use_is_rejected(tree: Path) -> None:
    """The count may not grow. This is the rule the whole mechanism rests on."""
    _set_workflow(
        tree,
        "      - uses: actions/checkout@v4",
        "      - uses: actions/setup-python@v5",
        "      - uses: actions/checkout@v4",
    )
    report = ActionPinAuditor(tree).audit()
    assert not report.ok
    assert any("baseline allows 1" in violation for violation in report.violations)


@pytest.mark.unit
def test_a_new_action_must_be_pinned_on_first_use(tree: Path) -> None:
    """No baseline entry means no allowance; a brand-new dependency starts pinned."""
    _set_workflow(
        tree,
        "      - uses: actions/checkout@v4",
        "      - uses: actions/setup-python@v5",
        "      - uses: some-vendor/risky-action@v1",
    )
    report = ActionPinAuditor(tree).audit()
    assert not report.ok
    assert any("has no baseline entry" in violation for violation in report.violations)
    assert any(str(ACTION_PIN_SHA_LENGTH) in violation for violation in report.violations)


@pytest.mark.unit
def test_a_new_action_pinned_to_a_sha_needs_no_baseline_entry(tree: Path) -> None:
    _set_workflow(
        tree,
        "      - uses: actions/checkout@v4",
        "      - uses: actions/setup-python@v5",
        f"      - uses: some-vendor/action@{PINNED_SHA}",
    )
    assert ActionPinAuditor(tree).audit().ok


@pytest.mark.unit
def test_slack_left_in_the_baseline_is_rejected(tree: Path) -> None:
    """Pinning an action without re-tightening the baseline leaves budget for the next regression.

    This is the rule that keeps the ratchet monotonic. Without it, one cleanup would silently
    authorise a future unpinned step.
    """
    _set_workflow(tree, f"      - uses: actions/checkout@{PINNED_SHA}", "      - uses: actions/setup-python@v5")
    report = ActionPinAuditor(tree).audit()
    assert not report.ok
    assert any("baseline is stale" in violation for violation in report.violations)


@pytest.mark.unit
def test_raising_the_baseline_does_not_launder_a_new_unpinned_use(tree: Path) -> None:
    """The obvious way to defeat a ratchet is to edit its baseline. The staleness rule blocks it.

    Raising ``actions/checkout`` to 5 while the tree has 1 makes the file stale, so the audit still
    fails — an author cannot pre-authorise headroom.
    """
    _write_baseline(tree, {"actions/checkout": 5, "actions/setup-python": 1})
    report = ActionPinAuditor(tree).audit()
    assert not report.ok
    assert any("baseline is stale" in violation for violation in report.violations)


@pytest.mark.unit
def test_fully_pinning_an_action_and_removing_its_entry_passes(tree: Path) -> None:
    _set_workflow(tree, f"      - uses: actions/checkout@{PINNED_SHA}", "      - uses: actions/setup-python@v5")
    _write_baseline(tree, {"actions/setup-python": 1})
    assert ActionPinAuditor(tree).audit().ok


# --------------------------------------------------------------- baseline hygiene


@pytest.mark.unit
def test_missing_baseline_is_reported_with_the_fix(tmp_path: Path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        _workflow("      - uses: actions/checkout@v4"), encoding="utf-8"
    )
    report = ActionPinAuditor(tmp_path).audit()
    assert not report.ok
    assert any("--write-baseline" in violation for violation in report.violations)


@pytest.mark.unit
def test_malformed_baseline_is_reported(tree: Path) -> None:
    (tree / ACTION_PIN_BASELINE_RELATIVE_PATH).write_text("{ nope", encoding="utf-8")
    assert any("could not parse" in violation for violation in ActionPinAuditor(tree).audit().violations)


@pytest.mark.unit
def test_baseline_schema_version_is_enforced(tree: Path) -> None:
    """An unversioned or future-versioned file is refused rather than half-understood."""
    _write_baseline(tree, {"actions/checkout": 1}, schema_version=99)
    assert any("schema_version" in violation for violation in ActionPinAuditor(tree).audit().violations)


@pytest.mark.unit
def test_baseline_with_a_non_object_map_is_reported(tree: Path) -> None:
    _write_baseline(tree, {}, unpinned_uses_by_action=["actions/checkout"])
    assert any("must be an object" in violation for violation in ActionPinAuditor(tree).audit().violations)


@pytest.mark.unit
@pytest.mark.parametrize("count", ["four", -1, 1.5, True])
def test_baseline_with_a_non_integer_count_is_reported(tree: Path, count: object) -> None:
    """``True`` is included on purpose: it is an ``int`` in Python and would otherwise mean 1."""
    _write_baseline(tree, {}, unpinned_uses_by_action={"actions/checkout": count})
    assert any("non-integer count" in violation for violation in ActionPinAuditor(tree).audit().violations)


# --------------------------------------------------------------- report and CLI


@pytest.mark.unit
def test_counts_are_sorted_for_stable_output(tree: Path) -> None:
    _set_workflow(tree, "      - uses: zzz/last@v1", "      - uses: aaa/first@v1")
    report = PinReport(ActionPinAuditor(tree).scan())
    assert list(report.counts) == sorted(report.counts)


@pytest.mark.unit
def test_as_dict_is_json_serialisable(tree: Path) -> None:
    payload = ActionPinAuditor(tree).audit().as_dict()
    assert json.loads(json.dumps(payload))["ok"] is True


@pytest.mark.unit
def test_cli_exits_zero_when_the_ratchet_holds(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--repo-root", str(tree)]) == 0
    assert "OK" in capsys.readouterr().out


@pytest.mark.unit
def test_cli_exits_one_and_explains_on_violation(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_baseline(tree, {})
    assert main(["--repo-root", str(tree)]) == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.err
    assert "no baseline entry" in captured.err


@pytest.mark.unit
def test_cli_json_output_is_stable(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--repo-root", str(tree), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["--repo-root", str(tree), "--json"]) == 0
    assert capsys.readouterr().out == first


@pytest.mark.unit
def test_write_baseline_records_the_current_tree(tree: Path) -> None:
    _set_workflow(tree, "      - uses: actions/checkout@v4", "      - uses: actions/checkout@v4")
    assert main(["--repo-root", str(tree), "--write-baseline"]) == 0
    written = json.loads((tree / ACTION_PIN_BASELINE_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["unpinned_uses_by_action"] == {"actions/checkout": 2}
    assert written["schema_version"] == ACTION_PIN_BASELINE_SCHEMA_VERSION
    assert ActionPinAuditor(tree).audit().ok


@pytest.mark.unit
def test_write_baseline_preserves_the_explanatory_readme(tree: Path) -> None:
    """The file's only documentation lives inside it; regenerating must not delete it."""
    _write_baseline(tree, {"actions/checkout": 1, "actions/setup-python": 1}, _README="why this file exists")
    assert main(["--repo-root", str(tree), "--write-baseline"]) == 0
    written = json.loads((tree / ACTION_PIN_BASELINE_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["_README"] == "why this file exists"


@pytest.mark.unit
def test_write_baseline_creates_a_missing_file(tmp_path: Path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        _workflow("      - uses: actions/checkout@v4"), encoding="utf-8"
    )
    assert main(["--repo-root", str(tmp_path), "--write-baseline"]) == 0
    assert (tmp_path / ACTION_PIN_BASELINE_RELATIVE_PATH).is_file()


@pytest.mark.unit
def test_debug_logging_names_skipped_references(tree: Path, caplog: pytest.LogCaptureFixture) -> None:
    _set_workflow(tree, "      - uses: ./local/action")
    with caplog.at_level(logging.DEBUG, logger="src.tools.action_pins"):
        ActionPinAuditor(tree).scan()
    assert any("skipping local/container reference" in record.getMessage() for record in caplog.records)


@pytest.mark.unit
def test_info_logging_states_the_verdict(tree: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="src.tools.action_pins"):
        ActionPinAuditor(tree).audit()
    assert any("action pin audit" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------- the live tree


@pytest.mark.unit
def test_repository_ratchet_holds() -> None:
    report = audit(REPO_ROOT)
    assert report.ok, "\n".join(report.violations)


@pytest.mark.unit
def test_repository_has_at_least_one_workflow_reference() -> None:
    """Guards against the audit passing because it found nothing to audit."""
    assert audit(REPO_ROOT).uses


@pytest.mark.unit
def test_action_use_equality_is_by_value() -> None:
    """The dataclass is frozen so uses can be deduplicated and compared in assertions."""
    first = ActionUse("ci.yml", 3, "actions/checkout", "v4")
    assert first == ActionUse("ci.yml", 3, "actions/checkout", "v4")
    assert len({first, ActionUse("ci.yml", 3, "actions/checkout", "v4")}) == 1
