"""Unit tests for the schema-v2 spec validator (``intent/spec_validator.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.framework.harness.intent import SPEC_STATUSES, SpecValidator
from src.framework.harness.intent.spec_validator import DONE_MARKER_RE

pytestmark = pytest.mark.unit

_VALID_BODY = (
    "\n# Goal\nDo a thing.\n\n" "# Acceptance Criteria\n- AC-1: one\n- AC-2: two\n\n" "# Constraints\n- safe\n"
)


def _write(
    tmp_path: Path,
    spec_id: str = "demo_spec",
    *,
    frontmatter: str | None = None,
    body: str = _VALID_BODY,
    filename: str | None = None,
) -> Path:
    front = (
        frontmatter if frontmatter is not None else (f"id: {spec_id}\ngoal: Do a thing\nmodule: src/\nstatus: approved")
    )
    path = tmp_path / (filename or f"{spec_id}.SPEC.md")
    path.write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")
    return path


def _errors(issues: list) -> set[str]:
    return {i.code for i in issues if i.severity == "error"}


def _warnings(issues: list) -> set[str]:
    return {i.code for i in issues if i.severity == "warning"}


def _paths_issues(paths: list[Path]) -> list:
    """Issues from a multi-path validation (report form)."""
    return SpecValidator().validate_paths(paths).issues


def test_valid_v2_spec_has_no_issues(tmp_path: Path) -> None:
    issues = SpecValidator().validate_file(_write(tmp_path))
    assert issues == []


def test_missing_id_errors(tmp_path: Path) -> None:
    spec = _write(tmp_path, frontmatter="goal: g\nmodule: src/\nstatus: draft")
    assert "missing-id" in _errors(SpecValidator().validate_file(spec))


def test_missing_status_errors(tmp_path: Path) -> None:
    spec = _write(tmp_path, frontmatter="id: demo_spec\ngoal: g\nmodule: src/")
    assert "missing-status" in _errors(SpecValidator().validate_file(spec))


def test_missing_goal_errors(tmp_path: Path) -> None:
    spec = _write(
        tmp_path,
        frontmatter="id: demo_spec\nmodule: src/\nstatus: draft",
        body="\n# Acceptance Criteria\n- AC-1: one\n",
    )
    assert "missing-goal" in _errors(SpecValidator().validate_file(spec))


def test_no_criteria_errors(tmp_path: Path) -> None:
    spec = _write(tmp_path, body="\n# Goal\nDo a thing.\n\n# Acceptance Criteria\n\n")
    assert "no-criteria" in _errors(SpecValidator().validate_file(spec))


@pytest.mark.parametrize("status", ["active", "bogus", "Implemented"])
def test_unknown_status_errors(tmp_path: Path, status: str) -> None:
    """The vocabulary is closed and case-sensitive; the message lists valid values."""
    spec = _write(tmp_path, frontmatter=f"id: demo_spec\ngoal: g\nmodule: src/\nstatus: {status}")
    issues = SpecValidator().validate_file(spec)
    assert "unknown-status" in _errors(issues)
    message = next(i.message for i in issues if i.code == "unknown-status")
    assert all(s in message for s in sorted(SPEC_STATUSES))


def test_filename_id_mismatch_errors(tmp_path: Path) -> None:
    spec = _write(tmp_path, filename="other_name.SPEC.md")
    assert "filename-id-mismatch" in _errors(SpecValidator().validate_file(spec))


def test_filename_without_spec_suffix_errors(tmp_path: Path) -> None:
    """A file not named ``<id>.SPEC.md`` fails the filename rule whenever id is present."""
    spec = _write(tmp_path, filename="demo_spec.md")
    assert "filename-id-mismatch" in _errors(SpecValidator().validate_file(spec))


def test_duplicate_exact_header_errors(tmp_path: Path) -> None:
    spec = _write(tmp_path, body=_VALID_BODY + "\n# Goal\nAgain.\n")
    assert "duplicate-section" in _errors(SpecValidator().validate_file(spec))


def test_alias_colliding_headers_error(tmp_path: Path) -> None:
    """Two distinct headers matching one alias group is ambiguous — the parser picks one."""
    spec = _write(tmp_path, body=_VALID_BODY + "\n# Criteria\n- AC-3: three\n")
    assert "duplicate-section" in _errors(SpecValidator().validate_file(spec))


def test_alias_suffix_header_is_not_a_collision(tmp_path: Path) -> None:
    """A single prefix-extended header (phase_8 style) matches its group exactly once."""
    body = (
        "\n# Goal\nDo a thing.\n\n"
        "# Acceptance Criteria\n- AC-1: one\n\n"
        "# Constraints (why the physical moves were rejected)\n- safe\n"
    )
    issues = SpecValidator().validate_file(_write(tmp_path, body=body))
    assert "duplicate-section" not in _errors(issues)


def test_frontmatter_comment_is_not_a_header(tmp_path: Path) -> None:
    """Frontmatter ``#`` comment lines must not be counted by the header walk."""
    spec = _write(
        tmp_path,
        frontmatter="# goal comment\nid: demo_spec\ngoal: g\nmodule: src/\nstatus: draft",
    )
    assert "duplicate-section" not in _errors(SpecValidator().validate_file(spec))


def test_done_marker_errors_with_line_number(tmp_path: Path) -> None:
    body = _VALID_BODY.replace("- AC-1: one", "- AC-1: **(8a — done)** one")
    issues = SpecValidator().validate_file(_write(tmp_path, body=body))
    assert "done-marker" in _errors(issues)
    message = next(i.message for i in issues if i.code == "done-marker")
    assert "line" in message


def test_harness_done_comment_not_flagged(tmp_path: Path) -> None:
    """Ralph's completion marker is a legitimate mechanism, not a changelog marker."""
    body = _VALID_BODY + "\n<!-- HARNESS:DONE -->\n"
    issues = SpecValidator().validate_file(_write(tmp_path, body=body))
    assert "done-marker" not in _errors(issues)


def test_done_marker_regex_ignores_overdone_class() -> None:
    assert DONE_MARKER_RE.search("**(8a — done)**")
    assert DONE_MARKER_RE.search("**(Done)**")
    assert not DONE_MARKER_RE.search("**(overdone)**")
    assert not DONE_MARKER_RE.search("<!-- HARNESS:DONE -->")


def test_mixed_criterion_ids_error(tmp_path: Path) -> None:
    body = _VALID_BODY.replace("- AC-2: two", "- two")
    assert "mixed-criterion-ids" in _errors(SpecValidator().validate_file(_write(tmp_path, body=body)))


def test_duplicate_criterion_id_error(tmp_path: Path) -> None:
    body = _VALID_BODY.replace("- AC-2: two", "- AC-1: two")
    assert "duplicate-criterion-id" in _errors(SpecValidator().validate_file(_write(tmp_path, body=body)))


def test_ac_prefix_without_space_falls_back_positional(tmp_path: Path) -> None:
    """``AC-1:text`` is not an authored ID; all-positional bullets warn, not error."""
    body = "\n# Goal\nDo a thing.\n\n# Acceptance Criteria\n- AC-1:one\n- AC-2:two\n"
    issues = SpecValidator().validate_file(_write(tmp_path, body=body))
    assert _errors(issues) == set()
    assert "positional-criterion-ids" in _warnings(issues)


def test_missing_module_warns(tmp_path: Path) -> None:
    spec = _write(tmp_path, frontmatter="id: demo_spec\ngoal: g\nstatus: draft")
    issues = SpecValidator().validate_file(spec)
    assert _errors(issues) == set()
    assert "missing-module" in _warnings(issues)


def test_all_positional_ids_warn(tmp_path: Path) -> None:
    body = "\n# Goal\nDo a thing.\n\n# Acceptance Criteria\n- one\n- two\n"
    issues = SpecValidator().validate_file(_write(tmp_path, body=body))
    assert _errors(issues) == set()
    assert "positional-criterion-ids" in _warnings(issues)


def test_validate_paths_report_carries_parsed_specs(tmp_path: Path) -> None:
    """The report exposes parsed specs so callers (the CLI) don't re-load files."""
    spec = _write(tmp_path)
    report = SpecValidator().validate_paths([spec])
    assert report.errors() == []
    assert report.specs[str(spec)].id == "demo_spec"
    assert report.specs[str(spec)].status == "approved"


def test_spec_loader_is_constructor_injected(tmp_path: Path) -> None:
    """A substitute loader passed via __init__ is actually used (DI guideline)."""
    from src.framework.harness.intent import SpecLoader

    class CountingLoader(SpecLoader):
        calls = 0

        def load(self, path: Path):  # type: ignore[override]
            CountingLoader.calls += 1
            return super().load(path)

    spec = _write(tmp_path)
    SpecValidator(spec_loader=CountingLoader()).validate_file(spec)
    assert CountingLoader.calls == 1


def test_duplicate_id_across_files_errors(tmp_path: Path) -> None:
    """Same spec id in two files (necessarily different directories, per the filename rule)."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    issues = _paths_issues([_write(a_dir), _write(b_dir)])
    assert "duplicate-id" in _errors(issues)


def test_duplicate_ac1_across_files_is_clean(tmp_path: Path) -> None:
    """Criterion IDs are file-scoped: every spec may have its own AC-1."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    issues = _paths_issues([_write(a_dir, "spec_a"), _write(b_dir, "spec_b")])
    assert _errors(issues) == set()


def test_missing_file_yields_error_issue(tmp_path: Path) -> None:
    issues = _paths_issues([tmp_path / "absent.SPEC.md"])
    assert "parse-error" in _errors(issues)


def test_binary_file_yields_error_not_traceback(tmp_path: Path) -> None:
    """Non-UTF-8 bytes become an 'unreadable' issue (UnicodeDecodeError is a ValueError)."""
    path = tmp_path / "binary.SPEC.md"
    path.write_bytes(b"\xff\xfe\x00\x01 not utf-8")
    issues = SpecValidator().validate_file(path)
    assert "unreadable" in _errors(issues)


def test_same_file_via_two_spellings_is_not_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Paths are resolved before duplicate-id comparison, so one file ≠ two declarations."""
    spec = _write(tmp_path)
    monkeypatch.chdir(tmp_path)
    issues = _paths_issues([Path(spec.name), Path(".") / spec.name])
    assert "duplicate-id" not in _errors(issues)


def test_fenced_code_comments_are_not_headers(tmp_path: Path) -> None:
    """``#`` comments inside ``` fences must not trip duplicate-section checks."""
    body = _VALID_BODY + "\n```bash\n# Goal\n# Goal\n```\n"
    issues = SpecValidator().validate_file(_write(tmp_path, body=body))
    assert "duplicate-section" not in _errors(issues)


def test_directory_yields_error_not_traceback(tmp_path: Path) -> None:
    """A directory (e.g. from a shell glob) becomes an error issue, never an exception."""
    target = tmp_path / "somedir"
    target.mkdir()
    issues = _paths_issues([target])
    assert "unreadable" in _errors(issues)


def test_crlf_file_validates_cleanly(tmp_path: Path) -> None:
    """CRLF files are safe: ``Path.read_text`` applies universal-newline translation,
    so the ``---\\n`` frontmatter delimiter matches and v2 fields populate normally."""
    path = tmp_path / "crlf.SPEC.md"
    text = "---\nid: crlf\ngoal: g\nmodule: src/\nstatus: draft\n---\n# Goal\ng\n# Acceptance Criteria\n- AC-1: x\n"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    issues = SpecValidator().validate_file(path)
    assert _errors(issues) == set()


def test_repo_specs_all_validate_clean() -> None:
    """The migrated specs/ directory must always pass — pins the same-PR migration invariant."""
    specs_dir = Path(__file__).parents[4] / "specs"
    if not specs_dir.is_dir():
        pytest.skip("specs/ not present in this checkout")
    paths = sorted(specs_dir.glob("*.SPEC.md"))
    assert paths, "expected specs/*.SPEC.md to exist"
    issues = _paths_issues(paths)
    assert _errors(issues) == set(), [i.render() for i in issues]
