"""Unit tests for spec scaffolding (``spec-new`` refusal rules + template)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.framework.harness.intent import SpecLoader, SpecValidator
from src.framework.harness.intent.spec_scaffold import (
    DEFAULT_GOAL,
    SpecScaffoldError,
    modules_overlap,
    normalize_module,
    scaffold_spec,
)

pytestmark = [pytest.mark.unit, pytest.mark.harness]


def _write_existing(specs_dir: Path, spec_id: str, module: str, status: str) -> Path:
    specs_dir.mkdir(parents=True, exist_ok=True)
    path = specs_dir / f"{spec_id}.SPEC.md"
    path.write_text(
        f"---\nid: {spec_id}\ngoal: g\nmodule: {module}\nstatus: {status}\n---\n\n"
        "# Goal\ng\n\n# Acceptance Criteria\n- AC-1: x\n",
        encoding="utf-8",
    )
    return path


def test_scaffold_creates_valid_draft(tmp_path: Path) -> None:
    """The template passes the schema-v2 validator with zero errors."""
    path = scaffold_spec(tmp_path / "specs", "my_spec", "src/api/", goal="Ship the thing")
    assert path.name == "my_spec.SPEC.md"
    spec = SpecLoader().load(path)
    assert spec.status == "draft"
    assert spec.module == "src/api/"
    assert spec.goal == "Ship the thing"
    errors = [i for i in SpecValidator().validate_file(path) if i.severity == "error"]
    assert errors == []


def test_scaffold_default_goal_when_omitted(tmp_path: Path) -> None:
    """missing-goal is a validator error, so the scaffold writes a placeholder."""
    path = scaffold_spec(tmp_path / "specs", "my_spec", "src/api/")
    assert SpecLoader().load(path).goal == DEFAULT_GOAL


@pytest.mark.parametrize("bad_id", ["MySpec", "my-spec", "x..y", "spec/x", "a b", ""])
def test_invalid_id_refused(tmp_path: Path, bad_id: str) -> None:
    with pytest.raises(SpecScaffoldError, match="invalid spec id"):
        scaffold_spec(tmp_path / "specs", bad_id, "src/")


def test_empty_module_refused(tmp_path: Path) -> None:
    with pytest.raises(SpecScaffoldError, match="module"):
        scaffold_spec(tmp_path / "specs", "my_spec", "   ")


@pytest.mark.parametrize("bad_module", ["/etc/passwd", "../evil/", "src/../../etc/"])
def test_traversal_module_refused(tmp_path: Path, bad_module: str) -> None:
    with pytest.raises(SpecScaffoldError, match="repo-relative"):
        scaffold_spec(tmp_path / "specs", "my_spec", bad_module)


def test_frontmatter_injection_via_module_refused(tmp_path: Path) -> None:
    """A module smuggling 'status: approved' must never scaffold a pre-approved spec."""
    with pytest.raises(SpecScaffoldError, match="single line"):
        scaffold_spec(tmp_path / "specs", "my_spec", "fake/\nstatus: approved\n---\n\nX")
    assert not (tmp_path / "specs" / "my_spec.SPEC.md").exists()


def test_frontmatter_delimiter_in_module_refused(tmp_path: Path) -> None:
    with pytest.raises(SpecScaffoldError, match="'---'"):
        scaffold_spec(tmp_path / "specs", "my_spec", "src/---x/")


def test_frontmatter_injection_via_goal_refused(tmp_path: Path) -> None:
    with pytest.raises(SpecScaffoldError, match="single line"):
        scaffold_spec(tmp_path / "specs", "my_spec", "src/", goal="g\nstatus: approved")


def test_unreadable_existing_spec_refuses_not_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed: an unparseable candidate could hide a genuine module collision."""
    from src.framework.harness.intent import spec_loader

    specs = tmp_path / "specs"
    _write_existing(specs, "broken_spec", "docs/", "draft")

    def _boom(self: spec_loader.SpecLoader, path: Path) -> spec_loader.Spec:
        raise spec_loader.SpecParseError(f"unreadable: {path}")

    monkeypatch.setattr(spec_loader.SpecLoader, "load", _boom)
    with pytest.raises(SpecScaffoldError, match="cannot check module overlap"):
        scaffold_spec(specs, "my_spec", "src/")
    assert not (specs / "my_spec.SPEC.md").exists()


def test_existing_file_refused(tmp_path: Path) -> None:
    specs = tmp_path / "specs"
    _write_existing(specs, "my_spec", "docs/", "implemented")
    with pytest.raises(SpecScaffoldError, match="already exists"):
        scaffold_spec(specs, "my_spec", "src/")


@pytest.mark.parametrize(
    ("existing", "new", "blocks"),
    [
        ("src/", "src/", True),
        ("src/", "src/api/", True),
        ("src/api/", "src/", True),
        ("src", "src2/", False),
        ("docs/", "src/", False),
        ("./src/", "src", True),
    ],
)
def test_overlap_matrix(tmp_path: Path, existing: str, new: str, blocks: bool) -> None:
    """Segment-wise, either-direction prefix overlap; ``src`` must not overlap ``src2/``."""
    specs = tmp_path / "specs"
    _write_existing(specs, "open_spec", existing, "draft")
    if blocks:
        with pytest.raises(SpecScaffoldError, match="overlaps open spec 'open_spec'"):
            scaffold_spec(specs, "my_spec", new)
    else:
        assert scaffold_spec(specs, "my_spec", new).exists()


@pytest.mark.parametrize(
    ("status", "blocks"),
    [("draft", True), ("approved", True), ("implemented", False), ("verified", False), ("superseded", False)],
)
def test_only_open_statuses_block(tmp_path: Path, status: str, blocks: bool) -> None:
    specs = tmp_path / "specs"
    _write_existing(specs, "other_spec", "src/", status)
    if blocks:
        with pytest.raises(SpecScaffoldError, match="overlaps open spec"):
            scaffold_spec(specs, "my_spec", "src/api/")
    else:
        assert scaffold_spec(specs, "my_spec", "src/api/").exists()


def test_refusal_leaves_nothing_on_disk(tmp_path: Path) -> None:
    specs = tmp_path / "specs"
    _write_existing(specs, "open_spec", "src/", "approved")
    with pytest.raises(SpecScaffoldError):
        scaffold_spec(specs, "my_spec", "src/")
    assert not (specs / "my_spec.SPEC.md").exists()


def test_normalize_module_forms() -> None:
    assert normalize_module("src") == "src/"
    assert normalize_module("./src/api") == "src/api/"
    assert normalize_module("src/api/") == "src/api/"


def test_modules_overlap_is_symmetric() -> None:
    assert modules_overlap("src/", "src/api/")
    assert modules_overlap("src/api/", "src/")
    assert not modules_overlap("src", "src2")
