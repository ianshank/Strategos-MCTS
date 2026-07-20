"""Tests for the deterministic context-doc validator (`src.tools.context_docs`).

Two kinds of test:

- **The real-repo gate** (`test_real_repo_has_no_drift`) runs the validator against this checkout — it
  is what fails CI when a `.claude` doc drifts.
- **Behavioural unit tests** build a *synthetic* repo under `tmp_path` and inject it as the repo root,
  so every branch (path classification, brace expansion, globbing, the value claims, frontmatter, the
  CLI) is exercised in isolation without depending on the live tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools import context_docs as cd

# --------------------------------------------------------------------------- real-repo gate


@pytest.mark.unit
def test_real_repo_has_no_drift():
    failures = cd.validate()
    assert failures == [], "Context-doc drift detected:\n" + "\n".join(f"  - {f}" for f in failures)


# --------------------------------------------------------------------------- brace expansion


@pytest.mark.unit
@pytest.mark.parametrize(
    "token,expected",
    [
        ("plain.py", ["plain.py"]),
        ("x/{a,b}.py", ["x/a.py", "x/b.py"]),
        ("llm/{base,resilience}.py", ["llm/base.py", "llm/resilience.py"]),
        ("x/{a,b}/{c,d}.py", ["x/a/c.py", "x/a/d.py", "x/b/c.py", "x/b/d.py"]),
    ],
)
def test_expand_braces(token, expected):
    assert cd._expand_braces(token) == expected


# --------------------------------------------------------------------------- synthetic-repo helpers

_SKILL = "---\nname: {name}\ndescription: >-\n  A folded description.\n---\n\n# {name}\n\n{body}\n"
_AGENT = "---\nname: {name}\ndescription: An inline description.\ntools: Read, Grep, Glob\n---\n\n{body}\n"

_ENV_FLAGS = (
    "ALLOW_MOCK_LLM_FALLBACK",
    "ALLOW_LIGHTWEIGHT_FRAMEWORK_FALLBACK",
    "ASSEMBLY_TRUST_LEGACY_PICKLE",
    "TRAINING_TRUST_LEGACY_PICKLE",
)
_STATUSES = ("draft", "approved", "implemented", "verified", "superseded")


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_repo(
    tmp_path: Path,
    *,
    fail_under: str = "85.0",
    scripts=("benchmark", "harness", "policy-lift"),
    env_flags=_ENV_FLAGS,
    statuses=_STATUSES,
    symbols=("class Settings", "def get_settings"),
    primer_body: str = "",
    guide_body: str = "",
) -> Path:
    """Build a minimal but internally-consistent repo so `validate()` passes by default."""
    scripts_block = "\n".join(f'{name} = "pkg:main"' for name in scripts)
    _write(
        tmp_path / "pyproject.toml",
        f"[tool.coverage.report]\nfail_under = {fail_under}\n\n[project.scripts]\n{scripts_block}\n",
    )
    _write(
        tmp_path / "src/config/settings.py", "\n".join(symbols) + "\n" + "\n".join(f"# {f}" for f in env_flags) + "\n"
    )
    _write(
        tmp_path / "src/framework/harness/intent/spec_validator.py",
        "SPEC_STATUSES = {" + ", ".join(f'"{s}"' for s in statuses) + "}\n",
    )
    primer = f"fail_under = {fail_under}\n" + "\n".join(env_flags) + "\n" + primer_body
    _write(tmp_path / cd.ContextDocValidator.PRIMER, _SKILL.format(name="strategos-primer", body=primer))
    _write(
        tmp_path / cd.ContextDocValidator.GUIDE,
        _AGENT.format(name="strategos-guide", body=f"fail_under = {fail_under}\n{guide_body}"),
    )
    return tmp_path


@pytest.mark.unit
def test_synthetic_repo_is_clean_by_default(tmp_path):
    assert cd.ContextDocValidator(_make_repo(tmp_path)).validate() == []


# --------------------------------------------------------------------------- path existence


def _paths_of(repo: Path, doc_body: str) -> list[cd.Failure]:
    """Run only the path check over a throwaway skill whose body is `doc_body`."""
    v = cd.ContextDocValidator(repo)
    doc = repo / ".claude/skills/probe/SKILL.md"
    return v.check_paths(doc, doc_body)


@pytest.mark.unit
def test_path_missing_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    assert any("not found" in f.message for f in _paths_of(repo, "See `src/does/not/exist.py`."))


@pytest.mark.unit
def test_rooted_dir_without_trailing_slash_sets_context(tmp_path):
    # C1 regression: `src/pkg` (a real dir, no slash) must set context so `mod.py` resolves under it.
    repo = _make_repo(tmp_path)
    _write(repo / "src/pkg/mod.py")
    assert _paths_of(repo, "`src/pkg` holds `mod.py`") == []


@pytest.mark.unit
def test_bare_directory_resolves_against_context(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo / "src/pkg/sub/x.py")
    assert _paths_of(repo, "`src/pkg/` contains (`sub/`)") == []


@pytest.mark.unit
def test_glob_and_recursive_globstar(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo / "src/pkg/a.py")
    _write(repo / "src/pkg/deep/nested/b.py")
    assert _paths_of(repo, "`src/pkg/*`") == []  # matches a.py
    assert _paths_of(repo, "`src/pkg/**/b.py`") == []  # C5: ** must recurse
    assert any("not found" in f.message for f in _paths_of(repo, "`src/pkg/*.md`"))  # no match


@pytest.mark.unit
def test_unprefixed_path_is_treated_as_prose(tmp_path):
    # The escape hatch: a first segment not in KNOWN_ROOTS is prose, so a "drifted" mention is ignored.
    assert _paths_of(_make_repo(tmp_path), "old layout was `framework/graph.py`") == []


@pytest.mark.unit
def test_case_variant_root_is_still_checked(tmp_path):
    # C2: a mistyped-case root is matched case-insensitively, so the broken path is *not* silently skipped.
    assert any("not found" in f.message for f in _paths_of(_make_repo(tmp_path), "`Src/config/settings.py`"))


@pytest.mark.unit
@pytest.mark.parametrize("span", ["`specs/<id>.SPEC.md`", "`a b/c.py`", "`x/$VAR.py`"])
def test_prose_guard_skips_placeholder_spans(tmp_path, span):
    assert _paths_of(_make_repo(tmp_path), f"see {span}") == []


@pytest.mark.unit
@pytest.mark.parametrize("suffix", [".", ",", ")", "]"])
def test_trailing_punctuation_is_stripped(tmp_path, suffix):
    # C6: `settings.py)` must still resolve.
    repo = _make_repo(tmp_path)
    assert _paths_of(repo, f"see `src/config/settings.py{suffix}`") == []


@pytest.mark.unit
def test_dotfile_suffix_is_skipped(tmp_path):
    assert _paths_of(_make_repo(tmp_path), "the `.SPEC.md` suffix and `.env`") == []


@pytest.mark.unit
def test_intentionally_absent_skip_and_reappearance(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(cd, "INTENTIONALLY_ABSENT", frozenset({"src/gone.py"}))
    assert _paths_of(repo, "gone: `src/gone.py`") == []  # absent → skipped, not flagged
    _write(repo / "src/gone.py")  # T3: it reappeared
    failures = cd.ContextDocValidator(repo).validate()
    assert any("INTENTIONALLY_ABSENT" in f.message for f in failures)


# --------------------------------------------------------------------------- frontmatter


@pytest.mark.unit
def test_frontmatter_name_mismatch(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo / ".claude/skills/probe/SKILL.md", _SKILL.format(name="wrong", body=""))
    v = cd.ContextDocValidator(repo)
    doc = repo / ".claude/skills/probe/SKILL.md"
    assert any("!= expected 'probe'" in f.message for f in v.check_frontmatter(doc, doc.read_text()))


@pytest.mark.unit
def test_agent_missing_tools(tmp_path):
    repo = _make_repo(tmp_path)
    doc = repo / ".claude/agents/x.md"
    _write(doc, "---\nname: x\ndescription: d\n---\nbody\n")
    assert any("'tools'" in f.message for f in cd.ContextDocValidator(repo).check_frontmatter(doc, doc.read_text()))


@pytest.mark.unit
def test_missing_frontmatter_block(tmp_path):
    repo = _make_repo(tmp_path)
    doc = repo / ".claude/skills/probe/SKILL.md"
    assert any(
        "missing YAML frontmatter" in f.message
        for f in cd.ContextDocValidator(repo).check_frontmatter(doc, "no frontmatter")
    )


@pytest.mark.unit
def test_empty_description_flagged_folded_accepted(tmp_path):
    repo = cd.ContextDocValidator(_make_repo(tmp_path))
    empty = "---\nname: probe\ndescription:\n---\nbody\n"
    folded = "---\nname: probe\ndescription: >-\n  Real text.\n---\nbody\n"
    doc = Path(tmp_path) / ".claude/skills/probe/SKILL.md"
    assert any("empty" in f.message for f in repo.check_frontmatter(doc, empty))
    assert repo.check_frontmatter(doc, folded) == []


# --------------------------------------------------------------------------- value claims


@pytest.mark.unit
def test_coverage_gate_drift(tmp_path):
    # Source says 90.0 but the primer still quotes 85.0 → drift on both primer and guide.
    repo = _make_repo(tmp_path)
    (repo / "pyproject.toml").write_text(
        '[tool.coverage.report]\nfail_under = 90.0\n\n[project.scripts]\nbenchmark = "x:m"\nharness = "x:m"\npolicy-lift = "x:m"\n'
    )
    failures = cd._check_coverage_gate(cd.ContextDocValidator(repo))
    assert {f.doc for f in failures} == {cd.ContextDocValidator.PRIMER, cd.ContextDocValidator.GUIDE}


@pytest.mark.unit
def test_missing_console_script_flagged(tmp_path):
    repo = _make_repo(tmp_path, scripts=("benchmark", "harness"))  # policy-lift removed
    assert any("policy-lift" in f.message for f in cd._check_console_scripts(cd.ContextDocValidator(repo)))


@pytest.mark.unit
def test_removed_env_flag_flagged(tmp_path):
    repo = _make_repo(tmp_path, env_flags=_ENV_FLAGS[:3])  # drop one from source + primer
    assert any("TRAINING_TRUST_LEGACY_PICKLE" in f.message for f in cd._check_env_flags(cd.ContextDocValidator(repo)))


@pytest.mark.unit
def test_removed_spec_status_flagged(tmp_path):
    repo = _make_repo(tmp_path, statuses=("draft", "approved", "implemented", "verified"))  # drop superseded
    assert any("superseded" in f.message for f in cd._check_spec_statuses(cd.ContextDocValidator(repo)))


@pytest.mark.unit
def test_removed_symbol_flagged(tmp_path):
    repo = _make_repo(tmp_path, symbols=("class Settings",))  # drop get_settings
    assert any("get_settings" in f.message for f in cd._check_settings_symbols(cd.ContextDocValidator(repo)))


@pytest.mark.unit
def test_moved_source_reports_cleanly(tmp_path):
    # T4: a vanished source file must yield a clean failure, not a stack trace.
    repo = _make_repo(tmp_path)
    (repo / "src/config/settings.py").unlink()
    failures = cd._check_env_flags(cd.ContextDocValidator(repo))
    assert failures and "cannot read" in failures[0].message


# --------------------------------------------------------------------------- driver + CLI


@pytest.mark.unit
def test_empty_repo_returns_sentinel(tmp_path):
    failures = cd.ContextDocValidator(tmp_path).validate()
    assert any("no context docs found" in f.message for f in failures)


@pytest.mark.unit
def test_failure_str_and_rel_outside_repo(tmp_path):
    assert str(cd.Failure("d", "path", "m")) == "[path] d: m"
    # rel() on a path outside the repo falls back to the raw string rather than raising.
    assert cd.ContextDocValidator(tmp_path).rel(Path("/elsewhere/x.md")) == "/elsewhere/x.md"


@pytest.mark.unit
def test_main_exit_codes_and_json(tmp_path, capsys):
    clean = _make_repo(tmp_path / "clean")
    assert cd.main(["--repo-root", str(clean)]) == 0
    assert "OK" in capsys.readouterr().out

    broken = _make_repo(tmp_path / "broken")
    (broken / cd.ContextDocValidator.PRIMER).write_text(
        _SKILL.format(name="strategos-primer", body="see `src/missing.py`")
    )
    assert cd.main(["--repo-root", str(broken), "--json"]) == 1
    assert '"category"' in capsys.readouterr().out


@pytest.mark.unit
def test_main_non_json_failure_path(tmp_path, capsys):
    broken = _make_repo(tmp_path)
    (broken / cd.ContextDocValidator.PRIMER).write_text(
        _SKILL.format(name="strategos-primer", body="see `src/missing.py`")
    )
    assert cd.main(["--repo-root", str(broken)]) == 1
    assert "FAILED" in capsys.readouterr().err


# --------------------------------------------------------------------------- defensive branches


@pytest.mark.unit
def test_bare_path_unresolved_is_flagged(tmp_path):
    assert any("bare path unresolved" in f.message for f in _paths_of(_make_repo(tmp_path), "`missing_bare.py`"))


@pytest.mark.unit
def test_malformed_brace_token_is_skipped(tmp_path):
    # An unbalanced brace never resolves to a real path, so it is dropped rather than flagged.
    assert _paths_of(_make_repo(tmp_path), "`src/{a.py`") == []


@pytest.mark.unit
def test_empty_description_before_next_key_is_flagged(tmp_path):
    v = cd.ContextDocValidator(_make_repo(tmp_path))
    doc = tmp_path / ".claude/skills/probe/SKILL.md"
    text = "---\nname: probe\ndescription:\ntools: Read\n---\nbody\n"
    assert any("empty" in f.message for f in v.check_frontmatter(doc, text))


@pytest.mark.unit
@pytest.mark.parametrize(
    "check,source,needle",
    [
        (cd._check_coverage_gate, "pyproject.toml", "cannot read"),
        (cd._check_console_scripts, "pyproject.toml", "cannot read"),
        (cd._check_spec_statuses, "src/framework/harness/intent/spec_validator.py", "cannot read"),
        (cd._check_settings_symbols, "src/config/settings.py", "cannot read"),
    ],
)
def test_value_claim_reports_missing_source_cleanly(tmp_path, check, source, needle):
    repo = _make_repo(tmp_path)
    (repo / source).unlink()
    failures = check(cd.ContextDocValidator(repo))
    assert failures and needle in failures[0].message


@pytest.mark.unit
def test_coverage_gate_missing_fail_under(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "pyproject.toml").write_text('[project.scripts]\nbenchmark = "x:m"\n')
    failures = cd._check_coverage_gate(cd.ContextDocValidator(repo))
    assert failures and "no coverage `fail_under` found" in failures[0].message
