"""Deterministic validation of the Claude context docs, run as part of the unit suite.

Wraps ``scripts/validate_context_docs.py`` so drift in ``.claude/skills/**`` or ``.claude/agents/*``
— a cited path that moved, a value claim that no longer matches source — fails in CI instead of
surviving until a human notices. The two "teeth" tests prove the validator actually reports
problems, so a green ``test_context_docs_have_no_drift`` means something.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_SCRIPT = REPO / "scripts" / "validate_context_docs.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_context_docs", _SCRIPT)
    assert spec and spec.loader, f"cannot load validator at {_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


@pytest.mark.unit
def test_context_docs_have_no_drift():
    """Every skill/agent doc's paths and pinned value claims still match the repo."""
    failures = validator.run()
    assert failures == [], "Context-doc validation failed:\n" + "\n".join(f"  - {f}" for f in failures)


@pytest.mark.unit
def test_validator_flags_a_missing_path(tmp_path):
    """A doc citing a path that does not exist must be reported (the check has teeth)."""
    doc = tmp_path / "SKILL.md"
    failures = validator.check_paths(doc, "See `src/definitely/not/here.py` for details.")
    assert any("not found" in f for f in failures), failures


@pytest.mark.unit
def test_validator_flags_missing_frontmatter_key(tmp_path):
    """An agent doc missing the required ``tools`` key must be reported."""
    doc = tmp_path / "agents" / "example.md"
    text = "---\nname: example\ndescription: a description\n---\nbody\n"
    failures = validator.check_frontmatter(doc, text)
    assert any("tools" in f for f in failures), failures
