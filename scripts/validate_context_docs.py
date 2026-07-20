#!/usr/bin/env python3
"""Deterministic validation of the repo's Claude context docs.

Checks every ``.claude/skills/**/SKILL.md`` and ``.claude/agents/*.md`` — the human-written
orientation docs (e.g. ``strategos-primer``, ``strategos-guide``) — against the real repository.
It exists because those docs make concrete, checkable claims (file paths, the coverage gate,
console-script names, env-var flags) that silently drift as the code moves. An independent review
once caught exactly this by hand — a doc citing a module that had moved; this makes the check
mechanical and repeatable so drift fails fast instead of surviving until someone notices.

Pure filesystem + regex. No network, no LLM, no imports from ``src`` — deterministic: same tree in,
same verdict out. Three layers:

  1. Frontmatter schema — every doc has the required keys (skills: name/description; agents also tools).
  2. Path existence     — every repo path a doc cites in backticks resolves on disk.
  3. Pinned value claims — coverage gate, console scripts, env flags, spec statuses, and key symbols
                           match their real sources.

Run standalone (exit 1 on any failure)::

    python scripts/validate_context_docs.py

It is also wrapped by ``tests/unit/docs/test_context_docs.py`` so the same checks gate CI.
"""

from __future__ import annotations

import glob as globmod
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Top-level directories that mark a token as a repo-rooted path claim. A path whose first segment is
# not one of these (e.g. a bare ``framework/graph.py`` written in prose to say "this drifted") is NOT
# treated as an existence claim — only fully-qualified paths are checked, which keeps the validator
# from flagging the docs' own deliberate references to paths that no longer exist.
KNOWN_ROOTS = {"src", "tests", "docs", "specs", "planning", "kubernetes", "scripts", ".claude", ".github"}

# Extensions that make a *bare* (slash-free) backtick token look like a filename worth resolving.
PATH_EXTS = (".py", ".md", ".toml", ".yaml", ".yml", ".cfg", ".json", ".txt", ".sh")

# Fully-qualified paths the docs intentionally cite as *gone* (to explain drift). Excluded from the
# existence check on purpose; keep this list tiny and commented.
INTENTIONALLY_ABSENT = {
    "src/framework/graph.py",  # primer notes CLAUDE.md still lists this; it is now the graph/ package
}

_BACKTICK = re.compile(r"`([^`]+)`")
_BRACE = re.compile(r"^(.*?)\{([^{}]*)\}(.*)$")


def iter_context_docs() -> list[Path]:
    """All skill and agent docs, sorted for stable output."""
    skills = sorted((REPO / ".claude" / "skills").glob("*/SKILL.md"))
    agents = sorted((REPO / ".claude" / "agents").glob("*.md"))
    return skills + agents


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)  # synthetic paths (e.g. from tests) live outside the repo root


# --------------------------------------------------------------------------- frontmatter


def _frontmatter(text: str) -> dict[str, str] | None:
    """Return a shallow {key: first-line-value} map for the YAML frontmatter block, or None.

    Deliberately lightweight (no YAML dependency): we only need presence + non-emptiness of a few
    top-level keys, and a folded ``>-`` description counts as present if any indented body follows.
    """
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        km = re.match(r"^([A-Za-z_][\w-]*):(.*)$", line)
        if km:
            fm[km.group(1)] = km.group(2).strip()
    return fm


def check_frontmatter(doc: Path, text: str) -> list[str]:
    fails: list[str] = []
    fm = _frontmatter(text)
    if fm is None:
        return [f"{_rel(doc)}: missing YAML frontmatter block"]
    is_agent = doc.parent.name == "agents"
    expected_name = doc.stem if is_agent else doc.parent.name
    required = ["name", "description", "tools"] if is_agent else ["name", "description"]
    for key in required:
        if key not in fm:
            fails.append(f"{_rel(doc)}: frontmatter missing required key '{key}'")
    if fm.get("name") and fm["name"] != expected_name:
        fails.append(f"{_rel(doc)}: frontmatter name '{fm['name']}' != expected '{expected_name}'")
    # A folded scalar leaves an empty value on the key line; treat that as present (body follows).
    if "description" in fm and not fm["description"] and ">" not in text.split("description:", 1)[1][:3]:
        fails.append(f"{_rel(doc)}: frontmatter 'description' is empty")
    return fails


# --------------------------------------------------------------------------- path existence


def _expand_braces(token: str) -> list[str]:
    """Expand a single ``{a,b,c}`` group (recursively) into concrete tokens."""
    m = _BRACE.match(token)
    if not m:
        return [token]
    pre, inner, post = m.groups()
    out: list[str] = []
    for part in inner.split(","):
        out.extend(_expand_braces(f"{pre}{part.strip()}{post}"))
    return out


def _exists(relpath: str) -> bool:
    if "*" in relpath:
        return bool(globmod.glob(str(REPO / relpath)))
    return (REPO / relpath).exists()


def _is_rooted_path(token: str) -> bool:
    return "/" in token and token.split("/", 1)[0] in KNOWN_ROOTS


def _is_bare_pathish(token: str) -> bool:
    """A slash-free filename (``state.py``) or bare directory (``loop/``) to resolve contextually."""
    if token.startswith("."):
        return False  # ``.SPEC.md`` / ``.env`` — a suffix or dotfile, not a contextual path claim
    if token.endswith("/"):
        return "/" not in token[:-1]
    return "/" not in token and token.endswith(PATH_EXTS)


def check_paths(doc: Path, text: str) -> list[str]:
    fails: list[str] = []
    for line in text.splitlines():
        current: str | None = None  # dir context for bare filenames, scoped to this line
        for span in _BACKTICK.findall(line):
            span = span.strip()
            if " " in span or "<" in span or ">" in span or "$" in span:
                continue  # multi-word spans / placeholders are prose, not a single path claim
            for token in _expand_braces(span):
                token = token.strip().rstrip(".,;:")
                if not token:
                    continue
                if _is_rooted_path(token):
                    if token not in INTENTIONALLY_ABSENT and not _exists(token):
                        fails.append(f"{_rel(doc)}: path not found: `{token}`")
                    current = token.rstrip("/") if token.endswith("/") else str(Path(token).parent)
                elif _is_bare_pathish(token):
                    candidates = ([f"{current}/{token}"] if current else []) + [token]
                    if not any(_exists(c) for c in candidates):
                        fails.append(f"{_rel(doc)}: bare path unresolved: `{token}` (tried {candidates})")
    return fails


# --------------------------------------------------------------------------- pinned value claims


def _scripts_block() -> set[str]:
    text = _read(REPO / "pyproject.toml")
    block = re.search(r"\[project\.scripts\](.*?)(?:\n\[|\Z)", text, re.S)
    if not block:
        return set()
    return set(re.findall(r"^([A-Za-z][\w-]*)\s*=", block.group(1), re.M))


def check_value_claims() -> list[str]:
    """Assert the primer's concrete claims still match their sources (and appear in the primer)."""
    fails: list[str] = []
    primer = REPO / ".claude" / "skills" / "strategos-primer" / "SKILL.md"
    if not primer.exists():
        return [f"{_rel(primer)}: primer skill missing"]
    doc = _read(primer)

    def want(literal: str, where: str) -> None:
        if literal not in doc:
            fails.append(f"primer no longer states {where}: expected `{literal}`")

    # Coverage gate — derive the number from pyproject; the primer must quote it.
    pyproject = _read(REPO / "pyproject.toml")
    m = re.search(r"fail_under\s*=\s*([0-9.]+)", pyproject)
    if not m:
        fails.append("pyproject.toml: no `fail_under` found for coverage gate")
    else:
        want(f"fail_under = {m.group(1)}", "the coverage gate")

    # Console scripts — every name the primer advertises must be a real entry point.
    scripts = _scripts_block()
    for name in ("benchmark", "harness", "policy-lift"):
        if name not in scripts:
            fails.append(f"pyproject [project.scripts] no longer defines `{name}`")
        want(name, f"console script `{name}`")

    # Env flags — opt-in fallback / legacy-pickle switches must still exist in settings.
    settings = _read(REPO / "src" / "config" / "settings.py")
    for flag in (
        "ALLOW_MOCK_LLM_FALLBACK",
        "ALLOW_LIGHTWEIGHT_FRAMEWORK_FALLBACK",
        "ASSEMBLY_TRUST_LEGACY_PICKLE",
        "TRAINING_TRUST_LEGACY_PICKLE",
    ):
        if flag not in settings:
            fails.append(f"src/config/settings.py no longer defines env flag `{flag}`")
        want(flag, f"env flag `{flag}`")

    # Spec lifecycle — the five statuses the primer lists must match SPEC_STATUSES.
    validator = _read(REPO / "src" / "framework" / "harness" / "intent" / "spec_validator.py")
    for status in ("draft", "approved", "implemented", "verified", "superseded"):
        if f'"{status}"' not in validator:
            fails.append(f"spec_validator.py SPEC_STATUSES no longer includes `{status}`")
        want(status, f"spec status `{status}`")

    # Key symbols the primer/guide reference.
    for symbol in ("class Settings", "def get_settings"):
        if symbol not in settings:
            fails.append(f"src/config/settings.py no longer defines `{symbol}`")

    return fails


# --------------------------------------------------------------------------- driver


def run() -> list[str]:
    """Return every validation failure (empty list == all good)."""
    failures: list[str] = []
    docs = iter_context_docs()
    if not docs:
        return ["no .claude context docs found — expected skills/agents to validate"]
    for doc in docs:
        text = _read(doc)
        failures += check_frontmatter(doc, text)
        failures += check_paths(doc, text)
    failures += check_value_claims()
    return failures


def main() -> int:
    failures = run()
    if failures:
        print(f"Context-doc validation FAILED ({len(failures)} issue(s)):\n", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    n = len(iter_context_docs())
    print(f"Context-doc validation OK — {n} doc(s) checked, all paths and value claims verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
