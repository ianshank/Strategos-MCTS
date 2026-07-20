"""Deterministic validation of the repository's Claude *context docs*.

The orientation docs under ``.claude`` — every ``skills/**/SKILL.md`` and ``agents/*.md`` (e.g.
``strategos-primer``, ``strategos-guide``) — make concrete, checkable claims: file paths, the
coverage gate, console-script names, env-var flags, spec statuses. Those drift silently as code
moves. This module makes the check mechanical so drift fails fast instead of surviving until a human
notices it.

Design notes:

- **Deterministic and dependency-light.** Pure filesystem + regex; no network, no LLM, and no imports
  from the rest of ``src`` — the same tree in always yields the same verdict, and the validator runs
  in a minimal environment (e.g. before heavy deps are installed). It deliberately uses the stdlib
  ``logging`` module rather than ``src.observability`` — there are no secrets or requests here, so the
  house correlation-id/sanitisation convention does not apply, and keeping it dependency-free is the
  point. Do not "fix" this to import the observability stack.
- **Reusable.** The engine is a class parameterised by ``repo_root`` (injected for tests, relocatable
  in principle), it returns structured :class:`Failure` objects rather than pre-formatted strings, and
  the pinned "value claims" are a declarative registry (:data:`VALUE_CLAIMS`) so adding a new claim is
  data, not a control-flow edit.

Three layers, in :meth:`ContextDocValidator.validate`:

1. **Frontmatter schema** — every doc has the required keys (skills: ``name``/``description``; agents
   also ``tools``) and ``name`` matches the file/dir.
2. **Path existence** — every backticked repo path a doc cites resolves on disk. Brace groups expand;
   a bare filename resolves against the nearest directory cited earlier on the same line; only
   *rooted* paths (first segment in :data:`KNOWN_ROOTS`) are treated as existence claims, so a doc can
   still discuss a path that intentionally no longer exists by writing it unprefixed.
3. **Pinned value claims** — the coverage gate, console scripts, env flags, spec statuses and key
   symbols still exist in their sources (``pyproject.toml`` / ``src/config/settings.py`` /
   ``spec_validator.py``); where a claim's literal is unambiguous (the coverage ``fail_under`` value
   and the env-flag names) the primer/guide are additionally required to still quote it.

Run standalone (exit 1 on any failure)::

    python -m src.tools.context_docs            # or the `validate-context-docs` console script
    python -m src.tools.context_docs --verbose  # -v/--debug trace; --json for machine output

Wrapped by ``tests/unit/tools/test_context_docs.py`` so drift fails CI.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Failure", "ContextDocValidator", "validate", "main"]

logger = logging.getLogger(__name__)

# This module lives at <repo>/src/tools/context_docs.py.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

#: First path segments that mark a backticked token as a repo-rooted *existence claim*. Matched
#: case-insensitively so a mistyped case (``Src/...``) is still checked rather than silently ignored.
#: A path whose first segment is not one of these (e.g. a bare ``framework/graph.py`` in prose) is
#: treated as prose — this is the escape hatch for discussing paths that intentionally no longer exist.
KNOWN_ROOTS = frozenset({"src", "tests", "docs", "specs", "planning", "kubernetes", "scripts", ".claude", ".github"})

#: Extensions that make a slash-free token look like a filename worth resolving contextually.
PATH_EXTENSIONS = (".py", ".md", ".toml", ".yaml", ".yml", ".cfg", ".json", ".txt", ".sh")

#: Fully-qualified paths the docs intentionally cite as *gone* (to explain drift). Excluded from the
#: existence check — but a validator run *fails* if one reappears (see :func:`_check_absent_paths`), so
#: the allowlist can never become a silent permanent blind spot.
INTENTIONALLY_ABSENT = frozenset({"src/framework/graph.py"})

_BACKTICK = re.compile(r"`([^`]+)`")
_ONE_BRACE = re.compile(r"^(.*?)\{([^{}]*)\}(.*)$")
_FRONTMATTER = re.compile(r"^﻿?\s*---\n(.*?)\n---\n", re.S)
_KEY_LINE = re.compile(r"^([A-Za-z_][\w-]*):(.*)$")
_GLOB_METACHARS = "*?["
_TRAILING_PUNCT = ".,;:)]}'\""


@dataclass(frozen=True)
class Failure:
    """One validation problem. ``category`` is one of ``frontmatter`` | ``path`` | ``value-claim``."""

    doc: str
    category: str
    message: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.doc}: {self.message}"


def _expand_braces(token: str) -> list[str]:
    """Expand brace groups (``{a,b}`` and sequential ``{a,b}/{c,d}``) into concrete tokens.

    Nested braces are best-effort: the innermost group expands first, which yields the correct set for
    the flat cases the docs use. Any token still containing a brace after expansion is dropped by the
    caller rather than treated as a literal path.
    """
    match = _ONE_BRACE.match(token)
    if not match:
        return [token]
    pre, inner, post = match.group(1), match.group(2), match.group(3)
    out: list[str] = []
    for part in inner.split(","):
        out.extend(_expand_braces(f"{pre}{part.strip()}{post}"))
    return out


def _strip_trailing_punct(token: str) -> str:
    """Strip trailing prose punctuation, but keep a ``]`` that closes a glob character class.

    ``settings.py)`` -> ``settings.py``; ``src/pkg/*[ab]`` is left intact so the ``[]`` glob still works.
    """
    while token and token[-1] in _TRAILING_PUNCT:
        if token[-1] == "]" and "[" in token[:-1]:
            break
        token = token[:-1]
    return token


class ContextDocValidator:
    """Validate the Claude context docs under a given ``repo_root`` (defaults to this repo)."""

    #: Docs that state the pinned value claims, checked by the :data:`VALUE_CLAIMS` registry.
    PRIMER = ".claude/skills/strategos-primer/SKILL.md"
    GUIDE = ".claude/agents/strategos-guide.md"

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self.repo = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT

    # -- filesystem helpers -------------------------------------------------------------------

    def exists(self, relpath: str) -> bool:
        """True if ``relpath`` resolves under the repo. Globs (``*``/``**``/``?``/``[]``) match ≥1 path.

        Case-sensitive even on case-insensitive filesystems (Windows NTFS):
        the token ``Src/config/settings.py`` must *not* match the real
        ``src/config/settings.py``, so that case-drifted mentions are
        flagged rather than silently accepted.
        """
        if any(ch in relpath for ch in _GLOB_METACHARS):
            # NOTE: glob.glob is case-insensitive on Windows NTFS; this may
            # match case-drifted paths.  Acceptable: CI runs on Linux and
            # the non-glob path below enforces case sensitivity.
            return bool(_glob.glob(str(self.repo / relpath), recursive=True))
        candidate = self.repo / relpath
        if not candidate.exists():
            return False
        # On case-insensitive filesystems (Windows NTFS), Path.exists() returns
        # True even for case-mismatched paths.  Enforce case-sensitivity by
        # comparing the relpath parts against the actual on-disk path parts.
        try:
            resolved = candidate.resolve()
            repo_resolved = self.repo.resolve()
            # Strip any trailing slash and split into parts.
            norm = relpath.rstrip("/\\").replace("\\", "/")
            rel_parts = tuple(norm.split("/")) if norm else ()
            resolved_rel = resolved.parts[len(repo_resolved.parts) :]
            return rel_parts == resolved_rel
        except OSError:
            return True  # Cannot resolve — assume original check was correct.

    def read(self, relpath: str) -> str:
        return (self.repo / relpath).read_text(encoding="utf-8")

    def try_read(self, relpath: str) -> str | None:
        """Read a source file, or ``None`` if it has moved/vanished (reported as a clean failure)."""
        try:
            return self.read(relpath)
        except OSError:
            return None

    def rel(self, path: Path) -> str:
        """Return a POSIX-style path relative to the repo root.

        Always uses forward slashes regardless of platform so that
        ``Failure`` strings are consistent across Windows and POSIX.
        Falls back to the raw path string (with forward slashes) when
        ``path`` is outside the repo.
        """
        try:
            return path.resolve().relative_to(self.repo.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def iter_docs(self) -> list[Path]:
        skills = sorted((self.repo / ".claude" / "skills").glob("*/SKILL.md"))
        agents = sorted((self.repo / ".claude" / "agents").glob("*.md"))
        return skills + agents

    # -- path-token classification ------------------------------------------------------------

    @staticmethod
    def _is_rooted(token: str) -> bool:
        return "/" in token and token.split("/", 1)[0].lower() in KNOWN_ROOTS

    @staticmethod
    def _is_bare(token: str) -> bool:
        if token.endswith("/") and "/" not in token[:-1]:
            return True  # bare directory, e.g. `loop/`
        if token.startswith("."):
            return False  # `.SPEC.md` / `.env` — a suffix or dotfile, not a contextual path claim
        return "/" not in token and token.endswith(PATH_EXTENSIONS)

    def _dir_context(self, token: str) -> str:
        """Directory a subsequent bare filename on the same line resolves against."""
        if token.endswith("/"):
            return token.rstrip("/")
        if not any(ch in token for ch in _GLOB_METACHARS) and (self.repo / token).is_dir():
            return token.rstrip("/")  # a directory cited without a trailing slash
        return str(Path(token).parent)

    # -- checks -------------------------------------------------------------------------------

    def check_paths(self, doc: Path, text: str) -> list[Failure]:
        rel = self.rel(doc)
        failures: list[Failure] = []
        for line in text.splitlines():
            current: str | None = None  # directory context for bare filenames, reset each line
            for span in _BACKTICK.findall(line):
                span = span.strip()
                if not span or any(ch in span for ch in " <>$"):
                    continue  # multi-word spans / placeholders are prose, not a single path claim
                for token in dict.fromkeys(_expand_braces(span)):  # dedupe, preserve order
                    token = _strip_trailing_punct(token)
                    if not token or "{" in token or "}" in token:
                        continue
                    if self._is_rooted(token):
                        if token in INTENTIONALLY_ABSENT:
                            logger.debug("%s: skip intentionally-absent `%s`", rel, token)
                        elif not self.exists(token):
                            failures.append(Failure(rel, "path", f"path not found: `{token}`"))
                        current = self._dir_context(token)
                        logger.debug("%s: rooted `%s` -> current=%s", rel, token, current)
                    elif self._is_bare(token):
                        candidates = ([f"{current}/{token}"] if current else []) + [token]
                        if not any(self.exists(c) for c in candidates):
                            failures.append(
                                Failure(rel, "path", f"bare path unresolved: `{token}` (tried {candidates})")
                            )
                        logger.debug("%s: bare `%s` tried %s", rel, token, candidates)
                    else:
                        logger.debug("%s: skip non-path token `%s`", rel, token)
        return failures

    def check_frontmatter(self, doc: Path, text: str) -> list[Failure]:
        rel = self.rel(doc)
        match = _FRONTMATTER.match(text)
        if not match:
            return [Failure(rel, "frontmatter", "missing YAML frontmatter block")]
        block = match.group(1)
        keys: dict[str, str] = {}
        for line in block.splitlines():
            key_match = _KEY_LINE.match(line)
            if key_match:
                keys[key_match.group(1)] = key_match.group(2).strip()

        is_agent = doc.parent.name == "agents"
        expected_name = doc.stem if is_agent else doc.parent.name
        required = ("name", "description", "tools") if is_agent else ("name", "description")

        failures: list[Failure] = []
        for key in required:
            if key not in keys:
                failures.append(Failure(rel, "frontmatter", f"missing required key '{key}'"))
        if keys.get("name") and keys["name"] != expected_name:
            failures.append(Failure(rel, "frontmatter", f"name '{keys['name']}' != expected '{expected_name}'"))
        if "description" in keys and not _description_present(block):
            failures.append(Failure(rel, "frontmatter", "'description' is present but empty"))
        return failures

    def validate(self) -> list[Failure]:
        """Return every validation failure across all context docs (empty list == all good)."""
        docs = self.iter_docs()
        if not docs:
            return [Failure(".claude", "path", "no context docs found under .claude/skills or .claude/agents")]
        failures: list[Failure] = []
        for doc in docs:
            text = doc.read_text(encoding="utf-8")
            failures += self.check_frontmatter(doc, text)
            failures += self.check_paths(doc, text)
        for claim in VALUE_CLAIMS:
            failures += claim(self)
        return failures


def _description_present(block: str) -> bool:
    """True if the frontmatter ``description`` has inline text or a folded/literal body that follows."""
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^description:(.*)$", line)
        if not match:
            continue
        inline = match.group(1).strip().lstrip(">|-").strip()
        if inline:
            return True
        for continuation in lines[index + 1 :]:
            if re.match(r"^\s+\S", continuation):
                return True  # an indented body line
            if continuation and not continuation[0].isspace():
                break  # next top-level key — no body
        return False
    return False


# --------------------------------------------------------------------------- value claims

ValueClaimCheck = Callable[[ContextDocValidator], "list[Failure]"]


def _check_coverage_gate(v: ContextDocValidator) -> list[Failure]:
    pyproject = v.try_read("pyproject.toml")
    if pyproject is None:
        return [Failure("pyproject.toml", "value-claim", "cannot read pyproject.toml")]
    match = re.search(r"fail_under\s*=\s*([0-9.]+)", pyproject)
    if not match:
        return [Failure("pyproject.toml", "value-claim", "no coverage `fail_under` found")]
    literal = f"fail_under = {match.group(1)}"
    failures: list[Failure] = []
    for rel in (v.PRIMER, v.GUIDE):
        text = v.try_read(rel)
        if text is not None and literal not in text:
            failures.append(Failure(rel, "value-claim", f"coverage gate drifted: expected `{literal}`"))
    return failures


def _check_console_scripts(v: ContextDocValidator) -> list[Failure]:
    pyproject = v.try_read("pyproject.toml")
    if pyproject is None:
        return [Failure("pyproject.toml", "value-claim", "cannot read pyproject.toml")]
    block = re.search(r"\[project\.scripts\](.*?)(?:\n\[|\Z)", pyproject, re.S)
    defined = set(re.findall(r"^([A-Za-z][\w-]*)\s*=", block.group(1), re.M)) if block else set()
    return [
        Failure("pyproject.toml", "value-claim", f"console script `{name}` no longer defined")
        for name in ("benchmark", "harness", "policy-lift")
        if name not in defined
    ]


def _check_env_flags(v: ContextDocValidator) -> list[Failure]:
    settings = v.try_read("src/config/settings.py")
    if settings is None:
        return [Failure("src/config/settings.py", "value-claim", "cannot read settings.py")]
    primer = v.try_read(v.PRIMER) or ""
    failures: list[Failure] = []
    for flag in (
        "ALLOW_MOCK_LLM_FALLBACK",
        "ALLOW_LIGHTWEIGHT_FRAMEWORK_FALLBACK",
        "ASSEMBLY_TRUST_LEGACY_PICKLE",
        "TRAINING_TRUST_LEGACY_PICKLE",
    ):
        if flag not in settings:
            failures.append(
                Failure("src/config/settings.py", "value-claim", f"env flag `{flag}` no longer in settings")
            )
        if flag not in primer:
            failures.append(Failure(v.PRIMER, "value-claim", f"env flag `{flag}` no longer documented in primer"))
    return failures


def _check_spec_statuses(v: ContextDocValidator) -> list[Failure]:
    source = "src/framework/harness/intent/spec_validator.py"
    validator = v.try_read(source)
    if validator is None:
        return [Failure(source, "value-claim", "cannot read spec_validator.py")]
    return [
        Failure(source, "value-claim", f"spec status `{status}` no longer in SPEC_STATUSES")
        for status in ("draft", "approved", "implemented", "verified", "superseded")
        if f'"{status}"' not in validator
    ]


def _check_settings_symbols(v: ContextDocValidator) -> list[Failure]:
    settings = v.try_read("src/config/settings.py")
    if settings is None:
        return [Failure("src/config/settings.py", "value-claim", "cannot read settings.py")]
    return [
        Failure("src/config/settings.py", "value-claim", f"`{symbol}` no longer defined")
        for symbol in ("class Settings", "def get_settings")
        if symbol not in settings
    ]


def _check_absent_paths(v: ContextDocValidator) -> list[Failure]:
    """Fail if an ``INTENTIONALLY_ABSENT`` path has reappeared — the allowlist must not go stale."""
    return [
        Failure(rel, "value-claim", "path is in INTENTIONALLY_ABSENT but now exists — update the allowlist/docs")
        for rel in sorted(INTENTIONALLY_ABSENT)
        if v.exists(rel)
    ]


#: Declarative registry — add a pinned claim here (data), not by editing control flow.
VALUE_CLAIMS: tuple[ValueClaimCheck, ...] = (
    _check_coverage_gate,
    _check_console_scripts,
    _check_env_flags,
    _check_spec_statuses,
    _check_settings_symbols,
    _check_absent_paths,
)


# --------------------------------------------------------------------------- entry points


def validate(repo_root: Path | str | None = None) -> list[Failure]:
    """Convenience wrapper: validate the context docs under ``repo_root`` and return failures."""
    return ContextDocValidator(repo_root).validate()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate-context-docs",
        description="Deterministically validate the repo's Claude context docs against the live tree.",
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="repo root to validate (default: this repo)")
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO-level trace to stderr")
    parser.add_argument("--debug", action="store_true", help="DEBUG-level trace (per-token classification)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit failures as JSON")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    validator = ContextDocValidator(args.repo_root)
    failures = validator.validate()

    if args.as_json:
        print(json.dumps([{"doc": f.doc, "category": f.category, "message": f.message} for f in failures], indent=2))
    elif failures:
        print(f"Context-doc validation FAILED ({len(failures)} issue(s)):\n", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
    else:
        print(f"Context-doc validation OK — {len(validator.iter_docs())} doc(s) checked, all claims verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
