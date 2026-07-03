"""Lightweight SPEC.md / AGENTS.md / SKILL.md parser.

We deliberately do *not* depend on a third-party markdown library. The spec
format is simple: a YAML-style frontmatter block (delimited by ``---``) and
a markdown body. Sections within the body are ATX headers (``#``).

The parser is forgiving — missing frontmatter is fine, missing sections are
fine — and returns a :class:`Spec` dataclass with the recognised fields.
Schema *enforcement* (required fields, status vocabulary, criterion-ID rules)
lives in :mod:`.spec_validator`, not here.

Spec schema v2 (see ``docs/plans/SDD_PLUGIN_EXTRACTION_PLAN.md`` §2) adds
frontmatter fields (``id``/``module``/``status``/``supersedes``), authored
acceptance-criterion IDs as bullet prefixes (``- AC-1: ...``), and optional
``# Invariants`` / ``# Out of Scope`` sections. Bullets without an ``AC-n:``
prefix keep the historical positional IDs (``c0``, ``c1``, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Authored criterion-ID prefix on an acceptance bullet: "- AC-1: <description>".
# The trailing whitespace requirement is deliberate — "AC-1:text" is not an
# authored ID and falls back to a positional one (the validator warns on it).
_AC_ID_RE = re.compile(r"^(AC-\d+):\s+(.*)$")


class SpecParseError(ValueError):
    """Raised when a spec file cannot be parsed."""


@dataclass(frozen=True)
class SpecCriterion:
    """One acceptance criterion with its authored (``AC-n``) or positional (``c{i}``) ID."""

    id: str
    description: str


@dataclass
class Spec:
    """Parsed spec document."""

    goal: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    frontmatter: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    # Schema v2 frontmatter fields (empty string when absent — legacy specs).
    id: str = ""
    module: str = ""
    status: str = ""
    supersedes: str = ""
    # Criteria with IDs; parallel to ``acceptance_criteria`` (which holds the
    # prefix-stripped descriptions — downstream verifiers match on them).
    criteria: list[SpecCriterion] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    # Markdown body with the frontmatter block removed. Validators that walk
    # headers must use this, not ``raw`` — frontmatter supports ``#`` comments.
    body: str = ""

    def criteria_payload(self) -> list[dict[str, str]]:
        """Criteria as intent-dict entries: authored IDs, positional fallback."""
        return [{"id": c.id, "description": c.description} for c in self.criteria]


class SpecLoader:
    """Parse markdown spec files into a :class:`Spec`."""

    _ACCEPTANCE_HEADERS = ("acceptance criteria", "acceptance", "criteria")
    _CONSTRAINTS_HEADERS = ("constraints", "constraint")
    _GOAL_HEADERS = ("goal", "objective", "summary")
    _INVARIANTS_HEADERS = ("invariants",)
    _OUT_OF_SCOPE_HEADERS = ("out of scope", "out-of-scope")

    def load(self, path: Path) -> Spec:
        """Read and parse ``path``."""
        if not path.exists():
            raise SpecParseError(f"spec file not found: {path}")
        return self.parse(path.read_text(encoding="utf-8"))

    def parse(self, text: str) -> Spec:
        """Parse a raw markdown string."""
        body, frontmatter = self._split_frontmatter(text)
        sections = self._split_sections(body)
        goal = self._first_match(sections, self._GOAL_HEADERS) or frontmatter.get("goal", "")
        bullets = self._extract_bullets(self._first_match(sections, self._ACCEPTANCE_HEADERS) or "")
        criteria = self._identify_criteria(bullets)
        constraints = self._extract_bullets(self._first_match(sections, self._CONSTRAINTS_HEADERS) or "")
        invariants = self._extract_bullets(self._first_match(sections, self._INVARIANTS_HEADERS) or "")
        out_of_scope = self._extract_bullets(self._first_match(sections, self._OUT_OF_SCOPE_HEADERS) or "")
        return Spec(
            goal=str(goal).strip(),
            acceptance_criteria=[c.description for c in criteria],
            constraints=constraints,
            sections=sections,
            frontmatter=frontmatter,
            raw=text,
            id=str(frontmatter.get("id", "")).strip(),
            module=str(frontmatter.get("module", "")).strip(),
            status=str(frontmatter.get("status", "")).strip(),
            supersedes=str(frontmatter.get("supersedes", "")).strip(),
            criteria=criteria,
            invariants=invariants,
            out_of_scope=out_of_scope,
            body=body,
        )

    @staticmethod
    def _identify_criteria(bullets: list[str]) -> list[SpecCriterion]:
        """Attach IDs to acceptance bullets: authored ``AC-n:`` prefix or positional ``c{i}``."""
        criteria: list[SpecCriterion] = []
        for i, bullet in enumerate(bullets):
            match = _AC_ID_RE.match(bullet)
            if match:
                criteria.append(SpecCriterion(id=match.group(1), description=match.group(2).strip()))
            else:
                criteria.append(SpecCriterion(id=f"c{i}", description=bullet))
        return criteria

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[str, dict[str, Any]]:
        """Split ``---``-delimited frontmatter from the body. Frontmatter is
        parsed with a tiny line-based ``key: value`` reader to avoid pulling
        in PyYAML; nested structures are not supported and are passed through
        as raw strings."""
        if not text.startswith("---\n"):
            return text, {}
        end = text.find("\n---\n", 4)
        if end == -1:
            return text, {}
        front = text[4:end]
        body = text[end + 5 :]
        front_dict: dict[str, Any] = {}
        for line in front.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped:
                key, _, value = stripped.partition(":")
                front_dict[key.strip()] = value.strip()
        return body, front_dict

    @staticmethod
    def _split_sections(body: str) -> dict[str, str]:
        """Split a markdown body into ``{title: content}`` keyed by ATX header."""
        sections: dict[str, str] = {}
        current_title: str | None = None
        current_lines: list[str] = []
        for line in body.splitlines():
            if line.startswith("#"):
                if current_title is not None:
                    sections[current_title] = "\n".join(current_lines).strip()
                title = line.lstrip("#").strip().lower()
                current_title = title
                current_lines = []
            else:
                current_lines.append(line)
        if current_title is not None:
            sections[current_title] = "\n".join(current_lines).strip()
        return sections

    @staticmethod
    def _first_match(sections: dict[str, str], aliases: tuple[str, ...]) -> str | None:
        for key, value in sections.items():
            if key in aliases:
                return value
            for alias in aliases:
                if key.startswith(alias):
                    return value
        return None

    @staticmethod
    def _extract_bullets(block: str) -> list[str]:
        out: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ", "+ ")):
                out.append(stripped[2:].strip())
                continue
            # Numbered lists with arbitrary digit width: "12. text" / "12) text".
            head, sep, rest = stripped.partition(" ")
            if rest and (head.endswith(".") or head.endswith(")")) and head[:-1].isdigit():
                out.append(rest.strip())
        return out


__all__ = ["Spec", "SpecCriterion", "SpecLoader", "SpecParseError"]
