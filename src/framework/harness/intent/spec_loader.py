"""Lightweight SPEC.md / AGENTS.md / SKILL.md parser.

We deliberately do *not* depend on a third-party markdown library. The spec
format is simple: a YAML-style frontmatter block (delimited by ``---``) and
a markdown body. Sections within the body are ATX headers (``#``).

The parser is forgiving — missing frontmatter is fine, missing sections are
fine — and returns a :class:`Spec` dataclass with the recognised fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SpecParseError(ValueError):
    """Raised when a spec file cannot be parsed."""


@dataclass
class Spec:
    """Parsed spec document."""

    goal: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    frontmatter: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


class SpecLoader:
    """Parse markdown spec files into a :class:`Spec`."""

    _ACCEPTANCE_HEADERS = ("acceptance criteria", "acceptance", "criteria")
    _CONSTRAINTS_HEADERS = ("constraints", "constraint")
    _GOAL_HEADERS = ("goal", "objective", "summary")

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
        criteria = self._extract_bullets(self._first_match(sections, self._ACCEPTANCE_HEADERS) or "")
        constraints = self._extract_bullets(self._first_match(sections, self._CONSTRAINTS_HEADERS) or "")
        return Spec(
            goal=str(goal).strip(),
            acceptance_criteria=criteria,
            constraints=constraints,
            sections=sections,
            frontmatter=frontmatter,
            raw=text,
        )

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


__all__ = ["Spec", "SpecLoader", "SpecParseError"]
