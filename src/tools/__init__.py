"""Developer tooling that ships with the package but is not part of the runtime system.

Currently: `context_docs` — a deterministic validator for the repo's Claude context docs
(`.claude/skills/**/SKILL.md`, `.claude/agents/*.md`). Kept under `src/` so it is importable,
type-checked, and covered by the same gates as product code.
"""
