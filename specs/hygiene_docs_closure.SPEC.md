---
id: hygiene_docs_closure
goal: Docs truth-up and program closure
module: docs/
status: draft
---

# Goal

Close the program: fix stale doc claims, consolidate migration notes, groom the changelog,
record the final measured baselines, and ensure every deferred item has a tracking artifact.

# Acceptance Criteria

- AC-1: CLAUDE.md is accurate: graph package row, harness table rows for deleted modules, factories row, final mypy statement; PROJECT_STRUCTURE.md's models/ size claim matches the LFS reality.
- AC-2: MIGRATION_NOTES is newest-first and complete; CHANGELOG is release-ready.
- AC-3: Final coverage-baseline and LOC counts are recorded in docs/STATUS.md; spec validation is green across specs/; every remaining draft or deferred item has a tracking artifact.

# Constraints

- Docs only; no src/** changes.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
