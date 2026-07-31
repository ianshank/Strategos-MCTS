---
id: hygiene_delete_enterprise_cluster
goal: Delete the dead enterprise, component_factory, and performance subtrees with their tests
module: src/enterprise/
status: draft
---

# Goal

src/enterprise/ (~4.8k LOC), src/framework/component_factory/ (~1.4k), and src/performance/
(~1.4k) have zero production importers and are pinned only by dedicated tests.

# Acceptance Criteria

- AC-1: The three subtrees and their pinned tests are deleted; repo-wide reachability re-verified at HEAD (imports, string literals, importlib, scripts/, demos/, examples/, notebooks/, Dockerfiles, compose, kubernetes/, workflows) before deletion.
- AC-2: All same-PR cleanups applied: package re-exports, __all__ entries, availability probes, orphaned settings fields, CLAUDE.md rows.
- AC-3: CHANGELOG Removed entries with replacement pointers where one exists; annotated rollback tag pre-hygiene-delete-enterprise with restore recipe in MIGRATION_NOTES.
- AC-4: Local coverage dry-run at or above 85% pasted into the PR before merge.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
