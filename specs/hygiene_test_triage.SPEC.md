---
id: hygiene_test_triage
goal: Classify all never-gated test files (fix / xfail-with-issue / delete-with-module)
module: tests/
status: draft
---

# Goal

About 84 test files run in no blocking gate. Triage each against a fixed matrix, cross-checked
against the deletion phases' kill list FIRST so no effort is spent fixing tests for code
scheduled for removal.

# Acceptance Criteria

- AC-1: Every never-gated test file is classified fix / xfail-with-issue / delete-with-module; zero unclassified.
- AC-2: The blocking job covers all surviving tests/unit/; tests/integration is PR-blocking if measured wall-time is under 10 minutes, else main-push and summary-gated.
- AC-3: Skipped/xfailed counts are reported in docs/STATUS.md.

# Constraints

- Time-boxed; no fixing tests for modules on the deletion kill list.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
