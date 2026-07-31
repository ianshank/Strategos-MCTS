---
id: hygiene_ci_mechanical
goal: Make the CI gate structurally honest (summary deps, strict markers, honest e2e, collection-gated server suites, corrected mypy claim)
module: .github/
status: draft
---

# Goal

The summary job ignores two jobs' failures, 19 of 22 pytest markers are unused with no
--strict-markers, the e2e workflow cannot fail, rest_server suites are suppressed in three
layers while their modules are omitted from coverage, and CLAUDE.md claims mypy --strict
which nothing runs.

# Acceptance Criteria

- AC-1: chess-tests and integration-test are in the summary job's needs and failure conditions.
- AC-2: --strict-markers is set; unused markers pruned (list re-derived at execution).
- AC-3: e2e_with_langsmith.yml has no '|| true'; its jobs are conditional on the LangSmith secret being present.
- AC-4: pre-commit pytest-quick can fail.
- AC-5: The rest_server suites are collection-gated (import+collect must succeed; known failures xfail with reasons) with all three suppression layers (ci.yml ignores, conftest collect_ignore_glob, pyproject coverage omits) addressed together; green-ness is deferred to the rest-split phase.
- AC-6: The post-change coverage gate is dry-run locally and the measured number pasted into the PR before the CI flip; CHANGELOG documents old vs new blocking set and baseline.
- AC-7: CLAUDE.md's mypy claim matches reality; a tracking draft spec exists for the strictness ratchet.

# Constraints

- No src/** changes.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
