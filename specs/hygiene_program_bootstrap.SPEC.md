---
id: hygiene_program_bootstrap
goal: Land the hygiene program's governance artifacts (plan, draft specs, PR template, LOC baseline)
module: docs/
status: draft
---

# Goal

Establish the peer-reviewed code-hygiene & modularity program in-repo: the plan document, one
draft spec per phase, a PR-body template for phase PRs, and a reproducible LOC baseline so
every destructive phase can be measured against a recorded starting point.

# Acceptance Criteria

- AC-1: The program plan exists at docs/plans/2026-07-30-code-hygiene-modularity.md.
- AC-2: Every phase spec of the program exists under specs/ with status draft, and schema-v2 validation of all specs reports zero errors.
- AC-3: A phase-PR body template exists at .github/PULL_REQUEST_TEMPLATE/hygiene_phase.md (phase id, spec link, AC checklist, gate summary, rollback tag).
- AC-4: A reproducible LOC baseline (per top-level tree, with the exact command) is recorded in docs/STATUS.md.

# Constraints

- No src/** changes in this phase.
- Draft specs are inert program-traceability artifacts; approval (and module-overlap arbitration with open specs) happens per phase via spec-review and a human status flip.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
