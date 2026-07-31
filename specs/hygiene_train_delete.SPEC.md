---
id: hygiene_train_delete
goal: Delete the superseded root training/ fork
module: training/
status: draft
---

# Goal

About 13k LOC of root training/ is superseded by src/ counterparts; src/ imports nothing from
it, so after extraction the fork (plus its never-run tests and conflicting requirements
files) is removed.

# Acceptance Criteria

- AC-1: Superseded modules, training/tests/ (including the sys.modules['pinecone'] clobber), training/requirements*.txt, and the extraction shims are deleted; the surviving training/ entry surface is enumerated or the tree is removed entirely with Dockerfile.train's CMD retargeted (decided at spec review with the Dockerfile.test ENTRYPOINT checked).
- AC-2: docker-deployment.yml is updated in the same PR; both training workflows are green.
- AC-3: Coverage and LOC deltas recorded in docs/STATUS.md.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
