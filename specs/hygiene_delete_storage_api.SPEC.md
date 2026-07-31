---
id: hygiene_delete_storage_api
goal: Delete dead storage and API modules (s3_client, faiss_store, health, inference_server, caching, dead utils)
module: src/storage/
status: draft
---

# Goal

storage/s3_client.py, storage/faiss_store.py, api/health.py, api/inference_server.py,
framework/caching.py, utils/planning_loader.py, and utils/mcts_debug.py have zero production
reachability; several are re-exported by package __init__ files which must be cleaned in the
same PR.

# Acceptance Criteria

- AC-1: Listed modules and their tests are deleted; storage/__init__.py and utils/__init__.py re-exports/__all__/lazy __getattr__ entries removed in the same PR; the s3 string in scripts/export_architecture_diagrams.py cleaned.
- AC-2: Before deleting api/health.py, kubernetes/compose probes are verified to hit rest_server's own /health route.
- AC-3: api/inference_server.py's ci.yml ignore and pyproject coverage omit are removed in the same PR.
- AC-4: Same-PR cleanup rule, CHANGELOG Removed with replacement pointers (mcts_debug -> src/observability/debug.MCTSDebugger; faiss_store -> src/api/local_embedding_store), rollback tag pre-hygiene-delete-storage-api, coverage dry-run pasted in PR.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
