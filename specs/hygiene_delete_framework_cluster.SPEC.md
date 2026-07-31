---
id: hygiene_delete_framework_cluster
goal: Delete dead framework/meta-controller/harness modules and trim live files (factories, validation, edge_cases harvest)
module: src/framework/
status: draft
---

# Goal

observability/facade.py, the hybrid/assembly meta-controllers, harness loop/facade.py,
memory/heartbeat.py, and topology/ are dead; framework/factories.py is 83% dead around the
live LLMClientFactory; models/validation.py is dead beyond QueryInput; edge_cases.py holds
enums worth harvesting into core.py.

# Acceptance Criteria

- AC-1: Deletions and their tests land with the same-PR cleanup rule: meta_controller/__init__ guards/__all__/probe, factories.py trimmed to LLMClientFactory in the SAME PR (no dangling dispatch), harness/__init__ re-export, orphaned settings fields (MEMORY_HEARTBEAT_INTERVAL_SECONDS, TOPOLOGY*), compressor docstring reference.
- AC-2: MCTSTerminationReason and MCTSSearchResult are harvested into core.py as str-Enums (string equality preserved), re-exported from the mcts package, before edge_cases.py is deleted; core's stringly termination reasons use the enum.
- AC-3: models/validation.py is trimmed to QueryInput plus transitive dependencies; the dead ProgressiveWideningConfig copy in policies.py is deleted.
- AC-4: CHANGELOG Removed; rollback tag pre-hygiene-delete-framework; coverage dry-run pasted in PR.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
