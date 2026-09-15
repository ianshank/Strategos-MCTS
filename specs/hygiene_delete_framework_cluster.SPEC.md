---
id: hygiene_delete_framework_cluster
goal: Delete dead framework/meta-controller modules and trim live files (factories, validation, edge_cases harvest) without removing the public harness facade or topology API
module: src/framework/
status: draft
---

# Goal

`observability/facade.py`, the hybrid/assembly meta-controllers, and
`harness/memory/heartbeat.py` are deletion candidates; `framework/factories.py` is mostly dead
around the live `LLMClientFactory`; `models/validation.py` is dead beyond `QueryInput`;
`edge_cases.py` holds enums worth harvesting into `core.py`.

`harness/loop/facade.py` (`HarnessAgentAdapter`) and `harness/topology/` (`HarnessFactory.create_topology`)
are live public API. They are not on this kill list.

# Acceptance Criteria

- AC-1: Deletions and their tests land with the same-PR cleanup rule: meta_controller/__init__ guards/__all__/probe, factories.py trimmed to LLMClientFactory in the SAME PR (no dangling dispatch), heartbeat module plus `MEMORY_HEARTBEAT_INTERVAL_SECONDS` and compressor docstring reference. Do not delete `HarnessAgentAdapter`, `harness/topology/`, or `TOPOLOGY*` settings.
- AC-2: MCTSTerminationReason and MCTSSearchResult are harvested into core.py as str-Enums (string equality preserved), re-exported from the mcts package, before edge_cases.py is deleted; core's stringly termination reasons use the enum.
- AC-3: models/validation.py is trimmed to QueryInput plus transitive dependencies; the dead ProgressiveWideningConfig copy in policies.py is deleted.
- AC-4: CHANGELOG Removed; rollback tag pre-hygiene-delete-framework; coverage dry-run pasted in PR.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Do not break public harness signatures (CHARTER.md NG-6): `HarnessAgentAdapter` and `create_topology` stay.
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
