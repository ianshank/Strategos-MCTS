---
id: hygiene_mcts_config
goal: One canonical MCTSConfig with compatibility wrappers
module: src/framework/mcts/
status: draft
---

# Goal

Three mutually incompatible MCTSConfig classes exist (framework/mcts/config.py,
src/training/system_config.py, src/models/validation.py). Consolidate on the framework one
with adapters and thin wrappers preserving import paths and field names.

# Acceptance Criteria

- AC-1: framework/mcts/config.MCTSConfig is canonical with from_training_config()/from_query_model() adapters; the other two become thin wrappers preserving import paths and field names, emitting DeprecationWarning via the shared helper.
- AC-2: All existing construction sites keep working (legacy kwargs preserved); the characterization suite is green.

# Constraints

- Gated on closure or re-scope of the open ddp_orchestrator and m5_policy_lift specs (src/training/system_config.py is inside their claimed module).
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
