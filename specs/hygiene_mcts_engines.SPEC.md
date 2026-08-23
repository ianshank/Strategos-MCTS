---
id: hygiene_mcts_engines
goal: One backpropagation/selection/stats implementation shared by the MCTS engines
module: src/framework/mcts/
status: approved
---

# Goal

Backprop exists in six copies with disagreeing sign conventions; selection loops, tree stats,
best-action, and tree-depth helpers are copied per engine (two copies still recursive).
Consolidate onto core with the two-player perspective flag from the value-semantics phase.

# Acceptance Criteria

- AC-1: Shared backprop on core.MCTSNode parameterized by the perspective flag; parallel and progressive-widening engines subclass the core engine/node (the spec names which classes).
- AC-2: One stats/best-action/tree-depth helper module; tree depth is iterative everywhere; raise_if_invalid(errors) replaces the 12 validation-boilerplate copies.
- AC-3: The characterization suite and the value-semantics regression suite are both green.

# Constraints

- Gated on closure or re-scope of the open strategos_risk_averse_subgoal_scorer spec.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
