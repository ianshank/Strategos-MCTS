---
goal: Clear the small, genuine residue of correctness and packaging issues
phase: "1"
milestone: M3
status: active
---

# Goal

Resolve the real (not stale) correctness and packaging defects: rollout-policy typing drift in tests,
a permanently-skipped chaos suite gated on non-existent modules, and one uncovered dataset-loader branch.

# Acceptance Criteria

- `mypy src/` is clean; the MCTS rollout-policy test subclasses match the source protocol via a shared
  `BaseRolloutPolicy` test helper, and the early-termination / framework / parallel-mcts / e2e suites pass.
- `tests/chaos/test_resilience.py` no longer skips silently on a non-existent `improved_hrm_agent` /
  `improved_trm_agent` guard: the resilience tests either execute real assertions against the real agent
  modules or are removed with documented rationale.
- A regression test asserts the dataset loader defaults to the `train` split and that an unknown split
  falls back to an available split with a warning.

# Constraints

- Do not change the public signatures of `src/framework/mcts/policies.py` (source is already consistent).
- Do not add `examples/__init__.py` (it would package `examples/` and break the bare
  `import langgraph_multi_agent_mcts` the chaos tests use).
- Backward compatible; no hardcoded values; full local gate green before push.
