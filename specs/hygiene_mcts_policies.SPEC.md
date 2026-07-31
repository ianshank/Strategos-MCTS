---
id: hygiene_mcts_policies
goal: Single canonical UCB1/PUCT implementations and one exploration-weight constant
module: src/framework/mcts/
status: draft
---

# Goal

UCB1 is inlined five times and PUCT four times; the 1.414 exploration literal appears 22 times
across 13 files. Consolidate on policies.puct/ucb1 and one settings-backed constant.

# Acceptance Criteria

- AC-1: policies.ucb1 and neural_policies.puct/puct_with_virtual_loss are the only implementations; all inline copies call them.
- AC-2: DEFAULT_EXPLORATION_WEIGHT = math.sqrt(2) lives in framework/mcts/config.py with settings.MCTS_C as the env override; the literal is gone (re-grep); llm_mcts re-exports are preserved for comparison_service.
- AC-3: Characterization suite (25 seeded scenarios, exact best action + visit counts, rel=1e-9 values, JSON goldens under tests/fixtures/characterization/) is green; any golden change is enumerated in the PR body.

# Constraints

- Gated on closure or re-scope of the open strategos_risk_averse_subgoal_scorer spec; its cited symbols are not moved or renamed.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
