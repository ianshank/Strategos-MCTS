---
id: hygiene_determinism
goal: One rank-aware seeding utility; reproducible Dirichlet root noise
module: src/utils/
status: approved
---

# Goal

Seed handling is hand-rolled at 12+ divergent sites (no two set the same RNG set) and
neural_mcts draws Dirichlet root noise from the unseeded global numpy RNG. Provide
src/utils/seeding.py and migrate call sites.

# Acceptance Criteria

- AC-1: src/utils/seeding.py provides set_all_seeds(seed, *, rank=0, deterministic_torch=False) (torch behind an import guard) and new_rng(seed), at 100% branch coverage.
- AC-2: The existing Settings.SEED field is reused; no new seed env var is introduced.
- AC-3: neural_mcts Dirichlet noise uses an injected numpy Generator; a same-machine fresh-process double-run yields identical visit counts and noise draws.
- AC-4: Divergent seed sites (re-grepped at execution; 12 at baseline) are migrated with legacy seed= kwargs preserved; the effective seed is logged at INFO on engine/trainer init.
- AC-5: tests/conftest.py gains an opt-in (non-autouse) global_seed fixture documented as the convention for new tests.

# Constraints

- src/training/ call-site edits are mechanical only (open approved specs claim that module); noted in the PR body.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
