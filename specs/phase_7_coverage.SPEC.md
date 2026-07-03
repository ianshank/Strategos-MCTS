---
id: phase_7_coverage
goal: Gap-analysis coverage lift for under-tested core modules; hold the >=85% gate
module: tests/unit/
phase: "7"
milestone: M5
status: implemented
---

# Goal

Run a gap analysis over the branch and raise the genuinely testable core modules that sit below the 85%
branch-coverage gate up to at least 85%, keeping the enforced global gate at 85% (currently satisfied at
~89.6%). Prioritize pure-logic modules (harness CLI/factories, `mcts/llm_guided` RAG, meta-controller,
benchmark adapters); modules whose remaining lines require optional dependencies (google-adk, chess) are
out of scope, like the coverage `omit` list.

# Acceptance Criteria

- AC-1: New unit tests raise the genuinely testable modules that were below the 85% gate up to at least 85%:
  `src/framework/harness/cli.py` (53.7%→~98%), `src/framework/harness/factories.py` (72.3%→~95%),
  `src/framework/mcts/llm_guided/rag/prompts.py` (71.3%→~97%),
  `src/benchmark/adapters/adk_adapter.py` (63%→~83%, remainder needs the optional `google-adk` dep), and
  `src/agents/meta_controller/hybrid_controller.py` (added get_statistics/adjust_weights/load-save/
  explain_decision coverage).
- AC-2: The enforced global gate stays at `fail_under = 85.0` in `pyproject.toml`, `--cov-fail-under=85` in
  `.github/workflows/ci.yml`, and `.claude/skills/quality-gate/SKILL.md`; overall branch coverage remains
  green at ~89.6% (no gate bump — a 90% raise is deferred until overall clears 90% with margin).
- AC-3: The current per-module baseline is recorded in `docs/STATUS.md`.

# Constraints

- No real network or API calls in unit tests; mock all I/O; reuse existing fixtures in `tests/fixtures/`.
- Neural training loops (`training/neural_trainer.py`, `meta_controller_trainer.py`) and optional-dep chess
  / google-adk modules already omitted or unavailable in CI are out of scope; the 85% global gate absorbs them.
- Backward compatible; no hardcoded values; full local gate green before push.
