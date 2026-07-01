---
goal: Gap-analysis coverage lift for under-tested core modules; hold the >=85% gate
phase: "7"
milestone: M5
status: active
---

# Goal

Run a gap analysis over the branch and raise the genuinely testable core modules that sit below the 85%
branch-coverage gate up to at least 85%, keeping the enforced global gate at 85% (currently satisfied at
~89.6%). Prioritize pure-logic modules (harness CLI/factories, `mcts/llm_guided` RAG, meta-controller,
benchmark adapters); modules whose remaining lines require optional dependencies (google-adk, chess) are
out of scope, like the coverage `omit` list.

# Acceptance Criteria

- New unit tests cover the harness topology package (`base`, `pipeline`, `fan_out_in`, `expert_pool`,
  `producer_reviewer`, `supervisor`, `hierarchical`), `context/{compressor,injector}.py`, and the
  under-tested `memory/`, `tools/`, `loop/`, `hooks/base.py` modules.
- `src/framework/harness/cli.py`, `src/framework/harness/factories.py`,
  `src/framework/mcts/llm_guided/rag/{context,prompts}.py`,
  `src/framework/mcts/llm_guided/benchmark/runner.py`, `src/benchmark/adapters/adk_adapter.py`, and
  `src/agents/meta_controller/hybrid_controller.py` reach ~95% branch coverage.
- The global gate is raised to `fail_under = 90.0` in `pyproject.toml`, `--cov-fail-under=90` in
  `.github/workflows/ci.yml`, and the `.claude/skills/quality-gate/SKILL.md` command — updated in the same
  commit as the final tests, only after overall coverage is confirmed at or above 90%.
- The per-module 95%-on-core targets are recorded in `docs/STATUS.md` (tracked, not gated).

# Constraints

- No real network or API calls in unit tests; mock all I/O; reuse existing fixtures in `tests/fixtures/`.
- Neural training loops (`training/neural_trainer.py`, `meta_controller_trainer.py`) and optional-dep chess
  modules already omitted in `pyproject.toml` are out of the 95% push; the 90% global gate absorbs them.
- Backward compatible; no hardcoded values; full local gate green before push.
