---
goal: Raise core-module coverage toward 95% and lift the enforced gate to 90%
phase: "7"
milestone: M5
status: active
---

# Goal

Close the genuinely testable coverage gaps in the core (`src/framework/harness/` and
`src/framework/mcts/llm_guided/` are the largest untested surfaces), target ~95% branch coverage on those
core modules, and raise the global CI gate from 85% to 90% once coverage clears it with margin.

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
