# AGENTS.md — Strategos-MCTS

Routing ledger for autonomous agents. Keep ≤150 lines; prefer pointers over prose.
Scope, non-goals, and invariants live in `CHARTER.md` — read it before planning.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,benchmark]"
cp .env.example .env  # then add OPENAI_API_KEY or ANTHROPIC_API_KEY
```

## Build / lint / type / test commands

```bash
black . --check --line-length 120
ruff check .
mypy src/
pytest tests/unit -v
pytest tests/integration -v
pytest tests/ -m harness
pytest tests/ -m "not slow" --cov=src --cov-report=term-missing
python scripts/validate_context_docs.py  # deterministic check: .claude skills/agents vs the tree
```

`ruff`/`mypy` are pinned in the `[dev]` extra (CI lint job installs `.[dev]`) for CI/local
parity — bump deliberately and re-validate. Coverage gate: 85% (`--cov-fail-under=85`).
Achieved: **89.65%** gate-scope (`tests/unit/`, 2026-08-04, `[dev,neural,api]` env — see
`docs/STATUS.md`). The full-suite 90.15% figure predates the 2026-08-04 denominator widening.
`mypy src/` must remain clean (336 files).
Async tests must use `@pytest.mark.asyncio` + `await` — never `asyncio.get_event_loop()`.

## Harness CLI

```bash
harness validate-spec specs/*.SPEC.md   # schema v2; errors exit 1
harness spec-new --id <id> --module <path> # scaffold a draft; refuses module overlap w/ open specs
harness spec-status <id> --require approved  # lifecycle check; exit 1 on mismatch
harness spec-trace --base-ref origin/main --branch <name>  # CI traceability rules (PR diffs)
harness dry-run --spec path/to/spec.md
harness run --spec path/to/spec.md
harness run --goal "describe the goal" --max-iterations 5
harness run --spec path/to/spec.md --ralph
harness replay --cassette-dir ./.harness/cassettes
```

All knobs come from `HARNESS_*` env vars; never hardcode in call sites.

## Benchmark CLI

```bash
python -m src.benchmark --dry-run
python -m src.benchmark --systems langgraph_mcts --tasks A1
```

## Code style

- Python ≥3.10, ruff-clean, mypy-clean (`src/`).
- Line length 120 (`pyproject.toml` enforces).
- Pydantic Settings for all config — no hardcoded values.
- Async-first; new I/O paths must be `async`.
- Protocol-based DI — prefer `runtime_checkable` Protocols over ABCs at boundaries.

## Architecture pointers

| Concern | Path |
| --- | --- |
| Settings | `src/config/settings.py`, `src/config/constants.py` (shared defaults/bounds), `src/framework/harness/settings.py` |
| Existing agents | `src/agents/`, `src/framework/agents/base.py` |
| LangGraph orchestration | `src/framework/graph/builder.py` |
| MCTS engine | `src/framework/mcts/core.py` (baseline), `src/framework/mcts/neural_mcts.py` (AlphaZero-style; `single_agent` flag) |
| Gameplay domains | `src/games/chess/` (chess), `src/games/connect_four/` (connect_four), `src/games/othello/` (othello) (adversarial), `src/framework/mcts/single_agent_domains.py` (reasoning, planning) |
| Neural self-play (M5) | `src/training/self_play_trainer.py` |
| Training profiles | `src/training/training_config.py` (`smoke`/`dev`/`full` profiles) |
| Self-play convergence driver | `src/training/self_play_convergence.py` (CLI entry, `--profile`/`--mixed-precision`/`--compile`) |
| System/device config | `src/training/system_config.py` (device resolution, AMP, compile, CUDA memory fraction) |
| GPU Introspection & Memory | `src/utils/gpu_utils.py` (`get_gpu_info`, `check_gpu_ready`, `GPUMemoryTracker`, memory fraction limit) |
| Distributed utilities (DDP) | `src/utils/distributed.py` (`init_distributed`, `is_main_process`, `wrap_ddp`, `unwrap_model`, process topology) |
| Meta-controller learning (M5) | `src/training/meta_controller_data_collector.py` (see `docs/META_CONTROLLER_TRAINING.md`) |
| API services (streaming/viz/compare) | `src/api/{streaming,graph_service,comparison_service}.py` (thin endpoints in `rest_server.py`) |
| LLM adapters | `src/adapters/llm/{base,resilience,openai_client,anthropic_client,lmstudio_client}.py` (`resilience.py` = shared `CircuitBreaker`) |
| API authentication | `src/api/auth.py` (`AUTH_MODE`: api_key default / jwt) |
| Observability | `src/observability/{logging,metrics,tracing}.py` |
| Benchmark harness | `src/benchmark/` (+ `policy_comparison.py` for trained-vs-baseline lift) |
| Agent harness framework | `src/framework/harness/` |
| · runner | `src/framework/harness/loop/runner.py` |
| · facade (`AsyncAgentBase` adapter) | `src/framework/harness/loop/facade.py` |
| · memory (event log + compactor) | `src/framework/harness/memory/` |
| · tools (registry + builtins) | `src/framework/harness/tools/` |
| · topologies | `src/framework/harness/topology/` |
| · ralph loop | `src/framework/harness/ralph/` |
| · replay (cassettes + clock) | `src/framework/harness/replay/` |

## Test layout

| Layer | Path | Marker |
| --- | --- | --- |
| Unit | `tests/unit/` | `@pytest.mark.unit` |
| Integration | `tests/integration/` | `@pytest.mark.integration` |
| Contract | `tests/contract/` | `@pytest.mark.contract` |
| Property | `tests/property/` | `@pytest.mark.property` |
| E2E | `tests/e2e/` | `@pytest.mark.e2e` |
| Harness suite | `tests/{unit,integration}/{framework/harness,harness}/` | `@pytest.mark.harness` |

Fixtures: `tests/fixtures/harness_fixtures.py` (helpers), `tests/integration/harness/conftest.py` (pytest fixtures).

## Permissions / secrets

- Never read API keys directly — use `Settings.get_api_key()`.
- Shell tool defaults disabled; opt in via `HARNESS_PERM_SHELL=true` and `--shell-allow <argv0>`.
- File edits use SHA-256 hash anchors via `file_edit_hashed_tool`; never bypass.
- Memory tools never escape `HARNESS_MEMORY_ROOT`.

## Pitfalls

- The Reason phase binds directly to `LLMClient`; do not route it through `AsyncAgentBase` — that path is the *outer* facade only.
- `MEMORY.md` is a derived view. Never write it directly; append events via `MarkdownMemoryStore.append_event` and let the compactor materialise.
- Hook ordering follows `cost_class` (cheap → expensive). Stable insertion order tie-breaks.
- `case_sensitive=True` on settings — env vars must match field names exactly.

## Pointers to deeper docs

- Active roadmap: `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md`
- Current test/coverage status (source of truth): `docs/STATUS.md`
- Spec-driven specs: `specs/<id>.SPEC.md`, schema v2 (validate with `harness validate-spec specs/*.SPEC.md`)
- Project skills: `.claude/skills/{quality-gate,validate-specs,coverage-baseline,strategos-primer,validate-context}`
- Codebase orientation: `strategos-primer` skill + `strategos-guide` agent (`.claude/agents/`) map the
  layers/subsystems/invariants; `validate-context` (`src/tools/context_docs.py`, `validate-context-docs`
  console script, in the unit suite) deterministically checks those docs' paths and value-claims vs the tree
- SDD enforcement: `/spec-new` + `/spec-implement` (`.claude/commands/`), `spec-review` subagent
  (`.claude/agents/`), PreToolUse gate `.claude/hooks/spec_gate.py` (warn mode; `SPEC_GATE_BYPASS=1`
  for hotfixes; src/** PRs need a `spec/<id>` branch with an approved spec or a `No-Spec: <reason>` trailer)
- Deep Research: `/deep-research` command (`.claude/commands/`), orchestrated by `research-planner`,
  `research-fetcher`, `research-critic`, and `research-synthesizer` agents (`.claude/agents/`) with
  reports output to `docs/reports/` using the `deep-research` skill (`.claude/skills/`).
- Implementation template: `docs/templates/MULTI_AGENT_MCTS_TEMPLATE.md`
- Architecture: `docs/C4_ARCHITECTURE.md`
- This file is a routing ledger, not an encyclopedia. Drill into a path above for detail.
