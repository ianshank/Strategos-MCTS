# AGENTS.md — Strategos-MCTS

Routing ledger for autonomous agents. Keep ≤150 lines; prefer pointers over prose.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,benchmark]"
cp .env.example .env  # then add OPENAI_API_KEY or ANTHROPIC_API_KEY
```

## Build / lint / type / test commands

```bash
black src/ tests/ --check --line-length 120
ruff check src/ tests/
mypy src/
pytest tests/unit -v
pytest tests/integration -v
pytest tests/ -m harness
pytest tests/ -m "not slow" --cov=src --cov-report=term-missing
```

`ruff`/`mypy` are pinned in the `[dev]` extra (CI lint job installs `.[dev]`) for CI/local
parity — bump deliberately and re-validate. Coverage gate: 85% (`--cov-fail-under=85`).

## Harness CLI

```bash
harness validate-spec path/to/spec.md
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
| MCTS engine | `src/framework/mcts/core.py` |
| LLM adapters | `src/adapters/llm/{base,resilience,openai_client,anthropic_client,lmstudio_client}.py` (`resilience.py` = shared `CircuitBreaker`) |
| Observability | `src/observability/{logging,metrics,tracing}.py` |
| Benchmark harness | `src/benchmark/` |
| Agent harness framework | `src/framework/harness/` |
|  · runner | `src/framework/harness/loop/runner.py` |
|  · facade (`AsyncAgentBase` adapter) | `src/framework/harness/loop/facade.py` |
|  · memory (event log + compactor) | `src/framework/harness/memory/` |
|  · tools (registry + builtins) | `src/framework/harness/tools/` |
|  · topologies | `src/framework/harness/topology/` |
|  · ralph loop | `src/framework/harness/ralph/` |
|  · replay (cassettes + clock) | `src/framework/harness/replay/` |

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

- Comprehensive template: `MULTI_AGENT_MCTS_TEMPLATE.md`
- Original patterns: `CLAUDE_CODE_IMPLEMENTATION_TEMPLATE.md`
- Architecture: `docs/C4_ARCHITECTURE.md`
- This file is a routing ledger, not an encyclopedia. Drill into a path above for detail.
