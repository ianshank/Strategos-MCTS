---
name: strategos-primer
description: >-
  Big-picture orientation to the Strategos-MCTS (langgraph-multi-agent-mcts) codebase — what the
  system is, how its layers fit together, where each subsystem lives (MCTS, agents, LLM adapters,
  LangGraph orchestration, observability, benchmark, agent harness, spec-driven development), the
  non-negotiable invariants a change must respect, and the map of deeper docs. Use this whenever you
  are getting oriented in this repo, explaining or diagramming its architecture, locating where a
  subsystem lives, planning or reviewing a cross-cutting change, or onboarding someone — any time you
  need the mental model rather than a single command. Load it before editing so you respect the
  config-via-settings, async, dependency-injection, spec-driven-development, and coverage-gate rules
  that a change will otherwise fail on.
---

# Strategos-MCTS Primer

Orientation to this codebase for anyone starting work, explaining the design, or making a change that
crosses subsystems. `CLAUDE.md` is the always-on quick reference (commands, patterns); this skill is
the map you consult when you need the whole picture and where things actually live. **Paths here are
verified against the current tree** — prefer them over `CLAUDE.md`'s "Key File Locations" table, which
predates two moves (orchestration is now the `framework/graph/` package, not `graph.py`; component
factories moved to `framework/component_factory/`).

## What the system is

A multi-agent framework (PyPI name `langgraph-multi-agent-mcts`, Python ≥3.10) that combines three
ideas into a DeepMind-style reasoning system:

- **Hierarchical + iterative agents** — HRM (Hierarchical Reasoning Module) decomposes problems; TRM
  (Task Refinement Module) iteratively refines solutions; a hybrid agent blends LLM + neural policies.
- **MCTS search** — Monte Carlo Tree Search (classic, LLM-guided, and AlphaZero-style neural) explores
  action/solution spaces, optionally guided by policy/value networks.
- **LangGraph orchestration** — a stateful async graph wires agents, search, and a neural
  meta-controller (router) into one flow with checkpointing.

Around that core sit a training pipeline, RAG, a benchmark harness, an autonomous agent harness, and a
spec-driven development toolchain. The README is explicit that components are production-quality but
**full end-to-end integration is still in progress** — treat `docs/STATUS.md` as the source of truth
for what actually works today, not marketing copy.

## The layer model

```
 request ─▶ LangGraph orchestration ─▶ meta-controller routes to an agent ─▶ agent reasons
             (framework/graph)           (agents/meta_controller)              (agents/*, framework/agents)
                                                     │
                                          agent may invoke MCTS search ──▶ LLM adapters (provider-agnostic)
                                            (framework/mcts)                 (adapters/llm)
             cross-cutting, present at every layer:
               • config (config/)   • observability (observability/)   • assembly-theory features (framework/assembly)
```

## Where each subsystem lives

| Subsystem | What it is | Entry points |
|-----------|-----------|--------------|
| **Config** | All tunables & secrets — the `Settings(BaseSettings)` model and the cached `get_settings()` accessor. | `src/config/settings.py`, `src/config/constants.py` |
| **Orchestration** | LangGraph graph construction + the `AgentState` TypedDict flowing through it. | `src/framework/graph/builder.py`, `integrated.py`, `state.py` |
| **MCTS engine** | Search core plus neural / LLM-guided / parallel variants and domain adapters. | `src/framework/mcts/core.py`, `neural_mcts.py`, `llm_mcts.py`, `parallel_mcts.py`, `policies.py`, `progressive_widening.py` |
| **Application agents** | HRM / TRM / hybrid agents. | `src/agents/hrm_agent.py`, `trm_agent.py`, `hybrid_agent.py` |
| **Meta-controller** | Neural router (BERT / RNN / hybrid / assembly) that assigns a task to the best agent. | `src/agents/meta_controller/` (`bert_controller.py`, `rnn_controller.py`, `hybrid_controller.py`, `assembly_router.py`) |
| **Framework agents** | LLM-backed agent base used by the graph. | `src/framework/agents/base.py`, `llm_hrm.py`, `llm_trm.py` |
| **LLM adapters** | Provider-agnostic clients behind a Protocol, with a shared circuit breaker. | `src/adapters/llm/base.py`, `resilience.py`, `openai_client.py`, `anthropic_client.py`, `lmstudio_client.py` |
| **Observability** | Structured logging, Prometheus metrics, OTel tracing, decorators/facade. | `src/observability/logging.py`, `metrics.py`, `tracing.py`, `facade.py`, `decorators.py` |
| **Assembly theory** | Assembly-index features / substructure library (a distinctive scoring signal). | `src/framework/assembly/` (`calculator.py`, `concept_extractor.py`, `substructure_library.py`) |
| **Component factories** | DI/registry factories for trainers, data loaders, metrics. | `src/framework/component_factory/registry.py`, `*_factory.py` |
| **Benchmark** | System-vs-system evaluation harness + policy-lift measurement. | `src/benchmark/cli.py`, `factory.py`, `policy_lift.py`, `tasks/`, `evaluation/`, `reporting/` |
| **Agent harness** | Deterministic autonomous agent loop, tools, hooks, topologies, Ralph outer loop, record/replay. | `src/framework/harness/` (`cli.py`, `loop/`, `tools/`, `hooks/`, `topology/`, `ralph/`, `replay/`, `intent/`) |
| **Spec-driven dev** | Spec schema, validator, tracer, scaffolder driving the SDD workflow. | `src/framework/harness/intent/spec_loader.py`, `spec_validator.py`, `spec_trace.py`; specs in `specs/` |

Peripheral areas you'll meet less often: `src/training/` (ML pipeline), `src/neuro_symbolic/`,
`src/games/chess/`, `src/api/` (REST + inference servers), `src/enterprise/`, `src/integrations/`.

## Non-negotiable invariants

These are the rules a change is judged against — violating one is how a PR fails CI or review.

1. **All config flows through Pydantic Settings.** Read tunables and secrets via `get_settings()`;
   put shared defaults/bounds in `constants.py`. Never hardcode API keys or magic numbers — a
   `git grep` for `sk-` literals runs in the gate.
2. **I/O is async.** New I/O paths use `async`/`await` (httpx, aioboto3), matching the graph's
   async execution model.
3. **Dependencies are injected.** Components take config, clients, and a logger in `__init__`; they
   don't construct their own. This is what makes them testable.
4. **Unit tests never touch the network.** Mock every external call. Real API/LLM calls belong in
   integration/e2e tests behind markers, not `tests/unit`.
5. **Coverage is a gate, not a report.** Branch coverage `fail_under = 85.0` (`pyproject.toml`).
   `src/api/rest_server.py`, `src/api/inference_server.py`, and three `src/games/chess/` modules are
   config-omitted and won't show as movable targets.
6. **Fail loud by default.** Mock LLM / lightweight-framework fallbacks are **opt-in** via
   `ALLOW_MOCK_LLM_FALLBACK` / `ALLOW_LIGHTWEIGHT_FRAMEWORK_FALLBACK`; without them the service errors
   rather than silently serving mock output.
7. **`src/**` changes are spec-gated.** A change under `src/**` needs either a `spec/<id>` branch whose
   spec is `approved`, or a `No-Spec: <reason>` commit trailer. A PreToolUse hook
   (`.claude/hooks/spec_gate.py`) warns in-editor; the CI `spec-validate` job enforces it. See below.
8. **Logs are structured and secret-safe.** Log with a `correlation_id` and pass sensitive data
   through `sanitize_dict()` so secrets are masked.

## Workflows & commands

Three console scripts (declared in `pyproject.toml [project.scripts]`):

```bash
benchmark      # = python -m src.benchmark  — system-vs-system evaluation
harness        # autonomous agent loop: run | dry-run | replay | validate-spec | spec-trace | spec-new
policy-lift    # measure MCTS policy improvement (the M5 ≥20%-lift acceptance metric)
```

Reusable project skills (invoke by name) cover the routine loops so you don't reconstruct them:

- **`/quality-gate`** — the full local CI-equivalent gate (black → ruff → mypy → pytest+branch-cov →
  secret grep). Run before every push; green locally means green in CI.
- **`/validate-specs`** — validate `specs/*.SPEC.md` against harness spec schema v2.
- **`/coverage-baseline`** — regenerate the evidence-backed `docs/STATUS.md` baseline.

## Spec-driven development in one screen

Work is specified as `specs/<id>.SPEC.md` (schema v2: frontmatter `id`/`goal`/`module`/`status`
lifecycle `draft → approved → implemented → verified → superseded`; body `# Goal` /
`# Acceptance Criteria` with `AC-n:` IDs / `# Constraints`). The loop:

- `/spec-new <id> <module>` scaffolds a `draft` (deterministic refusal on bad ids / module overlap).
- The `spec-review` subagent gates `draft → approved`; a human flips the status.
- `/spec-implement <id>` requires `approved`, then cuts/switches to the `spec/<id>` branch from
  `origin/main`.
- CI traceability (`harness spec-trace`) requires `src/**` diffs to map to an approved spec or carry a
  `No-Spec: <reason>` trailer. Until the first approved spec merges, the trailer is the expected channel.

Full detail: `CLAUDE.md` → "Spec-Driven Development", and `docs/plans/SDD_PLUGIN_EXTRACTION_PLAN.md`
(the plan to package this toolchain as the reusable `claude-code-foundry` plugin).

## Deeper docs (go here, don't reinvent)

| Need | Read |
|------|------|
| **Current reality — what passes, coverage, gaps** (source of truth) | `docs/STATUS.md` |
| Architecture diagrams | `docs/C4_ARCHITECTURE.md`, `docs/C4_MERMAID_ARCHITECTURE.md` |
| Comprehensive implementation template (C4 + all sub-agents) | `MULTI_AGENT_MCTS_TEMPLATE.md` |
| Active roadmap | `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md` |
| Autonomous-agent routing ledger | `AGENTS.md` |
| Repo layout in depth | `PROJECT_STRUCTURE.md` |
| Training pipeline | `docs/LOCAL_TRAINING_GUIDE.md` |
| Quick commands & config patterns | `CLAUDE.md` |

## Gotchas

- **`CLAUDE.md` path drift.** Its "Key File Locations" lists `framework/graph.py` and
  `framework/factories.py`; the current tree has the `framework/graph/` package and
  `framework/component_factory/` (plus `framework/harness/factories.py`). Trust the table above.
- **Local test skips.** LMStudio tests need a local server (`LMSTUDIO_SKIP=1` to skip); Pinecone tests
  need a key or mocks; neural MCTS is slow on CPU (use CUDA or fewer iterations).
- **Persisted-artifact formats changed.** Substructure library is JSON; experience buffer is
  `torch.save(weights_only=True)`. Legacy `pickle` is read only behind
  `ASSEMBLY_TRUST_LEGACY_PICKLE` / `TRAINING_TRUST_LEGACY_PICKLE`, then migrated in place.
- **Tooling is pinned** (`ruff`, `mypy` in the `[dev]` extra) for CI/local parity — bump deliberately
  and re-run the full gate.
