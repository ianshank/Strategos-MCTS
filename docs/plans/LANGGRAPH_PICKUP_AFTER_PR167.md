# LangGraph pickup plan after PR #167

> **Status:** living plan · **Branch context:** `qa/full-suite-regression-2026-09-04` / PR #167
> **Authority:** sequenced work still defers to `docs/plans/EVIDENCE_FIRST_PROGRAM.md` for product north-star; this file only covers LangGraph / orchestration follow-ons after the godfile decomposition.

## What PR #167 already landed

- `GraphBuilder` split into mixins under `src/framework/graph/builder_components/`
- `UnifiedTrainingOrchestrator` split into mixins under `src/training/orchestrator_components/`
- Training E2E runners, knowledge-graph module + tests, AQA / godfile agent skills
- Spec `strategos_langgraph_hardening` remains **implemented** (validation, retry, tracing, checkpoint, benchmark resume)

## Phase A — Unblock merge (CI)

| Failure | Cause | Fix |
| --- | --- | --- |
| Lint & Format | Black: 22 files (mixins + scaffolding scripts) | Format keepers; delete one-shot `refactor_*.py` / `check_ast.py` / `builder_imports.txt` |
| Type Check (mypy) | Mixin modules reference attrs on the composed class | File-level mypy disable for mixin attr-defined/misc/assignment pending Protocol stubs |
| Spec Validation | `godfile-decomposer.md` missing frontmatter + bare paths | Add YAML frontmatter; cite full repo paths |
| Pre-Deployment Sanity | `W293` in `scripts/run_e2e_inference.py` | Strip whitespace-only blank line in f-string |

Gate: CI Pipeline green on PR #167; do not stack follow-ons on a red base.

## Phase B — Immediate LangGraph follow-ons (post-merge)

1. Replace mixin mypy pragmas with `Protocol` / `self: "GraphBuilder"` annotations.
2. Streaming / viz hardening: live `ENABLE_STREAMING` / `ENABLE_GRAPH_VISUALIZATION` against real `IntegratedFramework` (not only mocks).
3. Documented sqlite checkpoint + `thread_id` kill/resume smoke.
4. Optional: wire knowledge-graph retrieval into `retrieve_context` behind settings (avoid a parallel island).

## Phase C — Evidence-First alignment

Do **not** jump to MuZero / distributed self-play. After #167 merges:

1. Confirm E0 artifacts already on tree (`docs/CLAIM_LEDGER.md`, `evidence_claim_ledger` spec implemented).
2. Continue E1b → E2 (determinism, MCTS value semantics) per `docs/plans/EVIDENCE_FIRST_PROGRAM.md`.
3. Use graph traces + provenance for golden-path runs rather than adding orchestration nodes.

## Suggested next PR titles

1. `fix(ci): green PR #167 — black, mypy mixins, context-docs, W293`
2. `refactor(graph): Protocol stubs for GraphBuilder mixins`
3. `test(graph): streaming + checkpoint resume smokes`
