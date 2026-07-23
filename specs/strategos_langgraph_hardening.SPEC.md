---
id: strategos_langgraph_hardening
goal: Harden the LangGraph orchestration layer with construction-time state validation, checkpoint/resume for long-running jobs, retry-with-backoff at node I/O boundaries, and structured execution traces
module: src/framework/graph/
status: draft
---

# Goal

Apply four practitioner hardening patterns (typed/validated state, checkpoint and
resume, retry with backoff, execution trace logging) to the existing LangGraph
orchestration in `src/framework/graph/` and to the long-running benchmark sweep in
`src/benchmark/`, without changing the MCTS algorithm, hierarchical reasoning
logic, or model architecture. The source for these patterns (arXiv:2607.19297) is
an engineering reference, not a research citation, and is classified as such in
`docs/related-work.md`. The long-running job this spec targets is the benchmark
sweep (`python -m src.benchmark`), which today holds all results in memory and
loses everything on a crash; training resume is owned by `m5_policy_lift` and out
of scope here.

# Acceptance Criteria

- AC-1: `GraphBuilder.build_graph()` validates the state schema and the wired
  topology at construction time: an unresolvable or malformed `AgentState`
  annotation, a reducer channel without a callable, a duplicate node name, an
  unregistered edge endpoint, or a conditional-routing target with no matching
  node raises `GraphConstructionError` before the graph is compiled; and
  `IntegratedFramework.process()` rejects an initial state missing required keys
  (or carrying unknown keys, unless the caller passes the `allow_extra_keys`
  escape hatch) with `StateValidationError` before the first node executes. Falsified by any of
  these defects surfacing mid-execution instead of at construction/invocation
  boundaries. Intended test: `tests/unit/test_graph_state_schema.py`.
- AC-2: Worker nodes retry transient failures at their I/O boundaries with
  exponential backoff — exception allowlist, max attempts, initial delay, and
  backoff factor all configurable via `GRAPH_NODE_RETRY_*` settings — such that a
  configured transient exception raised on the first attempt is retried and the
  node succeeds when a subsequent attempt succeeds; a non-allowlisted exception
  is not retried; and exhausting max attempts falls through to the node's
  existing degrade-or-propagate behavior unchanged. Falsified by a whole-node
  wrapper that never observes exceptions swallowed inside node bodies, by
  retries on non-allowlisted exceptions, or by altered post-exhaustion node
  semantics. Intended tests: `tests/unit/test_graph_node_retry.py`,
  `tests/integration/test_graph_retry_fault_injection.py`.
- AC-3: Every node transition of a graph execution emits a structured trace
  event carrying run id, thread id, node name, monotonic sequence number,
  timestamp, input/output state digests, duration, final status, and attempt
  count; events always flow to the structured logger and per-node timings to
  `MetricsCollector.record_node_timing`; and when a trace directory is
  configured via `GRAPH_TRACE_DIR`, `load_trace(root, run_id)` returns the
  ordered event sequence from which `reconstruct_path` yields the executed node
  path. Falsified by a
  completed run whose trace cannot reconstruct the execution path, by missing
  transitions, or by interleaved sequence numbers across concurrent runs.
  Intended test: `tests/unit/test_graph_tracing.py`.
- AC-4: `IntegratedFramework` accepts an injected checkpointer and
  interrupt-before/after node lists, selects its checkpoint backend via
  `GRAPH_CHECKPOINT_BACKEND` (`memory` default; `sqlite` uses the optional
  `langgraph-checkpoint-sqlite` dependency, and an explicit `sqlite` selection
  with the dependency absent raises `GraphConstructionError` at construction —
  never a silent fallback that would fake durability), and threads a
  caller-supplied per-job `thread_id` through `process()` in place of the
  previous hardcoded constant; graph state remains JSON-serializable (the MCTS
  node summary replaces the live tree object in state). Falsified by a hardcoded
  thread id, by a silent fallback or uncontrolled import error when `sqlite` is
  selected without the extra installed, or by non-serializable values written
  into checkpointed state. Intended test:
  `tests/unit/test_framework_graph.py` (extended).
- AC-5: A benchmark sweep persists each scored result durably before advancing
  (append-only JSONL run store with a versioned manifest), and
  `python -m src.benchmark --resume <run_id>` re-runs only the
  (iteration, system, task) cells absent from the store — after a hard kill
  mid-sweep, resuming completes the full matrix with no lost and no duplicated
  results and without re-scoring completed cells, while the final results
  artifact (`benchmark_results.json` by default) keeps its existing shape.
  Falsified by data loss after SIGKILL, duplicate cells after resume,
  re-execution of completed cells, or a changed results-artifact schema. Intended tests:
  `tests/unit/benchmark/test_run_store.py`,
  `tests/unit/benchmark/test_harness_resume.py`,
  `tests/integration/test_benchmark_kill_resume.py`.
- AC-6: `docs/MIGRATION_NOTES.md` documents every externally observable change
  introduced by this spec — strict initial-state validation, the `mcts_root`
  summary-dict content change, and the new versioned persistence formats
  (benchmark run store, trace files, sqlite checkpoints) — and states that the
  legacy results artifact (`benchmark_results.json`) and training checkpoints
  are unaffected. Falsified by the note being absent or by any of those
  enumerated changes missing from it. Intended test:
  `tests/unit/test_migration_notes.py` (asserts the note exists and contains
  each enumerated change).

# Constraints

- No changes to the MCTS algorithm, hierarchical reasoning (HRM/TRM) logic, or
  model architecture; `src/framework/mcts/llm_guided/engine.py` keeps its
  intentional no-checkpointer design.
- `AgentState` remains a `TypedDict` (recorded architecture decision); validation
  is additive, not a Pydantic conversion.
- All new tunables live in Pydantic Settings with bounds, defaults mirrored in
  `src/config/constants.py`; no hardcoded values.
- Governed footprint beyond the frontmatter `module` prefix: AC-5 touches
  `src/benchmark/evaluation/` (harness plus new run store) and
  `src/benchmark/cli.py`; settings work touches `src/config/settings.py` and
  `src/config/constants.py`. No other `src/` paths are in scope.
- Reuse existing building blocks — the `retry` decorator in
  `src/observability/decorators.py`, correlation IDs and the structured logger in
  `src/observability/logging.py`, `MetricsCollector.record_node_timing`, and the
  atomic O_APPEND JSONL idiom from `src/framework/harness/memory/events.py` — no
  parallel implementations.
- Unit tests carry the coverage gate (>=85% branch); graceful degradation when
  optional dependencies (langgraph, the sqlite checkpoint extra) are absent.

# Invariants

- The benchmark harness's existing per-task timeout/retry contract (error
  results, not raised exceptions) is unchanged; its hand-rolled backoff is
  deliberately not consolidated into the node retry mechanism.
- Self-play convergence checkpointing (`ckpt_iter_{n}.pt` + `.meta.json`) is the
  authority for training resume and is not modified.

# Out of Scope

- Durable checkpointing inside the LLM-guided MCTS graph (in-memory by design).
- Consolidating adapter-layer tenacity retries with node-layer retries.
- Harness topology orchestration (`src/framework/harness/topology/` is not
  LangGraph and is untouched).
- Human-in-the-loop review flows beyond exposing `interrupt_before`/`interrupt_after`.
