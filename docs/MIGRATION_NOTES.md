# Migration Notes

Externally observable changes that may affect callers, stored state, or output formats.
Newest first.

---

## 2026-07 — LangGraph orchestration hardening (`strategos_langgraph_hardening`)

Engineering-hardening of the LangGraph orchestration and the benchmark sweep. No changes to
the MCTS algorithm, hierarchical reasoning (HRM/TRM) logic, or model architecture. Behavior is
backward compatible except for the items below.

### Strict initial-state validation

`IntegratedFramework.process()` (and `astream` / `astream_events`) now validate the initial
state before the graph runs: a missing required key or an **unknown key** raises
`StateValidationError` rather than surfacing mid-execution. Callers that legitimately pass
extra keys can use the `allow_extra_keys=True` escape hatch on
`src.framework.graph.schema.validate_initial_state`. The public `process()` signature is
otherwise unchanged; a new optional `thread_id` parameter threads a per-job LangGraph thread
id in place of the previously hardcoded `"default"`.

### `mcts_root` state content change

The `mcts_root` channel of `AgentState` now carries a JSON-serializable summary dict
(`{"state_id", "tree_depth", "tree_node_count"}`) instead of the live `MCTSNode` object. The
live object was written to state but never read anywhere in `src/`, so this is safe; it keeps
checkpointed graph state serializable. `state.py` tightens the annotation to
`NotRequired[dict]`.

### New, versioned persistence formats (`schema_version: 1`)

These are additive and opt-in; nothing pre-existing is migrated:

- **Benchmark run store** — `<output_dir>/runs/<run_id>/results.jsonl` (append-only per-result
  log) plus a `run.json` manifest. Enabled by default via
  `BENCHMARK_RUN_INCREMENTAL_PERSISTENCE`; makes a crashed sweep resumable with
  `python -m src.benchmark --resume <run_id>`.
- **Execution trace files** — per-run `<GRAPH_TRACE_DIR>/<run_id>.jsonl`. Off by default
  (`GRAPH_TRACE_DIR` unset); structured-log/metric emission is enabled by default (`GRAPH_TRACE_ENABLED=true`) and can be disabled via `GRAPH_TRACE_ENABLED=false`.
- **SQLite graph checkpoints** — written only when `GRAPH_CHECKPOINT_BACKEND=sqlite` and the
  optional `langgraph-checkpoint-sqlite` extra is installed.

### Unaffected formats

- The legacy benchmark results artifact (`benchmark_results.json`, written by
  `save_results`) keeps its existing shape — it is produced from the full resumed+new result
  set.
- Self-play training checkpoints (`ckpt_iter_{n}.pt` + `.meta.json`) are not modified; training
  resume remains owned by the `m5_policy_lift` spec.
- LangGraph `MemorySaver` checkpoints were always ephemeral (in-process), so there is no prior
  on-disk graph checkpoint state to migrate.
