# Related Work & Source Classification

This document classifies external sources that have influenced Strategos-MCTS, so that
engineering references are not mistaken for research lineage (and vice versa). When a
source informs implementation work, cite it here with its classification rather than in
code comments or as academic related work.

## Classification levels

| Level | Meaning |
|---|---|
| **Research citation** | Novel algorithm/architecture the system's design descends from; citable as related work. |
| **Engineering reference** | Practitioner guidance that shaped implementation hygiene or patterns; NOT citable as research novelty. |
| **Dataset attribution** | Data sources with license terms — tracked separately in [`ATTRIBUTION.md`](../ATTRIBUTION.md). |

## Sources

### arXiv:2607.19297 — "Graph-Based Agentic AI with LangGraph: Workflow Pathways for Long-Running Stateful Business Processes"

**Classification: Engineering reference.** A practitioner guide to LangGraph workflow
patterns. It contains no novel contribution to multi-agent MCTS or hierarchical
reasoning and must not be cited as related work for this system's architecture. It is
the source of the hardening recipes applied by spec
[`strategos_langgraph_hardening`](../specs/strategos_langgraph_hardening.SPEC.md):

Paths marked *(new)* are modules introduced by the implementing change; unmarked
paths already exist in the tree and are reused.

| Guide recipe | Strategos-MCTS realization |
|---|---|
| Typed state schemas | `AgentState` remains a `TypedDict` (recorded architecture decision); construction-time schema/topology validation and pre-invoke initial-state validation in `src/framework/graph/schema.py` *(new)* |
| Checkpoints & interrupts | Pluggable checkpointer + `interrupt_before`/`interrupt_after` in `IntegratedFramework` (`src/framework/graph/integrated.py`); per-job `thread_id`; kill-safe incremental run store + `--resume` for the benchmark sweep (`src/benchmark/evaluation/run_store.py` *(new)*) |
| Retries for flaky tool/worker nodes | Reuse of the existing `retry` decorator (`src/observability/decorators.py`) applied at node I/O boundaries, configured via `GRAPH_NODE_RETRY_*` settings |
| Execution traces | Per-transition structured events (`src/framework/graph/tracing.py` *(new)*) feeding the structured logger, `MetricsCollector.record_node_timing`, and an opt-in per-run JSONL sink |

Deliberate deviations from a literal reading of the guide, with reasons:

- **Retry wraps node I/O, not whole nodes.** Several graph nodes intentionally
  catch-and-degrade (e.g. RAG retrieval falls back to empty context); a whole-node
  retry wrapper would never observe those exceptions. Retries therefore sit inside the
  node at the transient I/O call, and existing degrade/propagate semantics apply only
  after retries are exhausted.
- **The benchmark harness's own backoff is not consolidated.** Its contract returns
  error results rather than raising; folding it into the raise-through node retry
  mechanism would change benchmark output semantics for no functional gain.
- **No durable checkpointer inside the LLM-guided MCTS graph.** Its in-memory,
  no-checkpointer design is intentional (the search tree is ephemeral); durable
  resume belongs to the long-running sweep loop that wraps graph invocations.

### Dataset sources

See [`ATTRIBUTION.md`](../ATTRIBUTION.md) (DABStep, PRIMUS, MITRE ATT&CK) for dataset
attribution and license terms.
