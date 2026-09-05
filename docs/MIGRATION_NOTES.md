# Migration Notes

Externally observable changes that may affect callers, stored state, or output formats.
Newest first.

---

## 2026-09-04 — Central seeding utility (`hygiene_determinism`)

`src/utils/seeding.py` is now the single entry point for process-wide seeding.

### Behavior changes

- **NeuralMCTS Dirichlet / action sampling** no longer draws from NumPy's process-global legacy RNG. Noise comes from an engine-owned `numpy.random.Generator` created via `new_rng(seed)` (or an injected `rng=`). Same seed → same noise on the same machine; callers that previously relied on ambient `np.random.seed` alone to control NeuralMCTS noise must pass `seed=` / `rng=` (or rely on `config.seed` / `Settings.SEED` / `DEFAULT_SEED`).
- **Migrated trainers/CLIs** (`self_play_convergence`, `self_play_trainer`, `policy_lift`, `train_rnn`, `train_bert_lora`, `unified_orchestrator`, meta-controller data collector / RNN & BERT controllers) call `set_all_seeds`, which also seeds Python's `random` module. Sites that previously only set torch (or only torch+numpy) now seed all three when torch is installed.

### Not a breaking API change

Constructor and CLI `seed=` kwargs are preserved. `set_all_seeds` returns the effective rank-aware seed (`seed + rank`). No new seed environment variable was added — continue to use `Settings.SEED` / `DEFAULT_SEED`.

---

## 2026-08-22 — `DEPLOYMENT_ENV` and the fail-loud posture refusal (`evidence_claim_ledger`)

`Settings` gains a `DEPLOYMENT_ENV` field (one of `development`, `test`, `staging`, `production`;
default `development`) and a model validator, `validate_fail_loud_posture`.

### Two new startup errors

1. **An unrecognised `DEPLOYMENT_ENV` value is rejected.** Previously no such setting existed, so
   a typo in a manifest was impossible; now `DEPLOYMENT_ENV=prod` fails at construction rather
   than being silently treated as non-production. Use the exact literal `production`.
2. **`ALLOW_MOCK_LLM_FALLBACK=true` is refused when `DEPLOYMENT_ENV` is `staging` or `production`.**
   The error names the remedy. Previously the flag was honoured everywhere, so a staging deployment
   could serve mock output that was indistinguishable from a real answer.

### Who is affected

**No existing deployment changes behaviour on upgrade.** The default is `development`, no manifest
in the tree declares the variable, and the refusal only binds once it is declared. This is the
migration's deliberate limit and it is recorded honestly: `CL-9` in `docs/CLAIM_LEDGER.md` stays
`PARTIAL`, not `PROVEN`, because an operator who never sets `DEPLOYMENT_ENV` still gets the
permissive posture.

**Action required for production operators:** set `DEPLOYMENT_ENV=production` in your deployment
environment (Docker `-e`, k8s `env:`, or Space secret) and leave `ALLOW_MOCK_LLM_FALLBACK` unset.
If a startup error appears after doing so, it is reporting a pre-existing misconfiguration that was
previously silent.

### Not a breaking change for tests

`DEPLOYMENT_ENV` is unset in the test environment, so the permissive branch applies and existing
fixtures that enable the mock fallback continue to work. Tests that need the strict behaviour set
the variable explicitly — see `tests/unit/config/test_fail_loud_posture.py`.

---

## 2026-07-31 — MCTS negamax value-semantics fix (`hygiene_mcts_value_semantics`)

Fixes three proven, executable-proof-verified selection bugs in
`src/framework/mcts/{neural_policies,parallel_mcts,progressive_widening}.py`. **There is no
escape hatch back to the old behavior** — it was never correct, so this is not a configurable
change, it is a correction.

### `select_child_puct` no longer double-divides Q

`neural_policies.select_child_puct` computed `q_value = child.value / child.visits`, but
`MCTSNode.value` is already the mean (`value_sum / visits`), so Q was silently divided by
visits a second time — collapsing toward 0 as visits grew and turning PUCT into a near-pure
exploration bandit. It now delegates directly to the canonical `puct()` formula (same file),
so it cannot drift from it again. **Impact:** any caller of `select_child_puct` will see
different (correct) selections, particularly on well-visited trees where the old bug was most
severe.

### `ParallelMCTSEngine` and `ProgressiveWideningEngine` now select on the correct perspective

Both engines flip the backpropagated value's sign per ply (negamax), but selection
(`VirtualLossNode.select_child_with_vl`, `RAVENode.select_child_rave`) read the child's stored
value without negating it — selecting the move that is best **for the opponent**, not the
root. This mirrors a bug already found and fixed in `neural_mcts.NeuralMCTSNode.select_child`
(the `negate_child_value` parameter), which was never ported to these two engines.

Both selection methods now accept a `negate_child_value: bool = False` parameter (RAVE also
negates the RAVE/AMAF mixing term, which was equally unnegated). Both engines gained a
`two_player: bool` field/parameter (`ParallelMCTSConfig.two_player`,
`ProgressiveWideningEngine(..., two_player=...)`), **defaulting to `True`**, wired through to
selection as `negate_child_value=self.two_player`. A new settings field,
`Settings.MCTS_TWO_PLAYER` (default `True`), is the project-wide override for callers that
construct engines without an explicit config.

**Impact:** with the default (`two_player=True`), root action selection for adversarial
two-player search changes — this is the fix. Callers that were relying on the old (broken)
selection can set `two_player=False`, but that also disables the (now-correct) backprop-vs-
selection consistency and should only be used for genuinely single-agent, non-adversarial
search, matching `core.MCTSEngine`'s untouched, always-unflipped convention (see
`tests/unit/framework/mcts/test_value_semantics_regression.py::TestCrossEngineSingleAgentParity`
for the formal parity guarantee at `two_player=False`).

### `core.MCTSEngine` / `core.MCTSNode` are unchanged

Neither backpropagation nor selection in `core.py` ever flipped sign, so the two were already
mutually consistent; this phase does not touch its numerics. This preserves the bit-for-bit
baseline the (approved, not yet implemented) `strategos_risk_averse_subgoal_scorer` spec
depends on — this fix lands first and becomes part of that spec's baseline going forward.

### Benchmarks and the M5 policy-lift gate

No stored benchmark artifact currently depends on `ParallelMCTSEngine` or
`ProgressiveWideningEngine` output (`benchmarks/results/reasoning_smoke_lift.json` does not use
either engine, and `docs/STATUS.md`'s M5 section already states no `≥20%` lift claim exists
yet). **Flag for the `m5_policy_lift` spec thread:** if any future chess self-play or
evaluation run reuses either engine, its results must be generated (or regenerated) after this
fix, not before — do not treat any pre-2026-07-31 output from these two engines as a valid
baseline for that gate.

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
- **Execution trace files** — when tracing is enabled (`GRAPH_TRACE_ENABLED`, default true),
  every node transition emits to the structured logger and per-node metrics; setting
  `GRAPH_TRACE_DIR` additionally writes per-run `<GRAPH_TRACE_DIR>/<run_id>.jsonl` files (off by
  default). Set `GRAPH_TRACE_ENABLED=false` to disable all trace emission.
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
