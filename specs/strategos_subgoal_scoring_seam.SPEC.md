---
id: strategos_subgoal_scoring_seam
goal: Introduce an explicit, pluggable candidate-scoring seam in the LangGraph MCTS node that exposes real per-candidate value estimates and lets an opt-in scorer re-rank them, with the default (identity) path byte-for-byte unchanged, so later uncertainty and risk-penalty specs have a genuine value and a single injection point
module: src/framework/graph/
status: approved
---

# Goal

Today the LangGraph MCTS node (`src/framework/graph/builder.py`, `_mcts_simulator_node`) runs the
baseline `MCTSEngine`, which selects `best_action` internally via its `MAX_VISITS` policy, and the
node collapses the result into a scalar summary — so there is nowhere for a candidate scorer to
observe or influence selection. This spec adds an explicit candidate-scoring **seam**: the node
builds an ordered list of per-candidate records (`candidate_id`, `value`, `visits`) from the
engine's existing `action_stats`, and passes them through a `CandidateScorer` protocol before
emitting the final action. The **default** scorer is an identity pass-through that returns the
engine's own `MAX_VISITS` choice, so wiring the seam changes nothing observable; an **opt-in**
scorer (starting with a value-argmax scorer) can re-rank candidates without touching the search
core. No new subgoal semantics are invented and no new `AgentState`/node-output keys are added —
the exposure is at the `CandidateScorer` protocol boundary only. This is the enabling seam the
roadmap's MDN dispersion (`strategos_coarse_dynamics_mdn`) and risk-averse scorer
(`strategos_risk_averse_subgoal_scorer`) plug into.

# Acceptance Criteria

- AC-1: `_mcts_simulator_node` constructs an explicit ordered list of candidate records
  (`candidate_id`, `value`, `visits`) from `MCTSEngine.action_stats` at one injection point and
  passes them through a `CandidateScorer`; the **default** scorer is an identity pass-through that
  returns the engine's own `_select_best_action` (`MAX_VISITS`) choice, so with the default scorer
  and default settings the emitted `mcts_best_action`, the `agent_outputs` summary text, and the
  `confidence` value are byte-for-byte identical to the pre-change node on a fixed seed. Falsified by
  any divergence in selection, summary, or confidence on the default path. Intended test:
  `tests/unit/framework/graph/test_mcts_candidate_seam.py`.
- AC-2: The scorer is selected by a bounded Pydantic Settings enum
  (`GRAPH_MCTS_CANDIDATE_SCORER`, default `identity`) resolved through a factory; the opt-in
  `value` scorer re-ranks candidates by mean `value` with a deterministic first-wins tie-break, so
  when selected it makes `mcts_best_action` the highest-value candidate (authoritative over the
  engine's visit-argmax when they differ). An unrecognized scorer name raises at construction
  (a `ValueError`/`GraphConstructionError`), never a silent fallback to the default. Falsified by a
  `value` scorer that does not change selection on a value≠visits case, by a non-deterministic
  tie-break, or by an unknown name silently degrading to identity. Intended tests:
  `tests/unit/framework/mcts/test_candidate_scoring.py`,
  `tests/unit/framework/graph/test_mcts_candidate_seam.py`.
- AC-3: The seam preserves determinism on the default path: the default `CandidateScorer` draws no
  numbers from `MCTSEngine`'s shared RNG, adds no term to the value-sum accumulation, changes no
  child-insertion order, and resolves selection ties first-wins (matching `max(..., key=...)`), so
  two seeded runs of the default node are identical to each other and to the pre-change baseline;
  and importing/constructing the default (baseline) scoring path leaves `torch` out of
  `sys.modules`. Falsified by any RNG draw, altered accumulation, reordering, tie-break divergence,
  or a `torch` import on the baseline path. Intended test:
  `tests/unit/framework/graph/test_mcts_seam_determinism.py`.

# Constraints

- All new tunables live in Pydantic Settings with the default mirrored in `src/config/constants.py`;
  the scorer selector follows the `GraphHardeningSettings` `GRAPH_*` pattern in
  `src/config/graph_settings.py` (validated independently of the LLM API key), resolved to a
  `CandidateScorer` in `IntegratedFramework.__init__` and injected into `GraphBuilder` exactly like
  `retry_policy` / `trace_recorder`. No hardcoded values.
- The scoring module (`src/framework/mcts/scoring.py`) is pure and deterministic: no RNG, no I/O,
  no `torch`. Construction-time validation (unknown scorer name) happens in the factory when
  `IntegratedFramework` builds the scorer; a directly-constructed `GraphBuilder` with no scorer
  defaults to identity and is unchanged.
- Reuse existing surfaces — `MCTSEngine.action_stats`, `_select_best_action`, and the DI pattern of
  optional `GraphBuilder.__init__` params — no parallel selection logic.
- Unit tests carry the >=85% branch coverage gate; no network in unit tests.

# Invariants

- The baseline `MCTSEngine` selection semantics (`MAX_VISITS` default, UCB1) are unchanged; the seam
  observes and, only for a non-default scorer, re-ranks — it never alters backpropagation or the
  tree policy.
- `AgentState` shape, node-output keys, and the graph topology are unchanged; the per-candidate
  exposure exists only at the `CandidateScorer` protocol boundary inside the node.

# Out of Scope

- A `NeuralMCTS`-backed `ValueSource` (replacing each candidate's value with a network estimate).
  The graph node operates on placeholder string actions, not `GameState`, so a neural value source
  cannot run end-to-end in the node yet; it is deferred to a follow-up spec
  (`strategos_neural_value_source`) that also supplies the reasoning-domain state adapter.
- The MDN dispersion estimator and the risk-averse penalty (separate roadmap specs).
- Any change that would make a non-identity scorer active by default.
