---
id: strategos_subgoal_scoring_seam
goal: Introduce an explicit, pluggable candidate-scoring seam in the LangGraph MCTS node that exposes real per-candidate value estimates and lets a scorer rank candidates, with the baseline path bit-for-bit unchanged, so later uncertainty and risk-penalty specs have a genuine value and a single injection point
module: src/framework/graph/
status: draft
---

# Goal

Today the LangGraph MCTS node (`src/framework/graph/builder.py`, `_mcts_simulator_node`) runs
the baseline `MCTSEngine` over hardcoded placeholder actions and collapses the result into a
single scalar summary, so there is nowhere for a candidate scorer to observe or influence
selection. This spec adds an explicit candidate-scoring **seam**: the node builds a list of
per-candidate records from the engine's existing `action_stats`, passes them through a
`CandidateScorer` protocol (default: value-only, order-preserving) before final selection, and
exposes a `ValueSource` abstraction so a candidate's value can optionally come from the existing
`NeuralMCTS` head instead of the baseline mean. Default behavior — baseline engine, default
`MCTS_IMPL`, default scorer — is byte-for-byte identical to the pre-change node. No new subgoal
semantics are invented here; this is the enabling seam that the roadmap's MDN dispersion
(`strategos_coarse_dynamics_mdn`) and risk-averse scorer (`strategos_risk_averse_subgoal_scorer`)
plug into.

# Acceptance Criteria

- AC-1: `_mcts_simulator_node` constructs an explicit ordered list of candidate records
  (`candidate_id`, `value`, `visits`) from `MCTSEngine` `action_stats` at one injection point and
  routes them through a `CandidateScorer` protocol whose default implementation ranks by value only
  and preserves the engine's existing ordering; with the default scorer and default settings the
  selected candidate, the emitted result summary, and the `confidence` value are byte-for-byte
  identical to the pre-change node on a fixed seed. Falsified by any divergence in selection,
  summary, or confidence on the default path. Intended test:
  `tests/unit/framework/graph/test_mcts_candidate_seam.py`.
- AC-2: A `ValueSource` abstraction supplies each candidate's value; `MCTS_IMPL=neural` selects a
  `NeuralMCTS`-backed source (via `evaluate_state`) for domains that provide a network and a state
  adapter, and when the `neural` extra (torch) is absent or no network is configured the node raises
  `GraphConstructionError` at construction — never a silent fallback to the baseline mean; the
  default `MCTS_IMPL` is unchanged so non-neural installs behave exactly as before. Falsified by a
  silent fallback, a top-level torch import, or a changed default. Intended test:
  `tests/unit/framework/graph/test_value_source_neural.py` (importorskip torch).
- AC-3: The seam is deterministic-preserving when not exercised: the default `CandidateScorer` and
  `ValueSource` draw no numbers from `MCTSEngine`'s shared RNG, add no terms to the value-sum
  accumulation, and change no child-insertion order, so two seeded runs of the default node are
  identical and match the pre-change baseline. Falsified by any RNG draw, altered accumulation, or
  reordering on the default path. Intended test:
  `tests/unit/framework/graph/test_mcts_seam_determinism.py`.

# Constraints

- All new tunables (scorer selection, value-source selection) live in Pydantic Settings with bounds
  and defaults mirrored in `src/config/constants.py`; reuse the `MCTS_IMPL`/`MCTSImplementation`
  selector pattern in `src/config/settings.py`. No hardcoded values.
- torch stays optional: the `NeuralMCTS` value source is import-guarded and reachable only when
  `MCTS_IMPL=neural`; the baseline node path imports no torch.
- Reuse existing surfaces — `MCTSEngine.action_stats`, `_select_best_action`, and the
  `MCTS_IMPL` selector — no parallel selection logic.
- Unit tests carry the >=85% branch coverage gate; no network in unit tests.

# Invariants

- The baseline `MCTSEngine` selection semantics (`MAX_VISITS` default, UCB1) are unchanged; the seam
  observes and optionally re-ranks, it does not alter backpropagation or the tree policy.
- `AgentState` shape and the graph topology are unchanged; only the node's internal scoring path
  gains the seam.

# Out of Scope

- The MDN dispersion estimator and the risk-averse penalty (separate roadmap specs).
- A reasoning-domain adapter for `NeuralMCTS` (games only here; reasoning-domain wiring is a
  follow-up spec if pursued).
- Any change that would make the seam active by default.
