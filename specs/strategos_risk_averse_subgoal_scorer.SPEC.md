---
id: strategos_risk_averse_subgoal_scorer
goal: Add a risk-averse candidate scorer that ranks by score = value minus lambda times dispersion, plugged into the scoring seam behind an off-by-default flag that overrides the configured scorer, preserving the baseline selection bit-for-bit when disabled
module: src/framework/mcts/
status: approved
---

# Goal

Building on the scoring seam (`strategos_subgoal_scoring_seam`) and the dispersion metric
(`strategos_coarse_dynamics_mdn`), this spec adds a `RiskAverseSubgoalScorer` implementing the
seam's `CandidateScorer` protocol that ranks candidates by `score = value - lambda * dispersion`,
where `lambda >= 0` is a bounded configuration weight and `dispersion` is supplied by a pluggable
`DispersionSource`. It is gated by an off-by-default flag; when the flag is OFF the risk scorer is
never constructed, no dispersion is computed, and selection is byte-for-byte identical to the
configured baseline scorer (default `identity`).

# Acceptance Criteria

- AC-1: `RiskAverseSubgoalScorer.score(c) == c.value - lambda * dispersion(c)`, where `dispersion`
  comes from an injected `DispersionSource` (`dispersion_for(candidate) -> float`, non-negative;
  the default reads `candidate.metadata['dispersion']`, 0.0 if absent). `select_best` returns the
  highest-scoring candidate with a first-wins tie-break. Two behaviours are pinned: (a) equal value
  with `dispersion(A) < dispersion(B)` gives `score(A) > score(B)` for every `lambda > 0` (A wins;
  at `lambda = 0` scores tie and first-wins applies); (b) when `value(B) > value(A)` and
  `dispersion(B) > dispersion(A)`, selection is B for `lambda` below the crossover
  `lambda* = (value(B) - value(A)) / (dispersion(B) - dispersion(A))` and flips to A above it,
  recovering pure-value ranking at `lambda = 0`. Falsified by a ranking that ignores dispersion or
  `lambda`, a non-deterministic tie-break, or a flip at the wrong crossover. Intended test:
  `tests/unit/framework/mcts/test_risk_averse_scorer.py`.
- AC-2: The behaviour is gated by `ENABLE_UNCERTAINTY_SUBGOAL_PENALTY` (default False, in
  `graph_settings.py`). When OFF the risk scorer is not instantiated, no dispersion is computed, no
  draw is taken from the MCTS RNG, and the node's selection/summary/confidence are byte-for-byte
  identical on a fixed seed to the configured baseline scorer (default `identity`). When ON it
  overrides `GRAPH_MCTS_CANDIDATE_SCORER` with the `RiskAverseSubgoalScorer`. Falsified by any
  divergence from the baseline when OFF, by dispersion being computed-then-zero-weighted rather than
  skipped, or by the flag not overriding the configured scorer when ON. Intended test:
  `tests/integration/framework/graph/test_subgoal_penalty_off_identity.py`.
- AC-3: `SUBGOAL_UNCERTAINTY_LAMBDA` is a bounded Pydantic Settings float in `graph_settings.py`
  (`ge=MIN_SUBGOAL_UNCERTAINTY_LAMBDA`, `le=MAX_SUBGOAL_UNCERTAINTY_LAMBDA`), its default sourced from
  `constants.py`, validated at settings load; a negative or above-max value is rejected at
  construction. Falsified by an unbounded or hardcoded weight. Intended test:
  `tests/unit/config/test_subgoal_uncertainty_settings.py`.

# Constraints

- Primary module `src/framework/mcts/` (the scorer + `DispersionSource` in `risk_scoring.py`);
  additional in-scope paths: `src/config/graph_settings.py` and `src/config/constants.py` (the flag,
  `lambda`, bounds), and a thin selection branch in `src/framework/graph/integrated.py`. Config lives
  in `graph_settings.py` (validated independently of the LLM API key), not the monolithic `Settings`.
- Dispersion enters via a pluggable `DispersionSource` (a metadata-reading bridge by default, plus
  synthetic/callable sources for tests); the risk scorer imports no torch and does not import the MDN.
- The OFF path follows the `enable_early_termination` precedent: gated construction, gated arithmetic,
  no RNG draw, no accumulation change.
- Reuse the seam's `CandidateScorer` protocol and `CandidateRecord`; no parallel scoring implementation.
- Unit tests carry the >=85% branch coverage gate.

# Invariants

- With the flag OFF the system is bit-for-bit identical to the configured baseline scorer (default
  `identity`).
- The scorer only re-ranks candidates at the seam; it never modifies MCTS backpropagation, the tree
  policy, or the graph topology.
- `ENABLE_UNCERTAINTY_SUBGOAL_PENALTY` is owned (defined) by this spec; the benchmark spec only
  consumes it.

# Out of Scope

- Attaching *real* MDN-over-states dispersion to the graph's candidates: the node's candidates are
  placeholder string actions with no coarse state sequence, so with the flag ON in the node,
  `dispersion` defaults to 0.0 and the risk scorer equals value ranking until a follow-up wires real
  coarse states (same limitation that deferred the seam's neural value source). A test demonstrates
  the metadata bridge with a real MDN-produced dispersion value in isolation.
- Turning the flag on by default (gated on the benchmark spec's result).
- Training the MDN or producing a checkpoint.
