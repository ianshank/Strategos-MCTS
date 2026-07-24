---
id: strategos_risk_averse_subgoal_scorer
goal: Add a risk-averse candidate scorer that combines the MCTS value estimate with the MDN dispersion metric as score = value minus lambda times dispersion, plugged into the scoring seam behind an off-by-default flag, preserving baseline selection bit-for-bit when disabled
module: src/framework/mcts/
status: draft
---

# Goal

With a scoring seam (`strategos_subgoal_scoring_seam`) and a dispersion metric
(`strategos_coarse_dynamics_mdn`) in place, this spec adds the risk-averse combination the S3
proposal targets: a `RiskAverseSubgoalScorer` implementing the seam's `CandidateScorer` protocol
that ranks candidates by `score = value - lambda * dispersion`, where `lambda >= 0` is a bounded
configuration weight. It is gated by an off-by-default feature flag; when the flag is OFF the scorer
is never constructed, computes nothing, and selection is byte-for-byte identical to the value-only
baseline — not merely `lambda = 0`. This is the corrected form of the original proposal's two ADDED
requirements, now resting on real values and a real dispersion signal instead of a placeholder
subgoal-value interface that does not exist.

# Acceptance Criteria

- AC-1: `RiskAverseSubgoalScorer` ranks candidates by `value - lambda * dispersion`; for two
  candidates with equal value and `dispersion(A) < dispersion(B)`, `score(A) > score(B)` for every
  `lambda > 0`, and the order flips to favor B only at `lambda = 0` (pure value). Falsified by a
  ranking that does not penalize higher dispersion or that ignores `lambda`. Intended test:
  `tests/unit/framework/mcts/test_risk_averse_scorer.py`.
- AC-2: The behavior is gated by `ENABLE_UNCERTAINTY_SUBGOAL_PENALTY` (default False): when OFF the
  scorer is not instantiated, the dispersion path is not computed, no draw is taken from the MCTS
  RNG, and the selected candidate plus summary are byte-for-byte identical to the seam's default
  value-only path on a fixed seed. Falsified by any divergence from baseline when OFF, or by
  dispersion being computed-then-zero-weighted rather than skipped entirely. Intended test:
  `tests/integration/framework/graph/test_subgoal_penalty_off_identity.py`.
- AC-3: `lambda` is a bounded Pydantic Settings float (`SUBGOAL_UNCERTAINTY_LAMBDA`, `ge=0`) with its
  default in `src/config/constants.py`, validated at settings load; an out-of-range value is
  rejected. Falsified by an unbounded or hardcoded weight. Intended test:
  `tests/unit/config/test_subgoal_uncertainty_settings.py`.

# Constraints

- Primary module `src/framework/mcts/` (the scorer, beside existing policies); additional in-scope
  paths: `src/config/settings.py` and `src/config/constants.py` (the flag and `lambda`), and a thin
  registration in `src/framework/graph/` that selects this scorer at the seam. No other `src/` paths.
- The OFF path follows the `enable_early_termination` precedent: gated construction, gated arithmetic,
  no RNG draw, no accumulation change.
- Reuse the seam's `CandidateScorer` protocol and the MDN dispersion metric; no parallel scoring or
  uncertainty implementations.
- Implemented on its spec branch only after `strategos_subgoal_scoring_seam` and
  `strategos_coarse_dynamics_mdn` are merged (dependency order; one open graph-touching spec at a time).
- Unit tests carry the >=85% branch coverage gate.

# Invariants

- With the flag OFF the system is bit-for-bit identical to the pre-scorer baseline.
- The scorer only re-ranks candidates at the seam; it does not modify MCTS backpropagation, the tree
  policy, or the graph topology.

# Out of Scope

- Turning the flag on by default (gated on the benchmark spec's result).
- Producing or training the MDN checkpoint that supplies dispersion (owned upstream).
- Benchmark A/B measurement (the `strategos_subgoal_uncertainty_benchmark` spec).
