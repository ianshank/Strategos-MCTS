---
id: strategos_coarse_dynamics_mdn
goal: Provide a torch-guarded mixture-density uncertainty head that predicts a distribution over coarse (multi-step aggregated) state transitions and exposes a scalar dispersion metric, extending the existing optional scalar uncertainty head without changing default behavior
module: src/models/
status: draft
---

# Goal

The proposal's dispersion signal needs a learned model of *coarse* dynamics — environment
transitions aggregated over multiple low-level steps — and a scalar dispersion metric derived from
it. `src/models/value_network.py` already has an optional single-scalar epistemic uncertainty head
(`estimate_uncertainty`), but a single Softplus scalar is not a mixture density and cannot express
multi-modal predictive spread. This spec (a) defines the coarse-transition input contract — how a
sequence of low-level states is aggregated into one coarse transition — and (b) adds a Mixture
Density head with a configurable number of components that outputs a scalar dispersion metric
(mixture-variance trace or mixture entropy). It is inert unless explicitly enabled; the existing
scalar head remains available.

# Acceptance Criteria

- AC-1: A documented coarse-transition input contract exists — a deterministic aggregator turns an
  ordered sequence of low-level states into a single fixed-shape coarse-transition tensor — with the
  aggregation window configurable; the same input yields the same tensor. Falsified by a
  non-deterministic or shape-unstable aggregation. Intended test:
  `tests/unit/models/test_coarse_transition.py`.
- AC-2: A Mixture Density head with a configurable component count consumes the coarse-transition
  tensor and outputs mixture parameters plus a single non-negative scalar dispersion metric
  (variance trace or entropy); component count and metric choice are configuration, not hardcoded.
  Falsified by a fixed component count, a negative dispersion, or a non-scalar dispersion. Intended
  test: `tests/unit/models/test_coarse_dynamics_mdn.py` (importorskip torch).
- AC-3: The head is optional and backward compatible: with the mixture head disabled the existing
  `ValueNetwork` outputs (value, features, scalar uncertainty) are unchanged, and with torch or the
  `neural` extra absent the module imports without error and its tests skip. Falsified by a change to
  default `ValueNetwork` outputs or by a top-level torch import that breaks non-neural installs.
  Intended test: `tests/unit/models/test_value_network_mdn_compat.py`.

# Constraints

- torch-only code lives behind the `neural` extra and the existing optional-import guard pattern;
  tests use `pytest.importorskip("torch")`.
- Component count, aggregation window, and dispersion-metric choice are Pydantic Settings or
  constructor configuration with defaults in `src/config/constants.py`; no magic numbers.
- Extend `src/models/value_network.py` and siblings; do not fork a parallel network.
- Unit tests carry the >=85% branch coverage gate.

# Invariants

- The existing scalar `estimate_uncertainty` head and the `ValueOutput` contract remain available and
  unchanged for current callers.

# Out of Scope

- Wiring the dispersion metric into subgoal scoring (the `strategos_risk_averse_subgoal_scorer` spec).
- Training the MDN head or producing a checkpoint; this spec delivers the architecture and metric only.
- Any change to the MCTS engines or the graph node.
