---
id: strategos_coarse_dynamics_mdn
goal: Provide a torch-guarded Mixture Density head over coarse (multi-step aggregated) state transitions that exposes a non-negative scalar dispersion metric, as a standalone src/models module that leaves ValueNetwork/ValueOutput and their callers untouched
module: src/models/
status: approved
---

# Goal

The S3 dispersion signal needs (a) a coarse-transition representation — a deterministic
aggregation of a short sequence of low-level states into one fixed-shape vector — and (b) a
Mixture Density Network over that vector that yields a scalar predictive-dispersion metric.
`src/models/value_network.py` already has an *optional single-scalar* epistemic uncertainty head
(`estimate_uncertainty`), which motivates this but is not a mixture density and cannot express
multi-modal spread. Rather than extend `value_network.py` — which **hard-imports torch** and whose
`forward(state)` takes a single `[batch, state_dim]` state (not a sequence) — this spec adds a new
**standalone, torch-guarded** module `src/models/coarse_dynamics.py`: a torch-free
`CoarseTransitionAggregator` and a torch-backed `CoarseDynamicsMDN`. `ValueNetwork`, `ValueOutput`,
and their callers are untouched. The head is architecture-only (no training/checkpoint here) and
plugs into the risk-averse scorer in a later spec.

# Acceptance Criteria

- AC-1: `CoarseTransitionAggregator` deterministically reduces an ordered sequence of low-level
  state vectors (each length `state_dim`) over a configurable `window` into a single fixed-shape
  coarse-transition vector of length `4 * state_dim` — the concatenation of [first, last,
  element-wise mean, (last - first) delta] over the window. It is torch-free (numpy), so the same
  input always yields the same vector and the same shape regardless of `window`. Falsified by a
  non-deterministic result, a shape that varies with `window`, or a length other than
  `4 * state_dim`. Intended test: `tests/unit/models/test_coarse_transition.py` (no torch).
- AC-2: `CoarseDynamicsMDN` is a **diagonal-Gaussian** mixture head with a configurable component
  count `K`; its `forward(coarse)` returns mixture parameters (mixing logits `[B, K]`, means
  `[B, K, D]`, log-variances `[B, K, D]`), and its `dispersion(coarse)` returns a **non-negative**
  scalar per batch element `[B, 1]` computed as the trace of the total mixture covariance
  (law-of-total-variance: E[within-component var] + between-component variance), which is `>= 0` by
  construction. `K` is a constructor argument (default from `constants.py`), not hardcoded. Falsified
  by a fixed `K`, a negative or non-scalar dispersion, or wrong mixture-parameter shapes. Intended
  test: `tests/unit/models/test_coarse_dynamics_mdn.py` (`importorskip torch`).
- AC-3: The module is torch-optional and side-effect-free on existing code: `import
  src.models.coarse_dynamics` succeeds **without torch installed** (the aggregator is usable), while
  constructing `CoarseDynamicsMDN` without torch raises a clear `RuntimeError` (no silent no-op); and
  `src/models/value_network.py` — its `ValueOutput` fields and `forward` signature — and its callers
  (`neural_trainer.py`, `hybrid_agent.py`) are unchanged. Falsified by an import error when torch is
  absent, a silent MDN no-op, or any change to `ValueOutput`/`ValueNetwork`. Intended tests:
  `tests/unit/models/test_coarse_dynamics_import.py`, `tests/unit/models/test_value_output_unchanged.py`.

# Constraints

- torch is imported behind a `try/except ImportError` `_TORCH_AVAILABLE` guard (the idiom already in
  `src/framework/mcts/llm_guided/training/networks.py`); torch-requiring tests use
  `pytest.importorskip("torch")`. The aggregator uses only numpy.
- `K` (component count) and `window` are constructor arguments with defaults + bounds expressed as
  `Final` constants in `src/config/constants.py`; no magic numbers. (Matches the existing
  constructor-args + constants precedent for these networks; no new Settings class.)
- New standalone module `src/models/coarse_dynamics.py`; do not modify `value_network.py`.
- Unit tests carry the >=85% branch coverage gate; no network/training in unit tests.

# Invariants

- `src/models/value_network.py`, the `ValueOutput` dataclass (fields/order), and the existing scalar
  `estimate_uncertainty` head remain unchanged and available for current callers.
- The dispersion metric is `>= 0` for every input (variance-trace of a diagonal-Gaussian mixture).

# Out of Scope

- Wiring the dispersion metric into subgoal scoring (`strategos_risk_averse_subgoal_scorer`).
- Training the MDN or producing a checkpoint; this spec delivers the architecture + metric only.
- A differential-entropy dispersion metric (can be negative for a continuous mixture; excluded to
  keep the non-negativity invariant).
- Any change to the MCTS engines, the graph node, or `value_network.py`.
