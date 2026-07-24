---
id: strategos_subgoal_uncertainty_benchmark
goal: Measure the risk-averse subgoal penalty ON vs OFF with a kill-safe benchmark sweep against a pre-agreed regression threshold, and record the S3 source in docs/related-work.md as a classified citation, without flipping the feature on by default
module: src/benchmark/
status: draft
---

# Goal

Before the uncertainty penalty could ever become default-on, it needs an honest A/B measurement.
This spec runs the existing benchmark sweep with `ENABLE_UNCERTAINTY_SUBGOAL_PENALTY` ON vs OFF,
reusing the kill-safe run store, compares against a regression threshold that is agreed and recorded
before the run, and adds the S3 paper to `docs/related-work.md` with an explicit classification
(Research citation, distinct from the existing engineering reference) once its arXiv id is verified.
The feature flag stays OFF regardless; a default-on change is a separate decision that requires a
passing result here.

# Acceptance Criteria

- AC-1: A benchmark comparison runs the same task and system matrix twice (flag ON and OFF) reusing
  the kill-safe run store and `--resume`, and emits a single comparison artifact carrying both arms'
  metrics, the pre-agreed threshold, and a pass/fail verdict; a hard kill mid-sweep resumes with no
  lost or duplicated cells. Falsified by data loss on resume, a missing arm, or an artifact without
  the recorded threshold. Intended tests: `tests/unit/benchmark/test_uncertainty_ab_report.py`,
  `tests/integration/benchmark/test_uncertainty_ab_resume.py`.
- AC-2: `docs/related-work.md` gains an entry for the S3 source classified as a Research citation
  (separate from the existing engineering reference), recording the verified arXiv id and the
  signal-only mapping (classical HRL MDN dispersion, not neural MCTS); the entry is absent if the id
  cannot be verified. Falsified by a missing or misclassified entry, or a citation with an unverified
  id. Intended test: `tests/unit/test_related_work_s3_entry.py`.
- AC-3: The default remains OFF: no code path enables the penalty without an explicit configuration
  change, and no default-on flip lands in this spec. Falsified by the flag defaulting to True or by
  the sweep enabling it globally. Intended test:
  `tests/unit/config/test_uncertainty_flag_default_off.py`.

# Constraints

- Primary module `src/benchmark/`; additional in-scope path `docs/related-work.md`. No `src/` runtime
  behavior changes beyond the benchmark harness wiring.
- Reuse the kill-safe run store (`src/benchmark/evaluation/run_store.py`) and the existing benchmark
  CLI; no parallel sweep or persistence implementation.
- The regression threshold is recorded in the spec or PR before the run, not chosen to fit the result.
- Unit tests carry the >=85% branch coverage gate.

# Invariants

- The existing `benchmark_results.json` artifact schema is unchanged; the comparison artifact is
  additive.
- No decision-quality claim is recorded from a plumbing or smoke run; only a real ON-vs-OFF
  comparison counts.

# Out of Scope

- Flipping the penalty on by default (a follow-up decision gated on a passing result here).
- Building the scorer, the MDN, or the seam (upstream roadmap specs).
