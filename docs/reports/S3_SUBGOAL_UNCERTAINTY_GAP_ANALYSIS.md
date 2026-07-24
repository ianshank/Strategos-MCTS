# Gap Analysis & Peer Review — S3 Uncertainty-Aware Subgoal Selection

- **Scope**: the `strategos-s3-uncertainty-subgoal` roadmap (peer review + 4 sequenced specs).
- **Branches reviewed vs `main`**: `spec/strategos_subgoal_scoring_seam` (#94),
  `spec/strategos_coarse_dynamics_mdn` (#95), `spec/strategos_risk_averse_subgoal_scorer` (#96,
  stacked on #94+#95). Spec 4 (benchmark A/B) not yet started.
- **Date**: 2026-07-24.

## 1. What shipped (delivered vs roadmap)

| Spec | PR | Module(s) | Status |
|---|---|---|---|
| 1 — candidate-scoring seam | #94 | `src/framework/mcts/scoring.py`, wired in `graph/builder.py` + `graph/integrated.py`, `GRAPH_MCTS_CANDIDATE_SCORER` | Implemented, green |
| 2 — coarse-dynamics MDN | #95 | `src/models/coarse_dynamics.py` (aggregator + numpy dispersion reference + torch-guarded MDN) | Implemented, green |
| 3 — risk-averse scorer | #96 | `src/framework/mcts/risk_scoring.py`, `ENABLE_UNCERTAINTY_SUBGOAL_PENALTY` + `SUBGOAL_UNCERTAINTY_LAMBDA` | Implemented, green |
| 4 — benchmark A/B + citation | — | (planned) | Not started; blocked (see §5) |

All three were gated by the `spec-review` subagent (each returned REVISE; every finding was
incorporated) and by an objective peer review that established the original OpenSpec change could
not be implemented as written (no subgoal subsystem existed) — see
`docs/reviews/strategos-s3-uncertainty-subgoal-review.md`.

## 2. Objective peer review — strengths

- **Backwards compatible by construction.** The seam's default `IdentityCandidateScorer` returns the
  engine's own `MAX_VISITS` selection, so the graph MCTS node is byte-for-byte unchanged unless a
  non-default scorer is selected. The risk penalty is OFF by default. A directly-constructed
  `GraphBuilder`/`IntegratedFramework` behaves exactly as before.
- **Reusable, decoupled components.** `CandidateScorer` (protocol) + `DispersionSource` (protocol)
  are small, pure, deterministic abstractions with a registry-backed factory; new scorers/sources
  slot in without touching the search core. The MDN dispersion has a torch-free numpy reference
  (`mixture_variance_trace`) that both documents the math and lets the torch head be verified against
  it.
- **No hardcoded values.** All tunables (`GRAPH_MCTS_CANDIDATE_SCORER`, `ENABLE_UNCERTAINTY_SUBGOAL_PENALTY`,
  `SUBGOAL_UNCERTAINTY_LAMBDA`, MDN `K`/window/hidden dims + bounds, dispersion metadata key) live in
  `src/config/graph_settings.py` / `src/config/constants.py`.
- **torch-optional.** The seam and risk scorer import no torch; the MDN is behind the repo's
  `_TORCH_AVAILABLE` guard (imports without torch; constructing the MDN without torch raises a clear
  `RuntimeError`).
- **Determinism preserved.** No new draws from the MCTS RNG on the default path; first-wins tie-breaks
  match `max(...)`; two-seeded-run equality is asserted.
- **Logging/debugging.** Debug logs on scorer overrides and factory construction; fail-loud guards
  (unknown scorer name → `ValueError`; negative λ → `ValueError`; unknown-id override ignored).

## 3. Review findings addressed (post-merge review, PR #96)

Copilot's review surfaced three real robustness gaps, all fixed with tests:
- `builder.py` — non-mapping/`None` `action_stats` guarded; a scorer override to an unknown
  `candidate_id` is now ignored (no action↔stats desync).
- `risk_scoring.py` — `MetadataDispersionSource`/`CallableDispersionSource` clamp dispersion to `≥ 0`
  (a negative dispersion would otherwise increase the score and reward uncertainty).
- integration test — removed an unused `mcts_state=None` param that pytest would treat as an
  unresolvable fixture.

### 3.1 Gap-analysis follow-ups (this consolidation)

A multi-agent gap analysis over the three stacked branches surfaced further items, all fixed on
their source branch with tests and merged up the stack:

| ID | Branch | Fix |
|---|---|---|
| A1 | #94 | Emitted `mcts_stats` (`best_action`/`visits`/`value`) kept consistent after a scorer re-rank, so the synthesis value cannot go stale relative to the chosen action. |
| A3 | #95 | `CoarseTransitionAggregator` now enforces the `MAX_COARSE_WINDOW` upper bound (was lower-bound only). |
| A4 | #95 | `CoarseDynamicsMDN.__init__` validates `input_dim`/`hidden_dim`/`output_dim > 0` (clear `ValueError`, not an opaque torch failure). |
| A2 | #96 | `RiskAverseSubgoalScorer.select_best` logs a **one-time** WARNING when the penalty is active (`lambda > 0`) yet every candidate dispersion is 0 — the scorer silently collapses to value ranking (usually a missing-dispersion misconfiguration). The `ENABLE_UNCERTAINTY_SUBGOAL_PENALTY` field help documents this node-level no-op. |
| A6 | #96 | Commented `CANDIDATE_SCORER_RISK_AVERSE` to explain it is intentionally excluded from the name registry / enum (needs `lambda` + a dispersion source, so it is flag-gated and wired directly). |
| A7 | #96 | Crossover-boundary test at `lambda* = 0.625` (just below → higher-value B; just above → safer A; exactly → equal scores, first-wins → A). |

The gitignore `reports/` rule (training-artifacts intent) was also shadowing the tracked
`docs/reports/` directory; this consolidation adds `!docs/reports/` to un-shadow it so project
reports (including this one) stay tracked.

## 4. Test / AQA coverage

- Every acceptance criterion (`AC-1..3` across the three specs) has at least one intended test, with
  same-line `<spec-id> AC-n` mappings under `tests/` to enable a future `verified` flip.
- Torch-free logic (seam scorers, risk scorer, aggregator, numpy dispersion reference, settings,
  flag-gating integration) is verified **locally**; the torch MDN forward/dispersion + `ValueOutput`
  guard run in **CI's neural job** (`importorskip torch`). Local run: 364 passed / 12 skipped across
  the touched areas; 486-test regression across seam/MDN/risk/graph/integrated green earlier.
- `black`, `ruff`, and `mypy src/` are clean on all changes.

## 5. Tech debt & explicitly-deferred items

1. **Real MDN dispersion on graph candidates (deferred).** The graph MCTS node ranks placeholder
   string actions with no coarse *state sequence*, so with the penalty flag ON the dispersion still
   defaults to `0.0` (the risk scorer collapses to value ranking) until a follow-up wires real coarse
   states through `CandidateRecord.metadata`. This mirrors the seam's deferred `NeuralMCTS` value
   source. Both are documented in the specs' Out-of-Scope.
2. **arXiv:2607.19232 unverified.** The S3 source id could not be resolved through this environment's
   proxy (arXiv/HF return 403). Spec 4's related-work citation is conditional on verifying it.
3. **A class of tests requires optional deps that are absent in this local env** (torch / pinecone),
   so they fail *collection* or at runtime here — e.g. `test_domain_registry*.py`, `test_value_network*.py`,
   `test_factories.py` (BERT/hybrid meta-controller), `utils/test_gpu_utils.py`, `training/*`,
   `storage/test_pinecone_store.py`. All pre-existing (not from this work) and green in CI's
   `[dev,neural]` job; the S3 modules import none of them. Worth `importorskip`-guarding the torch ones.
4. **Local `pytest --cov` blocked by a numpy 2.4.6 + coverage 7.15.2 interaction** (the MCTS package
   import chain double-loads numpy's C extension under coverage: "cannot load module more than once").
   An env/tooling issue, not a code defect — CI enforces `--cov=src --cov-fail-under=85` in a proper
   env. Candidate fixes: pin/align numpy in the dev extra, or `COVERAGE_CORE=sysmon` on Python ≥3.12.
5. **Stacked PRs.** #96 is stacked on #94+#95; merge in order (#94 → #95 → #96) and rebase #96 so its
   diff shows only Spec 3.

## 6. Recommendations

- Merge #94 → #95 → #96 (each independently green on the gated checks).
- Before Spec 4: agree the A/B regression threshold and verify the arXiv id; and decide whether to
  land the deferred "real coarse-state dispersion" wiring so an ON-vs-OFF benchmark shows a real
  effect (otherwise the A/B is a no-op with today's placeholder candidates).
- Optionally address the pre-existing tech debt in §5.3–§5.4 (torch-guard the domain-registry tests;
  align the numpy/coverage tooling) as small standalone fixes.
