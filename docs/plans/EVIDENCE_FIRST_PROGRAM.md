# Evidence-First Program — the sequenced work for the rest of 2026 H2

> **Status:** proposed · **Supersedes:** the phased roadmap in
> `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md` §2 (Scaling → Architectures → Domains →
> DX). That document remains the charter-designated home of the *sequenced work* axis and now
> delegates to this file; the scaling epics are not cancelled, they are **re-gated**.
>
> **Authority:** this document is authoritative for *what next, in what order*, and for nothing
> else. Scope/non-goals/invariants stay in `CHARTER.md`. Measured status stays in `docs/STATUS.md`.
> Per-change contracts stay in `specs/`. Where this file asserts something on another document's
> axis, that is a bug in this file.
>
> **Provenance:** written after a full read of the tree at `7a51bcb` (`main`), an external
> two-model adversarial review, and independent re-verification of every load-bearing claim in
> that review. Claims the review got wrong are listed in §3 rather than quietly dropped.

---

## 0. The verdict, in one paragraph

This repository has unusually good process machinery — a charter with an amendment protocol, 50
specs on a validated schema, a deterministic documentation validator, a spec-traceability CI gate,
a coverage gate at 85% with 89.65% actually achieved, `mypy` clean across 332 files, `ruff` and
`black` clean repo-wide. What it does not have is a single **end-to-end result**. There is no
candidate-versus-champion promotion gate anywhere in the tree; there is no cost-normalised
comparison of search against no-search; the three MCTS engines implement three mutually
inconsistent value-perspective conventions while `CHARTER.md` §2 asserts they agree; and the
central artefact a reviewer would ask for — "show me one reproducible run where search beat the
raw network, at a stated cost, with a confidence interval" — does not exist. The correct next
move is therefore **not** distributed self-play, MuZero, or Ray Tune. It is to close the evidence
chain on the smallest possible domain, fix the correctness defects that would silently invalidate
any result produced later, and make "built" versus "proven" a machine-checked distinction rather
than a cultural one. The scaling epics resume the moment a promotion gate has demonstrably
rejected a checkpoint.

## 1. Why the previous plan is re-gated rather than executed

`docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md` sequences four epics: distributed self-play and
inference export, Transformer backbones plus HPO plus MuZero, chess and automated Elo, then MLflow
and a model registry. Each is individually reasonable. Together they share one fatal property:
**every one of them multiplies the cost of a wrong answer that the current tree cannot detect.**

- Distributed self-play (Phase 1.2) scales generation of training data whose *targets* are
  produced by an engine family with inconsistent sign conventions (§2.1). Wrong data, faster.
- ONNX/TensorRT export (Phase 1.3) optimises inference latency for a search whose advantage over
  no search has never been measured (§2.3). An optimisation with no denominator.
- Transformer backbones and MuZero (Phase 2) change the model class before any promotion gate
  exists to say whether a change helped (§2.2). Ablation without a referee.
- MLflow and a model registry (Phase 4) build lifecycle management for artefacts that carry no
  provenance record distinguishing mock, random-weight, and trained runs (§2.4).

The dependency arrow in the existing plan's Mermaid diagram points the wrong way. Measurement is
not Phase 4; it is Phase 0. This program inverts that, and states its own kill criteria in §8 so
the inversion is falsifiable rather than a matter of taste.

## 2. Findings that survived independent verification

Every item below was re-checked against the tree at `7a51bcb`. File and line references are exact.

### 2.1 Three engines, three value-perspective conventions — and a charter claim that is false

`CHARTER.md` §2 claims "PUCT selection that agree across the core, parallel, and
progressive-widening engines". The code does not support that claim:

| Engine | Selection | Backup | Convention |
|---|---|---|---|
| `MCTSEngine` (`src/framework/mcts/core.py:377-393`) | UCB1, no negation, no `two_player` knob at all | adds the **same** value at every ancestor | single-agent only |
| `ParallelMCTSEngine` (`src/framework/mcts/parallel_mcts.py:492`, `:535-539`) | negates **iff** `two_player` | `value = -value` **unconditionally** | two-player only, silently |
| `ProgressiveWideningEngine` (`src/framework/mcts/progressive_widening.py:350`, `:470-471`) | negates **iff** `two_player` | `value = -value` **unconditionally** | two-player only, silently |
| `NeuralMCTS` (`src/framework/mcts/neural_mcts.py:457-501`) | negates iff not `single_agent` | negates iff not `single_agent` | **correct and consistent** |

Only `NeuralMCTS` honours its own flag on both sides of the loop. The consequence is precise and
non-obvious: setting `MCTS_TWO_PLAYER=false` on the parallel or progressive-widening engine
produces a search where selection treats child values as same-perspective while backup alternates
their sign along the path. The tree is not merely mis-tuned; its statistics are meaningless. The
progressive-widening docstring at `src/framework/mcts/progressive_widening.py:293` explicitly
promises that `two_player` controls backpropagation, which makes this a divergence between
documented and actual behaviour, not an undocumented design choice.

This is the single highest-value defect in the repository, and it is invisible to a 89.65%
coverage gate because both engines are exercised only in their default `two_player=True`
configuration, where the bug is unobservable.

**Credit where it is due, because it changes the plan.** This defect is not undiscovered. Two
existing draft specs already name it: `specs/hygiene_mcts_value_semantics.SPEC.md` describes the
selection-sign half, and `specs/hygiene_mcts_engines.SPEC.md` opens with "Backprop exists in six
copies with disagreeing sign conventions." The failure is therefore **not analysis, it is
sequencing** — the two specs have sat at `draft` while the roadmap pointed at MuZero. That is a
more useful diagnosis than "nobody noticed," and it means E2 needs no new spec: it promotes and
executes two contracts that already exist. It also means the external review's framing — a project
that cannot see its own gaps — is wrong. This project sees them clearly and does not schedule
them.

### 2.2 No promotion gate exists

Grepping the tree for `elo`, `arena`, `promote`, `champion`, `candidate`, and `tournament` returns
exactly three near-misses, none of which is a gate:

- `SelfPlayEvaluator.evaluate(current_model, best_model)` in `src/training/agent_trainer.py:565-640`
  is a real head-to-head arena with alternating first player and an `is_better` win-threshold
  property. **Nothing calls it from the self-play path.** It is a working gate with no wiring.
- `src/benchmark/policy_comparison.py:78-91` computes `meets_target` from a confidence-interval
  lower bound. That is a *decision-quality acceptance test*, not a promotion operation — it never
  labels, replaces, or refuses a checkpoint.
- `src/games/chess/continuous_learning.py:153-165` maintains a scalar `elo_estimate` updated by
  `K * (actual - 0.5)`. Single-agent bookkeeping, not candidate-versus-champion rating.

Meanwhile `src/training/self_play_convergence.py:279-338` trains and writes `ckpt_iter_<n>.pt` for
every iteration, unconditionally. Every checkpoint is promoted by default because promotion is not
a concept in the loop. This is the cheapest high-value change in the repository: the arena already
exists, and ELF OpenGo's published protocol (400 games, promote at ≥55%) and KataGo's gated
stochastic-weight-averaging give a well-documented target design, the latter reaching ELF-level
strength in 1.4 GPU-years against ELF's 74 ([ELF OpenGo](https://ar5iv.labs.arxiv.org/html/1902.04522),
[KataGo](https://www.alphaxiv.org/overview/1902.10565v5)).

### 2.3 No cost normalisation, and no search-versus-no-search arm

`src/benchmark/policy_lift.py:70-123` compares a trained checkpoint against either an explicit
baseline checkpoint or a fresh seeded untrained instance. The generic harness in
`src/benchmark/evaluation/` does record latency, token counts, and estimated USD cost
(`src/benchmark/evaluation/cost_calculator.py:33-98`) — but reports them *beside* quality, never
as a denominator (`src/benchmark/evaluation/harness.py:275-294`). Neither path normalises lift by
GPU-seconds, tokens, or wall-clock.

More importantly, the arm that would falsify the project's premise is absent. There is no
"raw network policy, no search" arm and no matched-compute self-consistency arm, so no run in this
repository's history can distinguish "our search works" from "our network works and the search is
overhead". Speed is the under-reported drawback of MCTS methods generally
([MCTS review](https://arxiv.org/pdf/2103.04931.pdf)), which makes its absence here a
positioning risk as well as a scientific one.

### 2.4 No machine-readable status artefact; provenance is unrecorded

`docs/STATUS.md` is the designated single source of truth for measured status and is regenerated
rather than hand-edited — good. But it is prose, and it is already visibly drifting inside itself:
the headline table carries two coverage numbers with different scopes and one row marked stale
against the other, plus a full-suite figure the file itself says predates three
denominator-widening changes. There is no `artifacts/status.json`; `.gitignore:214-230` ignores
`artifacts/` entirely, and the only committed measurement artefact is
`benchmarks/results/reasoning_smoke_lift.json`. Nothing in the tree records, per result, whether
it was produced with mock components, random weights, or trained weights — so a reader cannot
distinguish a plumbing smoke test from a result. `src/benchmark/policy_lift.py` documents its
reasoning/planning domains as synthetic and gameable, which is exactly the honesty that a
machine-readable provenance field would make enforceable instead of advisory.

### 2.5 Reproducibility is broken in the neural path specifically

`NeuralMCTS` draws root Dirichlet noise from the process-global NumPy RNG
(`src/framework/mcts/neural_mcts.py:300-323`) and samples stochastic actions the same way
(`:505-533`). The classic engines all own injected generators
(`src/framework/mcts/core.py:171-226`, `parallel_mcts.py:330-347`,
`progressive_widening.py:276-305`). So the one engine used for training is the one engine whose
runs cannot be reproduced exactly. `specs/hygiene_determinism.SPEC.md` already specifies the fix
and is still `draft`; it is promoted into this program's critical path as E1 rather than left as
opportunistic hygiene.

### 2.6 A production image can still serve mock output

`ALLOW_MOCK_LLM_FALLBACK` defaults to `False` (`src/config/settings.py:421-428`) and
`src/api/framework_service.py:301-327` raises when it is unset — the default posture is correct
and the charter's fail-loud claim is honest about it. But `Settings` reads the environment
(`src/config/settings.py:56-70`), and nothing in `Dockerfile`, `entrypoint.sh`,
`docker-compose.yml`, or `kubernetes/deployment.yaml` rejects or overrides an operator-supplied
`ALLOW_MOCK_LLM_FALLBACK=true`. `Dockerfile.space:66-80` goes as far as *commenting* that the flag
is not set, which is documentation standing in for a mechanism. One environment variable can make
a production API serve `MockLLMClient` output as real. The remediation is structural, not
documentary: refuse the combination at startup.

### 2.7 CI hardening gaps that are real (see §3 for the one that is not)

- **Zero SHA-pinned actions.** Every `uses:` in all four workflows is a mutable tag, including
  third-party gitleaks (`.github/workflows/ci.yml:195`), Codecov (`:306`), Trivy (`:634`), and
  `jlumbroso` (`:523`).
- **No top-level `permissions:` block in any workflow.** Repository-default token scope applies
  everywhere except three jobs that opt in (`.github/workflows/ci.yml:512-521`,
  `docker-deployment.yml:119-127`, `:266-273`).
- **PR-triggered code execution holds real secrets.** `e2e_with_langsmith.yml:47-95` checks out
  PR code and runs it with `LANGSMITH_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY`.
  Fork PRs get no secrets under `pull_request`, so the exposure is same-repository branches — a
  narrower risk than a fork-facing one, but the blast radius is three live API keys.
- **`docker-deployment.yml:310-313`** interpolates a dispatch input directly into shell source.
  The input is a constrained `choice`, so this is a pattern to remove rather than a live hole.
- **`packages: write` is held by a PR-capable job** (`docker-deployment.yml:119-127`).

## 3. Peer review of the review — council findings that did *not* survive

Presenting the external review's recommendations unmodified would have propagated three errors.

1. **"Insecure `pull_request_target` exposes privileged context to untrusted PRs" — false here.**
   No workflow in this repository uses `pull_request_target`; all four PR-triggered workflows use
   `pull_request` (`.github/workflows/ci.yml:3-8`, `docker-deployment.yml:18-48`,
   `e2e_with_langsmith.yml:3-8`). The generic finding is sound and the cited industry survey is
   real, but the specific accusation does not apply. The real Actions findings are §2.7, and the
   most serious of them — secrets reaching PR-executed code — the review missed entirely. This is
   what happens when a review reasons from a README rather than from `.github/`.

2. **"Connect Four is the right golden-path domain because it has no C dependencies and no torch"
   — half wrong.** Connect Four is the right domain, but not for that reason:
   `src/games/connect_four/state.py:5-13` imports NumPy and Torch unconditionally, and
   `src/games/connect_four/__init__.py:5-12` re-exports the state, so the package is not
   importable without the `neural` extra. A verified import probe in a `.[dev,api]`-only
   environment fails at `ModuleNotFoundError: No module named 'numpy'`. The golden path is
   therefore *cheap*, not *dependency-free*, and E2 must either accept the `neural` extra as a
   golden-path prerequisite or add a torch-free tensor seam. This program chooses the former and
   says so, because pretending otherwise is exactly the kind of claim this program exists to stop.

3. **"Add a claim ledger and a status artefact" — correct, but the review under-specified the
   hard part.** A ledger of prose claims graded by the same agent that wrote the code is
   theatre. What makes it load-bearing is that grades are *derived* from executable evidence and
   that CI fails when a claim's grade is not reproducible from the tree. The repository already
   has the right machinery for this in `src/tools/context_docs.py` — a deterministic validator
   that checks pinned value claims against source. E1 extends that engine rather than inventing a
   parallel one, which is also what `CHARTER.md` §3 NG-7 ("no second planning system") demands.

Two further council positions are recorded as **noted but not adopted now**:

- **Archiving roughly a third of the surface** (four Dockerfiles, Kroki, Gradio, four domains,
  Pinecone, W&B, S3, k8s). Directionally right for a solo-maintainer repo, but archiving is a
  large diff with no evidentiary payoff, and `specs/` already contains eleven `hygiene_delete_*`
  and `hygiene_*_consolidation` specs covering much of it. Sequenced after E3, executed through
  the specs that already exist.
- **The HRM/TRM architecture verdict.** The published ablations are real and unflattering — ARC
  Prize found the hierarchical architecture contributed little against a same-size plain
  transformer, with the outer refinement loop carrying the gains
  ([ARC Prize](https://arcprize.org/blog/hrm-analysis)), and the TRM checkpoint analysis found
  strict puzzle-ID dependence with roughly 11 points of Pass@1 attributable to test-time
  augmentation and voting ([TRM analysis](https://arxiv.org/abs/2512.11847)). But these are
  ARC-grid results and this repository's use is agent decomposition, so "already refuted" is an
  overreach. `CHARTER.md` §2 already lists these as named-but-not-capabilities, which is the
  honest state. The action is a refinement-step sweep with architecture held fixed, scheduled as
  an E4 ablation arm, not a README rewrite today.

## 4. The evidence chain, stated as a contract

Every claim this project makes must be traceable along an unbroken chain. A break at any link
means the claim is not made.

```
spec  →  invariant  →  reproducible run  →  controlled baseline  →  statistical result  →  claim
 │          │               │                      │                        │              │
specs/   property        seeded, provenance     an arm that could      CI lower bound   CLAIM_LEDGER
*.SPEC   tests          -stamped artefact       lose                   not point est.   grade
```

Four rules make it mechanical:

- **R1 — Grades are derived, never asserted.** A claim is `PROVEN` only if the ledger names a
  command and an evidence artefact, and the artefact exists with a matching commit SHA. Otherwise
  the validator downgrades it. `PROVEN` cannot be typed by hand.
- **R2 — Provenance is mandatory.** Every result artefact records whether it used mock
  components, random weights, or trained weights, plus commit SHA, hardware, and installed
  extras. A result without provenance is not a result.
- **R3 — Gates measure integrated behaviour, not component coverage.** Unit-coverage percentage
  stays as a hygiene floor; it is never cited as evidence for a capability claim. Integration
  coverage is reported under its own key so E2E tests cannot dilute the unit denominator.
- **R4 — Separation of duties.** The agent that writes an implementation may not grade it. Only
  the referee agents defined in §7 may move a claim to `PROVEN` or promote a checkpoint, and they
  are forbidden from writing code under `src/`.

## 5. Milestones

Gates are exit criteria; each is a command with an exit code, not a judgement. Sizes are relative
effort for a solo maintainer with agent assistance, not calendar promises.

### E0 — Truth baseline · S · no `src/` changes

Establish what is actually claimed and actually true, before changing behaviour.

- `docs/CLAIM_LEDGER.md`: every `CHARTER.md` §2 mission bullet and every README capability bullet
  mapped to `PROVEN` / `PARTIAL` / `UNPROVEN` / `FALSE`, with the verification command and the
  evidence artefact for each. The engine-agreement bullet enters as `FALSE` on the evidence in
  §2.1; the fail-loud bullet enters as `PARTIAL` on §2.6.
- The one new spec in §6 authored, and the three existing drafts it depends on reviewed to
  `approved`, so that E1 and E2 run under the existing `spec/<id>` traceability rule rather than
  around it. Note that rule (c) of `harness spec-trace` forbids authoring and implementing a spec
  in the same pull request, so E0 is necessarily its own PR — a constraint this program obeys
  rather than exempting itself from with a `No-Spec` trailer.

**Gate:** ledger parses under the E1 validator's schema in dry-run mode; every spec passes
`harness validate-spec`; no `src/` file changed (so the spec-trace rule is satisfied trivially).

### E1 — Machine-checked claims and provenance · M · `spec/evidence_claim_ledger`

- `src/tools/claim_ledger.py`: deterministic ledger parser and validator. Enforces R1 — a
  `PROVEN` row without a resolvable evidence artefact is an error; a row whose cited paths do not
  exist is an error; an unknown grade is an error. `--json` for CI, `--debug` for per-row trace.
- `src/tools/status_artifact.py`: generates `artifacts/status.json` — commit SHA, dirty flag,
  timestamp, Python version, installed extras, hardware, per-package coverage, test pass/skip/xfail
  counts with skip reasons, and the capability-maturity matrix
  (`imports` → `tested` → `integrated` → `trains` → `benchmarked` → `gated`) derived from the
  ledger. No literal thresholds in code: everything reads `pyproject.toml` and the settings model.
- Console scripts `claim-ledger` and `status-artifact`; `make claims` and `make status`; both
  wired into the CI `spec-validate` job and into `make gate`.
- Structural mock-fallback refusal: a production-deployment marker plus `ALLOW_MOCK_LLM_FALLBACK`
  is rejected at settings-validation time, so §2.6 becomes impossible rather than discouraged.
  Default-off, backwards compatible — no existing configuration changes behaviour.
- New CI workflow invariants, extending `tests/unit/test_ci_workflow_invariants.py` rather than
  duplicating it: no `pull_request_target`; every workflow declares top-level `permissions`;
  no PR-triggered job receives a provider API key; a SHA-pinning **ratchet** whose baseline count
  lives in a committed data file so the count can only decrease.

**Gate:** `claim-ledger --json` exits 0; `artifacts/status.json` regenerates byte-identically
under a fixed commit and environment; a deliberately falsified ledger row makes CI fail (the gate
is demonstrated, not assumed).

### E1b — Reproducibility · S · `spec/hygiene_determinism`

Injected RNG for `NeuralMCTS` per the existing `specs/hygiene_determinism.SPEC.md` (§2.5),
promoted from `draft`. Sequenced immediately after E1 and before any milestone that generates a
result, because an irreproducible run is not evidence.

**Gate:** a fresh-process double run of `NeuralMCTS` yields identical visit counts and identical
Dirichlet draws.

### E2 — Value semantics unified and property-gated · M · `spec/hygiene_mcts_value_semantics` then `spec/hygiene_mcts_engines`

Fix §2.1 and gate it so it cannot regress. **No new spec is authored**: both existing drafts are
reviewed to `approved` and executed in that order, which is also what their own `Constraints`
sections already require (the engines spec is gated on the value-semantics spec landing first).
One amendment is needed to the value-semantics spec: its acceptance criteria cover the *selection*
sign but not the unconditional *backup* negation at `parallel_mcts.py:535-539` and
`progressive_widening.py:470-471`, so an AC for backup-side flag honouring is added before
approval.

- One shared perspective policy applied by all four engines, honouring `two_player` /
  `single_agent` on **both** selection and backup. `MCTSEngine` gains the flag with its current
  behaviour as the default, so existing callers are unaffected.
- Hypothesis property suite, extending `tests/property/test_mcts_invariants.py`: legal-move
  probability mass sums to one; value sign flips exactly once per perspective transition;
  a mirrored board produces the mirrored policy target under the action-index map; a seeded run
  reproduces exactly; and search beats the raw network policy on a fixed toy tree.
- Mutation testing scoped to the perspective and selection code paths, with a committed score
  floor — because line coverage demonstrably did not catch this defect.

**Gate:** every invariant above is a failing test *before* the fix and a passing test after, with
both states recorded in the PR; the four engines agree on a seeded cross-engine comparison in both
`two_player` settings; `CHARTER.md` §2's engine-agreement bullet moves `FALSE` → `PROVEN` in the
ledger, with the ledger validator — not the author — confirming it.

### E3 — Connect Four golden path, end to end · L · `spec/golden_path_connect_four`

One domain, no HRM, no TRM, no RAG, no LLM, no Kubernetes. From `GameState` through PUCT to
visit-count targets to arena, reproducibly, with the `neural` extra as a stated prerequisite (§3.2).

- A single command runs self-play → training → arena → artefact, writing a provenance-stamped
  result under `artifacts/`.
- Integration coverage reported under its own key per R3.
- A CI job asserting that both fallback flags are structurally off in the production image.

**Gate:** two runs from the same seed on the same machine produce identical results; the artefact
carries full provenance; the pipeline runs in CI within the job timeout on CPU at the `smoke`
training profile.

### E4 — Cost-normalised Pareto with a losing arm · L · `spec/pareto_cost_normalised`

`docs/PARETO.md` plus a generated artefact, reporting quality per GPU-second and per token with
confidence intervals across at least: random, heuristic, vanilla UCT, untrained network,
policy-only, value-only, full PUCT, and **raw network without search**. Frontier-LLM and
matched-token self-consistency arms included for the reasoning domains, because the
matched-budget majority-vote arm is the one that kills most search results and omitting it makes
the report advocacy rather than measurement.

**Gate:** the report contains at least one result unflattering to Strategos, or the milestone is
not complete. That is a real gate: if every arm flatters us, the arms are wrong.

### E5 — Promotion gating with a demonstrated rejection · M · `spec/promotion_gate_elo`

Wire the arena that already exists (§2.2) into the self-play loop behind Bayes-Elo with a
configurable threshold. Sample size configurable, defaulting to a value honest about single-GPU
hardware — 100–200 games with published error bars rather than 400 games skipped.

**Gate:** a live rejection over ≥10 generations, recorded in `artifacts/`, showing at least one
candidate refused promotion. A gate that has never said no is decoration.

### Resumption of the scaling roadmap

Phase 1.2 onward from `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md` resumes when E5's gate has
fired. At that point distributed self-play has a referee, inference optimisation has a
denominator, and architecture changes have an acceptance test. The DeepSearch direction — moving
search inside the training loop, reported at 5.7× fewer GPU hours
([DeepSearch](https://arxiv.org/html/2509.25454v2)) — is then the highest-value strategic bet, and
it sidesteps the inference-latency objection that E4 is likely to surface.

## 6. Spec map

Only **one** new spec is authored by this program. Everything else promotes a contract that
already exists — which is the point of `CHARTER.md` §3 NG-7 and the reason the repository has 50
specs and no second planning system.

| Spec id | Milestone | Module governed | Provenance |
|---|---|---|---|
| `evidence_claim_ledger` | E1 | `src/tools/` | **new** — no existing spec claims `src/tools/` |
| `hygiene_determinism` | E1b | `src/utils/` | existing draft, promoted |
| `hygiene_mcts_value_semantics` | E2 | `src/framework/mcts/` | existing draft, promoted + one AC added for the backup path |
| `hygiene_mcts_engines` | E2 | `src/framework/mcts/` | existing draft, promoted |
| `golden_path_connect_four` | E3 | `src/training/` | new, authored at E2 exit |
| `pareto_cost_normalised` | E4 | `src/benchmark/` | new, authored at E3 exit |
| `promotion_gate_elo` | E5 | `src/training/` | new, authored at E4 exit |

The E3–E5 specs are deliberately **not** written now. Writing five specs before the first one is
implemented is the same mistake as the scaling roadmap: planning outrunning evidence. Each is
authored at the exit of its predecessor, when its acceptance criteria can be stated against a
measured baseline instead of a guess.

The existing `hygiene_delete_*` and `hygiene_*_consolidation` specs cover the surface-reduction
work deferred in §3; they are sequenced after E3 and need no new specs.

**Module-overlap note.** `src/framework/mcts/` is also claimed by the open `approved`
`strategos_risk_averse_subgoal_scorer` spec, and `harness spec-new` refuses overlapping modules.
The value-semantics spec already records the resolution — it lands under a human-approved
`No-Spec` exception before the scorer's implementation begins. E2 inherits that resolution rather
than inventing a new one.

## 7. Agents, skills, and hooks

Separation of duties (R4) is the part of this program most likely to outlast the MCTS code, so it
is defined concretely rather than aspirationally.

**New agents.** `eval-warden` — the only agent permitted to change a ledger grade or accept an
evidence artefact; tool access excludes writes under `src/`. `selfplay-referee` — the only agent
permitted to promote a checkpoint or ratify an arena result; same write exclusion. Both are
adversarial by construction: their success condition is finding a reason to refuse.

**New skills.** `validate-claims` — the operational counterpart to the existing
`validate-context` skill, driving `claim-ledger` and `status-artifact`. `promotion-gate` — how to
run and interpret an arena gate, including what an honest error bar at n=100 looks like.

**Updated skills.** `quality-gate` gains the claims and status steps in CI order.
`coverage-baseline` gains the separate integration-coverage key required by R3.
`strategos-primer` gains the evidence-chain contract and the referee roles.

**Hooks.** `.claude/hooks/spec_gate.py` keeps its fail-open contract and gains a companion
`PostToolUse` hook that runs the claim validator after any write under `docs/` or `specs/`, so
ledger drift is caught at authoring time rather than in CI. The `spec_gate.py` warn→block flip is
the E1 exit action: the pilot has run long enough, and a warn-mode gate is a suggestion.

## 8. Risks, and what would falsify this program

| Risk | Why it is real | Mitigation |
|---|---|---|
| E4 shows search loses on cost-adjusted quality | Frontier models plus matched-budget voting are a strong baseline and the ablation literature is thin | This is the *intended* outcome to test. If it happens, the project's honest destiny is a measurement harness and training-signal generator, and the harness is the part nobody has built well. Reposition, do not hide. |
| The evidence chain becomes its own bureaucracy | Six specs and two referee agents is real overhead for one maintainer | Every gate is a command with an exit code. If a gate cannot be run by one person in one command, it is deleted. |
| E2's fix breaks callers depending on current behaviour | Two engines' single-agent path is currently incoherent, so "depending on it" is possible | Flag-gated, defaults preserve today's behaviour, migration note recorded. |
| Fixing correctness delays visible progress | Ten weeks of no new features is uncomfortable with one star on the repo | The deliverables *are* the visible progress: a Pareto report with a losing arm is more credible than four more subsystems. |

**Kill criteria for this program itself:** if E1 and E2 land and E3's golden path still cannot
produce two identical seeded runs, the problem is architectural rather than procedural, and the
correct response is to reduce scope to a reference implementation plus a measurement harness and
retire the multi-agent orchestration claim entirely.

## 9. Charter amendment request (not applied here)

This program does not edit `CHARTER.md`. It records that two of the charter's own mission bullets
are not currently supported by the tree, which is a §7 matter for the maintainer, not a passing
edit:

1. **§2, engine agreement.** The bullet claims the core, parallel, and progressive-widening
   engines agree on negamax sign handling. Per §2.1 they do not. Either the bullet is narrowed to
   `NeuralMCTS`, or it is marked as a target pending E2. The ledger carries it as `FALSE` in the
   interim, which is the honest state and requires no charter change today.
2. **§2, fail-loud posture.** The bullet is accurate about the default and INV-6 is candid about
   the current state, but §2.6 shows the property is not enforced. E1 makes it structural, after
   which the bullet is fully earned.

Recommended amendment, for the maintainer to accept or refuse: add an invariant requiring that
every §2 mission bullet carry a ledger row, and that the ledger validator run in CI. That makes
§2 self-policing and is the smallest possible charter change consistent with §1's "evidence over
assertion".
