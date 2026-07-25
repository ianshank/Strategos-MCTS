# Peer Review — Proposed Plan "Execute M5" (2026-07-24 draft)

- **Reviewed document**: the 2026-07-24 draft titled "Execute M5 — Strategos-MCTS", proposed for
  `docs/plans/2026-07-24-execute-m5.md`. The draft was never committed; it is reproduced verbatim as
  **Appendix B** so every verdict below is checkable against its exact wording.
- **Tree audited**: `main` @ `ce084ac` (2026-07-25). Baseline ledger `docs/STATUS.md` dated 2026-07-23.
- **Method**: claim-by-claim verification against the live tree; every `[Certain]`-tagged claim in the
  draft receives a CONFIRMED / PARTIAL / FALSE verdict with `path:line` evidence.
- **Reviewer verdict**: **Do NOT execute as written.** The P0 "decisive act" — "run GSM8K-50,
  `use_llm_judge: false`, via the policy-lift entrypoint" — is not executable: those domains and that
  flag belong to a legacy, mock-backed suite under `training/`, while the `policy-lift` entrypoint runs
  a different harness that cannot reach them. The *methodology* the draft describes is real and
  accurately summarized — it simply belongs to a harness measuring a different thing (a trained vs
  untrained network under identical search, not MCTS vs single-shot). The rewrite re-targeted to the
  approved chess gate supersedes the draft: `docs/plans/2026-07-24-execute-m5.md` (v2).

---

## 1. Verdict at a glance

| Draft element | Verdict |
|---|---|
| P0: "GSM8K-50, `use_llm_judge: false`, via the policy-lift entrypoint" | **FAILS — subsystem conflation** (§2). GSM8K is unreachable from `policy-lift`; `use_llm_judge` belongs to a third, mock-backed suite. |
| Methodology (lift = (win_rate−0.5)×2, Wilson CI, gate on CI lower bound ≥20%) | **CONFIRMED** (§3.1) — but it measures **trained vs untrained under identical MCTS**, not MCTS vs single-shot. |
| "Invalidates trivially-exploitable rewards" | **PARTIAL** (§3.2) — a documented invariant, not a code-enforced check. |
| Domain list "no technical blocker" | **FALSE as an executable path** (§3.3). |
| STATUS figures (10,090 / 93.65%) | **STALE** (§3.4) — superseded 2026-07-20 revision. |
| Mock ratio 206/372 (55%) | **PARTIAL** (§3.5) — accurate on its evidence date; 211/397 = 53.1% at HEAD. |
| Artifact records "config hash and seed" | **PARTIAL** (§3.6) — seed yes, config hash and checkpoint digest no. |
| P3 "add one real neural-MCTS e2e test" | **ALREADY EXISTS** (§3.7). |
| "Mouse-Droid" / "Tier-4" / "2026-07-24 external audit" | **UNSOURCED** (§3.8) — zero repo hits / not committed. |
| "Uncomfortable clause" (a finished benchmark left unrun out of fear) | **CONTRADICTED BY TIMELINE** (§4). |
| P2 pre-committed decision tree; "nothing lands first" discipline; STATUS-as-honest-ledger | **KEEP** (§5) — the best of the draft; preserved in v2. |

---

## 2. The crux: three benchmark subsystems, and the draft straddles them

The repository contains three disjoint benchmark subsystems. The draft's P0 takes its domain list and
`use_llm_judge` flag from subsystem **1**, its methodology and gate description from subsystem **3**, and
implies the MCTS-vs-single-shot comparison that only subsystem **2** gestures at — and none of the three
can execute the sentence "run GSM8K-50 with `use_llm_judge: false` via the policy-lift entrypoint."

| # | Subsystem / entrypoint | What it actually measures | Why it cannot serve P0 |
|---|---|---|---|
| 1 | `training/benchmark_config.yaml` + `training/benchmark_suite.py` + `scripts/run_benchmarks.py` | A legacy RAG / reasoning / codegen evaluation suite. | Owns the draft's exact numbers — `gsm8k_subset: 50`, `math_subset: 40`, `humaneval_subset: 30`, `mbpp_subset: 40` (`training/benchmark_config.yaml:32-56`) plus `dabstep_subset: 60`, which the draft silently omitted. But datasets load from LangSmith or fall back to a **2-item hardcoded mock** (`training/benchmark_suite.py:1125-1171`); no GSM8K data exists in the repo; the system-under-test is `create_example_*_function` stubs (`scripts/run_benchmarks.py:75-119`); the LLM judge is an unimplemented placeholder (`training/benchmark_suite.py:391-394`); there is no seed key, no MCTS arm, and no single-shot arm. A "GSM8K-50 run" here scores canned functions against two mock items. |
| 2 | `src/benchmark` CLI (`python -m src.benchmark`, tasks A1–C3) | Compares two adapters on hand-written tasks. | The adapter registry is `langgraph_mcts` and `vertex_adk` only (`src/benchmark/adapters/factory.py:21-24`) — no single-shot arm — and `mcts_iterations` has `ge=1`, so MCTS cannot even be disabled into one (`src/benchmark/config/benchmark_settings.py:279-283`). As CLI-wired, no framework or LLM client is injected (`src/benchmark/cli.py:199` → `src/benchmark/factory.py:156-175`), so the LangGraph arm returns the placeholder `"LLM client not configured for direct execution"` with zero tokens and still "succeeds" (`src/benchmark/adapters/langgraph_adapter.py:244-252`). |
| 3 | `src/benchmark/policy_comparison.py` + `policy_lift.py` (console script `policy-lift`, `python -m src.benchmark.policy_lift`) | The Wilson / CI-lower-bound gate the draft's methodology correctly describes. | It compares an **untrained vs trained policy/value network, both arms under `NeuralMCTS`** — not MCTS vs single-shot. It makes no LLM calls (a dummy `sk-` key satisfies Settings validation only, `docs/STATUS.md:139`). Its reachable domains are `reasoning`/`planning` (synthetic, smoke-only) plus lazily `chess`/`connect_four`/`othello` (`src/framework/domain_registry.py:46-50`). GSM8K/MATH/HumanEval/MBPP are unreachable from here. |

Where MCTS-vs-single-shot actually exists today: `src/api/comparison_service.py:104` (`ComparisonService`,
single-query, driven by `demo.py --provider mock|openai|anthropic`, `demo.py:507-509`) and
`src/framework/mcts/llm_guided/benchmark/runner.py` (a HumanEval-only library API with no CLI). Neither is
a dataset-scale benchmark harness.

**Consequence.** The draft's P0/P1 would *run*, produce numbers, and the numbers would mean nothing — which
is worse than not running, because the draft's own P0 then commits them into `docs/STATUS.md` as evidence,
next to figures that the repo has been scrupulous to keep honest (`docs/STATUS.md:106`: "No ≥20% claim
exists yet.").

---

## 3. Claim-by-claim verdicts

### 3.1 Lift formula, Wilson interval, CI-lower-bound gate — CONFIRMED

The gate is the CI lower bound, not the point estimate, and it fails closed. `meets_target` returns
`lift_ci_lower_pct >= target_lift_pct`, with `None` CI ⇒ not met
(`src/benchmark/policy_comparison.py:78-86`); the point estimate is demoted to reporting-only
(`:88-91`). The target defaults to `20.0` (`src/config/constants.py:337`, re-exported as
`DEFAULT_TARGET_LIFT_PCT` at `src/benchmark/policy_comparison.py:50`) — so the comparison is `lift_ci_lower_pct >= 20.0`
on a percent scale; the literal `0.20` never appears, but the semantics match the draft. For adversarial
(win-rate) domains, `lift_pct = (win_rate - 0.5) * 2.0 * 100.0` and the Wilson bounds map through affinely
(`:213`, `:216-218`), with the interval from `src/utils/stats.py:31-65`. Draws count as half-wins (`:209-210`).
One caveat the draft omits: mean-reward domains use a *different* formula with an absolute-points fallback
below a baseline floor of 0.05 (`:162-181`).

### 3.2 "Invalidates trivially-exploitable rewards" — PARTIAL

This is enforced by documentation and a spec invariant, not by code. The warning lives in prose
(`docs/STATUS.md:103-105`; `specs/m5_policy_lift.SPEC.md:36`: "Reasoning/planning domains remain
smoke-test-only"); **no code path flags a smoke-domain result as invalid** — `policy-lift --domain reasoning`
runs and would exit 0 if its CI lower bound cleared 20. What the code *does* guard is a different exploit:
divide-by-tiny-baseline inflation, via the absolute-points fallback (`src/benchmark/policy_comparison.py:174-177`), plus a
hard error on zero games played (`:204-208`) and a warning below the recommended sample size (`:152-160`).

### 3.3 "GSM8K-50 … MBPP-40, `use_llm_judge: false`, no technical blocker" — FALSE as an executable path

The numbers exist only in `training/benchmark_config.yaml:32-56` (the draft also dropped `dabstep_subset: 60`),
and `use_llm_judge: false` is at `:45` — both belonging to subsystem 1, which cannot be driven through the
`policy-lift` entrypoint (§2). The blocker is structural, not operational: there is no pipeline connecting
that config to that harness, and the datasets it names are not present in the repo.

### 3.4 STATUS pins 10,090 passed / 93.65% branch — STALE

That is the superseded 2026-07-20 revision, still visible at `CHANGELOG.md:110`. The current ledger pins
**10,136+ passed / 93.35% branch** (`docs/STATUS.md:18,20`, 2026-07-23 baseline). `docs/STATUS.md:189` itself
still carries a stale "93.65%" remnant, inconsistent with its own headline — corrected by the same PR that
lands this review.

### 3.5 206 of 372 test files (55%) use mocks — PARTIAL

Accurate on its evidence date: 205/372 = 55.1% on the 2026-07-23 tree (off by one file, same percentage).
At HEAD `ce084ac` it is **211/397 = 53.1%**, using the pattern
`unittest\.mock|MagicMock|AsyncMock|mocker\b|monkeypatch|patch\(` over `tests/**/test_*.py` (the figure is
pattern-sensitive; `monkeypatch` inclusion is the main judgment call). Directionally correct as a statement
about the suite's reliance on fakes.

### 3.6 Artifact records "config hash and seed" — PARTIAL

The artifact's `run` block records the seed (plus device, `num_simulations`, `max_moves`, checkpoint path,
and network shape) at `src/benchmark/policy_lift.py:388-399` — see the committed
`benchmarks/results/reasoning_smoke_lift.json` for the exact shape. It records **no config hash** and no
checkpoint content digest; checkpoint identity is a path string. The rewrite adds a git SHA and a checkpoint
SHA-256 manually in STATUS.md, with no change to `src/benchmark/`.

### 3.7 P3 "add one integration test running the real neural-MCTS path end-to-end" — ALREADY EXISTS

`tests/integration/benchmark/test_self_play_convergence_e2e.py` already runs the self-play driver → `policy_lift`
gate with a real torch network on the tiny `reasoning` domain and no mocks; its docstring states exactly this.
Also present: `tests/integration/benchmark/test_chess_gate_smoke.py` (real `NeuralMCTS` chess smoke, `@slow`,
by its own docstring "no CI home"), `test_m5_lift_artifact.py` (asserts the committed artifact's schema now),
and `test_m5_lift_gate.py` (auto-unskips once `benchmarks/results/m5_policy_lift.json` lands). A counter-example
the draft's premise misses: `tests/e2e/test_mcts_simulation_flow.py` imports nothing from `src/` — its name is
misleading. So P3 as written would duplicate existing coverage; v2 re-scopes it to giving the chess smoke a CI
home.

### 3.8 "Mouse-Droid" / "Tier-4" / "2026-07-24 external audit" — UNSOURCED

`grep -rli 'mouse.droid'` and a case-insensitive `tier[- ]?4` search return zero hits repo-wide; the analogy
is to something outside this repository and would be unexplained to a reader here. The draft's cited
"2026-07-24 external audit" is not committed anywhere — the only audit-shaped document in the tree
(`docs/reviews/strategos-s3-uncertainty-subgoal-review.md`) reviews a different subject, and its figures date
its body to the 07-20..07-23 tree. Cut the references or cite a resolvable source.

Note on form: the repo has no `[Certain]`/`[Guessing]` evidence-tag convention (zero hits); maintained docs use
a blockquote banner (Version / Date / Status / Supersedes) plus a `file:line` evidence appendix (see
`docs/STATUS.md:3-8`). The rewrite adopts that convention.

---

## 4. The "uncomfortable clause" vs. the timeline

The draft speculates that a finished benchmark sits unrun because running it risks showing MCTS does not beat
single-shot by 20%. Two verifiable facts make the speculation unnecessary as an explanation.

First, the gate the repo actually commits to (`specs/m5_policy_lift.SPEC.md`, `status: approved`) is not
MCTS-vs-single-shot at all — it is a trained vs untrained policy under identical search. There was no
single-shot comparison being avoided, because that was never the milestone.

Second, the harness could not have produced a *valid* number until 2026-07-23/24, when the arena's
win/loss-perspective inversion, its network-blind evaluation cache, and root-noise-during-evaluation were all
fixed (`docs/STATUS.md:155-162`) — and the gate still requires a GPU-trained chess checkpoint that does not
exist (`benchmarks/results/m5_policy_lift.json` is absent; the only committed artifact is the ~0-lift synthetic
smoke). Mechanical blockers — a harness that only just became valid, plus a checkpoint that has to be trained on
a GPU — fully account for the absence of a number. What deserves to survive from the clause is its *discipline*,
not its diagnosis: pre-commit to publishing either outcome. The rewrite keeps that rule (its P1) and separately
names the draft's real question — does search beat single-shot LLM inference? — as explicitly deferred with its
concrete building blocks.

---

## 5. What the draft got right (preserved in v2)

- The gate-semantics description (CI-lower-bound gating, Wilson interval, tie handling) is essentially accurate.
- The observation that `docs/STATUS.md` practices honest supersession is correct (`docs/STATUS.md:106`).
- The mock ratio was accurate as of its evidence date.
- The pre-committed P2 decision tree — publish either outcome; a negative result is a deliverable — is excellent
  practice and survives verbatim in spirit.
- The "nothing else lands until the number exists" discipline survives.
- Extending STATUS.md's pin-the-number norm survives, as the v2 provenance additions.

---

## 6. Corrections carried into the rewrite

| Draft element | v2 disposition |
|---|---|
| P0 "GSM8K via policy-lift" | Re-targeted to the approved chess gate; operator GPU runbook already at `docs/STATUS.md` §"M5 chess gate — operator runbook". |
| P1 reasoning/code LLM domains | Moved to an explicitly-deferred section naming its building blocks (`ComparisonService`, the llm_guided HumanEval runner). |
| P2 decision tree | Preserved; retargeted to artifact outcomes (exit 0 and exit 1 both valid per AC-3). |
| P3 "add the e2e test" | Reframed to "give the existing chess smoke a CI home + artifact-driven unskip". |
| `[Certain]`/`[Guessing]` tags | Replaced with the house banner + evidence appendix. |
| Unsourced references | Cut. |
| Stale figures | Refreshed to 10,136+ / 93.35%; mock ratio to 211/397 = 53.1%. |

---

## Appendix A — Evidence index (`path:line`)

- Gate / lift / CI: `src/benchmark/policy_comparison.py:78-91` (gate, point-estimate demotion), `:152-160`
  (below-min warning), `:162-181` (mean-reward + absolute fallback), `:204-208` (zero-games raise),
  `:209-210` (draws=half-wins), `:213`, `:216-218` (Wilson + lift formula); `src/utils/stats.py:31-65`
  (Wilson); `src/config/constants.py:337` (`M5_TARGET_LIFT_PCT = 20.0`).
- Subsystem 1: `training/benchmark_config.yaml:32-56` (domains + `use_llm_judge:45`);
  `training/benchmark_suite.py:1125-1171` (LangSmith-or-mock), `:391-394` (stub judge);
  `scripts/run_benchmarks.py:75-119` (canned system-under-test).
- Subsystem 2: `src/benchmark/adapters/factory.py:21-24`; `src/benchmark/config/benchmark_settings.py:279-283`;
  `src/benchmark/cli.py:199`; `src/benchmark/factory.py:156-175`;
  `src/benchmark/adapters/langgraph_adapter.py:244-252`.
- Subsystem 3 / entrypoint: `src/benchmark/policy_lift.py:388-399` (run provenance block);
  `src/framework/domain_registry.py:46-50` (lazy domains); `benchmarks/results/reasoning_smoke_lift.json`.
- MCTS-vs-single-shot building blocks: `src/api/comparison_service.py:104`; `demo.py:507-509`;
  `src/framework/mcts/llm_guided/benchmark/runner.py`.
- Ledger / spec: `docs/STATUS.md:18,20` (10,136+/93.35%), `:101-105` (n≈0.70 note, smoke-domain warning),
  `:106` (no ≥20% claim), `:108-111` (chess bit-rot), `:139` (dummy key), `:153-182` (operator runbook),
  `:155-162` (arena fixes), `:189` (stale remnant); `CHANGELOG.md:110` (superseded figures);
  `specs/m5_policy_lift.SPEC.md:6` (approved), `:21-24` (AC-1..4), `:31,35-36` (invariants).
- Tests: `tests/integration/benchmark/{test_self_play_convergence_e2e,test_chess_gate_smoke,test_m5_lift_artifact,test_m5_lift_gate}.py`;
  `tests/e2e/test_mcts_simulation_flow.py` (no `src/` imports).
- Process: `src/framework/harness/intent/spec_trace.py:86-88` ("no src/ changes; spec reference not required").
- Mock ratio: `unittest\.mock|MagicMock|AsyncMock|mocker\b|monkeypatch|patch\(` over `tests/**/test_*.py`
  → 211/397 = 53.1% at `ce084ac`.

All pins verified against `main` @ `ce084ac` on 2026-07-25.

---

## Appendix B — The reviewed draft, verbatim

> The following is the 2026-07-24 "Execute M5" draft **as submitted for review** (never committed to the
> repository). It is reproduced unaltered so the verdicts above can be checked against the exact text. It is
> **superseded** by `docs/plans/2026-07-24-execute-m5.md` (v2) and is not itself a plan of record.

```markdown
# Plan: Execute M5 — Strategos-MCTS (2026-07-24)

> Target path: `docs/plans/2026-07-24-execute-m5.md`. Status: proposed.
> Evidence basis: 2026-07-24 external audit of `benchmark_config.yaml`,
> `src/benchmark/policy_comparison.py`, STATUS.md, and test-suite composition.

## Objective

Run the benchmark. That is the plan. Everything else in this document is
sequencing around one fact: [Certain] a rigorous, finished measurement harness
has existed for months and has never produced its number.

## Ground truth

- [Certain] `policy_comparison.py` computes lift = (win_rate − 0.5) × 2 with a
  Wilson-score interval, gates on the **CI lower bound ≥ 20%** (not the point
  estimate), and invalidates trivially-exploitable rewards. The methodology is
  better than most published agent evals.
- [Certain] Domains are cheap and standard: GSM8K-50, MATH-40, HumanEval-30,
  MBPP-40, with `use_llm_judge: false` available. There is no technical
  blocker.
- [Certain] STATUS.md already practices honest supersession (it corrects its
  own stale optimistic figures and pins 10,090 pass / 93.65% branch with
  reproduction steps). This plan extends that norm to the missing number.
- [Certain] 206 of 372 test files (55%) mock/patch — the green suite mostly
  proves components against fakes, not the integrated search path.

## Phases

**P0 — The decisive act: one domain, one run.**
GSM8K-50, `use_llm_judge: false`, via the policy-lift entrypoint. Commit the
lift + CI into STATUS.md exactly as coverage is pinned, with config hash and
seed. Acceptance: STATUS.md contains a dated MCTS-vs-single-shot lift with a
Wilson interval. Nothing else in this plan starts before this lands.

**P1 — Remaining domains.**
MATH-40, HumanEval-30, MBPP-40. Per-domain table; exploitable-reward flags
reported, not suppressed. Compute cost recorded per domain (this becomes the
cost-per-lift denominator later).

**P2 — Framing follows the number (decision tree, pre-committed).**
- Clears the CI-lower ≥20% gate on ≥1 domain → that domain is the headline;
  short writeup with the harness as the methods section.
- Clears none → publish "tree-search lift is domain-dependent: clears on
  none of {math, code} at this scale" as the finding. [Certain] This outcome
  is a real, honest contribution; an unrun benchmark is neither.
The tree is written **before** P0 executes so the result cannot renegotiate
the framing.

**P3 — De-mock the integration story.**
One integration-tier test running the real neural-MCTS path end-to-end on a
tiny domain, so "10,090 green" includes at least one path through the actual
system. Target: the hardware-free analogue of Mouse-Droid's Tier-4 e2e.

## The uncomfortable clause, kept in the plan on purpose

[Guessing, informed] The most likely reason a finished benchmark sits unrun is
that running it risks showing MCTS does not beat single-shot by 20% on
reasoning/code — domains where expensive search often yields marginal lift.
This plan's P2 decision tree exists precisely so that outcome is a deliverable
rather than a threat. A negative result published beats a positive one feared.

## Kill / pivot criterion

None required for M5 itself — both branches of P2 are wins. Pivot only
applies to Strategos's *positioning*: if P1 shows no domain clearing the gate,
the repo's framing shifts from "search framework that improves outcomes" to
"measurement framework that quantifies when search pays," and future
engineering effort follows that identity.

## Explicitly deferred

- Coverage expansion, new domains, new engine features — nothing lands until
  P0's number exists. The repo's constraint is not code volume.
```
