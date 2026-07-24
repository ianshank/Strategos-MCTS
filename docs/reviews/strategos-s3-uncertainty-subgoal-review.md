# Peer Review — OpenSpec Change `strategos-s3-uncertainty-subgoal` (revision 2)

- **Reviewed change**: "Add MDN-Based Uncertainty-Aware Subgoal Selection to Strategos-MCTS"
- **Source paper (as cited)**: arXiv:2607.19232 — *"S3: Stable Subgoal Selection by Constraining Uncertainty of Coarse Dynamics in Hierarchical Reinforcement Learning"*
- **Review date**: 2026-07-23
- **Method**: multi-agent codebase audit (architecture-guide, existence-check, and MCTS deep-read agents), each verifying claims against the live tree; plus a repository/PR-state review.
- **Reviewer verdict**: **Do NOT implement as written.** The change's foundational precondition (its own Task 2.0) is **false**: there is no high-level policy, no subgoal-selection module, and no discrete "subgoal value" anywhere in `src/`. As scoped, this is not "subtract a penalty from an existing subgoal value" — it is **four net-new subsystems**. Recommend converting it to a **design spike / research spec**, not a direct code change.

---

## 1. Verdict at a glance

| Dimension | Assessment |
|---|---|
| Problem framing / motivation | **Sound.** Uncertainty-aware, risk-averse subgoal selection is a legitimate idea; the proposal is unusually self-aware (flags its own assumptions, feature-flags OFF, tier "Likely"). |
| Precondition (Task 2.0: "confirm a composable subgoal-value interface exists") | **FAILS.** No such interface exists. The proposal explicitly says that if this is the case, the change "requires a design discussion before implementation, not a direct code change." That branch is the correct one. |
| Scope accuracy ("compose with existing value") | **Materially understated.** Requires inventing (1) a subgoal search abstraction, (2) a coarse multi-step transition representation, (3) a real MDN, (4) a strict OFF-flag threaded through a numpy-only deterministic core. Only a leaf-level `.value` scalar pre-exists. |
| Feature-flag / rollback design | **Good instinct, underspecified.** "Default OFF, bit-for-bit identical baseline" is achievable but fragile here (shared RNG, float-accumulation order) — see §4.3. |
| Source citation | **Unverified in this environment.** Paper content is plausible and internally consistent, but the exact arXiv id `2607.19232` could not be independently resolved (see §6). Verify before Task 2.1. |
| Recommendation | Redirect to a **spec-driven design doc + spike**; do not open an implementation PR against the current architecture. |

---

## 2. The crux: the codebase does not match the proposal's assumed shape

The proposal is explicitly built on two assumptions (its "Assumptions and Unknowns" section), and both are contradicted by the tree.

### 2.1 There is no hierarchical / subgoal subsystem — at all

- `src/agents/hierarchical/` — **does not exist**. The assumed files `subgoal_selector.py` and `high_level_policy.py` — **do not exist**.
- `grep -rEi "subgoal|sub_goal|high_level_policy|subgoal_selector" src/` → **zero hits**. Same across `tests/` → **zero hits**. (`-E` for the alternation; the underlying searches were run with ripgrep, which alternates by default.)
- The only appearance of "subgoal" anywhere in the repo is an **illustrative tutorial snippet** in `docs/training/MODULE_8_ADVANCED_MCTS.md` (`_generate_subgoals`, `_concretize_subgoal`, `high_level_plan`) — pedagogical markdown, not code in `src/`.
- The nearest real concept is HRM's `SubProblem` dataclass + `decompose()` (`src/agents/hrm_agent.py:28-37`). But `HRMAgent` is a `torch.nn.Module` (`hrm_agent.py:223`) emitting latent decomposition tensors — it performs **no candidate ranking** and its `SubProblem`s are never fed to MCTS or value-ranked. Do not let the proposal conflate "subproblem decomposition" with a "subgoal-value interface."

Actual roles of the existing agents:
- **HRM** (`hrm_agent.py`) / **TRM** (`trm_agent.py`) — neural `nn.Module` reasoning nets returning `{response, metadata}` with a scalar quality/confidence.
- **meta_controller** (`src/agents/meta_controller/base.py:53-77`) — a **router** that selects *which single agent* (`hrm` | `trm` | `mcts`) handles a query. It emits one agent label, not a ranked subgoal set. It *consumes* one MCTS scalar as an input feature (`MetaControllerFeatures.mcts_value`, `base.py:28-29`); it does not rank MCTS candidates.

### 2.2 The graph's MCTS ranks hardcoded placeholder strings, not subgoals

The only production caller of the search core is `GraphBuilder._mcts_simulator_node` (`src/framework/graph/builder.py:749-896`). Its "actions" and "state transitions" are stubs:

```python
# builder.py:774-785 — candidate "actions" are hardcoded placeholders
if depth == 0:      return ["action_A", "action_B", "action_C", "action_D"]
elif depth < ...:   return ["continue", "refine", "fallback", "escalate"]
# builder.py:788-794 — the "transition" is a string concatenation
new_id = f"{mcts_state.state_id}_{action}"
```

The value backed up into these nodes comes from a `HybridRolloutPolicy` that blends an HRM/TRM-confidence heuristic with **random noise** (`builder.py:797-816`; `policies.py:160-210`). The MCTS result is then consumed downstream as **a single scalar summary + `confidence = visits/iterations`** (`builder.py:876-895`) — *nothing iterates candidate subgoals and compares their values*. There is no candidate-ranking stage where a `RiskAverseSubgoalScorer` would naturally plug in; one would have to be introduced.

### 2.3 A real value head exists — but it is not in the decision path the proposal targets

Two engines exist and **do not share a node type**:

| Engine | File | Value / selection | Wired into the LangGraph orchestration? |
|---|---|---|---|
| Baseline `MCTSEngine` | `src/framework/mcts/core.py:160` | `MCTSNode.value = value_sum/visits` (`core.py:82-87`), **UCB1** (`policies.py:29-60`), numpy-only | **Yes** — the graph uses this, over placeholder actions |
| `NeuralMCTS` (AlphaZero-style) | `src/framework/mcts/neural_mcts.py:264` | Q-value + **PUCT** (`:169-210`), `evaluate_state → (policy, value)` (`:326`), torch | **No** — used only in the training stack (`trainer_factory.py`, `single_agent_domains.py`, `domain_registry.py`) |

So the AlphaZero-style value head the proposal likely imagines (`NeuralMCTS`) is **not reachable from the graph** today. The one real per-candidate value in the production path is `MCTSNode.value` / `action_stats[...]["value"]` (`core.py:651-659`) — but it is a value *of a placeholder string under a noise-blended heuristic*, not of a subgoal.

### 2.4 There is no MDN and no "coarse dynamics" object

- `grep -rEi "MixtureDensity|mixture_density|GaussianMixture|MDN|dispersion|predictive_variance" src/` → **zero hits** (only unrelated SVG diagrams match "mixture"/"coarse"). (`-E` for the alternation.)
- No multi-step aggregated ("coarse") transition object exists — transitions are single-string concatenations (§2.2). The MDN's declared *input* does not exist as a code object either.
- The **one partial hook**: `src/models/value_network.py` has an *optional* single-scalar epistemic-uncertainty head — `estimate_uncertainty: bool = False` → `Linear→ReLU→Linear(→1)→Softplus` (`value_network.py:58, 85-92`), surfaced as `ValueOutput.uncertainty`. This is a **heteroscedastic scalar**, **not** a mixture density (no mixing weights, no multi-component means/variances). It is a hook to build *near*, not the requested estimator.

---

## 3. Scope reality — "compose" vs "build from scratch"

The "What Changes" section reads as an additive composition. In this tree it is a ground-up build:

| Proposal element | Reality in this repo |
|---|---|
| "Compose with existing MCTS value estimate" | Only a leaf-level `.value` scalar exists, over placeholder actions — **not** a subgoal value. |
| ADD MDN uncertainty estimator | **Build from scratch** (no MDN anywhere; drags torch into a torch-free core — §4.2). |
| ADD `RiskAverseSubgoalScorer` | **Build from scratch** + first define what a "subgoal" is and make search operate over subgoals. |
| MODIFY subgoal-selection interface | **No such interface to modify** — it must be invented, plus a graph node to host it. |
| "NO changes to MCTS rollout / graph structure" | **Not achievable** as stated: exposing subgoal candidates + a scoring node *is* a graph-structure and search change. |

Net: **four substantial pieces** (subgoal search abstraction; coarse-transition representation; MDN; strict OFF-flag through the deterministic core), of which only `MCTSNode.value` pre-exists.

---

## 4. Requirement-by-requirement review of the Spec Deltas

### 4.1 ADDED: "Coarse Dynamics Uncertainty Estimator"
- **Input undefined in code.** No "sequence of low-level states aggregated into one coarse transition" exists to feed the MDN. The estimator's contract can't be met until a coarse-transition object is designed and produced somewhere in the pipeline.
- The scenario ("output a scalar dispersion metric") is fine *as a target*, but presupposes a learned dynamics model that does not exist. This is a research/modeling task, not an interface addition.

### 4.2 ADDED: "Risk-Averse Subgoal Scoring" `score = value - lambda * dispersion`
- Mathematically clean and the ranking-flip scenario is testable — **once there are subgoal candidates with values**. There aren't (§2.2).
- **Layering violation risk:** the base MCTS core (`core.py`, `policies.py`) is deliberately **torch-free** (numpy only; torch is the optional `neural` extra, `pyproject.toml:83-84`). An MDN is torch. Placing the scorer/MDN in the core selection path converts optional-torch into a **hard requirement for every non-neural install and test**. It must live behind the same optional-import guard the neural engine/agents use, reachable **only** when the flag is ON.
- `lambda` should be a bounded Pydantic Settings float, mirroring `MCTS_C: float = Field(default=1.414, ge=0.0, le=10.0, ...)` (`settings.py:117-122`) — not a constructor literal.

### 4.3 ADDED scenario: "Feature flag OFF ⇒ bit-for-bit identical baseline"
This is the strongest part of the proposal (it demands *skip computation entirely, not zero-weight*), but the review found determinism hazards that make it fragile:
1. **Single shared, stateful RNG.** `self.rng = np.random.default_rng(seed)` (`core.py:192`) is consumed by rollouts **and** by cache-hit noise `self.rng.normal(0, 0.01)` on *every cache hit* (`core.py:337-339`). Any new draw from `self.rng` when OFF (MDN sampling, tie-break) shifts every subsequent rollout → outputs diverge. The scorer/MDN must use an **independent** generator and must not even be *constructed* in a way that touches `self.rng`.
2. **Order-sensitive float accumulation.** `value_sum += value` (`core.py:391`) is addition-order sensitive; `value - lambda*dispersion` upstream changes rounding *even at lambda≈0*. The flag must gate the **arithmetic**, not multiply by zero — the proposal's wording ("dispersion computation skipped entirely, not just zero-weighted") correctly anticipates this; the implementation must honor it exactly.
3. **Tie-breaking by insertion order.** `select_child` uses strict `score > best_score` and `_select_best_action` uses `max(..., key=...)` (`core.py:117, 609-627`) — ties resolve to first child in list order; any change to child insertion order breaks reproducibility.
4. **Neural path uses the *global* numpy RNG** (`np.random.dirichlet`, `neural_mcts.py:322`; `np.random.choice`, `:532`) — already not per-engine reproducible; an MDN here compounds it.
- **Good precedent to follow:** `enable_early_termination: bool = False` (`mcts/config.py:90`) gated in `builder.py:822-824` — disabled path short-circuits and never constructs the new logic. A RiskAverse flag should replicate this exactly (default False → scorer/MDN never instantiated, never draws RNG). Only then is "bit-for-bit" realistic.

---

## 5. Invariants the change must respect (from the project's own gates)

1. **Config via Pydantic Settings** — `lambda`, MDN toggle, dispersion weight must be `Settings` Fields / `constants.py`; no hardcoded tunables (secret + magic-number grep runs in the gate).
2. **Async I/O** — new I/O paths async (graph + `search`/`simulate` are already async).
3. **Dependency injection** — config/clients/logger passed into `__init__`, not constructed internally.
4. **Branch-coverage gate `fail_under = 85.0`** (`pyproject.toml`) with **no network in unit tests** — a new MDN/scorer needs mocked-I/O unit tests hitting 85% branch coverage.
5. **`src/**` is spec-gated** — needs a `spec/<id>` branch with an `approved` spec, or a `No-Spec:` trailer. Given this introduces a whole subsystem, **a spec is the required channel**, not a trailer.
6. **Fail-loud, no silent fallbacks; structured, secret-safe logging** (`correlation_id` + `sanitize_dict`).
7. **Optional-torch layering** — MDN belongs with the neural nets (`src/models/*`) behind the `neural` extra, guarded by `importorskip` in tests (project convention, e.g. `tests/unit/test_neural_mcts_signs.py:23`).

---

## 6. Source-citation check (arXiv:2607.19232)

- The main arXiv site and the HF papers mirror both return **HTTP 403 through this environment's proxy**, so the abstract could not be fetched directly.
- Two independent web searches returned a **content-consistent** description (S3; MDN dispersion metrics over multi-step "coarse" dynamics; risk-averse high-level policy; ALA workshop @ AAMAS 2026) — matching the proposal's summary.
- **Caveat:** neither search actually surfaced `2607.19232` as a returned arXiv link; one search resolved the near-id `2601.19232` to an unrelated paper ("Structure-based RNA Design…"). Closely related *real* papers do exist (e.g. arXiv:2505.21750 "HRL with Uncertainty-Guided Diffusional Subgoals"; arXiv:2406.16707 "Probabilistic Subgoal Representations for HRL").
- **Action:** before Task 2.1, confirm the exact arXiv id and that the paper is the intended one. The project already has a norm of scrutinizing arXiv references (see `docs/related-work.md`, which classifies arXiv:2607.19297 as an *engineering reference, not research lineage*). S3 would be a genuine **Research citation** if adopted — but only once verified.

---

## 7. Recommendation

**Reclassify from "implementation change" to "design spike + spec."** Concretely:

1. **Do not open an implementation PR** against the current architecture. Task 2.0's precondition is unmet; the proposal's own text says this triggers "a design discussion, not a direct code change."
2. **Write a design doc / research spec** (schema v2 under `specs/`) that answers the prerequisite questions the code can't currently answer:
   - What is a "subgoal" in Strategos-MCTS, concretely? (Candidate `SubProblem`s from HRM? A new discrete action space?)
   - Where does subgoal selection happen — a new graph node, or inside `NeuralMCTS` (which first has to be **wired into the graph**, itself unbuilt)?
   - What is a "coarse transition" object and where is it produced?
3. **Prefer the real hook over a from-scratch core change.** The lowest-risk landing zone is the *neural* stack, not the placeholder-action graph MCTS: extend the existing scalar `estimate_uncertainty` head (`value_network.py`) toward a mixture-density head, and integrate `NeuralMCTS` into the graph — as separate, independently-valuable specs — *before* layering a risk penalty on top.
4. **Keep the good bits:** the OFF-by-default flag, "skip-computation-entirely" semantics, `lambda` as a bounded setting, and the benchmark A/B with a pre-agreed threshold are all correct instincts — carry them into the spec.
5. **Split into sequenced specs**, each with acceptance criteria and 85% branch coverage:
   - `spec-A`: wire `NeuralMCTS` into the LangGraph orchestration (replace placeholder actions with real candidates + values).
   - `spec-B`: mixture-density uncertainty head (extend `value_network.py`), torch-guarded, `neural` extra.
   - `spec-C`: risk-averse scorer + flag (`score = value - lambda*dispersion`), default OFF, bit-for-bit bypass per the `enable_early_termination` precedent.
   - `spec-D`: benchmark A/B + `docs/related-work.md` S3 entry (Research citation, after id verification).

---

## 8. Task-by-task feedback (proposal §Tasks)

- **2.0** ✅ correctly listed first — and it **fails**; that outcome should gate everything else. Elevate to a hard STOP, not a checkbox.
- **2.1** Verify the arXiv id first (§6).
- **2.2 `CoarseDynamicsMDN`** — net-new; define the coarse-transition input object before the model. Put behind `neural` extra.
- **2.3 `RiskAverseSubgoalScorer`** — needs subgoal candidates + values to exist first (spec-A).
- **2.4 feature flag** — good; make it a Pydantic Settings bool (default False) + a bounded `lambda` float; replicate the `enable_early_termination` bypass.
- **2.5 / 2.6 tests** — the "bit-for-bit identical when OFF" integration test must pin RNG independence and no float-order change (§4.3), not just equal top-choice.
- **2.7 benchmark A/B** — there is a benchmark harness (`src/benchmark/…`) and a kill-safe run store (`run_store.py`) landing in PR #91; reuse it. Agree the regression threshold **before** running (proposal already says this — good).
- **2.8 `docs/related-work.md`** — the file exists and expects a **classification**; add S3 as a *Research citation* (distinct from the existing engineering reference), after id verification.

---

## Appendix A — Repository & PR state (as of 2026-07-24)

CI installs `.[dev]`, which **caps the formatter minor ranges** (`black>=26.3.0,<26.4.0`, `ruff>=0.15.0,<0.16.0` in `pyproject.toml:66,70`) — currently **resolving to** `black 26.3.1` / `ruff 0.15.22`. Under that black, `main` was briefly **not** clean: a whole-tree `black src/ tests/ --check` reformatted **5 files**, so the `Lint & Format` job failed for **every** PR against `main` until they were reformatted (MyPy, Spec Validation & Secret Scan, Bandit, Security all passed; the pytest job was skipped behind the lint gate). This was fixed on `main` by **PR #93** (`style: black-format pre-existing formatting drift to unblock CI lint`):

| File | Landed via |
|---|---|
| `src/benchmark/evaluation/harness.py` | #91 (merged) |
| `src/framework/harness/intent/spec_scaffold.py` | #87 (merged) |
| `src/framework/mcts/llm_guided/rag/prompts.py` | #87 (merged) |
| `src/games/chess/ui.py` | #87 (merged) |
| `tests/components/test_hrm_agent_traced.py` | #87 (merged) |

- **#87 and #91 have since merged** into `main` — their own `Lint & Format` was red on these files at merge time, so the drift now sits on `main`. The likely cause is a **local↔CI black mismatch**: a local black older than the `[dev]`-resolved 26.3.1 leaves these files "clean" locally while CI reformats them (CLAUDE.md warns about exactly this parity risk).
- **Resolved on `main` by PR #93**, which ran `black src/ tests/` over the same drift. This review PR was rebased onto that merge, so it carries **no formatting change** — its diff is docs + specs only, and `black --check src/ tests/` is clean on the rebased tree.
- **Recently merged (context):** #91/#90/#89 (LangGraph hardening spec + implementation), #88 (negamax value-sign fix in `NeuralMCTS`), #86/#84 (M5 policy-lift arena scoring + self-play convergence driver), #85 (GPU training + Connect Four/Othello domains), #83/#82 (tech-debt + quality gates), #81 (strategos-primer skill/agent).

---

## Appendix B — Evidence index (file:line)

- Absent subsystem: `src/agents/hierarchical/**` (none); `grep -ri subgoal src/ tests/` (zero). Only doc mention: `docs/training/MODULE_8_ADVANCED_MCTS.md`.
- Agents: `src/agents/hrm_agent.py:28-37,223`; `src/agents/trm_agent.py:103`; `src/agents/meta_controller/base.py:28-29,53-77`.
- Baseline MCTS: `src/framework/mcts/core.py:82-87` (`value`), `:192` (`self.rng`), `:337-339` (cache-hit noise), `:377-393` (backprop), `:589-629` (`_select_best_action`, `MAX_VALUE`), `:651-659` (`action_stats`); `src/framework/mcts/policies.py:29-60` (`ucb1`), `:160-210` (`HybridRolloutPolicy`).
- Neural MCTS (training-only): `src/framework/mcts/neural_mcts.py:107-143,169-210,264,322,326,532`.
- Graph caller / placeholder actions: `src/framework/graph/builder.py:749-896` (esp. `:774-794` actions/transition, `:797-816` value, `:876-895` scalar summary), `:822-824` (early-termination bypass precedent).
- Uncertainty hook (scalar, not MDN): `src/models/value_network.py:58,85-92,141-146`.
- Config / flags: `src/config/settings.py:109` (`MCTS_ENABLED`), `:111-122` (`MCTS_IMPL`, `MCTS_C`), `:367-399` (advanced MCTS); `src/framework/mcts/config.py:90` (`enable_early_termination`).
- Optional torch: `pyproject.toml:83-84` (`neural` extra); test guard convention `tests/unit/test_neural_mcts_signs.py:23`.
- Related work: `docs/related-work.md` (classifies arXiv:2607.19297 as engineering reference; no S3 entry).
