# Next-Steps Implementation Plan — 2026 H2

> **Implementation progress (branch `claude/implementation-plan-roadmap-nq1cwv`):**
> Phases **0–5 implemented**. 0–3: re-baseline (`docs/STATUS.md`) + doc reconciliation; correctness &
> packaging fixes; targeted coverage (all Google ADK agents ≥85%); production readiness (ESO secret
> hygiene, settings-driven JWT `AUTH_MODE`). **Phase 4**: MCTS early-termination wiring + streaming /
> graph-visualization / comparison services + REST endpoints + Gradio UI (`[ui]` extra). **Phase 5
> (M5)**: generalized `SelfPlayTrainer` with a single-agent path, domain registry + reasoning/planning
> domains, domain-type-aware policy-comparison benchmark (≥20%-lift harness), and meta-controller
> learning loop (`docs/META_CONTROLLER_TRAINING.md`). Workstream B (specs + `.claude/skills/` + CI
> `spec-validate`) complete. Deploy-time only: Phase **3.3 staging soak** (runbook in
> `docs/SECRETS_MANAGEMENT.md`); M5 lift target is to be run to convergence on a chosen domain.

> **Tech-debt cleanup (branch `claude/tech-debt-cicd-cleanup-5i2eec`, specs `phase_5..8`):**
> Phase **5** — green CI (ADK factory test + Trivy SARIF `security-events` permission). Phase **6** —
> config centralization (assembly-router constants, LMStudio model dedup; the codebase was already
> largely centralized). Phase **7** — coverage gap-analysis lifts holding the ≥85% gate (~89.6%):
> `harness/cli.py`, `harness/factories.py`, `adk_adapter`, `rag/prompts`, `HybridMetaController`; also
> fixed the inert `explain_decision` and the broken `harness replay` path. Phase **8a** — consolidated 36
> archival root docs into `docs/{reports,summaries,plans,quickstart}`. **Remaining (Phase 8b–e):** untrack
> committed model binaries (`models/*.pt`), consolidate the demo trees (`demo.py`/`chess_demo.py`/
> `demo_src`/`demos` → `examples/`), and disambiguate `config/` vs `src/config/`.

> **Version:** 2.0.0 · **Date:** 2026-06-30 · **Status:** Active
> **Supersedes:** `docs/plans/NEXT_STEPS_PLAN.md` (v1.0), the status sections of
> `docs/reports/GAP_ANALYSIS_REPORT.md`, and `docs/NEXT_STEPS_INVESTIGATION.md` where they conflict with
> the **Verified Status** table below. (Archival roadmap docs were relocated under `docs/plans/` and
> `docs/reports/` in Phase 8a.)
>
> This plan was produced by reconciling every roadmap/milestone/next-steps document
> (`docs/plans/IMPLEMENTATION_ROADMAP.md`, `docs/plans/MVP_ROADMAP.md`,
> `docs/plans/IMPLEMENTATION_PLAN_PRIORITY_TASKS.md`, `docs/reports/GAP_ANALYSIS_REPORT.md`,
> `docs/NEXT_STEPS_INVESTIGATION.md`, `docs/plans/PHASE_4_TEMPLATE_PLAN.md`,
> `planning/milestones.yaml`, `planning/epics/*.yaml`) **against the source tree**. Where a
> document's claim disagreed with the code, the code wins and the evidence is cited inline.

> **Correction (2026-06-30, verified):** three rows in the §1.1 table below have themselves gone
> stale and are superseded by `docs/STATUS.md`:
> - *"`examples` cannot export `HRMAgent`" / hard-imports `langchain_openai`* — **false now**:
>   `examples/langgraph_multi_agent_mcts.py:25-36` already guards `langchain_openai`/`langgraph`
>   behind `try/except ImportError`; the module imports without optional deps. The real residual
>   defect is the dead `improved_hrm_agent`/`improved_trm_agent` guard that silently skips
>   `tests/chaos/test_resilience.py` and `tests/performance/test_load.py` (Phase 1.2).
> - *"Google ADK per-agent implementations untested"* — **false now**:
>   `tests/unit/test_google_adk_agents.py` (793 lines) + `tests/integration/google_adk/` cover all
>   five agents; only `agents/data_science.py` (78.7%) is below the 85% line. No fan-out needed.
> - *"RolloutPolicy signature mismatch → 24 failures"* — **resolved**: `mypy src/` is clean and the
>   policy test subclasses already carry the `rng`/`max_depth` annotations; the suite is green
>   (7769 passed, 0 failed). See `docs/STATUS.md`.

---

## 1. Why this plan exists

The existing planning docs span Dec 2025 → Feb 2026 and **contradict each other and the
current code**. The repository has a documented history of re-reporting the same items as
both "done" and "placeholder." Before sequencing new work, every contested claim below was
checked directly in the source tree. The headline result: **most of the items the docs still
list as critical blockers are already implemented.** The genuinely-remaining frontier is
narrow and is what this plan schedules.

### 1.1 Verified Status (code-checked 2026-06-30)

| Claim in older docs | Doc verdict | **Verified reality** | Evidence |
|---|---|---|---|
| Iteration counter never incremented → infinite-loop risk | 🔴 Critical bug | ✅ **Fixed.** `graph.py` is now the `src/framework/graph/` package; counter is incremented in `_evaluate_consensus_node` and returned, then checked in `_check_consensus` | `src/framework/graph/builder.py:954-998` |
| MCTS `RolloutPolicy.evaluate()` signature mismatch → 24 failures | 🔴 Critical bug | ⚠️ **Reduced to a typing nit.** All 5 `evaluate()` defs in the source protocol/impls are consistent; only ~6 **test** subclasses omit the `rng: np.random.Generator` annotation | `src/framework/mcts/policies.py:83,117,147,193,232` |
| DABStep dataset missing `split` parameter → 6 failures | 🔴 Critical | ✅ **Implemented** with backward-compatible default + available-split fallback | `src/data/dataset_loader.py:64,98,219` |
| REST `/query` is mock-only, framework not wired (40%) | 🟠 Needs work | ✅ **Wired** to the real framework service | `src/api/rest_server.py:481` → `framework_service.process_query(...)` |
| Domain training uses hardcoded `win_rate=0.55` (30% placeholder) | 🟠 Placeholder | ✅ **Uses real eval metrics** (`eval_metrics.get("win_rate", 0.0)`), tracks `best_win_rate` | `src/training/unified_orchestrator.py:459-476` |
| RAG pipeline not wired into query flow | 🟠 In progress | ✅ **Wired**: retriever initialized and passed into the framework; confidences come from settings | `src/api/framework_service.py:329-414` |
| LLM-adapter / hybrid-agent / monitoring / observability untested (0%) | 🟠 High | ✅ **Tests exist** for all of these modules | `tests/unit/adapters/`, `tests/unit/agents/test_hybrid_agent.py`, `tests/unit/monitoring/`, `tests/unit/observability/` |
| K8s deployment ships placeholder secrets | 🟠 High | ❌ **Still true** — plaintext placeholders in VCS | `kubernetes/deployment.yaml:195-197` |
| `examples` cannot export `HRMAgent` | 🟠 Medium | ❌ **Still true (different root cause)** — `examples/langgraph_multi_agent_mcts.py` hard-imports `langchain_openai` at module top, so import fails without optional deps | `examples/langgraph_multi_agent_mcts.py:19` |
| Google ADK per-agent implementations untested | 🟠 Medium | ❌ **Still true** — only `base.py` is covered; 5 agent impls lack unit tests | `tests/integration/google_adk/` |

**Net:** the M3→M4 "44-failure blocker cluster" that every doc cites is largely a stale
artifact. The real remaining work is (a) re-baselining + retiring stale docs, (b) a handful of
small correctness/packaging fixes, (c) targeted coverage on ADK + REST/RAG paths, (d)
production secret hygiene, and (e) the genuinely-unbuilt **M5 advanced features**.

### 1.2 Engineering constraints (apply to every task below)

These are non-negotiable acceptance criteria for **all** work in this plan:

1. **Backward compatible.** No breaking changes to public signatures, env-var names, REST
   contracts, or persisted-artifact formats without a documented migration path and a
   default-preserving flag (follow the existing `ALLOW_*` / `*_TRUST_LEGACY_*` pattern).
2. **Modular / reusable / dynamic.** New behavior lands behind protocols/factories
   (`src/framework/factories.py`, adapter protocols), is dependency-injected, and is selected
   at runtime via config — not hard-wired branches.
3. **No hardcoded values.** Every tunable, model name, URL, threshold, and limit flows through
   `src/config/settings.py` (Pydantic Settings) or `src/config/constants.py`. CI grep gate:
   no literal `sk-` keys; no magic numbers introduced in `src/`.
4. **Full test suite, ≥85% coverage.** The `fail_under = 85.0` branch-coverage gate
   (`pyproject.toml`) must stay green. New modules ship with unit tests; no real network/API
   calls in unit tests (mock all I/O); use the existing markers.
5. **CI/local parity.** `ruff`/`mypy` stay pinned in `[dev]`; run the full local gate
   (`black --check`, `ruff check`, `mypy src/`, `pytest`) before every push.

---

## 2. Milestone alignment

This plan maps onto the existing `planning/milestones.yaml` scheme so the milestone file stays
the system of record:

| Milestone | Was | This plan moves it to |
|---|---|---|
| M3 Review & Polish (93%) | blocked by "test cluster" | **Unblock & close** — Phase 0–2 |
| M4 Deployment Readiness (95%) | blocked by secrets + tests | **Close** — Phase 3 |
| M5 Advanced Features (0%) | planned | **Start** — Phase 5 |
| (cross-cutting MVP demo polish) | scattered in `MVP_ROADMAP.md` | **Phase 4** |

`planning/milestones.yaml` will be updated (Phase 0) to reflect verified completion and to
point at this document.

---

## 3. Phased plan

Each task lists **Owner agent** (how to execute it with the available subagent/worktree/MCP
tooling — see §5), **Size**, and **Acceptance Criteria (AC)**. ACs inherit §1.2 implicitly.

### Phase 0 — Re-baseline & doc reconciliation 🔴 CRITICAL · ~0.5 day
*Goal: replace stale, contradictory status with one evidence-backed source of truth.*

- **0.1 Establish the real test/coverage baseline.** Provision a clean env
  (`pip install -e ".[dev,neural]"`), run the full suite with coverage, and record actual
  pass-rate + per-module coverage as a CI artifact.
  - **AC:** A committed `docs/STATUS.md` (or updated `INTEGRATION_STATUS.md`) with real numbers
    replacing the stale "88.4% / 44 failures"; coverage report attached to CI; numbers
    reproducible via `pytest tests/ --cov=src`.
  - **Owner:** `general-purpose` subagent in a **worktree** (isolated install).
- **0.2 Retire/annotate stale docs.** Add a supersede banner to `NEXT_STEPS_PLAN.md`; correct
  the status tables in `GAP_ANALYSIS_REPORT.md` and `docs/NEXT_STEPS_INVESTIGATION.md`; update
  `planning/milestones.yaml` completion fields.
  - **AC:** No doc asserts a status contradicted by §1.1 without a "superseded" pointer.

### Phase 1 — Real correctness & packaging fixes 🟠 HIGH · ~1 day
*Goal: clear the small, genuine residue of the old "blocker cluster."*

- **1.1 RolloutPolicy typing consistency.** Annotate `rng: np.random.Generator` (and
  `max_depth: int`) on the ~6 test subclasses; consider a shared `BaseRolloutPolicy` test
  helper to prevent drift.
  - **AC:** `mypy src/` clean; the policy test subclasses conform to the protocol; the
    early-termination + framework + parallel-mcts suites pass. (Files: `tests/unit/test_mcts_early_termination.py`,
    `tests/unit/test_mcts_framework.py`, `tests/unit/test_parallel_mcts_ext.py`, `tests/e2e/test_user_journeys.py`.)
- **1.2 Make the `examples/langgraph_multi_agent_mcts` module importable without optional deps.**
  The module hard-imports `langchain_openai` at top level (`examples/langgraph_multi_agent_mcts.py:19`),
  so its public class `LangGraphMultiAgentFramework` (and the `HRMAgent`/`TRMAgent` it re-imports
  from `src/agents/`) cannot load without optional embedding deps. Move the
  `langchain_openai`/embedding imports behind a lazy/`TYPE_CHECKING` guard (mirror the existing
  optional-import pattern used for `langgraph`/`chess`); optionally add `examples/__init__.py`
  re-exporting `LangGraphMultiAgentFramework`.
  - **AC:** `from langgraph_multi_agent_mcts import LangGraphMultiAgentFramework` succeeds in a
    `[dev]`-only env (no `langchain_openai`); the previously-skipped/erroring `AGENTS_AVAILABLE`
    tests (e.g. `tests/chaos/test_resilience.py`) run; no circular imports.
- **1.3 DABStep `split` regression test.** The feature exists (`dataset_loader.py:98`); ensure
  a regression test pins the backward-compatible default and the unknown-split fallback.
  - **AC:** Test asserts `load()` defaults to `train` and that an unknown split falls back to an
    available one with a warning.

### Phase 2 — Targeted coverage to hold ≥85% 🟠 HIGH · ~2–3 days
*Goal: cover the modules that are actually under-tested.*

- **2.1 Google ADK per-agent unit tests** for the 5 implementations (academic_research,
  deep_search, data_engineering, data_science, ml_engineering) with mocked ADK clients and
  skip markers when ADK deps are absent.
  - **AC:** Each agent has init/route/error/timeout tests; ≥85% line coverage on
    `src/integrations/google_adk/`; zero real network calls; documented skip conditions.
- **2.2 REST + RAG path integration tests** for the now-wired `/query` flow and
  `framework_service.process_query` RAG branch (retriever present vs absent).
  - **AC:** Tests cover `rag_available=True/False`, degraded-LLM surfacing, and auth-guarded
    access; all I/O mocked; overall gate stays ≥85%.

### Phase 3 — Production readiness (close M4) 🟠 HIGH · ~2–3 days
*Goal: ship-safe staging deployment.*

- **3.1 Secret hygiene.** Replace plaintext placeholders in `kubernetes/deployment.yaml` with
  an `ExternalSecret`/SealedSecret reference (operator-driven); document rotation in
  `docs/SECRETS_MANAGEMENT.md`. Values resolve via env → settings; nothing secret in VCS.
  - **AC:** `git grep` finds no plaintext key material in `kubernetes/`; manifests reference an
    external store; rotation runbook exists; CI secret-scan passes.
- **3.2 JWT auth path** alongside the existing API-key auth in `src/api/auth.py`, selected by
  settings (`AUTH_MODE`), **API-key path unchanged by default**.
  - **AC:** JWT issue/verify covered by tests; API-key clients keep working with no config
    change; algorithm/expiry/secret sourced from settings.
- **3.3 Staging deploy + 24h soak.** Deploy once Phase 0–2 are green; run smoke + soak.
  - **AC:** Health checks green for 24h; smoke suite passes in staging; rollback documented.

### Phase 4 — Feature enhancements & demo polish 🟡 MEDIUM · ~1 week
*Goal: the user-facing items from `MVP_ROADMAP.md` + `NEXT_STEPS_PLAN.md` §3.*

- **4.1 LangGraph streaming** — `astream_events()` on the integrated framework with
  node-level snapshots, config-gated (`GraphConfig`), plus streaming REST responses.
  - **AC:** Token + node-completion events tested; non-streaming path unchanged.
- **4.2 LangGraph graph visualization** — `visualize_graph()` → Mermaid/ASCII + endpoint and
  node metadata. **AC:** Deterministic output covered by a unit test.
- **4.3 MCTS early termination** — `test_mcts_early_termination.py` already exists; confirm the
  convergence check is actually wired in `core.py` (config present per docs) and finish if not.
  - **AC:** Search stops on stable best-value within `early_stop_patience`; thresholds from
    settings; benchmark shows iteration savings with no decision-quality regression.
- **4.4 Demo comparison mode** (MCTS vs single-shot) + tree viz + streaming output
  (`MVP_ROADMAP` M2–M4). **AC:** `demo.py`/`app.py` expose the comparison; no hardcoded prompts.

### Phase 5 — M5 Advanced Features (the real frontier) 🟢 STANDARD · 2–4 weeks
*Goal: build what is genuinely unbuilt. Building blocks exist (`neural_mcts.py`,
`policy_value_net.py`, `value_network.py`); the orchestration/loops do not.*

- **5.1 AlphaZero-style self-play training loop** generalized beyond chess: a `SelfPlayTrainer`
  base class + replay buffer + value/policy update loop wired to the existing neural MCTS.
  - **AC:** Demonstrated ≥20% decision-quality improvement on a defined benchmark vs the
    untrained policy (the M5 epic target); chess remains a passing instance; checkpoints
    versioned in the safe (non-pickle) format.
- **5.2 Domain adapters** — `DomainAdapter` protocol + **≥2 non-chess** adapters (reasoning,
  planning) selected via factory/config. **AC:** Both adapters pass a shared adapter contract
  test; registered dynamically (no hardcoded dispatch).
- **5.3 Meta-controller learning loop** — collect live routing decisions/outcomes, fine-tune,
  validate on a held-out set; document in `docs/META_CONTROLLER_TRAINING.md`.
  - **AC:** Training + validation loop with metrics; weight updates are reproducible (seeded);
    routing accuracy reported vs baseline.

---

## 4. Sequencing, dependencies & timeline

```
Phase 0 ─► Phase 1 ─► Phase 2 ─► Phase 3 ─► (staging) 
                          └─► Phase 4 ─┐
                                       ├─► Phase 5
            (Phase 5 depends on a green ≥85% baseline from 0–2)
```

| Phase | Focus | Est. | Gate to next |
|---|---|---|---|
| 0 | Re-baseline + doc truth | 0.5d | Real numbers committed |
| 1 | Correctness/packaging | 1d | mypy+suite green |
| 2 | Coverage (ADK, REST/RAG) | 2–3d | ≥85% holds |
| 3 | Secrets, JWT, staging | 2–3d | 24h soak green |
| 4 | Streaming/viz/demo | ~1wk | Features tested |
| 5 | Neural self-play + adapters | 2–4wk | M5 AC met |

---

## 5. Execution model (subagents / worktrees / MCPs)

This plan is built to be executed with the available tooling, with isolation so parallel work
never collides:

- **Worktrees** (`isolation: "worktree"`): any phase that installs the heavy ML stack or mutates
  files in parallel (Phase 0 baseline, Phase 2 test authoring, Phase 5 training) runs in its own
  git worktree so installs/edits don't conflict.
- **Subagents** follow the `planning/milestones.yaml` role keys (`planner`, `coder`, `sge`,
  `reviewer`, `orchestrator`), realized with the available tooling: `coder` → implementation
  (general-purpose subagent), `sge` (Software Quality Engineering) → test authoring (Phase 2),
  `reviewer` → a `code-review`/`security-review` skill pass before each push, `planner` →
  codebase verification (Explore subagent), `orchestrator` → cross-phase coordination.
- **Parallelizable fan-out:** Phase 2.1 (5 independent ADK agents) and Phase 5.2 (independent
  domain adapters) are natural per-item fan-outs — one subagent per agent/adapter, each in a
  worktree, results merged after the shared contract test passes.
- **MCP / GitHub:** each phase lands on the feature branch as a focused commit; a **draft PR**
  is opened and CI watched via the GitHub MCP tools. Hugging Face MCP is available for sourcing
  models/datasets in Phase 5 (neural MCTS / domain corpora).
- **Quality gate per push:** `black --check` → `ruff check` → `mypy src/` → `pytest --cov`
  (≥85%) → secret grep, matching the repo's CI/local-parity rule.

---

## 6. Success metrics

| Metric | Baseline (to confirm in 0.1) | Target |
|---|---|---|
| Coverage gate | 85.0% (`fail_under`) | ≥85% sustained, ≥85% on new modules |
| mypy (`src/`, strict) | green | green |
| Stale/contradictory status docs | several | 0 (single source of truth) |
| K8s plaintext secrets | present | 0 |
| Google ADK module coverage | base only | ≥85% |
| M5 neural decision-quality lift | n/a (0%) | ≥20% vs untrained |

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Docs keep drifting from code | High | Medium | Phase 0 single-source-of-truth + CI-attached coverage |
| Heavy ML install destabilizes CI runners | Medium | Medium | Worktree isolation; `[dev]`-only path for non-neural phases |
| Neural self-play doesn't hit 20% lift | Medium | Medium | Define benchmark + baseline first; iterate; keep chess instance green |
| Backward-compat regressions (auth/streaming) | Low | High | Default-preserving flags; contract tests; API-key path untouched |
| Secret migration breaks deploy | Low | High | Stage with external-secrets in non-prod first; documented rollback |

---

*Generated 2026-06-30 from a code-verified reconciliation of all roadmap, milestone, and
next-steps documents. Where this plan and an older doc disagree, this plan’s §1.1 evidence
governs.*
