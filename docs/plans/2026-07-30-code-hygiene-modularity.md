# Strategos-MCTS Code-Hygiene & Modularity Program — Peer-Reviewed Plan v2

> Rewritten after a 3-agent adversarial peer review (architecture critic, live-tree fact-checker,
> methodology reviewer — 46 findings, all resolved below). Adheres to the repo's SDD charter and
> schema-v2 spec format (`spec_validator.py`: frontmatter `id/goal/module/status`, sections
> `# Goal` / `# Acceptance Criteria` (`- AC-n: `) / `# Constraints`, optional `# Out of Scope`).
> Program docs land on `claude/code-hygiene-modularity-skvtl6`; implementation follows the charter below.

## Context

A verified two-pass audit (8 scan agents + executable proofs) found: 3 proven MCTS search bugs;
~30k LOC of `src/` and ~13k of root `training/` unreachable from any production entry point but
kept green by ~15k LOC of tests; a quality gate that blocks on only 79% of test files with a
double-discounted coverage denominator; a second unresilient LLM client stack on the live path;
3 incompatible `MCTSConfig`s; a chess shadow-settings system; and a training container whose pins
prevent it importing `src/`. User decisions (recorded): delete `llm_guided/`; delete all
confirmed-dead code; full `training/` de-fork; phased PR series; wire chess `ui.py` in; delete
`examples/chess_demo/`; runtime `DeprecationWarning` on shims; e2e workflow → skip-if-no-secret.

## Peer-review disposition (what changed in v2)

- **Charter (HIGH)**: the No-Spec-trailer blanket was invalid — its CLAUDE.md precondition ("until
  the first approved spec merges") lapsed; `NEXT_STEPS…2026H2.md:17` makes spec-first
  **non-negotiable**; `.claude/hooks/spec_gate.py:36-37` documents a planned flip to block-mode
  that would hard-stop a `claude/*` program. → v2 is **spec-first**: each phase below is a draft
  spec; `spec-review` subagent gates it; a human flips draft→approved; implementation runs on
  `spec/<id>` branches. Where `module:` overlaps an OPEN approved spec (`src/framework/mcts/`,
  `src/training/`, `src/framework/graph/`), the phase either waits for that spec to close or
  carries a **human-approved, documented No-Spec exception** (flagged per phase below).
- **`SyncLLMBridge` (HIGH)**: `asyncio.run` inside the running loop would crash the live async
  `/compare` route (`rest_server.py:673-693`). → P9b makes the pipeline **async-native** through
  `ComparisonService`; sync entry survives only at CLI/demo top level.
- **`.gitignore` (HIGH)**: `!benchmarks/results/` under an excluded parent dir is a no-op. → rule
  becomes `benchmarks/*` + `!benchmarks/results/`.
- **`s3_client` contradiction (HIGH)**: P2 no longer "fixes" the import that P5 deletes —
  `verify_setup.py:126` drops the S3 check; `storage/__init__.py:14` re-export dies with the module.
- **Deletion protocol (HIGH)**: several "zero-importer" targets have package-`__init__`
  re-exports / `__all__` entries / availability probes / factory dispatch (`harness/__init__.py:7`,
  `meta_controller/__init__.py:107-173`, `utils/__init__.py:8-25`, `factories.py:426-505`). →
  **same-PR rule**: a module's re-exports, `__all__`, probes, factory dispatch, orphaned settings
  fields, and CLAUDE.md rows are removed in the same PR; reachability greps are **repo-wide**
  (scripts/, demos/, examples/, notebooks/, Dockerfiles, compose, kubernetes/, workflows,
  string literals). Factories trim merges into the same cluster PR as the controllers it dispatches.
- **Retry misdiagnosis (MED)**: `scorer.py:113-126` retries judge-JSON parse failures;
  `evaluation/harness.py:322-349` retries task timeouts — application-level, NOT duplicates of
  transport retry. → kept; total multiplicative attempts bounded via settings.
- **Methodology (26 findings)**: no long-lived integration branch — one PR per phase/cluster direct
  to main; P5 split into 4 cluster PRs; rollback tag per destructive PR; characterization pinned
  (scenarios, tolerance, golden location, amendment protocol); P4 split mechanical/triage with a
  triage matrix cross-checked against the kill list; compat surfaces enumerated (Docker CMDs,
  `docker-deployment.yml`, k8s probes, OpenAPI schemas, MCP tool names, retired env vars);
  deferred work gets tracking artifacts; `cloc` recorded per destructive phase; PR-body template;
  `/security-review` on P2 and P11; shared `src/utils/deprecation.py` helper with warning tests.
- **Fact corrections**: use existing `SEED` (settings.py:125), not a new `RANDOM_SEED`; harness
  hook fields are `HOOK_*` under `env_prefix="HARNESS_"` (else `HARNESS_HARNESS_*`); `LLM_RETRY_*`
  must reconcile with existing `HTTP_MAX_RETRIES` (settings.py:197, wired via factories.py:93);
  the `1.414` literal count is 22 across 13 files (not 9); `datetime.utcnow` is 8 sites/10
  occurrences; canonical chess `RoutingDecision` is `src/games/chess/meta_controller.py:27`
  (assembly_router's dies in P5, llm_guided's in P6); ensemble-chain retention rests on user
  decision 5 + `ui.py` wiring, not the roadmap; counted inventories are re-grepped at execution.

## Program charter (governance)

1. **Spec-first.** Every phase = one `specs/<id>.SPEC.md` (schema v2). Flow: author draft (P0) →
   `spec-review` subagent review → human flips to `approved` → implement on branch `spec/<id>` →
   PR to main flips status to `implemented` in its own diff → `verified` only with same-line
   `<spec-id> AC-n` mappings under `tests/`.
2. **Module-overlap gating.** Phases touching `src/framework/mcts/` (P1, P8), `src/training/`
   (P3-part, P8c), `src/framework/graph/` (P11-adjacent) overlap OPEN approved specs
   (`strategos_risk_averse_subgoal_scorer`, `ddp_orchestrator`, `m5_policy_lift`,
   `strategos_subgoal_scoring_seam`). Resolution per phase: **P1** = human-approved No-Spec
   exception (urgent proven bugfix; and `strategos_risk_averse_subgoal_scorer` promises a
   bit-for-bit baseline, so P1 MUST land before that spec's implementation starts — coordinate
   re-baseline in its thread); **P8/P8c** = wait for those specs to close, or human re-scopes
   modules; others don't overlap.
3. **PR model.** One PR per phase (or per P5 cluster) direct to main. No long-lived integration
   branch; `claude/code-hygiene-modularity-skvtl6` carries program docs only (plan copy at
   `docs/plans/2026-07-30-code-hygiene-modularity.md`, draft specs, PR template).
4. **Per-PR protocol**: `quality-gate` skill green locally → `/code-review` → `/security-review`
   where flagged (P2, P11) → PR body from template (phase id, spec link, AC checklist, pasted gate
   summary, rollback tag if destructive) → CI green. Destructive PRs: annotated tag
   `pre-<id>` + restore recipe in MIGRATION_NOTES. `coverage-baseline` skill + `cloc src tests
   training` recorded in STATUS.md after every destructive phase and at close. CHANGELOG entry per
   PR; MIGRATION_NOTES for behavior changes; retired env vars get a warn-on-presence list.
5. **Coordination.** P8, P10, P12 re-read all open specs' `module:`/symbol claims as their first
   step (not just at plan time). Deferred items (StdlibLLMClient removal, mypy strictness ratchet,
   container-smoke promotion) each get a tracking issue or draft spec when deferred.

## Phase specs (schema v2 drafts — authored under `specs/` in P0)

### P0 — `hygiene_program_bootstrap` (module: `docs/` — no gate)
Goal: land the program's governance artifacts. AC-1: plan copied to
`docs/plans/2026-07-30-code-hygiene-modularity.md`. AC-2: all phase specs below exist as `draft`
and `harness validate-spec specs/*.SPEC.md` exits 0. AC-3: PR-body template at
`.github/PULL_REQUEST_TEMPLATE/hygiene_phase.md`. AC-4: `cloc` baseline recorded in STATUS.md.
Constraints: no `src/**` changes; this branch.

### P1 — `hygiene_mcts_value_semantics` (module: `src/framework/mcts/` — **No-Spec exception, human-approved; must precede risk_averse implementation**)
Goal: fix the three proven selection bugs (PUCT double-division `neural_policies.py:727`; negamax
selection sign in `parallel_mcts.py` and `progressive_widening.py` incl. RAVE) by adopting the
`negate_child_value` pattern (`neural_mcts.py:169-210`), with perspective as an explicit
settings-backed config field.
- AC-1: regression suite ported from the executable proofs — all three engines select the
  minimax-optimal child on a seeded 2-ply tree.
- AC-2: cross-engine parity — core/parallel/PW agree on root action for seeded small states.
- AC-3: `select_child_puct` equals canonical `puct()` on 1,000 seeded random inputs.
- AC-4: DEBUG structured per-child selection logging via `get_logger`.
- AC-5: affected benchmark baselines re-run and re-recorded; m5 lift artifact flagged for
  re-validation in its spec thread; MIGRATION_NOTES entry (no escape hatch — documented as intentional).
Constraints: no symbol renames/moves (open specs cite this module); config via settings; reuse
`mcts_config`/`MCTSStateBuilder`.

### P2 — `hygiene_small_fixes` (modules: `src/framework/`, `src/observability/`, `scripts/`, `.gitignore`)
Goal: wire the harness security hooks; fix tz-naive datetimes; unblock the M5 artifact path.
- AC-1: `HarnessFactory.create_hook_chain()` registers secret_scan/payload_size/required_keys from
  NEW `HarnessSettings` fields `HOOK_SECRET_SCAN`/`HOOK_PAYLOAD_SIZE_LIMIT`/`HOOK_REQUIRED_KEYS`
  (env `HARNESS_HOOK_*`; precedent `HOOK_SHORT_CIRCUIT_DEFAULT` at settings.py:147); default ON;
  OFF flags documented in MIGRATION_NOTES; hook-chain unit tests (contents, toggles, limits).
- AC-2: `datetime.utcnow` → `utc_now()` at surviving sites (re-grep; 8 sites baseline, skip those
  scheduled for deletion, list skips in PR body); ruff `DTZ` rules enabled and green.
- AC-3: `.gitignore:269` replaced with `benchmarks/*` + `!benchmarks/results/`;
  `!data/training_with_assembly.json` added; `git check-ignore` confirms the M5 path is trackable.
- AC-4: `verify_setup.py:126` S3 check **dropped** (module scheduled for deletion in P5-3).
- AC-5: NEW `src/utils/deprecation.py` (`warn_deprecated(old, new, stacklevel)`) + parametrized
  test asserting exactly one `DeprecationWarning` with correct stacklevel — used by all later shims.
Constraints: `/security-review` runs on this PR (secret-scan semantics); settings only, no literals.

### P3 — `hygiene_determinism` (module: `src/utils/` + touched call sites; `src/training/` sites coordinated with open specs)
Goal: one seeding utility; reproducible Dirichlet noise.
- AC-1: `src/utils/seeding.py` — `set_all_seeds(seed, *, rank=0, deterministic_torch=False)`
  (rank-aware for DDP; torch behind import guard) + `new_rng(seed)`; 100% branch coverage.
- AC-2: reuses existing `SEED` field (settings.py:125) — **no new env name**.
- AC-3: `neural_mcts.py:322` uses an injected `np.random.Generator`; double-run reproducibility
  test (identical visit counts + dirichlet draws; same-machine, fresh-process).
- AC-4: divergent seed sites migrated (re-grep at execution; 12 baseline), legacy `seed=` kwargs
  preserved; effective seed logged at INFO on init.
Constraints: `src/training/` call-site edits are mechanical only (open specs claim the module —
note in PR body); conftest gains an opt-in (not autouse) `global_seed` fixture.

### P4a — `hygiene_ci_mechanical` (module: `.github/`, `pyproject.toml` — no src gate)
Goal: make the gate structurally honest without waiting on test triage.
- AC-1: `chess-tests` + `integration-test` added to summary-job `needs` + failure conditions.
- AC-2: `--strict-markers`; unused markers pruned (re-derive list; 19 baseline).
- AC-3: e2e workflow: `|| true` removed; jobs conditional on LangSmith secret presence.
- AC-4: pre-commit `pytest-quick` can fail.
- AC-5: rest_server suites gated at **collection level** (must import/collect; known failures
  xfail with reasons) — all three suppression layers addressed together: `ci.yml:247-249` ignores,
  `tests/conftest.py:81-84` `collect_ignore_glob`, `pyproject.toml:361-362` coverage omits.
  Green-ness deferred to P11.
- AC-6: post-change coverage gate **dry-run locally, measured number pasted into the PR** before
  the CI flip; CHANGELOG "Quality-gate changes" section states old vs new blocking set + baseline.
- AC-7: `CLAUDE.md:41` mypy claim corrected to reality; strictness ratchet gets a tracking draft spec.

### P4b — `hygiene_test_triage` (module: `tests/` — runs parallel with Wave C)
Goal: classify the ~84 never-gated test files. AC-1: every file classified via the triage matrix —
**fix** / **xfail-with-issue** / **delete-with-module (cross-checked against the P5/P6/P7 kill
list FIRST)**; 0 unclassified. AC-2: blocking job covers all surviving `tests/unit/`; integration
promoted to PR-blocking if wall-time < 10 min (measured), else main-push + summary-gated. AC-3:
skipped/xfailed counts reported in STATUS.md. Constraints: time-boxed; no fixing tests for code on
the kill list.

### P5 — dead-code removal, four cluster PRs (each: own spec, rollback tag, repo-wide reachability re-verification, same-PR cleanup rule, coverage dry-run before merge)
- **P5-1 `hygiene_delete_enterprise_cluster`** (`src/enterprise/`, `src/framework/component_factory/`,
  `src/performance/` + their tests incl. `test_component_factory.py` 2,004 ln).
- **P5-2 `hygiene_delete_chess_dead`** (`src/games/chess/verification/` + `observability/`; NOT
  `engines/` — kept per decision 5; logger-name strings at `observability/logging.py:333,341`
  cleaned same PR).
- **P5-3 `hygiene_delete_storage_api`** (`storage/s3_client.py` + `storage/__init__.py:14`
  re-export + `export_architecture_diagrams.py:63` string; `storage/faiss_store.py`;
  `api/health.py` — verify k8s/compose probes hit `rest_server`'s own `/health` route first;
  `api/inference_server.py` + its ci.yml ignore + pyproject omit; `framework/caching.py`;
  `utils/planning_loader.py` + `utils/mcts_debug.py` + `utils/__init__.py:8-25` cleanup).
- **P5-4 `hygiene_delete_framework_cluster`** (`observability/facade.py`;
  `meta_controller/{hybrid_controller,assembly_router}.py` **plus** `meta_controller/__init__.py:107-173`
  guards/`__all__`/probe **plus** `factories.py` trim to `LLMClientFactory` in the SAME PR (no
  dangling dispatch window); `harness/loop/facade.py` + `harness/__init__.py:7,31` re-export;
  `harness/memory/heartbeat.py` + `MEMORY_HEARTBEAT_INTERVAL_SECONDS` (harness settings:106) +
  compressor docstring ref; `harness/topology/` + `create_topology` + `TOPOLOGY*` settings;
  `models/validation.py` trim to `QueryInput`+transitive; dead `ProgressiveWideningConfig`
  (policies.py:253-313); `edge_cases.py` → harvest `MCTSTerminationReason`/`MCTSSearchResult` into
  `core.py` as str-Enums (string `==` stays valid), re-export from `mcts/__init__`, delete rest).
- Shared ACs: AC-1 zero references to each deleted module repo-wide (excl. CHANGELOG/
  MIGRATION_NOTES/git history). AC-2 CLAUDE.md rows for deleted features removed in same PR.
  AC-3 CHANGELOG Removed with replacement pointers (`mcts_debug`→`observability/debug`,
  `faiss_store`→`local_embedding_store`, `facade`→direct APIs). AC-4 coverage ≥85 demonstrated by
  local dry-run pasted in PR. Exclusions (verified): `coarse_dynamics.py` (approved spec),
  `engines/stockfish_adapter.py`, chess ensemble chain, `llm_guided/` (P6).

### P6 — `hygiene_delete_llm_guided` (module: `src/framework/mcts/llm_guided/` + `src/config/`)
Goal: remove the parked successor stack. AC-1: subtree + tests + 2 comment mentions gone; tag
`pre-llm-guided-removal` + MIGRATION_NOTES restore recipe (notes RAG-context overlap with roadmap
2.x). AC-2: settings fields `MCTS_GENERATOR_MODEL`/`MCTS_REFLECTOR_MODEL`/`MCTS_EXECUTION_TIMEOUT`/
`MCTS_MAX_MEMORY_MB` (settings.py:388-404) removed; **pydantic `extra` policy verified here** so
stale `.env` entries are ignored-not-fatal; retired names added to the warn-on-presence list.
AC-3: coverage dry-run ≥85.

### P7 — training de-fork, three PRs
- **P7-fix `hygiene_train_container`** (early, Wave B; module `Dockerfile.train`): install from
  root requirements/pyproject (single pin source). AC-1: container builds and `python -c "import
  src"` succeeds inside it. AC-2: `docker-deployment.yml` training-demo container run stays green.
  AC-3: tag `pre-training-defork`.
- **P7a `hygiene_train_extract`** (**after P9a** so migrated code targets the unified adapter
  base): `checkpoint_loader.py` → `src/utils/checkpoint_loader.py` (+ adopt at the ~10 open-coded
  `torch.load` sites; AC: 0 open-coded `torch.load` outside it in `src/`);
  `synthetic_knowledge_generator.py` → `src/benchmark/synthetic/`; `benchmark_suite.py` →
  `src/benchmark/rag_metrics.py`; embedder abstraction → adapters-backed; knowledge-graph model
  half extracted with typed-exception LLM half. **Each migrated original becomes a re-import shim
  (with `warn_deprecated`) in the same PR** — no divergence window. The 6 CI test files' imports
  migrate atomically. Mocks: `MockPineconeClient` replaces the `sys.modules` clobber.
- **P7b `hygiene_train_delete`**: delete superseded modules + `training/tests/` +
  `training/requirements*.txt` + the P7a shims. AC-1: surviving `training/` entry surface
  enumerated (or `training/` removed entirely and `Dockerfile.train` CMD retargeted to the
  src-hosted CLI — decide at spec-review with the Dockerfile.test ENTRYPOINT
  (`scripts/run_comprehensive_training.py`) checked). AC-2: `docker-deployment.yml` updated same
  PR; both training workflows green. AC-3: coverage + cloc recorded.

### P8 — MCTS consolidation, three PRs (**gated on closure/re-scope of `strategos_risk_averse_subgoal_scorer`; P8c also on `ddp_orchestrator`/`m5_policy_lift` for `src/training/system_config.py`**)
- **P8a `hygiene_mcts_policies`**: canonical UCB1 (`policies.py:57`) + PUCT (`neural_policies.py:92,130`)
  everywhere; `1.414` literal (22 occurrences / 13 files, re-grep) → `DEFAULT_EXPLORATION_WEIGHT =
  math.sqrt(2)` in `framework/mcts/config.py`, `settings.MCTS_C` override; `llm_mcts` re-exports kept.
- **P8b `hygiene_mcts_engines`**: shared backprop parameterized by the P1 perspective flag; engines
  subclass `core.MCTSEngine`/`MCTSNode` (name which in the spec); one stats/best-action/tree-depth
  helper (iterative); `raise_if_invalid()` replaces 12 validation copies.
- **P8c `hygiene_mcts_config`**: canonical `MCTSConfig` = `framework/mcts/config.py:32`;
  `src/training/system_config.py:69` and `models/validation.py:110` become thin wrappers
  (import paths + field names preserved, `warn_deprecated`).
- Characterization protocol (pinned): 25 seeded scenarios via `MCTSStateBuilder`; exact-match on
  best action + visit counts; `rel=1e-9` on values; goldens as JSON under
  `tests/fixtures/characterization/`; any golden change enumerated old→new in the PR body.
  P1 regression suite is a required check for P8b.

### P9 — LLM stack, two PRs
- **P9a `hygiene_llm_base`**: shared `_get_client`/`close`/`_handle_error_response`/retry-decorator/
  breaker-wrap in `BaseLLMClient`; retry knobs **reconcile with existing `HTTP_MAX_RETRIES`
  (settings.py:197) + `constants.py:75-77,260`** — no duplicate semantics, aliases where renamed;
  `LMStudioClient(OpenAIClient)` (regains 401/429/404 mapping); mock unification on the conforming
  replay-client shape; protocol-conformance suite (keyword-only async `generate` → `LLMResponse`)
  parametrized over ALL clients (httpx MockTransport) + mocks — permanent gate.
- **P9b `hygiene_llm_pipeline`**: `llm_mcts` pipeline + `ComparisonService` become **async-native**
  (the live caller `rest_server.py:673-693` is an async route; no `asyncio.run` inside the loop);
  sync entry only at CLI/demo top level; `StdlibLLMClient` kept one cycle as a deprecated shim
  (tracking issue for removal); `generate_sync` protocol collapses; application-level retries in
  `scorer.py`/`evaluation/harness.py` KEPT with settings-bounded total attempts; unused
  `decorators.py:306` retry adopted-or-deleted (recorded). Route `/compare` behavior parity
  asserted by existing tests.

### P10 — `hygiene_config_consolidation` (module: `src/config/` + shims; re-read open specs first)
One shared `model_config` base (kills 5 env_file copies); singletons → sub-settings off
`get_settings()` with delegating shims; chess shadow config (`games/chess/config.py:391-447`) →
Pydantic CHESS_ fields via lazy accessors (`constants.py` pattern); assert-validation → validators;
dead fields removed post-grep. AC: NEW permanent guard test — no duplicate env names across all
Settings classes; env-precedence tests via `settings_override`; no mass case rename.

### P11 — API refactor, two PRs (after P9/P10; `/security-review` required)
- **P11a `hygiene_rest_split`**: `src/api/rest/` package (`models/deps/lifespan/routes/app.py`,
  `create_app(settings=None)`); `rest_server.py` stays as shim. AC: route-parity (method,path) set
  unchanged **and OpenAPI schema snapshot unchanged**; import-time `get_settings()` gone; the P4a
  collection-gated suites now pass green against `create_app()`.
- **P11b `hygiene_framework_service`**: `FlexibleLogger` → `get_logger`; `initialize`/
  `process_query` decomposed (state max function length in spec, e.g. ≤60 lines); prod
  `MockLLMClient` (framework_service.py:694) removed; `LightweightFramework` → own module,
  documented fallback.

### P12 — `hygiene_chess_consolidation` (after P5-2/P10; re-read open specs first)
Canonical `RoutingDecision` = **`src/games/chess/meta_controller.py:27`** (llm_chess_engine.py:200
re-exports it); one phase classifier in `state.py:304` with settings-backed thresholds
(anchors: llm_chess_engine.py:339, examples copy dies with decision 6); all piece-value consumers →
`constants.get_piece_values` (settings-backed, scale parameter; kills state.py:315/:349 +
llm_chess_engine.py:65 tables); public facade replaces the 3 private-member reaches
(mcp_chess_tools.py:307-330); `tools/mcp/server.py:145,185` de-coupled via a `register_tools` seam
— AC: 0 chess imports in the generic server; **MCP tool name+schema set unchanged** (parity test).
Wire `ui.py`: console script `chess-ui`; split `ui/render|controller|learning|app`; module-global
session state injected. Delete `examples/chess_demo/` + `tests/chess_demo/` (tag). Classifier PR
body enumerates every FEN whose classification changed (old→new).

### P13 — `hygiene_consistency_sweep` (five fan-out tracks, disjoint files)
(1) `benchmark/` bare `logging.getLogger` (9 modules) → `get_logger` (API shape untouched — draft
benchmark spec exists); AC: 0 bare getLogger in `src/benchmark/`. (2) library-code `print()` →
logger; AC: 0 `print(` outside CLIs. (3) sanitize dedupe → `sanitize_dict`. (4) import hygiene:
guard `feature_extractor.py:12`, settings not raw env; `utils/__init__` lazy `personality_response`.
(5) test hygiene: merge files matching **`_ext\d*\.py$` suffix rule only** (explicit list in spec —
the naive `*_ext*` glob over-matches `test_feature_extractor.py` etc.); MANDATORY
`pytest --collect-only -q` count-parity per merged pair + per-module coverage non-decrease; rename
duplicate basenames; relocate 11 loose root tests.

### P14 — `hygiene_docs_closure` (docs only)
CLAUDE.md: `:88` graph.py row, harness table rows for deleted facade/topology, factories row,
final mypy statement. PROJECT_STRUCTURE.md:165 (~1.49GB LFS). MIGRATION_NOTES ordering pass.
CHANGELOG grooming. Final `coverage-baseline` + cloc → STATUS.md (program acceptance artifact).
`validate-specs` all green; every remaining draft/deferred item has a tracking artifact.
`validate-context` after CLAUDE.md edits.

## Wave order

```
Wave A:  P0 → {P1*, P2, P3}        (* No-Spec exception, human-approved)
Wave B:  P4a, P7-fix               (mechanical; unblocks everything)
Wave C:  P4b ∥ P5-1..4 ∥ P6       (triage parallel to deletions; kill-list cross-check)
Wave D:  P8a→P8b→P8c† ∥ P9a→P9b ∥ P10    († gated on open-spec closure)
Wave E:  P7a (after P9a) → P7b;  P11a→P11b;  P12
Wave F:  P13 → P14
```

## Verification protocol

Per PR: quality-gate → /code-review → (/security-review for P2, P11) → template PR body with AC
checklist + pasted gate output → CI green (post-P4a hardened). Destructive PRs: tag + restore
recipe + coverage/cloc dry-run pasted before merge. Determinism: same-machine fresh-process
double-run of seeded MCTS suites after P3 and P8b. Conformance: P9a suite green across all
clients/mocks permanently. Parity: P11a route+OpenAPI snapshots; P12 MCP tool set; P13 collect-count
per merged pair. Baselines: STATUS.md refreshed (coverage-baseline skill) after P4, each P5 cluster,
P6, P7b, P13, P14 — real numbers, never gamed.

## Net effect (prediction, measured per-phase via cloc)

≈ −50k src LOC / −18k test LOC; gate honest (84 files triaged into the gate, no unfailable
workflows, true denominator); 3 proven search bugs fixed with permanent regression suites; one LLM
stack (async-native live path), one MCTS engine hierarchy + config, one settings system, structured
logging + correlation IDs across `benchmark/`, a reproducibility story (rank-aware seeding), and
every deletion tagged, verified repo-wide, and documented.

## Progress log

- **2026-07-30 — P0 (`hygiene_program_bootstrap`)**: this document, all 25 phase specs (`draft`),
  the `hygiene_phase.md` PR template, and the LOC baseline in `docs/STATUS.md` landed on
  `claude/code-hygiene-modularity-skvtl6`.
