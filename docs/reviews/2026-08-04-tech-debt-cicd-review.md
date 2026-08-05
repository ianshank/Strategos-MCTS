# Tech-Debt & CI/CD Plan Review — two drafts withdrawn (2026-08-04)

- **Reviewed documents**: two successive drafts of a tech-debt / CI-hardening plan, authored in
  session on `claude/tech-debt-cicd-plan-q09i8e`. Neither was merged; both are withdrawn. This
  document is their disposition record.
- **Tree audited**: `main` @ `74877fa` (2026-08-03), 43 specs, 327 Python files. Status ledger:
  `docs/STATUS.md`, content-dated 2026-07-25, committed 2026-07-31 (`09d82a3`).
- **Method**: 11 agents — six parallel fact-checkers (CI reality, coverage arithmetic, code
  organization, config/hardcoded values, governance, backwards compatibility), four adversarial
  critics over the verified facts (charter compliance, blast radius, value-per-effort, CI/CD
  engineering correctness), one synthesizer. Every quantitative claim in both drafts was
  re-measured against the tree rather than inherited from documentation.
- **Reviewer verdict**: **the drafts were planning artifacts written on top of an unexamined
  baseline.** 49 claims were refuted or downgraded; 46 real issues appeared in neither draft; 16
  proposals were cut outright. The single most consequential finding is not any individual number
  but that **both drafts asserted `main` was green while `main` was red**, and that the first draft
  re-proposed, in worse form, a program the project had already written and already peer-reviewed.

---

## 1. The finding both drafts missed

`main` has been red since at least 2026-07-31. Measured directly against the Actions API:

```text
docker-deployment.yml on main: 30 of the last 30 runs → failure
Latest: run 30863130835 · HEAD 74877fa · push → failure
```

The cause is deterministic, not a flake:

```text
training/requirements.txt:21    datasets~=5.0.1     # requires requests>=2.32.2
training/requirements.txt:121   requests~=2.31.0
→ pip ResolutionImpossible at Dockerfile.train:84
```

PR #127 (`dependabot/pip/training/requests-approx-eq-2.34.2`) is the one-line fix and has been open
since 2026-07-31. A plan whose stated goal was "get CI/CD into a green state" did not find it.

## 2. Why the first draft was structurally wrong

It proposed a new six-phase, four-to-six-week tech-debt program. That program already exists:

- `docs/plans/2026-07-30-code-hygiene-modularity.md` — already **v2 after a three-agent adversarial
  peer review that resolved 46 findings**, with a governance charter, per-PR protocol, rollback-tag
  rules, and a Wave A–F execution order.
- **25 `hygiene_*.SPEC.md` files already authored**, one per phase, all schema-v2 valid.
- `CHARTER.md` §5 already themes the work as milestone **M5 — Governance closure**, gate **G-M5**.

`CHARTER.md` **NG-7** reads: *"No second planning system. Work is planned in `specs/` and
`docs/plans/`."* The draft violated the charter it was written to serve. Four of its phases
(`green CI`, `config centralization`, `coverage`, `repo organization`) re-proposed specs already
marked `implemented`.

## 3. Refuted claims

Recorded so they are not re-derived. Draft 1 unless marked.

| Claim | Measured reality |
|---|---|
| "CI is sequential; parallelizing gains 30%" | Six jobs have no `needs:` and already start together (`ci.yml:19/45/106/132/173/190`). The proposed restructure *adds* serialization. |
| "CI runs ~7–8 minutes" | Last successful `main` run: **2935s (48m55s)**; range 1360–2935s. Off by 3–6×. |
| "Docker build 2–3m, main-only" | `docker-build` is **2738s = 93.3%** of wall clock, has **no job-level `if:`** (`ci.yml:397`) so it runs on every PR, and **builds the same image twice** (`:445`, then `:537`). |
| "`pytest -n auto` gains 25%" | ~**0%** — the test job finishes ~41 minutes before the run ends. `pytest-xdist` was already installed and unused. |
| "mypy takes 60s" | mypy is **6–9s**; the 148s job is ~100s of `pip install` pulling torch. |
| "Coverage 90.15% → 92% via five modules" | Against a measured 41,471-unit denominator, all five to 85% = **+0.162pp**; to an impossible 100% = **+0.619pp**. 92% needs 767 units; those modules hold 257. **Overstated 11.4×.** |
| "Five modules below 85%" | At least **21**. Three genuine gaps were dropped from STATUS.md's own list. |
| "`llm_guided/` is 30k+ dead LOC" | **9,883 LOC**. The 30k figure is the repo-wide dead total, mis-attributed to one subtree. |
| "Move `src/observability/` — ~70 imports" | **173 files, 539 occurrences.** `src/observability` imports nothing from `src/framework`; nesting a dependency-free leaf under one of its consumers *reduces* cohesion. |
| "Four config classes → one `AppConfig`" | **16** `BaseSettings` classes across **15 disjoint env prefixes**. Merging breaks INV-3 dependency injection. |
| "Timeouts 30/60/300 are magic numbers" | `300` appears **zero** times; `60` twice, both already named constants. Exactly **one** unnamed timeout literal exists in `src/`: `s3_client.py:90`. |
| "Batch sizes 32/64/128" | `128` does not occur as a batch size. Five of seven real sites are correct design. |
| "`vulture` will find the dead code" | 19 findings, ~14 of them `exc_tb`-shaped false positives, and **zero** real orphans — every dead module has a test importing it. Dead code here is package-level orphaning, which needs import-graph reachability. |
| "No central exceptions module" | `src/api/exceptions.py` (324 LOC) already defines `MCTSError` and `ConfigurationError`, and is already imported cross-layer. |
| "Create `tests/shared/fixtures.py`" | `tests/conftest.py` is 829 LOC / 37 fixtures; `tests/fixtures/` is 1,713 LOC. The problem is under-adoption, not absence. |
| "Deprecate, remove in v0.3.0" | **0 git tags, 0 releases**, no publish step, `version = "0.1.0"`. The class is `ParallelMCTSEngine`, not `ParallelMCTS`. |
| "Verify by pushing this branch to CI" | `ci.yml` triggers only on `main`/`develop`. This branch runs nothing. |
| "Dependencies & blockers: none" | At least four, three of them governance-level. |
| *(Draft 2)* "Stable and green; needs no rescue plan" | 30/30 red — see §1. |
| *(Draft 2)* "27 hygiene specs; 9 implemented / 29 draft" | **25** hygiene specs; ledger is **10 implemented / 5 approved / 28 draft** = 43. |
| *(Draft 2)* "Approve `hygiene_mcts_value_semantics` and `security_secret_scan_hardening`" | **Both already shipped.** `negate_child_value` is live at `parallel_mcts.py:228/259/492`; `.gitleaks.toml` and the regression suite exist; carve-out CO-1 is CLOSED. Their frontmatter is stale — they need reconciling, not approving. |

Draft 1's only correct finding was that `1.414` occurs 22× across 13 files — but its diagnosis was
wrong: `src/config/constants.py:24 DEFAULT_MCTS_C` already exists with **zero uses in `src/`**. The
debt is an unadopted canonical constant. Already assigned to P8a.

## 4. Real defects found during the review, filed here

Not scheduled by this review; recorded so they are not lost.

1. **The `summary` job does not gate what it prints.** `ci.yml:561` `needs:` omits `chess-tests` and
   `integration-test` entirely. `security-scan` and `dependency-audit` *are* in `needs:` and *are*
   printed (`:570`, `:572`) but are **absent from the failure condition** (`:578-586`), and each
   additionally fails open via `|| true` (`:149`, `:209`) plus an `if [ -f ... ]` with no `else`.
   **A bandit HIGH finding does not fail CI today.** → folded into `hygiene_ci_mechanical` AC-1.
2. **No `timeout-minutes` anywhere.** All 23 jobs across all three workflows inherit the 360-minute
   default, on a `docker-build` that routinely runs 45m and has already hung. → AC-8.
3. **Only `ci.yml` declares a `concurrency:` group**, and `docker-deployment.yml`'s `paths:` filter
   covers its `push:` trigger only, so docs-only PRs run the full `Dockerfile.train` matrix. → AC-9.
4. **Coverage `exclude_lines` contains a bare `pass`** (`pyproject.toml:381`), applied by coverage as
   `re.search` over raw lines: **359 lines in `src/` match, 291 of which are not `pass` statements**
   (docstrings reading "forward pass", dataclass fields `num_passed`/`pass_at_1`). This is NG-5
   inverted — the gate silently moved to meet the code. → AC-11.
5. **Trivy cannot fail anything** while costing 176s per run: it carries both
   `continue-on-error: true` and `exit-code: '0'` (`ci.yml:511-521`). → AC-12.
6. **`ParallelMCTSEngine`'s two constructor paths are not equivalent.** The legacy branch sets
   `two_player=get_settings().MCTS_TWO_PLAYER` (`parallel_mcts.py:325`) but `ParallelMCTSConfig`
   hardcodes `True` (`:115`). Removing the legacy branch as a "no behavior change" cleanup would
   silently make `ParallelMCTSEngine()` ignore `MCTS_TWO_PLAYER`. Filed, unfixed — **NG-6**.
7. **`scripts/verification/verify_setup.py:126` imports a symbol that does not exist** (`S3Client`;
   the module exports `S3Config`/`S3StorageClient`), so the probe can never succeed and fails
   silently inside a `try`. Already in scope for `hygiene_small_fixes`.
8. **A spec-module collision the gate cannot see.** `hygiene_small_fixes` declares
   `module: src/framework/harness/` but its AC-5 writes `src/utils/deprecation.py`, colliding with
   `hygiene_determinism`'s `src/utils/` claim. `modules_overlap()` reads only the declared `module:`
   field, so nothing catches it.

## 5. Decisions recorded (do not re-propose)

Two successive drafts independently proposed the `config/` merge that `CHARTER.md` §3 already
records as *considered and explicitly not adopted*. This section exists to make the refusals
greppable.

| Proposal | Standing decision |
|---|---|
| Consolidating YAML configs into `config/` | **Refused.** Of 39 tracked YAMLs, 19 are external-tool path contracts and 17 are charter-blocked, leaving 3. `CHARTER.md` §3 records the `config/`→`src/config/` merge as not adopted; **NG-10** forbids it as incidental tidying. |
| Moving `src/observability/` under `src/framework/` | **Refused.** 173 files / 539 occurrences, reduces cohesion, breaks a path pinned by `src/tools/context_docs.py`. **NG-10**. |
| Merging the 16 `BaseSettings` classes into one `AppConfig` | **Refused.** 15 disjoint env prefixes; couples `benchmark`/`enterprise`/`harness`; breaks INV-3. |
| A coverage target of 91.5% or 92% | **Refused.** Arithmetically unreachable from the named work. **NG-5** (budget 0/0) protects the gate; `hygiene_ci_mechanical` will legitimately move the measured number *down*. |
| A `load_profile()` YAML loader in `src/config/settings.py` | **Refused.** Makes the module INV-1 designates for tunables resolve filesystem paths at call time. `phase_6_config_centralization` is already `implemented`. |
| Writing tests for `llm_guided/benchmark/runner.py`, `llm_guided/rag/context.py`, `src/api/health.py` | **Refused.** 105 of the 172 targeted lines sit in subtrees that `hygiene_delete_llm_guided` and `hygiene_delete_storage_api` remove; the tests would be deleted with them and make the deletion PR read as a regression. |
| Ten new advisory CI jobs (vulture, radon, pydeps, hadolint, cyclonedx, Snyk, pdoc, towncrier, pytest-benchmark, coverage-badge) | **Refused.** They would be permanently informational and therefore ignored — the same defect AC-12 removes from Trivy. All are unpinned and absent from `[dev]`, reintroducing the drift the `ruff`/`mypy` pins exist to prevent. |
| Restructuring CI job dependencies for parallelism | **Refused.** Six jobs already start together; the proposal adds serialization and omits four jobs from its own diagram. |
| `pytest -n auto` | **Deferred, not refused.** ~0% wall-clock gain today. Real ordering hazard: `tests/unit/test_observability_metrics_ext2.py:27-46` has an autouse fixture that unregisters every collector from the global Prometheus registry. Revisit with measurement. |
| `mypy --incremental` cache in CI | **Refused.** mypy is 6–9s of a 148s job; the cost is the torch install. Stale-cache risk against INV-11 for no gain. |
| A new `src/exceptions.py` | **Refused as proposed.** Two hierarchies already exist with two name collisions between them; a third adds more. Whether the existing two merge is an **NG-6** review question. |
| A new `tests/shared/fixtures.py` | **Refused.** Under-adoption of `tests/conftest.py` and `tests/fixtures/`, not absence. |
| `parallel_mcts` deprecation ceremony with a v0.3.0 removal | **Refused.** No release process, no external consumers, no `filterwarnings` policy so the warnings fail nothing, and the project's own factory would trigger them. Removal is also the latent behavior change in §4.6. |
| Re-stating the Wave C–F sequence in a new document | **Refused.** A second copy of `docs/plans/2026-07-30-code-hygiene-modularity.md` would drift — NG-7 in miniature. |

## 6. Disposition

The review produced no new plan document, deliberately. Its outputs were:

1. An amendment to `specs/hygiene_ci_mechanical.SPEC.md` — **AC-8 through AC-12** plus a widened
   AC-1, covering §4 items 1–5. Amending a `draft` spec is free; the `draft` → `approved` flip is
   reserved to the maintainer (`CHARTER.md` §7.5).
2. This document.

Everything else routes into the existing M5 program. The sequencing authority remains
`docs/plans/2026-07-30-code-hygiene-modularity.md`; the status ledger remains `specs/`; the measured
baseline remains `docs/STATUS.md`, which should be regenerated via the `coverage-baseline` skill
before any coverage figure is quoted again — the current one predates the code it measures.

**Maintainer actions this review is blocked on:** merge PR #127 (§1), then review and approve
`hygiene_ci_mechanical` as amended.
