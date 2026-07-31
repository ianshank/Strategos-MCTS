# Charter Alignment Audit — CHARTER.md versus the tree (2026-07-31)

- **Reviewed document**: `CHARTER.md`, introduced by this change. Governing spec:
  `specs/charter_alignment.SPEC.md`.
- **Tree audited**: `claude/project-charter-alignment-0xdtsd` @ `1533120` (2026-07-31), `src/`
  holding 327 Python files. Status ledger: `docs/STATUS.md` dated 2026-07-25.
- **Method**: claim-by-claim verification against the live tree. Every finding carries a
  path-and-line reference and a verdict of CONFIRMED, PARTIAL, or FALSE. Six passes: invariant
  probes, non-goal probes, charter self-verification (each mission demo clause executed once),
  mechanical documentation sweep, authority-stack conflict scan, and exception-ledger
  reconciliation.
- **Auditor verdict**: **the documentation was materially ahead of the code.** Nine documentation
  claims were false or stale and are corrected in this change. Six code-side divergences are real
  and are filed here unfixed, per the spec's constraint that no behavior changes. The single most
  consequential finding is not any individual drift but a structural one: **three of the eight
  invariants the project advertises as non-negotiable have no enforcement mechanism at all**, and a
  fourth is contradicted by a default value. A rule nobody checks is a preference, and the project
  had no way to tell the difference until now.

---

## 1. Verdict at a glance

| # | Finding | Class | Verdict | Disposition |
|---|---|---|---|---|
| F-1 | README coverage badge claimed 93% against a measured 90.15% | doc | CONFIRMED | Fixed here |
| F-2 | `docs/STATUS.md` contradicted its own headline (93.35%) | doc | CONFIRMED | Fixed here |
| F-3 | Three documents stated a lapsed `No-Spec:` precondition | doc | CONFIRMED | Fixed here |
| F-4 | Five live documents cited `src/framework/graph.py`, which does not exist | doc | CONFIRMED | Fixed here |
| F-5 | Primer claimed three console scripts; five are declared | doc | CONFIRMED | Fixed here |
| F-6 | `docs/plans/MVP_ROADMAP.md` reads as current but is stale | doc | CONFIRMED | Banner added |
| F-7 | `docs/SLA.md` is unratified boilerplate presented as an SLA | doc | CONFIRMED | Banner added |
| F-8 | `planning/` encodes a coverage floor and Python floor that never applied | doc | CONFIRMED | Banner added; values left uncorrected by design |
| F-9 | `planning/epics/epic_5_1_neural_mcts.yaml` artifact paths are fictional | doc | CONFIRMED | Banner added |
| **F-10** | **INV-6 fail-loud is contradicted by a default value** | **code** | **CONFIRMED** | **Filed — needs a decision** |
| **F-11** | **INV-2, INV-3, INV-8 have no structural enforcement** | **code** | **CONFIRMED** | **Filed** |
| F-12 | INV-1 config discipline is bypassed in 97 places | code | CONFIRMED | Filed |
| F-13 | The coverage gate's scope is narrower than the headline implies | code | CONFIRMED (by design) | Filed as a documentation-of-scope issue, now stated in INV-5 |
| F-14 | The e2e workflow cannot fail | code | CONFIRMED | Filed |
| F-15 | Two `src/` packages are unreachable from any production entry point | code | CONFIRMED | Filed — already scoped by hygiene specs |
| F-16 | 58 commits carried the `No-Spec:` exception; it was the default channel | governance | CONFIRMED | Recorded once in `CHARTER.md` §8 |
| **F-17** | **A live Weights & Biases API key is committed in `docs/`, outside every scan's scope** | **security** | **CONFIRMED** | **Redacted here — the key still requires rotation** |
| F-18 | An *active* plan doc and `ATTRIBUTION.md` carried further drift the first sweep missed | doc | CONFIRMED | Fixed here |

---

## 2. Documentation-side findings (corrected in this change)

**F-1 — Coverage badge overstated by ~3 points.** `README.md:8` rendered
`coverage-93%25`; `docs/STATUS.md:20` records **90.15%** against a gate of 85.0. The 93% figure
traces to a superseded 2026-07-23 baseline. *Fixed:* badge now reads 90.15%.

**F-2 — `docs/STATUS.md` contradicted itself.** Its headline states 90.15%, but the "Implications
for the plan" section still argued from **93.35%** — the exact figure the file's own banner says it
supersedes. A source-of-truth document disagreeing with itself is worse than one that is merely
stale, because both numbers carry its authority. *Fixed.*

**F-3 — A lapsed precondition still instructed contributors.** `.github/CONTRIBUTING.md:61-62`,
`.claude/skills/strategos-primer/SKILL.md:138`, and `CLAUDE.md:338-340` each stated that "until the
first approved spec merges, the `No-Spec:` trailer is the expected channel for `src/**` work". That
precondition has lapsed: `specs/` currently holds 5 approved and 10 implemented specs.
`docs/plans/2026-07-30-code-hygiene-modularity.md:22-27` had already identified this and declared the
blanket invalid, but the three documents were never updated — so the repository's own governance
change had not propagated to the documents contributors actually read. *Fixed:* all three now state
that the `spec/<id>` branch is the default channel and the trailer is the written exception.

**F-4 — A path that does not exist, cited in five live documents.** `CLAUDE.md`'s "Key File
Locations" listed `src/framework/graph.py`; orchestration is the package `src/framework/graph/`. The
primer narrated this disagreement in *two* separate places (its opening orientation and its
"Gotchas" section) rather than either document being fixed, so correcting one alone would have
orphaned the others into fresh drift. `docs/KEY_CODE_SNIPPETS.md:200,421` cited the same dead path in
a code-provenance comment and a component table.

*All five fixed together.* Two further classes of citation were found and **deliberately not
touched**: the superseded plans under `docs/plans/` (`MVP_ROADMAP.md`,
`IMPLEMENTATION_PLAN_COMPREHENSIVE.md`, `IMPLEMENTATION_PLAN_PRIORITY_TASKS.md`,
`NEXT_STEPS_PLAN.md`), which carry historical banners and are accurate as records of their own
moment; and `docs/templates/MULTI_AGENT_MCTS_TEMPLATE.md:182,778,2105`, where the path is
*illustrative of a generic architecture a new project would build*, not a claim about this tree.
`docs/training/MODULE_6_ASSESSMENT.md:330` is a dated coverage assessment and is likewise left as a
historical record.

The `INTENTIONALLY_ABSENT` allowlist entry in `src/tools/context_docs.py:74` that covered the
primer's narration is now vestigial. It is deliberately left in place, because `_check_absent_paths`
only fires if the path *reappears* on disk — which remains the behavior we want, and removing the
entry would buy nothing.

This finding is the clearest argument for F-11's remedy: only the two `.claude/` citations were
caught mechanically, because only `.claude/` documents are in the validator's scope. `CLAUDE.md`,
`docs/KEY_CODE_SNIPPETS.md`, and the template are not — and the drift survived there until a manual
sweep. Widening `GOVERNANCE_DOCS` to cover more of `docs/` is the obvious next increment.

**F-5 — Console-script count drifted silently.** The primer claimed "Three console scripts";
`pyproject.toml:140-145` declares five. The reason it went undetected is itself the finding, and it
is subtler than "the pinned tuple was short": `_check_console_scripts` asserts that named scripts
still *exist* in pyproject. It reads in the wrong direction. No amount of widening that tuple can
notice that a document enumerates a stale subset — the drift is in the prose, and pyproject is fine.

*Fixed in the document, and the checker now closes the loop properly.* The tuple is widened to all
five (so a removed script still fails), **and** a new `_check_console_scripts_documented` runs from
pyproject to the prose: every declared script must be named in the primer. That is the check that
would actually have caught this, and a negative test proves it fires.

**F-6, F-7, F-8, F-9 — Stale documents reading as current.** `docs/plans/MVP_ROADMAP.md:10` claims
"217 Python source files" (actual: 327) and `:56` "No CI/CD pipeline" (a twelve-job pipeline exists),
yet unlike its sibling plans it carried no supersession banner. `docs/plans/PHASE_4_TEMPLATE_PLAN.md`
similarly reads "Status: Implementation Ready" for work that has landed. `docs/SLA.md` presents
uptime tiers, error budgets, and service credits with placeholder contacts, a placeholder
jurisdiction, and an unsigned approval block. `planning/milestones.yaml` declares
`min_test_coverage: 80` and `python>=3.11` against a real gate of 85 and a real floor of ≥3.10, and
`planning/epics/epic_5_1_neural_mcts.yaml` lists eight artifact paths, none of which exist.

*Fixed by banner, not by correction.* The `planning/` values are **deliberately left wrong**.
Correcting them would imply that abandoned parallel planning system is maintained, which is exactly
the state `CHARTER.md` §3 NG-7 exists to prevent. The banner records where the work actually landed.

---

## 3. Code-side findings (filed, not fixed)

None of these are touched by this change. Each names the vehicle that should carry the fix.

**F-10 — The fail-loud invariant is contradicted by a default. (HIGH)**
`src/config/settings.py:420-427` sets `ALLOW_MOCK_LLM_FALLBACK` to `False`, matching the documented
intent. But `:429-436` sets `ALLOW_LIGHTWEIGHT_FRAMEWORK_FALLBACK` to **`True`**, with the field's own
description acknowledging this ("Default True preserves the documented zero-dependency path; set
False to fail loud"). The primer, `.claude/agents/strategos-guide.md`, and `README.md:52-55` all
described *both* fallbacks as opt-in. So the project's stated fail-loud posture held for one flag and
not the other, and the documentation asserted the stronger claim.

The documentation is corrected here to state the actual behavior. **The underlying decision is
yours**: either change the default to `False` (a behavior change, needing its own spec, and likely
breaking the "documented zero-dependency path" the description protects), or narrow the invariant so
it only claims what is true. This audit does not choose. Recorded against `CHARTER.md` §4 INV-6 and
§3 NG-2.

**F-11 — Three advertised invariants have no enforcement. (HIGH)**
`.claude/skills/strategos-primer/SKILL.md:85-108` introduces its eight invariants as "the rules a
change is judged against — violating one is how a PR fails CI or review". For three of them, no CI
job, hook, or test fails:

- *Async I/O* — nothing checks that a new I/O path is async. Four synchronous sleep sites exist in
  `src/` (`src/observability/decorators.py:265,357`,
  `src/framework/harness/memory/compactor.py:63,68`); all four are defensible in context, which is
  the point — there is no mechanism that would distinguish a defensible one from a regression.
- *Dependency injection* — enforced only by review and, indirectly, by the coverage gate.
- *Structured, secret-safe logging* — the only gate is the `sk-` literal grep in the `spec-validate`
  job, which catches secrets committed to source, not secrets reaching a log sink. 73 bare `print(`
  calls exist under `src/` (e.g. `src/monitoring/otel_tracing.py:459-460`,
  `src/monitoring/prometheus_metrics.py:513-514`), predominantly in module demo blocks.

These are now labelled **ASPIRATIONAL** in `CHARTER.md` §4 rather than asserted as enforced. That
labelling is the fix for the honesty problem; building the gates is separate work and belongs behind
its own spec.

**F-12 — Config discipline is bypassed in 97 places. (MEDIUM)**
INV-1 requires tunables to flow through Pydantic Settings. `grep` finds **97** `os.environ.get` /
`os.getenv` call sites under `src/` outside `src/config/` — for example
`src/neuro_symbolic/config.py:70-99`, which reads six tunables directly from the environment with
inline string defaults. Separately, five distinct `BaseSettings` classes exist
(`src/config/settings.py`, `src/config/graph_settings.py`, `src/framework/harness/settings.py`,
`src/benchmark/config/benchmark_settings.py`, `src/enterprise/config/enterprise_settings.py`), and
three mutually incompatible `MCTSConfig` classes (`src/framework/mcts/config.py:32`,
`src/training/system_config.py:69`, `src/models/validation.py:110`).

*Already scoped* by the draft spec `specs/hygiene_config_consolidation.SPEC.md` and, for the config
classes, `specs/hygiene_mcts_config.SPEC.md`. No competing finding is opened; this entry exists so
the charter's INV-1 verdict of PARTIAL is traceable to evidence.

**F-13 — The coverage gate's scope is narrower than the headline. (MEDIUM, by design)**
`pyproject.toml:374` sets `fail_under = 85.0` and `.github/workflows/ci.yml:244` enforces it — but
the CI `test` job runs `tests/unit/` only, with three test modules ignored, while
`[tool.coverage.run] omit` excludes `src/api/rest_server.py` and `src/api/inference_server.py` (a
5,198-line package) plus three chess modules. `docs/STATUS.md`'s 90.15% headline is the **full**
suite. Both numbers are honestly reported in their own contexts; the risk is a reader concluding
that 85% is enforced over all of `src/`. `CHARTER.md` §4 INV-5 now states the scope explicitly. No
code change is proposed — narrowing coverage scope is a legitimate engineering choice, and NG-5
already forbids *widening* the omit list to make the gate pass.

**F-14 — The e2e workflow cannot fail on test results. (MEDIUM)**
`.github/workflows/e2e_with_langsmith.yml:40` and `:142` terminate their pytest invocations with
`|| true`, so the job reports success regardless of outcome. A workflow that always passes provides
no signal. Filed against NG-3 (claims must be reproducible): a green check that cannot go red is a
claim of health that no command substantiates.

**F-15 — Two `src/` packages are unreachable from production. (LOW — already scoped)**
`src/performance/` and `src/integrations/` have **zero** importers anywhere under `src/`; they are
referenced only by tests, `scripts/run_e2e_workflow.py`, `examples/`, and README code samples.
`src/data/`, `src/neuro_symbolic/`, and `src/enterprise/` have one or two `src/` importers each.
Separately, `src/games/connect_four/` (242 lines) and `src/games/othello/` (313 lines) are thin
registrations relative to `src/games/chess/` (12,109 lines), while `README.md:94` lists all three
together as "Fast Gameplay Domains" without signalling the difference.

*Already scoped* by the draft delete-cluster specs (`specs/hygiene_delete_framework_cluster.SPEC.md`,
`specs/hygiene_delete_storage_api.SPEC.md`, `specs/hygiene_delete_enterprise_cluster.SPEC.md` and
siblings). Closing them is Gate G-M5 in `CHARTER.md` §5.

**F-16 — The exception was the rule. (HIGH, governance)**
`git log --grep="No-Spec" --oneline` returns **58 commits**. NG-4 describes the trailer as a written
exception to spec-gated development; in practice it has been the default channel for `src/**` work.
This is recorded once, in aggregate, in `CHARTER.md` §8 rather than as 58 retroactive ledger rows,
and is explicitly *not* retroactively ratified. It is the reason NG-4 carries a carve-out budget at
all.

---

**F-17 — A live credential is committed in `docs/`, where nothing was looking. (HIGH, security)**

`docs/API_CONFIGURATION_GUIDE.md:87` contained a real-format Weights & Biases API key — a 40-character
hex string presented as a `.env` example — committed to history and present on the remote.

The mechanism failure is the point, and it is exactly the shape of F-11. The repository *has* a secret
scan, and the charter's INV-1 cites it as enforcement. But `.github/workflows/ci.yml:100` runs
`git grep -nE "sk-[A-Za-z0-9]{20,}" -- src/ kubernetes/`: it is scoped to two directories that exclude
`docs/`, and its pattern matches only OpenAI-shaped `sk-` keys, so a hex-format W&B key in a
documentation file is invisible to it on **both** axes. The guard existed, was believed to cover this,
and structurally could not.

*Disposition:* the value is **redacted here** and replaced with a placeholder. **Redaction is not
remediation** — the key remains in git history and on the remote, so it must be treated as
compromised and **rotated at wandb.ai/authorize**. That action is the maintainer's; this audit cannot
perform it.

*Filed, not fixed:* widening the scan. Making it repo-wide and format-agnostic requires an allowlist
for the legitimate placeholder and test-fixture keys (`docs/SECRETS_MANAGEMENT.md:102`,
`docs/API_QUICK_REFERENCE.md:8,16` — all truncated or obvious placeholders — plus roughly a dozen
`sk-ant-test-…` fixtures under `tests/`). That is a real piece of work with a false-positive budget,
and it belongs behind its own spec rather than being bolted onto a governance PR. It is the single
highest-value item this audit surfaces for the next spec cycle.

**F-18 — Two further drift sites the first sweep missed. (LOW)**
`docs/plans/2026-07-24-execute-m5.md:7,186` cited the superseded 93.35% coverage figure. Unlike the
other plans carrying it, this one is **Active** ("P0 awaits the operator GPU run"), so it is a live
claim rather than a historical record. *Fixed.* Separately, `ATTRIBUTION.md:15` expanded TRM as
"Tactical Reasoning Module" where every other document in the repository says "Task Refinement
Module". *Fixed.* Occurrences of 93.35% inside `CHANGELOG.md` are deliberately left alone — they are
dated release records and are accurate as such.

---

## 4. Charter self-verification (Pass C)

Every mission demo clause in `CHARTER.md` §2 was executed once against this tree. This pass is the
charter earning NG-3 on itself: a demo clause that does not run is a defect in the charter, not an
inconvenience.

| Demo clause | Result |
|---|---|
| `pytest tests/unit/framework/mcts/test_value_semantics_regression.py -q` | **21 passed** |
| `pytest tests/unit/framework/mcts/test_domain_adapters.py -q` | **24 passed** |
| `python -m src.benchmark --dry-run` | **exit 0** (20 planned executions) |
| `harness validate-spec specs/charter_alignment.SPEC.md` | **exit 0** |
| `python -m src.tools.context_docs` | **exit 0** after this change (14 documents checked, up from 13) |
| `GET /graph/structure`, `/graph/mermaid` | **NOT EXECUTED** — routes confirmed present at `src/api/rest_server.py:611,632` and their `ENABLE_GRAPH_VISUALIZATION` flag confirmed default-on at `src/config/settings.py:474`; a live service was not started in this environment |
| `GET /health`, `/ready`, `/metrics` | **NOT EXECUTED** — routes confirmed present at `src/api/rest_server.py:393,416,455` |
| `self-play-convergence`, `policy-lift` console scripts | **NOT EXECUTED** — entry points confirmed declared at `pyproject.toml:144,143` and their targets confirmed to exist; both require the optional neural extra and meaningful runtime |

One demo clause was **rewritten during this audit rather than reported as passing**:
`tests/unit/framework/test_domain_registry.py` requires the optional neural extra and fails
collection on a default install, so the charter now cites
`tests/unit/framework/mcts/test_domain_adapters.py`, which runs everywhere, and names the registry
test's dependency explicitly.

**Deliberately excluded from §2:** the `/compare` endpoint. `ENABLE_DEMO_COMPARISON` defaults to
`False` (`src/config/settings.py:482`), so it is not observable on a default install and does not
meet the falsifiability bar a demo clause sets — even though `README.md:89` lists it among the
serving features.

---

## 5. Verification performed, and its limits

Stated plainly, because NG-3 applies to this document too.

This session began with neither the package nor its dependencies installed; `pip install -e ".[dev]"`
and the optional neural extra were installed in order to run the checks below.

**Ran, with results:**

| Check | Result |
|---|---|
| `black . --check --line-length 120` | clean (889 files; one file this change touched was reformatted first) |
| `ruff check .` | **All checks passed** |
| `mypy src/` | **Success: no issues found in 327 source files** |
| CI-equivalent unit gate — `pytest tests/unit/` with `--cov=src --cov-fail-under=85` and the three CI ignores, under the CI environment variables | **8,472 passed, 62 skipped; TOTAL coverage 89.88%** (gate 85%), 207s |
| `git grep -nE "sk-[A-Za-z0-9]{20,}" -- src/ kubernetes/` | no matches |
| `harness validate-spec specs/*.SPEC.md` | exit 0 over all 42 specs |
| `python -m src.tools.context_docs` | exit 0 — **14 documents checked**, up from 13 |
| `pytest tests/unit/tools/test_context_docs.py` | **53 passed**, up from 48 (five new tests cover the governance-doc lane) |

The 89.88% figure is this branch's measurement of the **gated** scope (`tests/unit/` with the CI
ignores and coverage omits). It is not comparable to, and does not supersede, the 90.15% full-suite
headline in `docs/STATUS.md`, which is measured differently — see F-13. `docs/STATUS.md` is not
amended by this change; regenerating it is the `coverage-baseline` skill's job.

**Not run:** the full suite (`pytest tests/ -m "not slow"`), which includes integration, e2e, chess,
and property tests beyond the CI gate's scope; and the live-service demo clauses noted in §4.

This change touches no runtime code path. The one file modified under `src/` is
`src/tools/context_docs.py`, a build-time documentation validator with no production importer.

This change touches no runtime code path. The one file modified under `src/` is
`src/tools/context_docs.py`, a build-time documentation validator with no production importer.

---

## 6. What this audit did not do

- It did not fix any code-side finding. F-10 through F-15 remain open by design.
- It did not open findings that duplicate the code-hygiene program's 25 `hygiene_*` draft specs;
  where a divergence is already scoped there, this audit links to that spec instead. (`specs/` now
  holds 42 specs: 27 draft — including this change's own — 5 approved, 10 implemented, 0 verified.)
- It did not correct the stale values inside `planning/`, for the reason given under F-8.
- It did not retroactively ratify the 58 pre-charter exceptions in F-16.
