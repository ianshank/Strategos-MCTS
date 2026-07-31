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
| F-16 | 57 commits carried the `No-Spec:` exception; it was the default channel | governance | CONFIRMED | Recorded once in `CHARTER.md` §8 |
| **F-17** | **A live Weights & Biases API key is committed in `docs/`, outside every scan's scope** | **security** | **CONFIRMED** | **Redacted here (rotation still required); scan gap closed via `specs/security_secret_scan_hardening.SPEC.md`** |
| F-18 | An *active* plan doc and `ATTRIBUTION.md` carried further drift the first sweep missed | doc | CONFIRMED | Fixed here |
| F-19 | A fourth stale gate-status snapshot, and an axis rule that singled out one of four peer architecture docs | doc | CONFIRMED | Fixed here |
| **F-20** | **The F-17 gitleaks fix had a whole-file allowlist entry covering the exact file where a second copy of the leaked key survived** | **security** | **CONFIRMED** | **Fixed here — see §5a** |
| F-21 | Four governance-accuracy defects in `CHARTER.md` itself: two overstated invariant verdicts (INV-4, INV-7), a carve-out marked closed while its PR is still open, an unanchored/stale commit count, and an NG-1 violation in `docs/runbooks/` the SLA banner never reached | doc | CONFIRMED | Fixed here |

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
`git log --grep="^No-Spec:" -E --oneline origin/main` returns **57 commits** (an unanchored substring
match overcounts — first drafted with that looser count, corrected here). NG-4 describes the trailer
as a written exception to spec-gated development; in practice it has been the default channel for
`src/**` work. This is recorded once, in aggregate, in `CHARTER.md` §8 rather than as 57 retroactive
ledger rows, and is explicitly *not* retroactively ratified. It is the reason NG-4 carries a carve-out
budget at all.

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

*Widened, in the same branch, under its own spec:* `specs/security_secret_scan_hardening.SPEC.md`
adds `.gitleaks.toml` (extending gitleaks' built-in ruleset, allowlisting only the specific
test-fixture and documentation-placeholder values already verified as non-secrets — not a broad
path or pattern exclusion) and a `secret-scan-gitleaks` CI job, repo-wide and pattern-agnostic,
wired into the `summary` job's failure check so it cannot fail silently. The `git grep` step is
left untouched — the two layers are complementary, not a replacement of one by the other. This
environment has no `gitleaks` binary, so the configuration's *syntax* was validated locally
(TOML and workflow YAML both parse); its actual behavior against the live repository is verified
by the first CI run, not asserted here.

**F-18 — Two further drift sites the first sweep missed. (LOW)**
`docs/plans/2026-07-24-execute-m5.md:7,186` cited the superseded 93.35% coverage figure. Unlike the
other plans carrying it, this one is **Active** ("P0 awaits the operator GPU run"), so it is a live
claim rather than a historical record. *Fixed.* Separately, `ATTRIBUTION.md:15` expanded TRM as
"Tactical Reasoning Module" where every other document in the repository says "Task Refinement
Module". *Fixed.* Occurrences of 93.35% inside `CHANGELOG.md` are deliberately left alone — they are
dated release records and are accurate as such.

**F-19 — A third architecture doc carried its own hardcoded, stale gate snapshot; the charter's
axis rule cited only one of four peer architecture docs. (LOW)**
`docs/C4_ARCHITECTURE.md`'s closing section hardcoded a "Current gate status (2026-07-20)" table —
305 source files, 93.82% coverage, 10,101 tests passed — the same shape of drift as F-1/F-2, just in
a fourth location the first sweep didn't reach. *Fixed*, by the same principle applied everywhere
else in this change: the table is replaced with a pointer to `docs/STATUS.md` rather than a
corrected snapshot, so it cannot go stale the same way again.

Separately, `CHARTER.md`'s original axis rule named `docs/C4_ARCHITECTURE.md` alone as governing
"where code lives" — but `docs/README.md`'s own index lists **four** architecture documents
(`architecture.md`, `C4_ARCHITECTURE.md`, `C4_MERMAID_ARCHITECTURE.md`,
`langgraph_mcts_architecture.md`) as peers with no stated precedence, and `PROJECT_STRUCTURE.md`'s
Quick Navigation points to `architecture.md`, not `C4_ARCHITECTURE.md`. Singling one out was itself
inaccurate. *Fixed* by pointing the axis rule at `docs/README.md`'s index instead of picking a
single file — the same move the whole charter makes everywhere else, delegate rather than duplicate.
**Fully reconciling the four architecture docs against the current module layout — cross-checking
every diagram, establishing an actual precedence, possibly merging some — is a substantially larger
effort or its own spec, not attempted here.**

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
| `harness validate-spec specs/*.SPEC.md` | exit 0 over all 43 specs |
| `python -m src.tools.context_docs` | exit 0 — **14 documents checked**, up from 13 |
| `pytest tests/unit/tools/test_context_docs.py` | **54 passed**, up from 48 (six new tests cover the governance-doc lane and the console-scripts-documented check) |
| `pytest tests/ -m "not slow"` (full suite, `.[dev,neural]`) | **9,552 passed, 209 skipped, 32 deselected, 31 failed**, 351s |
| `.gitleaks.toml` / `.github/workflows/ci.yml` syntax | valid TOML (`tomllib`) and valid YAML (`PyYAML`); the scan's actual behavior is unverified locally — no `gitleaks` binary in this environment |

The 89.88% figure is this branch's measurement of the **gated** scope (`tests/unit/` with the CI
ignores and coverage omits). It is not comparable to, and does not supersede, the 90.15% full-suite
headline in `docs/STATUS.md`, which is measured differently — see F-13. `docs/STATUS.md` is not
amended by this change; regenerating it is the `coverage-baseline` skill's job.

**The 31 full-suite failures were checked, not assumed, to be pre-existing.** `git diff
origin/main...HEAD` touches none of the failing test files, and the only `src/**` file this branch
changes (`src/tools/context_docs.py`) has zero production importers — so nothing in this branch can
plausibly cause them. To confirm rather than infer that, a disposable worktree of `origin/main` was
checked out and a sample of the failures re-run in isolation: `test_local_embedding_store.py`,
`test_deployed_models.py::test_meta_controller_loading`, and
`test_ui_e2e.py::test_query_submission_sync` failed **identically on main**, for the same
environment causes — `torch.load(..., weights_only=True)` rejecting an LFS-pointer checkpoint under
PyTorch 2.6's new default, no outbound network to reach the HuggingFace Hub, and `gradio` not
installed. 23 of the 31 failures are `tests/e2e/test_ui_e2e.py` (needs a live Gradio/Selenium
environment — `docs/STATUS.md` already documents this class), 4 are
`tests/api/test_local_embedding_store.py` (the same torch/HF-offline cause), and 1 is
`test_deployed_models.py::test_meta_controller_loading` — a file CI's `integration-test` job already
excludes via `--ignore` for exactly this reason. The remaining 2 property-test and 1
`test_config_loading_performance` failures did **not** reproduce in the isolated re-run (they passed
on main in isolation), consistent with Hypothesis health-check timing sensitivity under the full
suite's resource contention rather than a deterministic failure — flaky, not caused by this branch.
Live-service demo clauses noted in §4 remain unexecuted; running a server was out of scope here.

This change touches no runtime code path. The one file modified under `src/` is
`src/tools/context_docs.py`, a build-time documentation validator with no production importer.

---

## 5a. F-20 — the gitleaks fix (F-17) had its own near-miss, found by adversarial review after merge readiness was claimed

Recorded against the same discipline the rest of this document applies to everyone else: after F-17
was fixed and this PR reported CI-green, a separately-running adversarial peer review of the branch
found that the fix was incomplete, and its own mechanism made the gap *worse*, not better.

**What was missed.** The leaked Weights & Biases key redacted from `docs/API_CONFIGURATION_GUIDE.md`
had a second occurrence: `docs/API_QUICK_REFERENCE.md:23` carried the same key's first 16 hex
characters, truncated with an ellipsis in the style of that file's other two example lines. The
initial F-17 fix checked the one file the original grep hit named, and never swept the repo for
other occurrences of the *value* before writing the allowlist.

**Why the fix made it worse.** `.gitleaks.toml`'s first version allowlisted
`docs/API_QUICK_REFERENCE.md` by **whole file path** — reasoned from its other two lines (both
genuinely truncated placeholder-shaped examples) without checking the third. That path entry would
have made the new, repo-wide scanner **structurally blind** to the exact file where the real
exposure survived — the identical failure shape as F-17 itself (a guard that exists and cannot fire),
introduced fresh in the guard meant to prevent it. The config's own header comment claimed the
allowlist was "narrow scope... named literally, not by broad path or pattern exclusion" while the
`paths` array was exactly that.

**Disposition, fixed:**
- `docs/API_QUICK_REFERENCE.md` — all three example lines (OpenAI, Anthropic, and the W&B fragment)
  replaced with generic placeholders. On inspection the section was titled "Your Configured
  Providers" with a specific project name and "✅ Working"/"✅ Configured" statuses — consistent with
  a real local setup snapshot having been committed, not synthetic documentation. All three should be
  treated as **rotation candidates**, not just the one already flagged; this audit cannot determine
  whether the OpenAI/Anthropic fragments were ever live, and says so rather than assuming they are
  safe because they are truncated.
- `.gitleaks.toml` rewritten: the `paths` array is reduced to one entry, `.secrets.baseline`, kept on
  a **structural** argument (it stores SHA1 fingerprints of prior findings, not secret values — not
  exploitable regardless of content) rather than a one-time eyeball check of a doc's current
  contents. Every other allowlist entry is now a literal-value `regexes` match, so it protects only
  the specific string it names and cannot blind the scanner to anything else in the same file.
- A repo-wide grep for the leaked value's fragment (`26a08535`, `sk-proj-uJN73wUtmD`,
  `sk-ant-api03-OsXSRo`) confirms zero remaining occurrences in the tracked tree.

**Why this belongs in the audit rather than a quiet follow-up commit.** This document's whole
argument is that a guard which exists on paper and cannot fire is worse than no guard, because it is
trusted. Fixing this silently would repeat exactly that pattern one level up — a security fix whose
own defect goes unrecorded. It is recorded here, at the same severity class as F-17, not folded into
F-17's text as if it had been caught the first time.

---

## 5b. F-21 — four accuracy defects in `CHARTER.md`, caught by continuing to check the charter against itself

The same adversarial review that found F-20 checked `CHARTER.md`'s own claims for the class of error
this whole document exists to catch elsewhere: an assertion that outran what its cited evidence
actually supports.

- **INV-4 (unit tests are hermetic) was labelled ENFORCED; it is PARTIAL.** The cited mechanism —
  CI forcing offline HF Hub / W&B / LangChain-tracing modes and a dummy API key — disables the
  *common accidental* network paths. It is not a socket block: no `pytest-socket` or equivalent
  exists anywhere in `pyproject.toml` or the test configuration (checked directly — no match). A
  test making a raw call to an arbitrary host would not be stopped. Downgraded to PARTIAL with that
  distinction stated, matching the charter's own definition of the tier.
- **INV-7 (`src/**` changes are spec-gated) was labelled ENFORCED; it is PARTIAL.** CI genuinely
  blocks a diff with *neither* an approved spec nor a trailer — but the trailer path, read directly
  from `spec_trace.py`, accepts any non-empty `No-Spec: <reason>` string with no check on the
  reason's substance. Given F-16 (57 pre-charter commits used exactly this path), "enforced" claimed
  more than the mechanism delivers.
- **CO-2 was recorded `CLOSED (merged)` while its own PR is still open.** §7.2 defines a carve-out's
  closure as landing "one merged change" — this PR had not merged when that row was written. Fixed
  to `OPEN — closes on merge`, with §0's summary, the NG-4 row's active-carve-out count, and the
  budget-state line all updated to match (0 → 1 active carve-out) rather than only the ledger row.
- **The "58 commits" pre-charter count was already imprecise, and would go stale on every commit
  this PR itself makes.** `git log --grep="No-Spec" --oneline` (unanchored substring) overcounts
  against the trailer's actual format (`^No-Spec:\s*(\S.*)$`); anchored, `origin/main` measures
  **57**. Corrected in both `CHARTER.md` §8 and this document's F-16, with the exact command pinned
  so a future reader can re-measure rather than trust a number that necessarily drifts.
- **NG-1's banner reached `docs/SLA.md` but not `docs/runbooks/incident-response.md`**, which makes
  the identical unstaffed-commitment claim — an "Operations Team" owner, PagerDuty on-call, a CTO
  escalation ladder, an on-call email — under the same "Version 1.0.0 / 2025-01-15" template stamp.
  The other three files under `docs/runbooks/` were checked and do **not** share this pattern — they
  are alert-response technical procedures (Prometheus queries, remediation steps) that stand on
  their own regardless of staffing, so only `incident-response.md` gets the banner, not the whole
  directory.

**Not chased**, and stated as such rather than silently dropped: this same review raised whether the
charter-alignment work should itself be spec-first rather than riding a trailer (already disclosed in
the spec's Constraints and this PR's description — a documented tradeoff, not a miss), whether one
32-file PR with a solo reviewer is the right shape (already disclosed in §7.5), and whether
`GOVERNANCE_DOCS` should widen beyond `CHARTER.md` (already named as the obvious next increment at
the end of F-4). None of these have a single unambiguous fix available to apply unilaterally in this
PR, unlike the five items above.

---

## 6. What this audit did not do

- It did not fix most code-side findings. F-10 through F-15 remain open by design. F-17 is the one
  exception: its scan-coverage gap was closed under its own spec
  (`specs/security_secret_scan_hardening.SPEC.md`) rather than left filed, because leaving a known,
  currently-exploitable detection gap open for a future cycle was a worse trade than the scope
  increase of fixing it now.
- It did not open findings that duplicate the code-hygiene program's 25 `hygiene_*` draft specs;
  where a divergence is already scoped there, this audit links to that spec instead. (`specs/` now
  holds 43 specs: 28 draft — `charter_alignment` and `security_secret_scan_hardening` among them —
  5 approved, 10 implemented, 0 verified.)
- It did not correct the stale values inside `planning/`, for the reason given under F-8.
- It did not retroactively ratify the 58 pre-charter exceptions in F-16.
- It did not scan git history for secrets, and it did not attempt to rotate the F-17 key —
  rotation is the maintainer's action, not this audit's.
