# Project Charter — Strategos-MCTS (stable long-term plan)

> **Status: STABLE / SLOW-CHANGING.** This document is the north-star that humans and AI agents
> (and sub-agents) read first to stay on track. It changes rarely and only by deliberate decision —
> not per task. If a task seems to require changing this charter, **stop and raise it explicitly**
> rather than editing it in passing.
>
> **Charter integrity is governed by §7 (Amendment Protocol).** Amendments are budgeted. When a
> budget or trigger fires, the amendment path closes and a full charter review is mandatory.
> Procedural tidiness is not a substitute for scope discipline.

> **Axis of authority.** This charter is authoritative for **why and never** — vision, mission
> boundaries, scope, non-goals, invariants, and the amendment protocol. It is authoritative for
> *nothing else*, and it deliberately restates no measured value that another artifact generates.
>
> | Axis | Governing document |
> |---|---|
> | Measured status (pass counts, coverage, lint) | `docs/STATUS.md` |
> | Sequenced work — what next, in what order | `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md` |
> | Where code lives / architecture | `docs/README.md`'s "Explanation" index (`architecture.md`, `C4_ARCHITECTURE.md`, `C4_MERMAID_ARCHITECTURE.md`, `langgraph_mcts_architecture.md`), `.claude/skills/strategos-primer/SKILL.md` |
> | Commands and day-to-day patterns | `CLAUDE.md`, `AGENTS.md` |
> | Per-change process | `.github/CONTRIBUTING.md` |
> | Layout duplications | `PROJECT_STRUCTURE.md` |
> | Per-change contracts | `specs/` (schema v2) |
>
> Where one of those conflicts with the charter **on the charter's own axis**, the charter governs
> and the other document is drift to be fixed. Where the charter asserts something on **another
> document's axis**, that is a charter bug — file it, do not obey it. The single deliberate
> exception is the coverage literal in §4 (INV-5), which the deterministic doc validator
> mechanically pins to `pyproject.toml`.

---

## 0. Executive Summary

Strategos-MCTS is a LangGraph multi-agent Neural MCTS / AlphaZero self-play framework, distributed
on PyPI as `langgraph-multi-agent-mcts`. It serves a solo maintainer and the AI agents that work
this repository; secondarily, it serves people who install the package, run the container images, or
read the code as a reference implementation. It runs on Python (supported ≥3.10, CI-verified on 3.11
only), LangGraph, Pydantic Settings, FastAPI, and — behind the optional `neural` extra — PyTorch.

Its individual components are engineered and tested to a production standard; **full end-to-end
integration is still in progress**, and the project says so rather than implying otherwise. It
deliberately does **not** operate as a hosted service with uptime commitments, does not ship silent
fallbacks, does not accept capability claims that no command can reproduce, and does not maintain a
second planning system beside `specs/`.

The charter has **0 active carve-outs against 10 non-goals** (2 closed, recorded in §8). No full
charter review has yet occurred. Execution detail lives in
`docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md`; this document defines intent, boundaries, and
invariants only.

---

## 1. Vision

Strategos-MCTS exists so that DeepMind-style neural Monte Carlo Tree Search techniques can be
implemented, measured, and compared inside one honest codebase — where every claimed capability is
reproducible from the tree by a named command, and where the difference between "built" and "proven"
is never blurred.

The one thing this project must never sacrifice to get there is **evidence over assertion**. A
faster path that requires claiming an unmeasured result is not a faster path; it is a different,
worse project. This is already the operating culture — `docs/STATUS.md` is regenerated from a real
run rather than edited by hand, `src/tools/context_docs.py` mechanically checks documentation claims
against the tree, and the harness checks code against its specs. §1 only names what the tooling
already enforces.

---

## 2. Mission (what we are building)

Each bullet is a durable outcome, and each demo clause is a command that runs against this tree, so
anyone can falsify the bullet without reading code.

- **Neural MCTS search with correct game-theoretic semantics**: AlphaZero-style tree search guided
  by policy/value networks, with negamax sign handling and PUCT selection that agree across the
  core, parallel, and progressive-widening engines. *(Demo: `pytest
  tests/unit/framework/mcts/test_value_semantics_regression.py -q` — pins the negamax sign and PUCT
  agreement on seeded trees.)*
- **LangGraph orchestration with introspection**: a typed, checkpointed agent graph whose structure
  is inspectable at runtime. *(Demo: start the service and `GET /graph/structure` or
  `/graph/mermaid`, defined in `src/api/rest_server.py`; both are on by default.)*
- **Multi-domain play through a domain registry**: Connect Four, Othello, and Chess register against
  one adapter contract, with single-agent reasoning domains alongside them. *(Demo: `pytest
  tests/unit/framework/mcts/test_domain_adapters.py -q` — runs on the default install;
  `tests/unit/framework/test_domain_registry.py` covers the registry itself but needs the optional
  neural extra.)*
- **Self-play training on one node**: a generalized self-play trainer supporting DDP, mixed
  precision, and compiled models. *(Demo: the `self-play-convergence` console script, declared in
  `pyproject.toml` and implemented in `src/training/self_play_convergence.py`.)*
- **Policy-improvement measurement**: a decision-quality lift metric that says what search actually
  bought. *(Demo: the `policy-lift` console script → `src/benchmark/policy_lift.py`; the acceptance
  contract lives in `specs/m5_policy_lift.SPEC.md`.)*
- **System-versus-system benchmarking**: a harness that scores different agent systems on a shared
  task set. *(Demo: `python -m src.benchmark --dry-run` → `src/benchmark/cli.py`.)*
- **Spec-driven development toolchain**: every change under `src/` is traceable to a written
  contract, mechanically. *(Demo: `harness validate-spec specs/charter_alignment.SPEC.md` →
  `src/framework/harness/cli.py`.)*
- **Deterministic documentation validation**: documentation claims are checked against the tree, not
  trusted. *(Demo: `python -m src.tools.context_docs` — exit 0 means every cited path and pinned
  value still holds.)*
- **Fail-loud operational posture**: with the mock-LLM fallback unset, the service errors rather than
  silently serving mock output. *(Demo: the fallback flags and their defaults are declared in
  `src/config/settings.py`; see INV-6 for the honest current state.)*
- **Operational observability**: liveness, readiness, and Prometheus metrics endpoints.
  *(Demo: `GET /health`, `/ready`, `/metrics` in `src/api/rest_server.py`.)*

**Named here, but not yet capabilities** — each is gated in §5 and may be prototyped behind a
default-off flag but not advertised: distributed self-play, ONNX/TensorRT inference export,
Transformer backbones, Ray Tune hyper-parameter search, MuZero, UCI/ELO engine evaluation,
stochastic and imperfect-information domains, MLflow experiment tracking, and zero-downtime model
hot-swap.

---

## 3. Scope

**In scope:** everything in §2, plus the packaging, container images, and Kubernetes manifests that
ship it, the observability stack that instruments it, and the agent harness and spec tooling that
govern changes to it.

**Downstream consumers (context, not deliverables):** LLM providers (OpenAI, Anthropic, LM Studio),
Pinecone as a vector store, Weights & Biases and Braintrust for experiment tracking, and S3 for
artifacts. These systems are **consumed, never built**, by this project. Building a replacement for
any of them is out of scope by default, and adopting a new one is a §7 decision, not a task-level
one.

**Platform baseline (fixed by decision, 2026-07-31):** Python is *supported* at ≥3.10 per
`pyproject.toml`, but CI verifies **only 3.11** — the support claim beyond 3.11 is untested and must
not be strengthened without adding a matrix. Formatting is black at line length 120; ruff and mypy
are pinned in the `dev` extra so local and CI verdicts match. PyTorch is optional (`neural` extra)
and python-chess is optional, so the default install must remain functional without either.

> **Recorded divergence.** The following alternatives were considered and explicitly **not**
> adopted; adopting one requires deliberately reopening this decision, not an edit in passing.
> Their rationale is maintained in `PROJECT_STRUCTURE.md`, which governs the detail:
> merging `config/` into `src/config/` (runtime YAML versus Pydantic code, copied separately by the
> image build); merging root `training/` into `src/training/` (root carries its own test suite
> excluded from the default test paths); giving `examples/` a root package marker (the chaos and
> performance suites import the distribution as a bare module); relocating `demo_src/` (a script
> imports it top-level); untracking `models/` (small reference artifacts consumed by integration
> tests on fresh clones); reviving `planning/milestones.yaml` and its epics as a parallel planning
> system; and ratifying `docs/SLA.md` as an operative commitment.

**Out of scope (non-goals).** Each row is a boundary an agent or contributor checks a proposed task
against *before* starting it. The carve-out budget is the maximum number of simultaneously active
exceptions §7 permits; **a budget of 0 means the boundary cannot be carved out at all** — crossing it
requires a full charter review under §7.4.

| ID | Non-goal | Status | Active carve-outs |
|------|----------|--------|-------------------|
| NG-1 | Strategos-MCTS is not a hosted or managed service, and makes no uptime, latency, or support-response commitment to anyone. *(Rationale: an availability promise the maintainer cannot staff is a promise that will be broken; `docs/SLA.md` is an unratified template retained for reference only.)* | ACTIVE | 0 / 0 |
| NG-2 | No silent fallback. A degraded path never becomes the default without being named in this charter and defaulted off. *(Rationale: silently serving mock or lightweight output makes every downstream measurement untrustworthy, and the damage is discovered long after the fact.)* | ACTIVE | 0 / 0 |
| NG-3 | No capability, metric, or status claim appears in any document unless a named command reproduces it from the tree. *(Rationale: this is §1's never-sacrifice restated as an enforceable boundary; an unreproducible claim is indistinguishable from a false one.)* | ACTIVE | 0 / 0 |
| NG-4 | No change under `src/` lands without either an approved spec on its own branch or a written, reasoned exception recorded in the commit. *(Rationale: untraceable changes are why the invariants below drifted from the code in the first place.)* | ACTIVE | 0 / 2 |
| NG-5 | The branch-coverage gate is never lowered, and modules are never added to the coverage omit list to make it pass. *(Rationale: a gate that moves to meet the code is a report, not a gate.)* | ACTIVE | 0 / 0 |
| NG-6 | No breaking change to public signatures or the domain registry; CPU-only and single-GPU paths stay functional. *(Rationale: the framework's value is as a comparable baseline, which requires that old experiments still run.)* | ACTIVE | 0 / 2 |
| NG-7 | No second planning system. Work is planned in `specs/` and `docs/plans/`; `planning/` is not revived. *(Rationale: two planning systems means neither is trusted, which is exactly what happened to `planning/`.)* | ACTIVE | 0 / 1 |
| NG-8 | Unit tests never perform real network or API calls. *(Rationale: a test suite whose result depends on the network cannot gate anything.)* | ACTIVE | 0 / 0 |
| NG-9 | Strategos-MCTS is not a general-purpose LLM agent framework, chat product, or prompt library. *(Rationale: the project's edge is search quality under measurement; breadth here trades that edge for features other projects already provide.)* | ACTIVE | 0 / 1 |
| NG-10 | The layout duplications recorded above are not "cleaned up" as incidental tidying. *(Rationale: each exists to satisfy a real import, build, or test constraint, and merging one breaks something that is not obviously connected to it.)* | ACTIVE | 0 / 2 |

---

## 4. Invariants (must not drift — enforce in review)

These are the rules a change is judged against. Each names the mechanism that enforces it and
carries an honest verdict: **ENFORCED** means a gate fails when it is violated; **PARTIAL** means the
gate covers only part of the claim; **ASPIRATIONAL** means no gate fires at all and only review
catches a violation. The aspirational rows are stated as such deliberately — an invariant everyone
believes is enforced but is not is worse than one honestly labelled.

1. **Configuration discipline.** All tunables and secrets flow through Pydantic Settings; shared
   defaults and bounds live in `src/config/constants.py`. No hardcoded keys or magic numbers.
   *Enforced by:* the secret scan in the `spec-validate` CI job (`.github/workflows/ci.yml`,
   scoped to `src/` and `kubernetes/`), the repo-wide `secret-scan-gitleaks` job
   (`.gitleaks.toml`), and the settings-symbol pin in `src/tools/context_docs.py`.
   **Verdict: PARTIAL** — the two secret scans catch literal keys (one narrow-scope and
   pattern-specific, one repo-wide and pattern-agnostic — see F-17), but nothing prevents a new
   settings class outside `src/config/`, and several exist.
2. **Asynchronous I/O.** New I/O paths use async/await, matching the graph's execution model.
   *Enforced by:* review only. **Verdict: ASPIRATIONAL.**
3. **Dependency injection.** Components take their config, clients, and logger in the constructor
   rather than constructing their own, which is what makes them testable. *Enforced by:* review, and
   indirectly by INV-4 (code that cannot be faked cannot reach the coverage gate).
   **Verdict: ASPIRATIONAL.**
4. **Unit tests are hermetic.** No real network or API calls under `tests/unit/`; all external I/O is
   mocked. *Enforced by:* the CI test job's environment, which forces offline hub and tracing modes
   and injects a dummy API key (`.github/workflows/ci.yml`). **Verdict: ENFORCED.**
5. **Coverage is a gate, not a report.** Branch coverage must stay at or above `fail_under = 85.0`,
   declared in `pyproject.toml` and enforced in CI. *Scope, stated honestly:* the CI gate measures
   `tests/unit/` only, and the coverage configuration omits the two `src/api/` server modules and
   three chess modules — so the gated number is narrower than the headline in `docs/STATUS.md`, which
   covers the full suite. **Verdict: ENFORCED (narrow scope).**
6. **Fail loud by default.** Degraded execution paths are opt-in, never the default.
   *Enforced by:* the fallback flag defaults in `src/config/settings.py`, whose names are pinned by
   `src/tools/context_docs.py`. **Verdict: PARTIAL** — the mock-LLM fallback defaults off as
   documented, but the lightweight-framework fallback defaults **on**, so the general claim that both
   are opt-in is currently false. Recorded in `docs/reviews/2026-07-31-charter-alignment-audit.md`;
   resolving it means either changing the default or narrowing this invariant, and that is a
   deliberate decision, not a drive-by fix.
7. **Changes under `src/` are spec-gated.** A change needs an approved spec on its own branch or a
   written exception trailer. *Enforced by:* the `spec-validate` CI job's traceability step and the
   editor-time hook at `.claude/hooks/spec_gate.py`. **Verdict: ENFORCED (hook warns, CI enforces).**
8. **Logs are structured and secret-safe.** Log with a correlation id and pass sensitive data through
   the sanitizer so secrets are masked. *Enforced by:* the secret scan only, which catches literals
   in source rather than secrets reaching a log sink. **Verdict: ASPIRATIONAL.**
9. **Backward compatibility.** No breaking changes to public signatures or the domain registry;
   CPU-only and single-GPU paths keep working. *Enforced by:* the domain-registry tests
   (`tests/unit/framework/test_domain_registry.py`) and CI running CPU-only.
   **Verdict: PARTIAL** — signature stability itself is review-enforced.
10. **Documentation claims are mechanically checked.** Every repository path and pinned value cited
    by a governed document must resolve against the tree. *Enforced by:* `src/tools/context_docs.py`,
    wrapped by `tests/unit/tools/test_context_docs.py`, which runs inside the CI test job.
    **Verdict: ENFORCED.**
11. **Format, lint, and types stay clean repo-wide.** *Enforced by:* the `lint` and `type-check` CI
    jobs. **Verdict: ENFORCED.**

---

## 5. Long-term roadmap (themes, not dates)

Detailed milestones live in `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md`; keep the two aligned —
milestone detail changes there, not here. **Gates are hard:** work themed under a gated milestone may
be prototyped behind a default-off flag, but shipping past a gate — making it the default,
documenting it in §2, or advertising it in `README.md` — is not permitted until the gate clears.

- **M0 — Foundation and evidence:** the component stack, the reproducible test/coverage baseline, and
  the tooling that keeps documentation honest. *(status: complete)*
- **M1 — Scaling and throughput:** distributed data-parallel training (landed), distributed
  self-play, and compiled-engine inference. **Gate G-M1:** a distributed end-to-end test green in CI
  on at least two ranks, **and** a scaling measurement recorded in `docs/STATUS.md` together with the
  command that reproduces it.
- **M2 — Architectural evolutions:** Transformer backbones, hyper-parameter search, MuZero.
  **Gate G-M2:** each new architecture selectable through configuration with parity tests against the
  existing backbone, and any search reproducible from a recorded seed.
- **M3 — Advanced domains and benchmarking:** full chess action-space encoding, UCI engine
  evaluation with ELO, and stochastic / imperfect-information domains. **Gate G-M3:** the
  policy-improvement artifact promised by `specs/m5_policy_lift.SPEC.md` is measured and recorded —
  `docs/STATUS.md` currently states that no such result exists yet, and no lift claim may appear
  anywhere until it does.
- **M4 — Harness and developer experience:** experiment tracking and a model registry with
  checkpoint lifecycle. **Gate G-M4:** a hot-swap demonstrated under an automated zero-downtime test.
- **M5 — Governance closure:** the code-hygiene and modularity program
  (`docs/plans/2026-07-30-code-hygiene-modularity.md`) run to completion. **Gate G-M5:** every draft
  spec of that program resolved — approved and implemented, or superseded — with the unreachable
  subsystems it identifies either deleted or given a production entry point.

---

## 6. How agents and contributors use this document

- Read this **before** planning a task; keep changes consistent with §3 scope and §4 invariants.
- Put concrete, changeable to-dos in `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md` or a spec under
  `specs/`, not here.
- When a change would violate an invariant or expand scope, **surface it for a human decision** — do
  not implement first and ratify later.
- Treat the §7 budgets as hard constraints. If a proposed task requires a carve-out and the relevant
  budget is exhausted or a trigger has fired, the correct output is a *charter-review request*, not
  an amendment draft.
- Gates in §5 are checked before deploy-facing work under the milestone that references them.
  "Prototype behind a flag" is permitted; "ship past a gate" is not.
- Respect the axis rule in the header. If this charter appears to contradict `docs/STATUS.md` on a
  number, `docs/STATUS.md` wins and the charter has a bug worth filing.
- Claims in this charter must stay reproducible. Any §2 demo clause that stops working is drift to be
  fixed in this document, not an inconvenience to route around.

---

## 7. Amendment Protocol (carve-out budget and review triggers)

Non-goals are amended only through **carve-outs**: dated, ratified, bounded exceptions appended as
blockquotes under §3. The default stance always remains the non-goal; the carve-out is the exception.

### 7.1 Required carve-out format

Every carve-out MUST contain all of the following, or it is invalid and unenforceable:

> **Carve-out CO-{n} (deliberate amendment — ratified by {ROLE} on {DATE} per §7, against
> NG-{id}): {short title}.**
> {One paragraph: exactly what is now permitted, and for which component only.}
> Constraints (all required):
> - **{Bounding constraint}** — what the exception explicitly does NOT permit.
> - **{Mechanism constraint}** — how it is added, via an extension seam rather than by editing core.
> - **{Safety/audit constraint}** — the gates, confirmations, logging, or default-off flags that
>   contain it, **and the mechanical check it defeats**, named explicitly.
> - **{Sequencing constraint}** — what must land before this ships, if anything.
> - **{Expiry}** — the date or merged change at which it lapses.
>
> This remains a bounded exception; every other §3 non-goal still stands.

### 7.2 Budgets (hard limits, not guidelines)

- **Per-non-goal budget:** as stated in the §3 table. NG-1, NG-2, NG-3, NG-5, and NG-8 carry a budget
  of **0** — they cannot be carved out, only amended through a full review. A proposed amendment
  beyond a budget **automatically closes the amendment path** for that non-goal and forces a full
  charter review (§7.4), which must choose one of: (a) rewrite the non-goal to honestly reflect the
  new scope, (b) reject the proposal and reaffirm the boundary, or (c) split the capability into a
  separate project with its own charter. "Ratify one more carve-out anyway" is not an available
  outcome.
- **Global density trigger:** **more than 3 ratified carve-outs within any rolling 30-day window** —
  across all non-goals — forces a full charter review.
- **Cumulative trigger:** when the ledger in §8 reaches **8 rows**, open or closed, the next
  amendment proposal forces a review regardless of the other two triggers. A charter that has needed
  eight exceptions is describing a project it no longer matches.
- **Mandatory expiry:** every carve-out lapses at **90 days or one merged change, whichever comes
  first**. A lapsed carve-out auto-closes; the deviation must then be removed or requested again from
  scratch.

### 7.3 Anti-gaming rules

- **No splitting.** A proposal may not be divided into multiple smaller carve-outs to stay under
  budget; reviewers count intent, not paperwork.
- **No non-goal inflation.** Adding narrowly-worded non-goals solely to reset denominators for the
  cumulative trigger is itself a charter change requiring review.
- **Carve-outs expire on rewrite.** After a review rewrites a non-goal, its prior carve-outs are
  folded into the new text and retired; counters reset only through review, never administratively.

### 7.4 Full charter review (what a trigger forces)

A full charter review is a deliberate, human-led session — not an agent task — that:

1. Re-reads §1–§3 against what the system has actually become.
2. Rewrites vision, mission, and scope so the *default text* — not the exception list — describes
   reality.
3. Re-ratifies or retires each invariant in §4, including re-checking every enforcement verdict
   against the tree.
4. Updates the Executive Summary and the Carve-out Ledger (§8).
5. Records the review date in §0. Until the review completes, **no new carve-outs may be ratified**
   and agents must treat pending amendment proposals as blocked.

### 7.5 Ratification authority

**There is no independent second reviewer for this project.** `CITATION.cff` names a single author
and `.github/CODEOWNERS` assigns a single owner to every path. Amendments are therefore ratified by
the **maintainer alone**, acting explicitly and in writing. That is a real governance gap, named here
rather than papered over: the template's usual "maintainer plus one independent reviewer" checkpoint
does not exist, and the budgets in §7.2 are deliberately tight *because* a single reviewer cannot be
relied on to notice gradual drift in their own project.

Compensating controls, all of which this repository can actually run:

1. Every carve-out names **the mechanical check it defeats**, and its replacement check where one is
   possible.
2. A carve-out is ratified in a commit **separate from the change it permits**, so the deviation is
   legible in the history rather than buried in a feature diff.
3. Any proposal touching a zero-budget non-goal (NG-1, NG-2, NG-3, NG-5, NG-8) requires an
   adversarial agent review before ratification — the `spec-review` agent
   (`.claude/agents/spec-review.md`) or an equivalent multi-agent critique — with its findings
   recorded in the ledger row.
4. **AI agents may draft carve-outs; they may never ratify one.** An agent-authored carve-out with no
   human ratification commit is void, regardless of how well-formed it is.

---

## 8. Carve-out Ledger (append-only)

| CO | Date | NG | Title | Ratified by | Status |
|----|------|----|-------|-------------|--------|
| CO-1 | 2026-07-30 | NG-4 | Documented exception for the MCTS value-semantics bugfix, which had to precede an open approved spec's implementation. Recorded in `CHANGELOG.md` and the hygiene program's governance section. | maintainer | CLOSED (merged) |
| CO-2 | 2026-07-31 | NG-4 | Tooling-only extension of the deterministic documentation validator so this charter's own claims are mechanically checked. Defeats no check; adds one. | maintainer | CLOSED (merged) |

**Budget state (2026-07-31):** 0 active carve-outs / 10 non-goals · last 30-day window: 2
ratifications of a maximum 3 · cumulative granted: 2 of 8 · last full review: never.

> **Pre-charter history, recorded once and not retroactively ratified.** Before this charter existed,
> 58 commits carried a written exception trailer for changes under `src/`, which means the exception
> was the de-facto default channel rather than the exception NG-4 describes. Those changes are not
> re-opened and are not entered in this ledger individually; the ledger begins with the rows above.
> The gap between NG-4's intent and that history is the reason NG-4 carries a budget at all, and
> closing it is tracked as part of Gate G-M5.
