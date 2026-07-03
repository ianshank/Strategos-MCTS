# SDD Plugin Extraction Plan — Spec-Driven Development as an Enforceable Toolchain

> **Version:** 2.0.0 · **Date:** 2026-07-03 · **Status:** Active
> **Supersedes:** the v1 draft plan (chat, 2026-07-02).
> **Scope:** this document covers only the SDD-plugin track. `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md`
> remains the roadmap source of truth for everything else.
>
> This v2 was produced by a two-pass objective review of the v1 draft: (1) a fact-check of every
> claim against the repository, and (2) an independent adversarial review of the regenerated plan
> itself. The corrections from both passes are recorded in §1 so future revisions don't
> re-litigate them.

---

## 0. Thesis (unchanged from v1)

Strategos-MCTS already runs a working spec-driven development loop — Workstream B
(specs + `.claude/skills/` + CI `spec-validate`) is complete. The goal is **not** to define an SDD
process; it is to harden the one that exists, give it enforcement teeth, and package it for reuse
as the `claude-code-foundry` plugin so it is consumed rather than reimplemented per repo.

What v2 changes is the honest framing of the work: **this is mostly new construction, not
extraction.** The schema fields, lifecycle, enforcement hook, slash commands, subagent, and CI
teeth in v1 do not exist yet (§1). The plan below builds them in-repo first — against the nine
real specs and live CI — and extracts to the plugin only once they demonstrably work.

---

## 1. Review findings that shaped this revision

### 1.1 v1 claims that verified clean

| Claim | Evidence |
|-------|----------|
| Workstream B complete | `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md:10-11`; 9 specs in `specs/`, 3 skills in `.claude/skills/`, `spec-validate` job at `.github/workflows/ci.yml:45` |
| Specs drift into changelogs without a schema rule | Precedent: `docs/plans/NEXT_STEPS_PLAN.md` (superseded 2026-06-30) accreted inline status updates; drift is already visible in the specs themselves — `specs/phase_8_repo_organization.SPEC.md` carries inline `**(8a — done)**` markers while its frontmatter stays `status: active` |
| M5 lift work is the right pilot | Phase 5.1 has a crisp, falsifiable, exit-code-gated AC (≥20% lift as 95%-CI lower bound, chess, ≥100 games, `python -m src.benchmark.policy_lift`) and is genuinely open — `docs/STATUS.md`: "No ≥20% claim exists yet" |
| Scope enforcement to contractual repos/paths; exclude maker projects | Kept as-is (§6) |

### 1.2 v1 claims corrected here

1. **"Extraction" overstates what exists.** None of the v1 schema fields exist: no `id`, no
   authored per-criterion IDs, no `invariants`, no `out_of_scope`, no `supersedes`, no
   draft→approved→implemented→verified lifecycle. Actual frontmatter is
   `goal`/`phase`/`milestone`/`status`; all nine specs are `status: active`; criterion IDs are
   synthesized positionally (`f"c{i}"`, `src/framework/harness/cli.py`) — reordering bullets
   silently remaps them. Phase 0 is a schema redesign plus a nine-spec migration.
2. **Existing validation is warn-only.** `harness validate-spec` fails only on `SpecParseError`;
   a missing goal warns and returns 0; malformed frontmatter passes silently
   (`src/framework/harness/intent/spec_loader.py`). "Port the existing `spec-validate` job" would
   port a toothless check — hardening is new work.
3. **Zero Claude Code plugin infrastructure exists.** No `.claude-plugin/`, no `plugin.json`, no
   `.claude/commands/`, no `.claude/agents/`, no settings hooks. `claude-code-foundry` appears
   nowhere in this repo; whether the repo exists elsewhere is unverified from here — it must be
   provisioned (or access added) at Phase 3.
4. **The "six repos, three incompatible conventions" audit is unverifiable from this repo.** The
   success metric (§7) is rescoped to what is checkable.
5. **CI auto-flipping spec status via bot commits is race-prone.** Replaced with diff-inspection
   rules (§4, CI).
6. **A "block unless the session started via `/spec-implement`" hook is infeasible as described.**
   A slash command is a prompt — it can only instruct the model to record state; marker files go
   stale after crashed sessions and are not keyed to the session; subagents, CI, and hotfixes
   break. Replaced with a stateless branch+status check (§4, gate).
7. **Timeline was optimistic.** Phase 0 is 1.5–2 days, not half a day (parser work, nine-spec
   migration, and known test breakage: `tests/unit/framework/harness/test_intent.py` asserts
   `id == "c0"`). Phase 1 is 4–6 days, not 2–3 (greenfield commands, hook, subagent, two CI
   checks).
8. **v1's pilot framing was wrong in one detail:** no M5 lift spec exists in `specs/` today
   (`phase_5_green_ci.SPEC.md` is unrelated CI-fix work). Phase 2 begins by *authoring* that spec.

---

## 2. Phase 0 — Spec contract v2, in-repo (1.5–2 days)

Define the machine-readable spec format and migrate the existing specs to it, in one PR.

**Schema.** YAML-style frontmatter + markdown body, extending the current format:

- New frontmatter fields: `id` (stable, unique across `specs/`), `status` with lifecycle
  `draft → approved → implemented → verified`, plus `superseded`; `module` (the repo-relative
  path prefix the spec governs, e.g. `src/framework/harness/` — the overlap key for `/spec-new`,
  §3); optional `supersedes: <id>`. Existing `goal`/`phase`/`milestone` stay.
- Authored acceptance-criterion IDs live in the **body bullets**, not nested YAML:
  `- AC-1: <criterion>`. Rationale: the parser is deliberately dependency-free and its line-based
  frontmatter reader does not support nesting — a YAML list under `acceptance_criteria:` would be
  silently dropped. Bullet prefixes are a small, contained extension of the existing
  bullet-extraction path.
- New optional sections: `# Invariants`, `# Out of Scope`.
- **No-changelog rule:** status lives only in frontmatter; specs state future contracts, not
  completed work. Enforcement is split: CI carries a *warning* on a narrow regex (inline
  done-marker patterns like `**(… — done)**` only — a broad prose heuristic would false-positive
  on legitimate text); the `spec-review` subagent (§4) is the real enforcement point.

**Parser and CLI** (`src/framework/harness/intent/spec_loader.py`, `src/framework/harness/cli.py`):

- Extend the `Spec` dataclass with defaulted fields (safe for existing consumers:
  `src/framework/harness/ralph/loop.py`, `intent/__init__.py`).
- Replace positional `c{i}` criterion-ID synthesis with authored `AC-n` IDs, keeping a positional
  fallback for legacy specs.
- Budget the known breakage: `tests/unit/framework/harness/test_intent.py` (asserts `id == "c0"`)
  and any replay cassettes keyed on criterion IDs.

**Validator hardening** (`harness validate-spec`, and therefore the CI `spec-validate` job):
error — not warn — on missing `id`/goal/criteria/status, unknown status value, duplicate `id`
across `specs/`, filename↔`id` mismatch (new specs are named `<id>.SPEC.md`), and duplicate
matching section headers (the parser's `startswith` header aliasing would otherwise pick one
arbitrarily).

**Migration.** Migrate all nine specs in the **same PR** as the validator hardening — otherwise CI
and the Ralph loop hard-fail mid-transition. Semantics: legacy specs migrate to at most
`implemented`, never `verified` — no legacy spec has AC-mapped tests, and mass-assigning
`verified` would debase the status on day one.

---

## 3. Phase 1 — Enforcement layer, repo-native `.claude/`, plugin-shaped (4–6 days)

**Loading mechanism — the fix for v1's silent no-op.** A bare `plugins/sdd/` directory with a
`plugin.json` is loaded by nothing; a plugin only activates via an installed marketplace or a dev
flag, i.e. per-developer opt-in. During the pilot, enforcement must load for **every contributor
who trusts the repo**, so Phase 1 builds on the surfaces Claude Code auto-loads from the
repository itself:

- `.claude/commands/` — the slash commands below;
- `.claude/agents/` — the `spec-review` subagent;
- committed `.claude/settings.json` — the PreToolUse hook.

The internal file layout mirrors plugin structure so Phase 3 packaging is a move, not a rewrite.
`.claude-plugin/plugin.json` + `marketplace.json` are the Phase 3 packaging step, not the pilot
mechanism.

**Components:**

1. **`/spec-new`** — scaffolds a spec from the v2 schema; refuses to create one if an open
   (`draft`/`approved`) spec already covers the same module, determined by comparing the new
   spec's `module` frontmatter path prefix against every open spec's.
2. **`/spec-implement <id>`** — requires `status: approved`; creates or switches to branch
   `spec/<id>`; loads the spec into context.
3. **PreToolUse gate, stateless.** On Edit/Write/MultiEdit/NotebookEdit targeting `src/**`, the
   hook passes iff the current git branch matches `spec/<id>` **and** that spec's frontmatter is
   `approved` or `implemented` — the latter so that the completing PR, which flips the status in
   its own branch (see CI rules below), can still take review-feedback edits afterwards. No
   marker file: the check survives session resume and works in subagents and worktrees. Env-var bypass for hotfixes. **Ships in warn mode**; flipped to block after the
   pilot. Known, stated v0 hole: Bash-based writes (`sed -i`, `tee`, heredocs) are not gated —
   warn-mode telemetry decides whether closing it is worth the complexity.
4. **`spec-review` subagent** — before a human flips `draft → approved`, checks each AC is
   falsifiable and **names an intended test path**. (Tests don't exist at approval time; existence
   and passing are enforced at the `verified` gate, not here.) Also the enforcement point for the
   no-changelog rule.
5. **CI** — extends the existing `spec-validate` job; diff steps check out with `fetch-depth: 0`:
   - A PR touching `src/**` must reference a spec ID via its `spec/<id>` branch name whose
     status is `approved` **on the base branch** (so the completing PR's own
     `approved → implemented` flip doesn't fail this check), **or** carry a documented
     `No-Spec: <reason>` exemption (label or commit trailer) — the hotfix/refactor channel.
   - Multi-PR specs: only the PR that completes the spec flips `approved → implemented`, declared
     by containing the frontmatter flip in its diff; earlier PRs just reference the ID.
   - `implemented → verified` is flipped by a human/agent PR once the AC-named tests exist and
     pass; CI checks the AC↔test mapping. **No bot commits anywhere.**

---

## 4. Phase 2 — Pilot on one real feature (1 week)

Don't pilot on a toy. The pilot **begins by authoring the M5 lift spec via `/spec-new`** — no such
spec exists today; the crisp AC lives in `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md` (§5.1)
and `docs/STATUS.md` ("M5 measurement (policy-lift gate)"): ≥20% decision-quality lift vs the
untrained policy, measured as the 95%-CI lower bound on chess win-rate over ≥100 games, gated by
`python -m src.benchmark.policy_lift` exit code.

Caveat: the convergence run is compute-bound, not code-bound — the pilot could stall on GPU time
rather than exercise the SDD loop. Fallback code-bound pilot: the Phase 8b–e repo-organization
work (untrack model binaries, consolidate demo trees, disambiguate `config/` vs `src/config/`).

Pilot exit criteria: one spec driven `draft → approved → implemented` (or `verified` if the run
converges) entirely through the new toolchain, with the gate in warn mode and its telemetry
reviewed.

---

## 5. Phase 3 — Extraction and forced migration

1. Provision the `claude-code-foundry` repo (or add access to it).
2. Package the `.claude/`-native components into a real plugin: `.claude-plugin/plugin.json`,
   `marketplace.json`, commands/agents/hooks/skills moved intact. Tag **v0.1.0**.
3. Strategos consumes the plugin and deletes its vendored copies.
4. Delete the vendored spec/quality-gate skills from at least one other repo and consume the
   plugin there. If migration hurts, that is signal about the plugin API, not a reason to defer
   (kept from v1).

---

## 6. Scoping (kept from v1)

SDD overhead is negative-value for solo exploratory work. The enforcement hook applies only to
repos/paths where agents implement autonomously and correctness is contractual (e.g. Strategos
`src/**`). Maker projects stay out.

---

## 7. Success metric

Within 30 days of the foundry repo existing:

- Strategos consumes the shared components with **zero vendored duplicates**;
- every merged PR touching `src/**` is traceable to a spec ID **or a documented `No-Spec`
  exemption**;
- one additional repo consumes the plugin at v0.1.0.

---

## 8. Sequencing summary

| Phase | Deliverable | Effort | Gate to next phase |
|-------|-------------|--------|--------------------|
| 0 | Spec schema v2 + hardened validator + 9-spec migration (one PR) | 1.5–2 days | CI `spec-validate` green with errors enabled |
| 1 | `.claude/`-native commands, gate (warn mode), subagent, CI checks | 4–6 days | Gate telemetry visible on a real PR |
| 2 | M5 lift spec driven through the full lifecycle | 1 week | Pilot exit criteria (§4) |
| 3 | `claude-code-foundry` v0.1.0, vendored skills deleted, second consumer | 2–3 days + coordination | Success metric (§7) |
