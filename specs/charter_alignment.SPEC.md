---
id: charter_alignment
goal: Establish CHARTER.md as the project's durable-intent authority and reconcile the documentation with the tree
module: docs/
status: draft
---

# Goal

Strategos-MCTS has no charter. Its durable intent — vision, scope, non-goals, invariants — is
spread across nine documents that each claim partial authority (`docs/STATUS.md`,
`docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md`, `.claude/skills/strategos-primer/SKILL.md`,
`.claude/agents/strategos-guide.md`, `README.md`, `PROJECT_STRUCTURE.md`, `CLAUDE.md`,
`AGENTS.md`, `.github/CONTRIBUTING.md`), with no stated precedence when they disagree — and they
already do.

This spec lands a single root-level `CHARTER.md` that is authoritative for intent and boundaries
only, delegating every other axis to the document that already owns it; reconciles the
documentation-side drift that a charter makes visible; and records the code-side divergences as an
evidence-backed audit rather than silently fixing them.

# Acceptance Criteria

- AC-1: CHARTER.md exists at the repository root with sections 0 through 8 in the charter
  template's numbering, no bracketed placeholder tokens, and a stated axis-of-authority rule
  naming which document governs each axis.
- AC-2: Every repository path CHARTER.md cites resolves against the tree — `python -m
  src.tools.context_docs` exits 0 and reports one more checked document than the pre-change
  baseline of 13.
- AC-3: Every mission bullet in CHARTER.md section 2 carries a falsifiable demo clause naming a
  command that runs against this tree, and each such command's observed exit code is recorded in
  the audit report.
- AC-4: An audit exists at docs/reviews/2026-07-31-charter-alignment-audit.md giving every finding
  a path-and-line reference, the invariant or non-goal it violates, a documentation-side or
  code-side classification, and a disposition.
- AC-5: Every documentation-side finding has a corresponding hunk in the diff, and no file under
  src/ changes behavior — the only permitted src/ diff is the deterministic context-doc validator
  extension that makes AC-2 enforceable.
- AC-6: The invariants in CHARTER.md section 4 each name the mechanism that enforces them with a
  path-and-line reference, and each carries an honest enforcement verdict; invariants with no
  structural gate are labelled as such rather than asserted as enforced.

# Constraints

- Module overlap is disclosed, not hidden: `module: docs/` prefix-overlaps two open draft specs
  (`hygiene_program_bootstrap`, `hygiene_docs_closure`). This spec governs only CHARTER.md, the
  charter pointers in existing docs, and the audit report; it does not touch the artifacts those
  specs own. `src/` is likewise claimed by the open drafts `hygiene_consistency_sweep` and
  `hygiene_train_extract`, which is why the validator extension rides a documented `No-Spec`
  trailer rather than a second, refused spec.
- No behavior changes under src/. Code-side violations are filed as audit findings with a named
  remediation vehicle, never fixed here.
- Findings already scoped by an open hygiene spec link to that spec instead of opening a competing
  finding, so this work does not fork the in-flight code-hygiene program.
- The charter restates no measured value that another artifact generates, with one exception: the
  coverage literal `fail_under = 85.0`, which the context-doc validator mechanically pins to
  pyproject.toml.
- Stale numbers in planning/ are not corrected — correcting them would imply that abandoned
  parallel planning system is alive. It gets a deprecation banner instead.
- CHANGELOG.md gains an entry under `[Unreleased]`; no MIGRATION_NOTES entry is required because
  no behavior changes.

# Out of Scope

- Adopting the third-party OpenSpec or Spec-Kit tooling. Neither is present in this repository and
  introducing one is a separate decision.
- Fixing the code-side divergences the audit records, including the fallback-default contradiction,
  the configuration-module fragmentation, and the ungated invariants.
- Deleting planning/ or docs/SLA.md. Both are retained with banners, matching the repository's
  existing convention for superseded documents.
- Adding a new CI job. The charter becomes CI-gated through the existing unit-test job.
