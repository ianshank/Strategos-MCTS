<!--
PR template for a phase of the Code-Hygiene & Modularity program
(docs/plans/2026-07-30-code-hygiene-modularity.md). Select this template with
?template=hygiene_phase.md when opening the PR, or via the "..." dropdown next
to the "Create pull request" button. See CONTRIBUTING.md for the full workflow.
-->

## Phase

- Phase ID: <!-- e.g. hygiene_mcts_value_semantics -->
- Spec: <!-- specs/<id>.SPEC.md — link, and its status at merge time (approved/implemented) -->
- Plan section: <!-- link to the matching phase in docs/plans/2026-07-30-code-hygiene-modularity.md -->

## Summary

<!-- What does this PR change, and why? -->

## Acceptance criteria addressed

<!-- Check off each AC-n from the phase spec this PR satisfies; note any deferred with a reason. -->

- [ ] AC-1:
- [ ] AC-2:
- [ ] AC-3:

## Spec linkage

<!--
Diffs touching src/** need one of:
  - a spec/<id> branch whose spec is approved on the base branch; or
  - a `No-Spec: <reason>` trailer on the commit(s) — e.g. a human-approved program exception
    documented in the phase's Constraints section.
-->

- Spec ID / branch: <!-- spec/<id>, or "No-Spec: <reason>" -->

## Quality gate summary

<!-- Paste the tail of each command's output (or "N/A — docs only"). -->

```
black . --check --line-length 120:
ruff check .:
mypy src/:
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85:
secret grep (git grep -nE "sk-[A-Za-z0-9]{20,}" -- src/ kubernetes/):
```

## Destructive-change record

<!-- Required if this PR deletes or moves code; otherwise state "N/A — non-destructive". -->

- Rollback tag: <!-- e.g. pre-hygiene-delete-enterprise -->
- Restore recipe location: <!-- e.g. docs/MIGRATION_NOTES.md entry -->
- Repo-wide reachability re-verified at HEAD (imports, string literals, importlib, scripts/,
  demos/, examples/, notebooks/, Dockerfiles, compose, kubernetes/, workflows): <!-- yes/no -->
- Coverage dry-run result pasted above meets the ≥85% gate: <!-- yes/no -->

## Checklist

- [ ] Full local quality gate passes (`black --check`, `ruff`, `mypy src/`, `pytest --cov-fail-under=85`)
- [ ] Branch coverage stays ≥ 85% (measured, not assumed — see gate summary above)
- [ ] Same-PR cleanup rule applied (package re-exports, `__all__`, availability probes, factory
      dispatch, orphaned settings fields, and CLAUDE.md rows for anything this PR deletes)
- [ ] `harness validate-spec specs/*.SPEC.md` passes
- [ ] `validate-context-docs` passes (if `.claude/` was touched)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] `docs/MIGRATION_NOTES.md` updated (if behavior changed)
- [ ] `docs/STATUS.md` refreshed via the `coverage-baseline` skill (if this is a destructive or
      gate-changing phase)
- [ ] No hardcoded secrets or configuration values

## Notes for reviewers

<!-- Anything else reviewers should know: risks, follow-ups, deferred items and their tracking artifact. -->
