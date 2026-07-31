<!--
Thanks for contributing to Strategos-MCTS! Please fill in the sections below.
See CONTRIBUTING.md for the full workflow and quality gate.
-->

## Summary

<!-- What does this PR change, and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation / governance / tooling
- [ ] Refactor / internal change
- [ ] Breaking change

## Spec linkage

<!--
Diffs touching src/** need one of:
  - a spec/<id> branch whose spec is approved on the base branch, listing the AC IDs addressed; or
  - a `No-Spec: <reason>` trailer on the commit(s).
Docs/governance/tooling-only PRs can state "N/A — no src/** changes".
-->

- Spec ID / branch: <!-- spec/<id>, or "N/A — no src/** changes" -->
- Acceptance criteria addressed: <!-- AC-1, AC-2, ... or n/a -->

## Charter impact

<!--
CHARTER.md §3 lists the project's non-goals and §4 its invariants. If this PR crosses one, it needs
a carve-out ratified under §7 and recorded in the §8 ledger — not a silent exception.
-->

- [ ] No charter amendment required — or a carve-out is filed in `CHARTER.md` §8

## Checklist

- [ ] Full local quality gate passes (`black --check`, `ruff`, `mypy src/`, `pytest --cov-fail-under=85`)
- [ ] Branch coverage stays ≥ 85%
- [ ] `validate-context-docs` passes (if `.claude/` was touched)
- [ ] `harness validate-spec specs/*.SPEC.md` passes (if `specs/` was touched)
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (for user-facing changes)
- [ ] Documentation updated (if behavior or interfaces changed)
- [ ] No hardcoded secrets or configuration values

## Notes for reviewers

<!-- Anything else reviewers should know: risks, follow-ups, screenshots, etc. -->
