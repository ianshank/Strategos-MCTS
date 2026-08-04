---
id: hygiene_ci_mechanical
goal: Make the CI gate structurally honest (summary deps, strict markers, honest e2e, collection-gated server suites, corrected mypy claim, job timeouts, concurrency, non-duplicated docker build)
module: .github/
status: draft
---

# Goal

The summary job ignores two jobs' failures, 19 of 22 pytest markers are unused with no
--strict-markers, the e2e workflow cannot fail, rest_server suites are suppressed in three
layers while their modules are omitted from coverage, and CLAUDE.md claims mypy --strict
which nothing runs.

Amended 2026-08-04 after a measured review of all three workflows (AC-8..AC-12). A PR to main
fires ci.yml, docker-deployment.yml and e2e_with_langsmith.yml; the latter two were outside the
original scope. Measured at HEAD 74877fa: no workflow sets timeout-minutes on any of its 23 jobs;
only ci.yml declares a concurrency group; docker-build is 2738s of a 2935s run (93.3%) and builds
the same image twice; security-scan and dependency-audit are printed by the summary job but absent
from its failure condition, so a bandit HIGH finding does not fail CI; and the coverage
exclude_lines entry "pass" is applied as a substring regex that matches 359 lines in src/, 291 of
which are not pass statements.

# Acceptance Criteria

- AC-1: chess-tests and integration-test are in the summary job's needs and failure conditions; security-scan and dependency-audit (already in needs and already printed at ci.yml:570,572) are added to the failure condition at ci.yml:578-586, and their fail-open paths are closed (the '|| true' at ci.yml:149 and :209, and the 'if [ -f ... ]' guards at :159 and :219 that pass silently when the report file is absent).
- AC-2: --strict-markers is set; unused markers pruned (list re-derived at execution).
- AC-3: e2e_with_langsmith.yml has no '|| true'; its jobs are conditional on the LangSmith secret being present.
- AC-4: pre-commit pytest-quick can fail.
- AC-5: The rest_server suites are collection-gated (import+collect must succeed; known failures xfail with reasons) with all three suppression layers (ci.yml ignores, conftest collect_ignore_glob, pyproject coverage omits) addressed together; green-ness is deferred to the rest-split phase.
- AC-6: The post-change coverage gate is dry-run locally and the measured number pasted into the PR before the CI flip; CHANGELOG documents old vs new blocking set and baseline.
- AC-7: CLAUDE.md's mypy claim matches reality; a tracking draft spec exists for the strictness ratchet.
- AC-8: Every job in ci.yml, docker-deployment.yml and e2e_with_langsmith.yml declares timeout-minutes (baseline: 0 of 23 do today, so all inherit the 360-minute default on a docker-build that routinely runs 45m and has hung). Suggested budgets: 15 for lint/spec-validate/type-check/security-scan/dependency-audit/secret-scan-gitleaks, 25 for test/chess-tests/integration-test, 60 for docker-build.
- AC-9: docker-deployment.yml and e2e_with_langsmith.yml each declare a concurrency group with cancel-in-progress (only ci.yml does today; the 2026-07-31 dependabot burst left ~10 overlapping Docker runs at 1766-2586s each while ci.yml correctly cancelled its own at 14-65s), and docker-deployment.yml's pull_request trigger carries the same paths filter its push trigger already has, so docs-only PRs stop running the Dockerfile.train matrix.
- AC-10: docker-build stops building the same image twice — the push step (ci.yml:537) reuses the image already loaded by ci.yml:445 instead of re-invoking build-push-action against the same context/file/target, and 'ignore-error=true' is removed from both cache-to lines (ci.yml:456, :546) so a failing cache export surfaces instead of silently costing a second full build.
- AC-11: pyproject.toml's coverage exclude_lines entry "pass" is anchored to '^\\s*pass\\s*$'. Coverage applies these as re.search over raw source lines, so the bare form currently excludes 359 lines in src/ of which 291 are not pass statements (docstrings and comments containing "forward pass", dataclass fields such as num_passed/pass_at_1). This is NG-5 inverted — the gate has silently moved to meet the code — so the reported number is expected to drop and fail_under must not be lowered in response.
- AC-12: The Trivy step (ci.yml:511-521) either fails the job on CRITICAL with a documented .trivyignore for accepted findings, or is removed. It currently carries both continue-on-error: true and exit-code: '0', so it cannot fail anything while costing 176s per run — the same permanently-informational defect this phase exists to remove.

# Constraints

- No src/** changes.
- AC-5 and AC-11 both widen the coverage denominator and must land in a single commit; an intermediate state red-lines CI at 85%. The coverage gate fail_under stays at 85.0 and no module is added to the omit list in response to the drop (CHARTER.md NG-5, budget 0/0 — it cannot be carved out).
- The unit-only coverage number the CI gate actually measures is unpublished and differs from docs/STATUS.md's full-suite headline (CHARTER.md INV-5). AC-6 measures and records the gate-scope number; no fail_under discussion precedes that measurement.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
