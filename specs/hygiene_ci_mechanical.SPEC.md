---
id: hygiene_ci_mechanical
goal: Make the CI gate structurally honest across all three workflows (summary gates what it prints, strict markers, honest e2e, collection-gated server suites, corrected mypy claim, job timeouts, concurrency groups, anchored coverage excludes, a blocking image scan)
module: .github/
status: draft
---

# Goal

The summary job ignores two jobs' failures, there is no --strict-markers so a typo'd marker is a
silent no-op, the e2e workflow cannot fail, rest_server suites are suppressed in three layers while
their modules are omitted from coverage, and CLAUDE.md claims mypy --strict which nothing runs.

Amended 2026-08-04 after a measured review of all three workflows (AC-8..AC-13). A PR to main fires
ci.yml, docker-deployment.yml and e2e_with_langsmith.yml; the latter two were outside the original
scope. Measured at HEAD 74877fa:

- No workflow sets timeout-minutes on any of its 23 jobs, so all inherit the 360-minute default.
- Only ci.yml declares a concurrency group, and docker-deployment.yml's paths filter covers its
  push trigger only, so a docs-only PR runs the full Dockerfile.train matrix.
- security-scan and dependency-audit are printed by the summary job but absent from its failure
  condition, so a bandit HIGH finding does not fail CI; chess-tests and integration-test are absent
  from needs entirely.
- The coverage exclude_lines entry "pass" is applied with re.search against the raw line, matching
  359 lines in src/, 291 of which are not pass statements.
- CLAUDE.md documents `mypy src/ --strict`; that command reports 545 errors in 92 files, while the
  gate's actual `mypy src/` is clean.
- The Trivy step carries both continue-on-error: true and exit-code: '0'.

Two figures from the original amendment were WRONG and are corrected in the ACs below rather than
silently dropped: "19 of 22 markers unused" (re-derived: 0 unused of 21) and "docker-build builds
the same image twice" (the push step is 28s of cached layers, not a second build — see AC-10).

# Acceptance Criteria

- AC-1: chess-tests and integration-test are in the summary job's needs and failure conditions; security-scan and dependency-audit (already in needs and already printed at ci.yml:570,572) are added to the failure condition at ci.yml:578-586, and their fail-open paths are closed (the '|| true' at ci.yml:149 and :209, and the 'if [ -f ... ]' guards at :159 and :219 that pass silently when the report file is absent).
- AC-2: --strict-markers is set in pyproject addopts. The "prune unused markers" half of this AC is WITHDRAWN as based on a stale count: re-derived 2026-08-04, all 21 declared markers are used under tests/ (0 unused), and 0 markers are used-but-undeclared, so there is nothing to prune and enabling strict mode is safe. The Goal's "19 of 22 unused" figure was inherited from an earlier tree state; re-measure before reinstating.
- AC-3: e2e_with_langsmith.yml has no '|| true'; its jobs are conditional on the LangSmith secret being present.
- AC-4: pre-commit pytest-quick can fail.
- AC-5: The rest_server/inference_server suites are collection-gated, with all three suppression layers addressed in one commit: the ci.yml --ignore flags, the conftest collect_ignore_glob, and the pyproject coverage omits for src/api/rest_server.py and src/api/inference_server.py. The load-bearing prerequisite is that the CI test job installs the `api` extra — fastapi/uvicorn live only there, so without it collection cannot succeed at all. MEASURED 2026-08-04: all three suites collect and pass (115 tests, 0 failures), so the "known failures xfail with reasons" clause is vacuous and is withdrawn — there are none. Gate-scope coverage lands at 89.65% (from 89.87%), i.e. 4.65pp of headroom above fail_under=85.0; the two un-omitted modules add 502 statements + 100 branches at 71.99%/81.36% covered, which is the whole of the -0.20pp drag. Degenerate half-landing (omits removed but suites still ignored) yields 88.59% — still green, so the failure mode is silent coverage loss rather than a red build, which is why the conftest guard is made strict under CI rather than left to degrade to a skip.
- AC-6: The post-change coverage gate is dry-run locally and the measured number pasted into the PR before the CI flip; CHANGELOG documents old vs new blocking set and baseline.
- AC-7: CLAUDE.md's mypy claim matches reality; a tracking draft spec exists for the strictness ratchet.
- AC-8: Every job in ci.yml, docker-deployment.yml and e2e_with_langsmith.yml declares timeout-minutes (baseline: 0 of 23 do today, so all inherit the 360-minute default on a docker-build that routinely runs 45m and has hung). Suggested budgets: 15 for lint/spec-validate/type-check/security-scan/dependency-audit/secret-scan-gitleaks, 25 for test/chess-tests/integration-test, 60 for docker-build.
- AC-9: docker-deployment.yml and e2e_with_langsmith.yml each declare a concurrency group with cancel-in-progress (only ci.yml does today; the 2026-07-31 dependabot burst left ~10 overlapping Docker runs at 1766-2586s each while ci.yml correctly cancelled its own at 14-65s), and docker-deployment.yml's pull_request trigger carries the same paths filter its push trigger already has, so docs-only PRs stop running the Dockerfile.train matrix.
- AC-10: The GHA build cache stops costing more than it saves. CORRECTED 2026-08-04 after measuring run 30863130823 job 91849618520 — the first draft of this AC was wrong on both of its clauses and is recorded here so it is not re-attempted:
  (a) WRONG: "collapse the duplicate build". The push step (ci.yml:535-548) is NOT a second build in any compute sense — it takes 28s with all fifteen layers CACHED from the same builder instance, against 17m34s for ci.yml:444. Re-tagging and pushing the locally-loaded image would save ~2s and LOSE the SLSA provenance attestation that build-push-action auto-injects (--attest type=provenance,mode=max), all 8 org.opencontainers.image.* labels, and the OCI image-index media type. Both build steps stay.
  (b) WRONG: "remove ignore-error=true from both cache-to lines". ignore-error is a cache-EXPORT option; removing it would newly hard-fail the job on a GHA cache-service hiccup, which is precisely what commit ba63eaf fixed. It stays on whichever cache-to survives.
  The real defect, measured: 'preparing build cache for export' + 'sending cache export' is 714.4s = 11m54s = 68% of the 17m34s step, and the cache it writes then FAILS TO IMPORT on the next run ('#20/#21/#22 ERROR: blob sha256:... not found', 14 such lines), so the build falls back to a full pip re-resolve. Root cause is mode=max against the 10GB per-repo GHA cache limit with LRU eviction, compounded by both cache-to lines targeting the same scope=ci-production so the push step overwrites the manifest the build step just wrote.
  Required: (i) delete the redundant cache-to on the push step (ci.yml:546) only, keeping its cache-from — that build is 100% local-cache hits, so its export only clobbers the scope; (ii) change the surviving cache-to (ci.yml:456) from mode=max to mode=min, or remove it, with the decision justified by a measured before/after on a main-branch run pasted into the PR per AC-6. Do not guess which; measure.
- AC-13: ci.yml:556 prints 'docker pull ghcr.io/${{ github.repository }}:latest', which renders the mixed-case 'ianshank/Strategos-MCTS'; the published image is lowercase (env.IMAGE_NAME). The printed command must be one a reader can paste and run.
- AC-11: pyproject.toml's coverage exclude_lines entry "pass" is anchored to '^\\s*pass\\s*$'. Coverage applies these as re.search over raw source lines, so the bare form currently excludes 359 lines in src/ of which 291 are not pass statements (docstrings and comments containing "forward pass", dataclass fields such as num_passed/pass_at_1). This is NG-5 inverted — the gate has silently moved to meet the code — so the reported number is expected to drop and fail_under must not be lowered in response.
- AC-12: The Trivy step (ci.yml:511-521) either fails the job on CRITICAL with a documented .trivyignore for accepted findings, or is removed. It currently carries both continue-on-error: true and exit-code: '0', so it cannot fail anything while costing 176s per run — the same permanently-informational defect this phase exists to remove.

# Constraints

- No src/** changes.
- AC-5 and AC-11 both widen the coverage denominator. MEASURED 2026-08-04 rather than assumed: AC-11 alone moves the gate-scope figure 89.87% -> 89.85% (restoring 245 units: +161 statements, +84 branches), so it is safe to land on its own and the earlier "must land in a single commit or CI red-lines at 85%" constraint was over-cautious. AC-5's effect is larger and unmeasured; measure it before landing it, and if the combined figure would approach 85% the two land together.
- The coverage gate fail_under stays at 85.0 and no module is added to the omit list in response to any drop (CHARTER.md NG-5, budget 0/0 — it cannot be carved out).
- The unit-only coverage number the CI gate actually measures is unpublished and differs from docs/STATUS.md's full-suite headline (CHARTER.md INV-5). MEASURED baseline for this spec, `.[dev,neural]`, `pytest tests/unit/` with the three --ignore flags: 8473 passed / 62 skipped / 0 failed, 89.87% pre-AC-11 and 89.85% post-AC-11, wall time 3m12s.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
