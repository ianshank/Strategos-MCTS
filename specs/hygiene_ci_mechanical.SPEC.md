---
id: hygiene_ci_mechanical
goal: Keep CI structurally honest, and stop the surviving GitHub Actions docker cache-to export from using mode=max
module: .github/
status: draft
---

# Goal

Keep the CI Pipeline structurally honest. The remaining open contract is the `cache-to` on
the CI Pipeline (`ci.yml`) `docker-build` job step named `Build Docker image`: it must use
`mode=min` or the `cache-to` key must be absent, with a measured before/after on a
main-branch run. Other `cache-to` exports (including `docker-deployment.yml`) are out of
scope unless an AC names them.

# Acceptance Criteria

- AC-1: The CI Pipeline `summary` job `needs` and failure conditions cover every job it prints, including chess-tests, integration-test, security-scan, and dependency-audit; missing bandit/pip-audit reports fail the job. Intended test: `tests/unit/test_ci_workflow_invariants.py::test_summary_gates_every_job_it_depends_on` and `test_ci_summary_covers_the_test_bearing_jobs`.
- AC-2: `pyproject.toml` `[tool.pytest.ini_options] addopts` includes `--strict-markers`. Intended test: `tests/unit/test_ci_workflow_invariants.py` (parse addopts; add `test_strict_markers_in_addopts` if absent).
- AC-3: Result-bearing steps in `.github/workflows/` do not swallow their exit code with `|| true`; `e2e_with_langsmith.yml` jobs that need the LangSmith secret are conditional on it. Intended test: `tests/unit/test_ci_workflow_invariants.py::test_no_result_bearing_step_swallows_its_exit_code`.
- AC-4: `.pre-commit-config.yaml` `pytest-quick` fails when pytest fails. Intended test: parse the hook `entry` in `tests/unit/test_ci_workflow_invariants.py` (add `test_precommit_pytest_quick_can_fail` if absent).
- AC-5: The CI `test` job installs `.[dev,neural,api]` and does not `--ignore` the API-server suites. Intended test: `tests/unit/test_ci_workflow_invariants.py::test_test_job_installs_the_extras_its_suites_require` and `test_suppressed_api_suites_are_no_longer_ignored`.
- AC-6: `fail_under` stays `85.0`; any gate-set change pastes a local coverage dry-run into the PR and records old vs new blocking set in CHANGELOG. Intended test: `tests/unit/test_ci_workflow_invariants.py` coverage-config assertions plus PR process.
- AC-7: `CLAUDE.md` documents `mypy src/` (not `--strict`); further strictness lives in `specs/hygiene_mypy_strictness_ratchet.SPEC.md`. Intended test: `tests/unit/tools/test_context_docs.py` (or the CLAUDE.md pin in `src/tools/context_docs.py`).
- AC-8: Every job in `ci.yml`, `docker-deployment.yml`, and `e2e_with_langsmith.yml` sets `timeout-minutes`. Intended test: `tests/unit/test_ci_workflow_invariants.py::test_every_job_declares_a_timeout`.
- AC-9: Every workflow has a concurrency group with `cancel-in-progress`; `docker-deployment.yml` pull_request `paths` match push. Intended test: `tests/unit/test_ci_workflow_invariants.py::test_every_workflow_has_a_concurrency_group` and `test_expensive_workflow_filters_pull_requests_by_path`.
- AC-10: In `.github/workflows/ci.yml`, the `Build Docker image` step's `cache-to` is `mode=min` or the `cache-to` key is absent; `ignore-error=true` may remain on a surviving export. The `Build and push Docker image` step must not declare `cache-to`. A measured before/after on a main-branch run is pasted into the implementing PR. Intended test: `tests/unit/test_ci_workflow_invariants.py::test_ci_build_step_cache_to_is_not_mode_max`.
- AC-11: Coverage `exclude_lines` for a pass statement is the anchored pattern `^\\s*pass\\s*$`. Intended test: `tests/unit/test_ci_workflow_invariants.py::test_coverage_exclude_patterns_are_anchored`.
- AC-12: Image CRITICAL findings can fail the job (blocking scan with `.trivyignore`, `exit-code: '1'`). Intended test: `tests/unit/test_ci_workflow_invariants.py::test_image_scan_can_actually_fail`.
- AC-13: Printed `docker pull` image names are lowercase. Intended test: `tests/unit/test_ci_workflow_invariants.py` (add `test_printed_docker_pull_is_lowercase` if absent).

# Constraints

- No `src/**` changes.
- `fail_under` stays `85.0`; no module is added to the coverage omit list (CHARTER.md NG-5).
- Do not guess `mode=min` vs removing `cache-to`; measure.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.

# Out of Scope

- `pytest-socket` (owned by `hygiene_test_triage`).
- Production image OS-package upgrades (Dockerfile, not this module).
- Dockerfile.train pin source (`hygiene_train_container`).
