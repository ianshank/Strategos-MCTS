---
id: hygiene_test_triage
goal: Classify all never-gated test files (fix / xfail-with-issue / delete-with-module) and fail-closed unit tests on unexpected sockets
module: tests/
status: draft
---

# Goal

Never-gated test files are classified against a fixed matrix, cross-checked against the
deletion phases' kill list FIRST so no effort is spent fixing tests for code scheduled for
removal. Unit tests must not open real network sockets unless they opt in.

# Acceptance Criteria

- AC-1: Every never-gated test file is classified fix / xfail-with-issue / delete-with-module; zero unclassified. Intended artefact: a checked-in matrix under `tests/` or `docs/` named by the implementing PR, plus zero unclassified files left in the tree.
- AC-2: The blocking job covers all surviving tests/unit/; tests/integration is PR-blocking if measured wall-time is under 10 minutes, else main-push and summary-gated. Intended test: `tests/unit/test_ci_workflow_invariants.py::test_the_end_to_end_suite_runs_on_pull_requests` plus the measured wall-time paste in the PR.
- AC-3: Skipped/xfailed counts are reported in docs/STATUS.md. Intended path: `docs/STATUS.md` refreshed via the coverage-baseline skill (no `planning/` writes).
- AC-4: `pytest-socket` is declared in the `[dev]` extra. Collection of `tests/unit/` disables sockets by default; tests that need a real socket carry `@pytest.mark.enable_socket`. Intended tests: `tests/unit/test_dev_extra_pytest_socket.py::test_pytest_socket_is_declared_in_dev_extra` (dependency pin) and `tests/unit/test_dev_extra_pytest_socket.py::test_unmarked_unit_socket_create_connection_fails_closed` (unmarked raw `socket.create_connection` under `tests/unit/` fails closed).

# Constraints

- Time-boxed; no fixing tests for modules on the deletion kill list.
- Do not put pytest-socket on `hygiene_ci_mechanical` (`module: .github/`).
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
