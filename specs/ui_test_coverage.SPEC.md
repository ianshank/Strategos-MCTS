---
id: ui_test_coverage
goal: Make UI tests actually execute in CI, and make the ones that run assert something
module: tests/
milestone: M4
status: draft
---

# Goal

A launch-blocking `TypeError` in the chess UI reached `main` behind 66 passing tests. That is not an
accident of review: no CI job installed the `[ui]` extra, no test anywhere constructed a Blocks graph,
and the tests that did exist exercised pure helper functions only.

The wider picture measured during the audit: of 143 UI-adjacent tests, 48 never executed (double-gated,
and they hard-ERROR the moment the gate is lifted because a module-local fixture shadows the graceful
skip), 66 were collected by no CI job at all, `tests/ui` appeared in zero workflows, and of the four
`tests/e2e/test_ui_e2e.py` tests that survived a totally broken backend, all four asserted nothing.

This spec closes the gap between "tests exist" and "tests run and mean something".

# Acceptance Criteria

- AC-1: A CI job installs the `[ui]` and `[chess]` extras and runs the UI suites. It is present in the
  summary job's `needs`, its env map and its gated `JOBS` list, so the job is enforced rather than merely
  printed.
- AC-2: A test constructs the chess UI's Blocks graph, so a launch-blocking error fails CI rather than
  passing behind helper-only tests.
- AC-3: `tests/ui` degrades cleanly when a prerequisite genuinely cannot be met in a hermetic
  environment: an explicit skip naming the cause, never an opaque failure and never a silent
  disappearance. The gate is narrow enough that a real regression still fails.
- AC-4: The module-local `driver` fixture in `tests/games/chess/test_ui_selenium.py` no longer shadows
  the graceful skip in `tests/games/chess/conftest.py`, so lifting `--runslow` without a browser yields
  skips rather than 48 errors.
- AC-5: No UI test asserts nothing. The `if not found: pass` loop and the
  `except Exception as e: assert str(e) is not None` handler in `tests/e2e/test_ui_e2e.py` are replaced
  with assertions that can fail.
- AC-6: `src/games/chess/ui.py` is measured by a coverage gate at no less than the main gate's
  threshold, and UI logic that must be measured lives under `src/` rather than the root `app.py`.

  This AC originally read "is removed from the coverage omit list once its logic is testable". That
  was written before the omit's cause was checked, and it is not achievable as stated: the module
  imports `chess.pgn` and `gradio` at module scope (`src/games/chess/ui.py:19-20`), so the
  coverage-gated `test` job could only measure it by installing `[ui,chess]` — pulling gradio into
  the gated job and duplicating the `ui-tests` job that already owns those extras. Keeping the omit
  while leaving the module unmeasured was the worse half of the trade: ~90 tests exercised it and no
  job scored it. Satisfied instead by a second gate (`.coveragerc.ui`) in `ui-tests`, scoped to
  exactly the modules the main gate omits, at the same 85% threshold. Measured 91.48% branch coverage.

# Constraints

- No new pytest markers: `--strict-markers` is enabled and `ui`, `selenium`, `slow` and `e2e` already
  exist. Registration is split between `pyproject.toml` and `tests/conftest.py`; check both.
- No real network or provider calls; a test that needs a downloadable model skips with a reason.
- Skips must name their cause. A skip that hides a real failure is worse than the failure.
- The repo-wide 85% branch-coverage gate stays owned by the main test job. The UI job gates only the
  modules the main job's `omit` list excludes, and at the same threshold — it must never become a
  second, weaker gate over code the main job already measures. `test_every_module_omitted_from_the_
  main_gate_is_measured_somewhere` enforces both halves.
- Full local gate green before push.
