---
goal: Establish an evidence-backed test/coverage baseline and reconcile all status docs to the code
phase: "0"
milestone: M3
status: active
---

# Goal

Replace the stale, contradictory project status with one evidence-backed source of truth. Provision a
clean environment, run the full test suite with branch coverage, record the real numbers, and correct
every roadmap/milestone doc that disagrees with the current source tree.

# Acceptance Criteria

- A committed `docs/STATUS.md` records the real pass-rate and per-module branch coverage, reproducible via
  `pytest tests/ --cov=src --cov-report=term-missing`.
- The per-module coverage table in `docs/STATUS.md` is explicit enough to select Phase 2 targets.
- `planning/milestones.yaml` `quality_metrics` reflect the measured numbers (no stale `88.4` literal).
- No doc asserts a status contradicted by the source tree without a "superseded" or corrected pointer,
  including the stale rows in `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md` §1.1 (examples-import,
  JWT-placeholder, ADK-untested).
- `CLAUDE.md` and `AGENTS.md` are refreshed to verified reality and point at `docs/STATUS.md`.

# Constraints

- No source-code behavior changes in this phase — documentation and measurement only.
- Coverage gate stays at `fail_under = 85.0` (branch coverage) in `pyproject.toml`.
- All tunables remain in `src/config/settings.py` / `src/config/constants.py`; no hardcoded values added.
- Run the full local gate (black, ruff, mypy, pytest) before any push.
