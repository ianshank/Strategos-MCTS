---
name: coverage-baseline
description: >-
  Produce a local, evidence-backed coverage report. Refresh docs/STATUS.md only
  after main is green. Never revive planning/milestones.yaml (CHARTER NG-7).
---

# Coverage Baseline

Generate a reproducible coverage report so docs never invent a number the tree cannot run.
**Do not write `docs/STATUS.md` on a branch that has not landed on green main.** CHARTER NG-7
forbids a second planning system: never update `planning/milestones.yaml`.

```bash
pip install -e ".[dev,neural,api]"

# Prefer the Makefile so TEST_ENV matches CI (STRICT_OPTIONAL_DEPS=1, offline hub)
make coverage
```

`make coverage` runs unit tests with branch coverage and writes an HTML report under htmlcov
(gitignored) — it does **not** edit `docs/STATUS.md`. After **green main**, refresh `docs/STATUS.md` in a dedicated follow-up using measured test output; `python -m src.tools.status_artifact --strict` only writes `artifacts/status.json`.

Notes:
- **Coverage is not evidence of capability.** Recorded as `FALSE` (`CL-29`) in
  `docs/CLAIM_LEDGER.md`. When writing STATUS, report the number and stop.
- `docs/STATUS.md` is a live claim surface: `.claude/hooks/evidence_gate.py` warns on
  promotion language that no ledger row supports.
- Coverage gate is branch coverage, `fail_under = 85.0` (`pyproject.toml`). Do not lower it.
- Three `src/games/chess/` modules are omitted from coverage by config.
