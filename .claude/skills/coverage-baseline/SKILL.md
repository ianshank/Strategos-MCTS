---
name: coverage-baseline
description: >-
  Produce or refresh the evidence-backed test/coverage baseline in docs/STATUS.md
  (pass-rate plus per-module branch coverage). Use to re-establish the source of
  truth after dependency or test changes, and to pick coverage targets.
---

# Coverage Baseline

Generate the real, reproducible project status so docs never drift from the code. The per-module table
this produces is what selects coverage work — never assume targets from older docs.

```bash
# Clean, full install (heavy ML stack)
pip install -e ".[dev,neural]"

# Full suite with branch coverage; term-missing shows per-line gaps, xml feeds CI/Codecov
pytest tests/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --junitxml=junit.xml

# Per-module summary (sorted by coverage ascending — lowest first = work targets)
python -m coverage report --sort=cover
```

Then update `docs/STATUS.md` with: date, pass/fail/skip counts, overall branch coverage, and the
per-module table (lowest first). Update `planning/milestones.yaml` `quality_metrics` to match.

Notes:
- Coverage gate is branch coverage, `fail_under = 85.0` (`pyproject.toml`).
- Three `src/games/chess/` modules (`ui.py`, `verification/game_verifier.py`,
  `verification/move_validator.py`) are omitted from coverage by config — they will not appear as
  movable targets. The two `src/api/` server modules were **un-omitted** on 2026-08-04 and are now
  measured.
