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

Then regenerate the machine-readable side, which is what other tooling reads:

```bash
python -m src.tools.claim_ledger        # grades must still validate after any doc edit
python -m src.tools.status_artifact --strict
```

Notes:
- **Coverage is not evidence of capability.** It measures which lines the tests execute, not
  whether the system works — recorded deliberately as a `FALSE` row (`CL-29`) in
  `docs/CLAIM_LEDGER.md` so the confusion cannot recur. When writing `docs/STATUS.md`, report the
  number and stop; capability language belongs in the ledger, where it needs evidence.
- `docs/STATUS.md` is a live claim surface: the `.claude/hooks/evidence_gate.py` PostToolUse hook
  scans it after every edit and warns on promotion language that no ledger row supports.
- Coverage gate is branch coverage, `fail_under = 85.0` (`pyproject.toml`).
- Three `src/games/chess/` modules (`ui.py`, `verification/game_verifier.py`,
  `verification/move_validator.py`) are omitted from coverage by config — they will not appear as
  movable targets. The two `src/api/` server modules were **un-omitted** on 2026-08-04 and are now
  measured.
