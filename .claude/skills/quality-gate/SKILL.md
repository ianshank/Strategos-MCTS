---
name: quality-gate
description: >-
  Run the full Strategos-MCTS local quality gate before any commit or push:
  format check, lint, type check, tests with branch coverage, and a hardcoded-secret
  grep. Use whenever you are about to push, open a PR, or want CI/local parity.
---

# Quality Gate

Run the repo's CI-equivalent gate locally, in order. Stop at the first failure and fix before
continuing. Mirrors `.github/workflows/ci.yml` so green locally means green in CI.

```bash
# 1. Format (check only; run without --check to auto-fix)
black src/ tests/ --check --line-length 120

# 2. Lint
ruff check src/ tests/

# 3. Types (pinned mypy; see [dev] extra)
mypy src/

# 4. Tests with branch coverage (gate = fail_under 85.0 in pyproject.toml)
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85

# 5. No hardcoded secrets in source or manifests.
# Use if/else with a subshell `(exit 1)` so a match returns a non-zero status
# (for CI / `set -e`) without the `A && B || C` pitfall and without closing an
# interactive shell.
if git grep -nE "sk-[A-Za-z0-9]{20,}" -- src/ kubernetes/; then echo "FAIL: hardcoded key"; (exit 1); else echo "OK: no keys"; fi
```

Notes:
- Install deps first if missing: `pip install -e ".[dev,neural]"`.
- All tunables must come from `src/config/settings.py` / `src/config/constants.py` — never hardcode.
- Unit tests must not make real network/API calls; mock all I/O.
