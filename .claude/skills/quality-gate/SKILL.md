---
name: quality-gate
description: >-
  Run the full Strategos-MCTS local quality gate before any commit or push:
  format check, lint, type check, tests with branch coverage, and the two
  secret-detection layers (a fast src/+kubernetes/ grep, and a repo-wide gitleaks
  scan if the binary is installed). Use whenever you are about to push, open a
  PR, or want CI/local parity.
---

# Quality Gate

Run the repo's CI-equivalent gate locally, in order. Stop at the first failure and fix before
continuing. Mirrors `.github/workflows/ci.yml` so green locally means green in CI.

```bash
# 1. Format (check only; run without --check to auto-fix) — repo-wide, matching CI
black . --check --line-length 120

# 2. Lint — repo-wide, matching CI (notebooks are excluded via pyproject)
ruff check .

# 3. Types (pinned mypy; see [dev] extra)
mypy src/

# 4. Tests with branch coverage (gate = fail_under 85.0 in pyproject.toml)
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85

# 5. No hardcoded secrets in source or manifests (fast, narrow-scope layer).
# Use if/else with a subshell `(exit 1)` so a match returns a non-zero status
# (for CI / `set -e`) without the `A && B || C` pitfall and without closing an
# interactive shell.
if git grep -nE "sk-[A-Za-z0-9]{20,}" -- src/ kubernetes/; then echo "FAIL: hardcoded key"; (exit 1); else echo "OK: no keys"; fi

# 6. Repo-wide, pattern-agnostic secret scan (CI-enforced via secret-scan-gitleaks;
# run locally only if the gitleaks binary is installed — it is not part of the [dev] extra).
command -v gitleaks >/dev/null && gitleaks detect --config .gitleaks.toml --source . --no-git -v || echo "gitleaks not installed locally — this layer runs in CI"
```

Notes:
- Install deps first if missing: `pip install -e ".[dev,neural]"`.
- All tunables must come from `src/config/settings.py` / `src/config/constants.py` — never hardcode.
- Unit tests must not make real network/API calls; mock all I/O.
- The step-5 and step-6 secret scans are complementary, not redundant: step 5 is instant and
  dependency-free but scoped to `src/`/`kubernetes/` and `sk-`-shaped keys only; step 6 is
  repo-wide and pattern-agnostic (see `docs/reviews/2026-07-31-charter-alignment-audit.md` F-17 for
  why both are needed — a real key sat in `docs/` for a while, invisible to step 5 alone).
