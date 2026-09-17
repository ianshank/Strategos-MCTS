---
description: Run the full pre-PR quality gate (format, lint, mypy, specs, docs, claims, unit/e2e, secrets)
allowed-tools: Bash, Read
---

# Pre-PR Validation Pipeline

Do **not** maintain a second recipe here. Run the Makefile gate in CI order:

```bash
make gate
```

That is `format-check lint lint-ratchet typecheck specs docs claims status pins test test-e2e secrets` with `TEST_ENV` (`STRICT_OPTIONAL_DEPS=1`, offline hub, dummy `OPENAI_API_KEY`). E2E uses `-m "not ui"`. Gitleaks uses the Makefile if/else shape (absent binary is not a green scan).

Overlay pins (also in `aqa-regression`):

```bash
pytest tests/unit/framework/mcts/test_value_semantics_regression.py -q
if [ -f tests/unit/test_dockerfile_perl_base.py ] && [ -f tests/unit/test_deploy_sanity_smoke_paths.py ]; then
  pytest tests/unit/test_dockerfile_perl_base.py tests/unit/test_deploy_sanity_smoke_paths.py -q
else
  echo "Overlay lanes from #172/#173 are not present in this tree yet."
fi
```

Do not regenerate `docs/STATUS.md` on a red main. Do not add a `No-Spec:` trailer on `spec/<id>` branches.
