---
name: aqa-regression
description: Executes full automated quality assurance suites, categorizes failures, and generates RCA reports.
---
# aqa-regression

Use this skill to execute the full test suite and verify no regressions exist.

## When to use
- After completing a complex refactoring or adding a new feature.
- To ensure no existing functionality is broken.
- To run the full matrix of tests (unit, integration, ui, e2e).

## Lanes

Run these in order. Do not treat a CPU-only green e2e run as a GPU claim (CHARTER NG-3).
Live LM Studio is adapter QA, not a distillation teacher.

1. **Hermetic unit** (`make test`): `tests/unit/` with `--cov=src --cov-fail-under=85`. Dummy `OPENAI_API_KEY`, `HF_HUB_OFFLINE=1`, `ALLOW_MOCK_LLM_FALLBACK` default false. Distillation units live under `tests/unit/scripts/local_distillation/` (also `make test-local-distillation`). Coverage gate does not measure `scripts/`.
2. **E2E device matrix**: `E2E_DEVICES=cpu make test-e2e` then `E2E_DEVICES=cuda,cpu make test-e2e`. CUDA must *fail* if missing when named. Distillation CLI is `tests/e2e/test_local_distillation_cli_e2e.py` (`python -m scripts.local_distillation`). Placement is `tests/e2e/test_local_distillation_device_e2e.py`. Same-device NeuralMCTS reproducibility is `tests/e2e/test_neural_mcts_device_e2e.py`.
3. **Regression / slow**: `make test-regression` then `pytest tests/ --runslow -ra`.
4. **UI**: `make test-ui` (record env skips; Gradio/LFS/offline-hub misses matching origin/main are RCA-ENV).
5. **Live LM Studio** (opt-in, never `tests/unit/`): `REQUIRE_LMSTUDIO=1`, `LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1`, `ALLOW_MOCK_LLM_FALLBACK=false`. Tests: `tests/e2e/test_lmstudio_live_e2e.py`. Fail-loud closed port: `tests/integration/test_lmstudio_fail_loud.py`. Health `GET /v1/models` must be 200 when required; skip unless `REQUIRE_LMSTUDIO=1`.

## RCA ids

Classify every failure before patching. Env-identical-to-`origin/main` is not a #169 bug.

| Id | Meaning |
|---|---|
| RCA-DIST | Distillation π/z/sidecar/device placement |
| RCA-NUM | CUDA bitwise / floating-point |
| RCA-LLM | Adapter/factory (model id, timeout, omni `content`) |
| RCA-MOCK | Mock fallback when fail-loud was required |
| RCA-DEV | Device mismatch (CPU vs CUDA, NeuralMCTS.device) |
| RCA-URL | Missing `/v1`, `localhost` vs `127.0.0.1` |
| RCA-ENV | Gradio/LFS/offline Hub/missing optional extra |

## Instructions

1. **Run Unit Tests & Mypy**:
   ```bash
   mypy src/
   mypy -p scripts.local_distillation
   pytest tests/unit -v --cov=src --cov-fail-under=85
   pytest tests/unit/scripts/local_distillation -v
   ```
   On Windows, invoke pytest as `python -m pytest ... -p no:randomly` (user-site pytest-randomly + thinc can crash collection).
2. **Run Integration Tests**:
   ```bash
   pytest tests/integration -v
   ```
3. **Run E2E and UI Tests**:
   ```bash
   E2E_DEVICES=cpu pytest tests/e2e -m "not ui" -ra
   E2E_DEVICES=cuda,cpu pytest tests/e2e -m "not ui" -ra
   pytest tests/e2e/test_ui_e2e.py
   pytest tests/e2e/test_user_journeys.py
   ```
4. **Run Status Artifact and Claim Ledger Check**:
   ```bash
   python -m src.tools.claim_ledger
   python -m src.tools.status_artifact --strict
   ```
5. **Parse Results**:
   Classify into the RCA ids above (plus typing / serialization / logger leaks when they appear) and write `docs/reviews/` plus `NOTES.md`.
