---
name: e2e-device-matrix
description: >-
  Run the end-to-end suite and decide honestly whether a green run actually exercised the
  GPU path. Use whenever you are about to report that the e2e suite passed, whenever you
  need to verify the CUDA or MPS path on real hardware, whenever a device case is skipped
  and you must say why, and before any document claims a device is tested. Covers the
  device matrix in tests/utils/device_matrix.py, the E2E_DEVICES pin/require contract, and
  the difference between "skipped with a reason" and "not tested".
---

# Interpreting an end-to-end run

The suite under `tests/e2e/` is parametrized over `cpu`, `cuda` and `mps`. On a CPU-only
machine two of those three cases skip. **A green run on such a machine is not evidence that
the accelerator path works**, and the single most common way to make a false claim about
this repository is to read "89 passed" as "the GPU path is tested".

This skill exists to make that distinction mechanical rather than a matter of memory.

## Run it

```bash
make test-e2e                    # every device the host offers; others skip with a reason
E2E_DEVICES=cpu make test-e2e    # pin the matrix to exactly cpu
E2E_DEVICES=cuda make test-e2e   # REQUIRE cuda: fails, never skips, if the host lacks it
```

`make test-e2e` runs exactly what the CI `test` job runs (`pytest tests/e2e -m "not ui"`),
with the same environment. `-m "not gpu"` deselects every accelerator case if you want the
CPU subset only.

## Read the result before reporting it

Always run with `-ra` (the Makefile target already does) and read the skip lines. Then
classify what you saw:

| What the report shows | What it means | What you may claim |
|---|---|---|
| `N passed` with **no** skips | every device in the matrix ran | the devices that ran, by name |
| `SKIPPED … cuda is not available on this host` | the host has no CUDA | **nothing** about CUDA |
| `SKIPPED … the device matrix contains no accelerator` | `E2E_DEVICES` was pinned to a CPU-only set | nothing about any accelerator |
| `FAILED … E2E_DEVICES names 'cuda' but this host does not provide it` | a pinned device is absent — a broken host, not a skip | fix the host or the pin |

The rule the whole design serves: **an unreasoned skip is a bug.** If a case skips without
naming its cause, that is a defect in the matrix, not an acceptable outcome — see
`tests/utils/device_matrix.py` and its invariants in
`tests/unit/tooling/test_e2e_harness_helpers.py`.

## Verifying the GPU path on real hardware

This is the one procedure that converts "written but unverified" into evidence:

```bash
pip install -e ".[dev,neural,api]"
E2E_DEVICES=cuda pytest tests/e2e -m "not ui" -ra --junitxml=e2e-cuda.xml
```

It must **fail**, not skip, if the machine has no CUDA — that is the point of the require
semantics. On success the junit report is the artefact: it carries the per-case outcomes,
so it is what a reviewer can check. Until such a report exists and is committed, `CHARTER.md`
NG-3 forbids any document claiming the CUDA path is tested, and `docs/STATUS.md` says so
explicitly.

## Writing a new device-parametrized test

Take the fixtures from `tests/e2e/conftest.py`; never name a device:

- `device` — a device string, one test run per device in the matrix.
- `device_case` — the same, as a `DeviceCase` when you need `available` / `requested`.
- `accelerator_case` — non-CPU devices only, for tests that compare an accelerator
  *against* CPU. Collects as a single reasoned skip where no accelerator exists.
- `run_script` / `run_module` — hermetic subprocesses with credentials stripped.

A hard-coded `device="cpu"` makes a test pass identically everywhere while proving nothing,
so `.claude/hooks/device_literal_gate.py` warns on one written under `tests/e2e/`. The
written exception, when a literal is genuinely required, is a trailing
`# device-literal: <reason>` comment.

## Do not assert cross-device determinism

CPU and accelerator kernels reduce in different orders. Assert instead:

- same device, same seed → identical results (bitwise on CPU only; the drivers set neither
  `cudnn.deterministic` nor `CUBLAS_WORKSPACE_CONFIG`, so bitwise CUDA equality is not a
  property the code claims);
- valid output on every device;
- accelerator vs CPU agreement within a **stated tolerance**, with TF32 disabled — otherwise
  the comparison silently becomes 10-bit against 24-bit and the tolerance describes the GPU
  rather than the code.

## Related

- `tests/README.md` — the e2e contract and the fixtures.
- `docs/plans/2026-09-04-e2e-device-agnostic.md` — why the matrix is static, what two
  adversarial reviews changed, and the defects the suite surfaced.
- `quality-gate` skill — the full local gate, which includes `make test-e2e`.
