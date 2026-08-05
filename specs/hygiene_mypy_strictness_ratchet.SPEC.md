---
id: hygiene_mypy_strictness_ratchet
goal: Track the mypy strictness ratchet as a deliberate, measured programme rather than an implied claim — raising [tool.mypy] strictness in bounded steps, each one landing green, never by blanket ignores
module: pyproject.toml
status: draft
---

# Goal

`hygiene_ci_mechanical` AC-7 required two things: that CLAUDE.md's mypy claim match what CI runs, and
that a tracking spec exist for the strictness ratchet. The first shipped in PR #142 (the documented
command is now `mypy src/`, which is what the type-check job actually runs). The second did not. This
spec is that tracker, created so the ratchet is a scheduled programme instead of an implied promise.

The gap being tracked, MEASURED 2026-08-04 at HEAD 74877fa and unchanged at 743d5e3:

- The CI gate is **`mypy src/ --no-error-summary`** (`.github/workflows/ci.yml`, the `Run mypy` step
  of the type-check job), with the settings in `pyproject.toml` `[tool.mypy]`, which deliberately set
  `disallow_untyped_defs = false` and `disallow_incomplete_defs = false`. It is **clean** — 0 issues
  across 327 source files. `--no-error-summary` suppresses only the trailing "Found N errors" line
  and changes no diagnosis; CLAUDE.md and the Makefile document the command without it because a
  human running it wants that trailer. This spec's *measurements* deliberately run WITHOUT the flag,
  since the trailer is the count being recorded.
- `mypy src/ --strict` reports **545 errors in 92 files**. CLAUDE.md previously documented the
  `--strict` form as the gate, which is the NG-3 failure mode: a capability claim no command
  reproduces.

## Measurement procedure (binding — every count in this spec is produced this way)

Stated exactly, because a ratchet whose numbers cannot be reproduced is the thing this spec exists to
avoid:

- **Resolved mypy version:** whatever the `dev` extra's `mypy>=1.19.0,<1.20.0` pin resolves to.
  Record the output of `mypy --version` alongside any count; a count from a different minor is not
  comparable and must be re-taken.
- **Base configuration:** `pyproject.toml` `[tool.mypy]` exactly as committed, unmodified. Flags
  under test are added on the command line (`mypy src/ --disallow-untyped-defs`), never by editing
  the config for a measurement — the config is edited only when a flag actually lands (AC-2).
- **Counts are absolute, not deltas.** Every number recorded is the total error count that one
  command produced. Per-flag figures will therefore overlap, because a single unannotated function
  can trip several flags at once; they will not sum to 545, and a decomposition that does should be
  treated as suspect rather than as confirmation.
- **Residual.** `--strict` enables more than the five flags tracked in AC-1. Whatever remains after
  those five is the *residual* — recorded as its own figure, not silently folded into the last flag.
  Until AC-1 runs, the residual is unmeasured and this spec does not guess at it.
- **Environment:** the same extras the CI type-check job installs, so a local count and a CI count
  are the same measurement.

The 545 is not a defect count to be cleared in one pass. It is the distance between two configurations,
most of it concentrated in the flags below. Closing it wholesale in a single change would be
unreviewable and would invite exactly the blanket-`# type: ignore` response CLAUDE.md already warns
against.

# Acceptance Criteria

- AC-1: The 545-error figure is decomposed per strictness flag before any flag is enabled — how many errors each of `disallow_untyped_defs`, `disallow_incomplete_defs`, `disallow_untyped_calls`, `warn_return_any`, and `no_implicit_reexport` produces on its own, measured by adding exactly one flag at a time to the committed base config per the Measurement procedure above, plus the residual (`--strict` minus those five). Counts are absolute and will overlap rather than sum to 545. The recorded set must reproduce the 545 baseline when all flags are applied together, and the resolved `mypy --version` is recorded with them. Ordering the ratchet without this is guesswork, so no flag lands under AC-2 until this table exists.
- AC-2: Each flag is enabled in its own change, landing with `mypy src/` green under the new setting. No change enables a flag while leaving errors suppressed.
- AC-3: No blanket suppression. A `# type: ignore` added by this programme carries a specific error code (`# type: ignore[arg-type]`, never bare) and a comment naming why the annotation is not possible. A module-level `[[tool.mypy.overrides]]` that relaxes a flag for a whole subtree is permitted ONLY with the subtree named, the reason recorded, and a follow-up AC to remove it.
- AC-4: The mypy pin in the `dev` extra is not loosened to make a step pass. It is pinned to a validated minor precisely because `no-redef`/`unused-ignore` diagnoses shift across releases; a bump is a separate, deliberately revalidated change.
- AC-5: CLAUDE.md's type-check table row and the `Makefile` `typecheck` recipe are updated in the same change as any flag flip, so the documented command and the enforced command never diverge again. `tests/unit/test_ci_workflow_invariants.py::test_makefile_gate_matches_ci_flags` already asserts the Makefile/CI half of this and must stay green. The one permitted divergence is `--no-error-summary`, which CI passes and the documented commands omit: it is output-only, and the reason is recorded above rather than left as an unexplained mismatch.
- AC-6: Progress is recorded as measured error counts in this spec, not as prose. A step that does not move the count is reported as such rather than dropped.

# Constraints

- The `mypy src/` gate stays green at every commit. This programme never lands a red type-check job "to be fixed in the next step".
- `--strict` is NOT adopted wholesale as an intermediate state, and is not documented as the gate until it actually is one (CHARTER.md NG-3, budget 0/0 — capability claims must be reproducible by a named command).
- No src/** behaviour changes. Annotations, `TYPE_CHECKING` imports, and typing-only refactors are in scope; changing what code does at runtime to satisfy the checker is not — that is a separate spec.
- Backward compatible; no hardcoded values.
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- CHANGELOG [Unreleased] entry per landed step.
- Module claim is `pyproject.toml` deliberately: `[tool.mypy]` is the artefact this programme edits, and every open `src/`-prefixed claim would collide under `modules_overlap()`, which matches prefixes bidirectionally.
