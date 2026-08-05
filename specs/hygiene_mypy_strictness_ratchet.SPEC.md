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

- The CI gate is `mypy src/` with the settings in `pyproject.toml` `[tool.mypy]`, which deliberately
  set `disallow_untyped_defs = false` and `disallow_incomplete_defs = false`. It is **clean** — 0
  issues across 327 source files.
- `mypy src/ --strict` reports **545 errors in 92 files**. CLAUDE.md previously documented the
  `--strict` form as the gate, which is the NG-3 failure mode: a capability claim no command
  reproduces.

The 545 is not a defect count to be cleared in one pass. It is the distance between two configurations,
most of it concentrated in the flags below. Closing it wholesale in a single change would be
unreviewable and would invite exactly the blanket-`# type: ignore` response CLAUDE.md already warns
against.

# Acceptance Criteria

- AC-1: The 545-error figure is decomposed per strictness flag before any flag is enabled — how many of the 545 each of `disallow_untyped_defs`, `disallow_incomplete_defs`, `disallow_untyped_calls`, `warn_return_any`, and `no_implicit_reexport` is individually responsible for, measured by toggling one flag at a time against the current tree and recording the counts here. Ordering the ratchet without this is guesswork.
- AC-2: Each flag is enabled in its own change, landing with `mypy src/` green under the new setting. No change enables a flag while leaving errors suppressed.
- AC-3: No blanket suppression. A `# type: ignore` added by this programme carries a specific error code (`# type: ignore[arg-type]`, never bare) and a comment naming why the annotation is not possible. A module-level `[[tool.mypy.overrides]]` that relaxes a flag for a whole subtree is permitted ONLY with the subtree named, the reason recorded, and a follow-up AC to remove it.
- AC-4: The mypy pin in the `dev` extra is not loosened to make a step pass. It is pinned to a validated minor precisely because `no-redef`/`unused-ignore` diagnoses shift across releases; a bump is a separate, deliberately revalidated change.
- AC-5: CLAUDE.md's type-check table row and the `Makefile` `typecheck` recipe are updated in the same change as any flag flip, so the documented command and the enforced command never diverge again. `tests/unit/test_ci_workflow_invariants.py::test_makefile_gate_matches_ci_flags` already asserts the Makefile/CI half of this and must stay green.
- AC-6: Progress is recorded as measured error counts in this spec, not as prose. A step that does not move the count is reported as such rather than dropped.

# Constraints

- The `mypy src/` gate stays green at every commit. This programme never lands a red type-check job "to be fixed in the next step".
- `--strict` is NOT adopted wholesale as an intermediate state, and is not documented as the gate until it actually is one (CHARTER.md NG-3, budget 0/0 — capability claims must be reproducible by a named command).
- No src/** behaviour changes. Annotations, `TYPE_CHECKING` imports, and typing-only refactors are in scope; changing what code does at runtime to satisfy the checker is not — that is a separate spec.
- Backward compatible; no hardcoded values.
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- CHANGELOG [Unreleased] entry per landed step.
- Module claim is `pyproject.toml` deliberately: `[tool.mypy]` is the artefact this programme edits, and every open `src/`-prefixed claim would collide under `modules_overlap()`, which matches prefixes bidirectionally.
