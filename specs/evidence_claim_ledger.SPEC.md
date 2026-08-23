---
id: evidence_claim_ledger
goal: Machine-checked claim ledger and provenance-stamped status artifact, so PROVEN is derived from evidence rather than asserted in prose
module: src/tools/
status: approved
---

# Goal

Every capability claim in README.md and CHARTER.md section 2 is prose. Nothing mechanically
distinguishes a claim backed by a reproducible run from one backed by component unit tests, and the
89.65 percent coverage gate is routinely read as evidence for capabilities it cannot speak to. The
proof that this matters already exists in the tree: CHARTER.md section 2 asserts that the core,
parallel, and progressive-widening engines agree on negamax sign handling, while
src/framework/mcts/core.py never negates on backpropagation and the parallel and
progressive-widening engines negate unconditionally regardless of their own two_player flag.

Add docs/CLAIM_LEDGER.md plus a deterministic validator that grades claims against the tree, and a
generator that emits artifacts/status.json carrying the provenance a reader needs to tell a
plumbing smoke test from a result. The validator extends the existing deterministic-documentation
engine pattern in src/tools/context_docs.py rather than introducing a second one.

Ship src/tools/claim_ledger.py and src/tools/status_artifact.py with console entry points, wire
both into the Makefile gate and CI, and make an unsubstantiated PROVEN grade a CI failure.

# Acceptance Criteria

- AC-1: src/tools/claim_ledger.py parses docs/CLAIM_LEDGER.md deterministically and exits 1 on any of: a malformed or duplicate claim id, an unknown grade, a Source path that does not resolve on disk, a PROVEN row whose Evidence path does not resolve, a PARTIAL or UNPROVEN row with an empty Notes cell, or a FALSE row whose Notes cite no location. Exit 0 otherwise. --json emits the parsed ledger and the verdict; --debug emits a per-row trace.
- AC-2: The promotion rule is enforced structurally, not by convention: PROVEN requires both a non-empty Verify command and a resolvable Evidence path, and there is no flag that relaxes it. A test asserts that a hand-edited PROVEN row without evidence fails.
- AC-3: src/tools/status_artifact.py writes artifacts/status.json containing, at minimum: commit sha, working-tree dirty flag, generation timestamp, Python version, platform, the installed optional-dependency extras it detected, the coverage fail_under read from pyproject.toml, the measured coverage when a coverage report is supplied, per-grade claim counts derived from the ledger, and a capability maturity matrix over the stages imports, tested, integrated, trains, benchmarked, gated. Every threshold and path is resolved from configuration or arguments; no literal thresholds appear in the module.
- AC-4: Provenance is a required field on every result entry in the artifact, drawn from a closed set covering at least mock, random-weights, and trained-weights, so an artifact cannot omit how it was produced. A test asserts that an entry missing provenance is rejected.
- AC-5: Byte-stable output. Two invocations at the same commit, with the same inputs and an injected clock, produce identical bytes; a test pins this by injecting the timestamp and comparing serialized output.
- AC-6: Console entry points claim-ledger and status-artifact are declared in pyproject.toml, Makefile targets claims and status exist and are documented in the help output, and both are added to the make gate target in CI order. The existing test that pins Makefile and CI agreement stays green.
- AC-7: The CI spec-validation job runs the ledger validator, and a deliberately falsified ledger row is shown to fail that job. The demonstration is recorded in the pull request body, not merely asserted.
- AC-8: tests/unit/test_ci_workflow_invariants.py gains invariants: no workflow uses pull_request_target; every workflow declares a top-level permissions block; no job reachable from a pull_request trigger receives an LLM provider API key; and the count of action references that are not pinned to a full commit sha may only decrease relative to a committed baseline value read from a data file.
- AC-9: Setting ALLOW_MOCK_LLM_FALLBACK together with a production deployment marker is rejected at settings validation time with an actionable message. Defaults are unchanged, so no existing configuration alters behavior, and a test covers both the rejected and the permitted combinations.
- AC-10: Structured logging at INFO for each validator verdict and at DEBUG for per-row decisions, via the project logger; branch coverage on both new modules is at or above the repository gate.

# Constraints

- Backward compatible; no hardcoded values. Thresholds, paths, and grade vocabularies come from src/config/settings.py, src/config/constants.py, or pyproject.toml.
- No second planning system: the ledger records claim status only. Sequenced work stays in docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md and its delegate, and measured test status stays in docs/STATUS.md. The ledger must not restate either.
- The validator is pure filesystem plus parsing. No network, no LLM, no imports of the framework or training packages, so it runs on a default install.
- artifacts/ stays git-ignored. The artifact is a CI output, not a committed file; only its schema and generator are committed.
- Full local quality gate green before push: black at line length 120, ruff, mypy on src, pytest with the coverage gate, and the secret grep.
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG Unreleased entry; MIGRATION_NOTES entry for the settings validation change.

# Invariants

- A grade of PROVEN is derived from a resolvable evidence artifact and can never be produced by editing prose alone.
- Every claim in README.md and CHARTER.md section 2 has exactly one ledger row; the row count is asserted against the extracted claim count so a new claim cannot be added without a grade.
- The validator is deterministic: same tree in, same verdict out.

# Out of Scope

- Fixing the value-semantics defect the ledger records as FALSE. That is specs/hygiene_mcts_value_semantics.SPEC.md and specs/hygiene_mcts_engines.SPEC.md.
- Injecting the NeuralMCTS random generator. That is specs/hygiene_determinism.SPEC.md.
- Cost-normalized benchmarking, the Connect Four golden path, and promotion gating. Those are later milestones in docs/plans/EVIDENCE_FIRST_PROGRAM.md and get their own specs at their predecessor's exit.
- SHA-pinning the GitHub Actions references. This spec adds the ratchet that measures them; the migration itself is a separate change.
