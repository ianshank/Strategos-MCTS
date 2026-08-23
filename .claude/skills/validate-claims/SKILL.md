---
name: validate-claims
description: >-
  Grade a capability claim against the Strategos-MCTS tree and record it in
  docs/CLAIM_LEDGER.md. Use whenever a README.md or CHARTER.md capability bullet
  is added, reworded, or disputed, whenever someone asks whether a capability is
  "proven", and before any prose promotes a capability from built to working.
  Applies the promotion rule mechanically instead of arguing about it.
---

# Validate Claims

Every capability claim in `README.md` and `CHARTER.md` §2 has exactly one row in
`docs/CLAIM_LEDGER.md`. This skill produces or corrects that row. The contract it enforces is
`docs/plans/EVIDENCE_FIRST_PROGRAM.md` §4; the validator is `src/tools/claim_ledger.py`.

## The rule (not negotiable, not overridable)

| Grade | Requires |
| --- | --- |
| `PROVEN` | a `Verify` command **and** an `Evidence` path that resolves on disk |
| `PARTIAL` | a `Verify` command and a `Notes` cell naming the **missing link** |
| `UNPROVEN` | a `Notes` cell naming what would have to exist |
| `FALSE` | a `Notes` cell citing the contradicting `file:line` |

`PROVEN` is derived, never asserted. No flag, env var, or reviewer opinion relaxes it — the CI
`spec-validate` job proves this each run by falsifying a `PROVEN` row in a scratch copy and
asserting rejection.

## Procedure

1. **Locate the claim verbatim.** `grep` the exact bullet in `README.md` / `CHARTER.md`. If the
   ledger has no row for it, the ledger is incomplete: add one.
2. **Find the implementation, then find its caller.** Code that exists but is never invoked
   supports `PARTIAL` at best. `SelfPlayEvaluator.evaluate()` in `src/training/agent_trainer.py`
   is the reference example: fully implemented, never called from a training path.
3. **Run the verification command yourself.** A command you have not run is not evidence. Record
   the exact invocation, not a paraphrase.
4. **Check the evidence artefact resolves.** Committed path only. the git-ignored artifacts directory is not committed, so
   a path under it cannot support `PROVEN`; cite the test module or a committed JSON instead.
5. **Grade down on doubt.** If the claim's wording is stronger than the tree supports, you have
   two honest moves: narrow the wording, or produce the evidence. Never split the difference in
   the `Notes`.
6. **Validate.** `python -m src.tools.claim_ledger` — schema, cited paths, and the promotion rule.
   Then `python -m src.tools.status_artifact --strict`, because grades feed the maturity ladder in
   `docs/capability_maturity.json` and a stage above its supporting grades is a build failure.

## Failure modes this skill exists to prevent

- **Prose promotion.** Rewriting a bullet to sound more confident while the row stays `PARTIAL`.
- **Test-count as evidence.** Coverage percentage measures the tests, not the capability — see
  the `CL-29` row, recorded `FALSE` deliberately as a standing reminder.
- **Self-consistency mistaken for agreement.** One engine agreeing with itself is not
  cross-engine agreement.
- **Evidence pointing at an ignored path.** Green locally, empty in CI.
