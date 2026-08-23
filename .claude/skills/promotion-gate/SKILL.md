---
name: promotion-gate
description: >-
  The checklist a measured result must pass in Strategos-MCTS before any
  document, changelog entry, or checkpoint promotion calls a capability better,
  stronger, or proven. Use when reporting a benchmark, a win rate, an Elo delta,
  a policy-lift number, or when deciding whether a trained checkpoint replaces
  the incumbent.
---

# Promotion Gate

A number is not a result. This skill is the difference between "the benchmark ran" and "the
capability improved". Contract: `docs/plans/EVIDENCE_FIRST_PROGRAM.md` §5.

## Checklist — every item, or the result is not reportable

1. **Provenance label.** Which weights produced this? One of `mock`, `static-analysis`,
   `random-weights`, `trained-weights` (`src/config/constants.py`, `EVIDENCE_PROVENANCES`). A
   number from random weights is a plumbing check; say so in the same sentence as the number.
   `src/tools/status_artifact.py` refuses an unlabelled entry structurally, so if you are writing
   the number into prose and cannot name the label, you do not have a result.
2. **Cost denominator.** Search strength is buyable. Report the compute budget the comparison held
   fixed — node count, wall-clock, or token spend — and hold it equal across arms. An
   un-normalized win rate measures the budget, not the method.
3. **Seeds and interval.** A single seed is an anecdote. State the seed count and a dispersion
   measure; a difference inside the noise band is a null result and must be reported as one.
4. **Opponent identity.** Name the incumbent exactly, including its checkpoint id. Self-play
   against the same checkpoint measures nothing.
5. **Gate direction.** State in advance what result would have *stopped* the promotion. A
   criterion written after seeing the number is not a gate.

## Checkpoint promotion

A candidate replaces the incumbent only on a **gated comparison**: fixed budget, declared seeds,
declared threshold, decided before the run. Record the rejected candidates — a gate that has never
rejected anything has not been shown to work, which is exactly why `gated` is the top rung of the
maturity ladder in `docs/capability_maturity.json` and why `CAPABILITY_MATURITY_STAGES` treats
`benchmarked` as strictly weaker.

## Language discipline

| Do not write | Unless |
| --- | --- |
| "improves" / "outperforms" | cost-normalized, multi-seed, versus a named incumbent |
| "validated" / "proven" | a `PROVEN` row exists in `docs/CLAIM_LEDGER.md` |
| "converges" | a lift metric was computed, not merely a loss curve logged |
| "agrees with" | two *independent* engines were compared |

When in doubt, report the weaker claim. Under-claiming costs a sentence; over-claiming costs the
credibility of every other number in the document.
