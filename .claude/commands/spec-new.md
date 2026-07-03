---
description: Scaffold a new schema-v2 spec as a draft, after deterministic id/overlap refusal checks
argument-hint: <id> <module-path>
arguments: [id, module]
allowed-tools: Bash(cd "${CLAUDE_PROJECT_DIR}" && python3 -m src.framework.harness.cli spec-new*), Read, Edit
---

Scaffold result (deterministic — the CLI refuses malformed ids, existing files, and module
overlap with any open draft/approved spec):

!`cd "${CLAUDE_PROJECT_DIR}" && python3 -m src.framework.harness.cli spec-new --id "$id" --module "$module"`

If the command above printed an `error:` line, report that refusal to the user **verbatim** and
stop — do not create or edit any file.

Otherwise the draft spec now exists at `specs/$id.SPEC.md`. Work with the user to complete it:

1. Read the scaffolded file.
2. Replace the Goal placeholder with the contract this spec enforces (future work only — the
   no-changelog rule makes inline done-markers a validation error).
3. Replace the `AC-1` placeholder and add further `- AC-n:` bullets. Every criterion must be
   falsifiable and name an intended test path (e.g. `tests/unit/...`); the tests need not exist
   yet — existence and passing gate the later `verified` flip.
4. Fill Constraints / Invariants / Out of Scope, or delete the optional sections if empty.
5. Leave `status: draft`. Before anyone flips it to `approved`, the `spec-review` subagent must
   review it (delegate proactively) and a human makes the flip.
6. Validate: `harness validate-spec specs/$id.SPEC.md`.
