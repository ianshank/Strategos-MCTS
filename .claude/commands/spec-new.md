---
description: Scaffold a new schema-v2 spec as a draft, after deterministic id/overlap refusal checks
argument-hint: <id> <module-path>
arguments: [id, module]
allowed-tools: Bash(cd "${CLAUDE_PROJECT_DIR}" && python3 -m src.framework.harness.cli spec-new*), Read, Edit
---

Scaffold result (deterministic — the CLI refuses malformed ids, existing files, and module
overlap with any open draft/approved spec):

<!-- Args are substituted as literal text before the shell parses this line. Single quotes
     stop $(...)/backtick expansion inside the quotes (double quotes would not); an argument
     containing a single quote can still break out — accepted residual risk: slash-command
     args are typed by the local operator into their own session (no privilege boundary),
     and the CLI additionally rejects any id/module that survives to it malformed. The
     complete fix (no arg interpolation in !-lines) is deferred to Phase 3 packaging. -->

!`cd "${CLAUDE_PROJECT_DIR}" && python3 -m src.framework.harness.cli spec-new --id '$id' --module '$module'`

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
