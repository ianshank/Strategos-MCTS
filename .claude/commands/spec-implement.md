---
description: Enter implementation for an approved spec on its spec/<id> branch (refuses anything else)
argument-hint: <id>
arguments: [id]
allowed-tools: Bash(cd "${CLAUDE_PROJECT_DIR}" && python3 -m src.framework.harness.cli spec-status*), Bash(git fetch:*), Bash(git cat-file:*), Bash(git show-ref:*), Bash(git switch:*), Read
---

Gate result (single &&-gated chain: nothing mutates unless the spec is `approved`, merged to
`origin/main`, and the branch switch is unambiguous):

<!-- Args are substituted as literal text before the shell parses this line. Single quotes
     stop $(...)/backtick expansion inside the quotes (double quotes would not); an argument
     containing a single quote can still break out — accepted residual risk: slash-command
     args are typed by the local operator into their own session (no privilege boundary),
     and spec-status rejects any id that survives to it malformed. The complete fix (no arg
     interpolation in !-lines) is deferred to Phase 3 packaging. -->

!`cd "${CLAUDE_PROJECT_DIR}" && python3 -m src.framework.harness.cli spec-status '$id' --require approved && git fetch origin main && git cat-file -e 'origin/main:specs/$id.SPEC.md' && { git show-ref --verify --quiet 'refs/heads/spec/$id' && git switch 'spec/$id' || git switch -c 'spec/$id' origin/main; }`

If the chain above failed, refuse with this exact framing and make no changes:

> Spec `$id` is not ready for implementation. Either it is not `approved` (only approved specs
> may enter implementation — use `/spec-new` to draft one, run `spec-review`, and have a human
> flip draft→approved), or the approved spec has not merged to `origin/main` yet (merge the spec
> PR first — a `spec/$id` branch cut from main would not contain it), or the branch switch
> failed (e.g. uncommitted changes — resolve them and retry).

On success:

1. Read `specs/$id.SPEC.md` and restate its goal, acceptance criteria (with AC ids),
   constraints, and out-of-scope items.
2. Implement strictly within the spec's `module` scope, driving toward the acceptance criteria.
3. The PreToolUse spec gate recognizes this branch; CI traceability passes because the spec is
   `approved` on the base branch. The PR that completes the spec flips `status: implemented` in
   its own diff.
4. Note: after that flip merges, this command refuses the spec (`--require approved`) — re-enter
   the branch with plain `git switch spec/$id`, and any follow-up `src/` PR on it needs a
   `No-Spec: <reason>` commit trailer for CI.
