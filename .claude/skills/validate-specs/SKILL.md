---
name: validate-specs
description: >-
  Validate the spec-driven development specs under specs/*.SPEC.md against the
  harness spec schema v2. Use when adding or editing a phase spec, or before
  driving a phase with the harness Ralph loop.
---

# Validate Specs

The project uses spec-driven development. Each spec is a `specs/<id>.SPEC.md` parsed by
`src/framework/harness/intent/spec_loader.py` and validated by
`src/framework/harness/intent/spec_validator.py` (schema v2, defined in
`docs/plans/SDD_PLUGIN_EXTRACTION_PLAN.md` §2):

- **Frontmatter (required):** `id` (must equal the filename minus `.SPEC.md`, unique across
  `specs/`), `goal`, `status` — one of `draft | approved | implemented | verified | superseded`
  (case-sensitive). Optional: `module` (repo-relative path prefix the spec governs — warned when
  absent), `supersedes`, plus legacy `phase`/`milestone`.
- **Body sections:** `# Goal`, `# Acceptance Criteria` (bullets prefixed `- AC-1: …`, `- AC-2: …`;
  IDs unique within the file), `# Constraints`; optional `# Invariants`, `# Out of Scope`.
- **No-changelog rule:** status lives only in frontmatter; inline done-markers like
  `**(8a — done)**` are a validation error. Specs state future contracts, not completed work.

```bash
# Validate every spec in one invocation (also runs cross-file duplicate-id checks)
harness validate-spec specs/*.SPEC.md

# Validate one spec
harness validate-spec specs/phase_0_baseline.SPEC.md

# Plan-only preview of a spec (no LLM calls)
harness dry-run --spec specs/phase_1_correctness.SPEC.md
```

Notes:
- `validate-spec` **errors (exit 1)** on schema violations: missing id/goal/status/criteria,
  unknown status, filename↔id mismatch, duplicate or alias-colliding section headers, inline
  done-markers, mixed or duplicate `AC-n` IDs, and duplicate spec ids across files.
- Warnings (exit 0): missing `module`, all-positional criterion IDs (no `AC-n:` prefixes).
- With an empty `specs/`, the unquoted interactive glob passes the literal pattern and the CLI
  exits 1; CI's nullglob guard skips instead — don't be surprised by the difference.
- `harness run`/`dry-run` (and the Ralph loop) stay permissive on legacy/ad-hoc spec files;
  only `validate-spec` and CI enforce the schema.
- The Ralph loop (`harness run --spec <spec> --ralph`) is a lightweight single-LLM test-iterate
  assist for tightening one change — it is not the multi-file executor for a whole phase.
