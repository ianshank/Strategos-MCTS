---
name: validate-specs
description: >-
  Validate the spec-driven development specs under specs/*.SPEC.md against the
  harness spec schema. Use when adding or editing a phase spec, or before driving
  a phase with the harness Ralph loop.
---

# Validate Specs

The project uses spec-driven development: each phase has a `specs/phase_N_*.SPEC.md` parsed by
`src/framework/harness/intent/spec_loader.py` (frontmatter `goal:` plus `# Goal`, `# Acceptance Criteria`,
`# Constraints` sections). Validate every spec parses cleanly and declares a goal + criteria.

```bash
# Validate one spec
harness validate-spec specs/phase_0_baseline.SPEC.md

# Validate all specs (exit non-zero if any fail to parse)
for f in specs/*.SPEC.md; do echo "== $f =="; harness validate-spec "$f" || exit 1; done

# Plan-only preview of a spec (no LLM calls)
harness dry-run --spec specs/phase_1_correctness.SPEC.md
```

Notes:
- `validate-spec` warns (does not fail) when a spec has no goal section; treat that warning as a defect.
- The Ralph loop (`harness run --spec <spec> --ralph`) is a lightweight single-LLM test-iterate assist for
  tightening one change — it is not the multi-file executor for a whole phase.
