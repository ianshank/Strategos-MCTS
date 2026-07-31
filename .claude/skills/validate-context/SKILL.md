---
name: validate-context
description: >-
  Deterministically validate the repo's Claude context docs — every skill under .claude/skills and
  every agent under .claude/agents (e.g. strategos-primer, strategos-guide) — plus the root
  governance doc CHARTER.md, against the real tree: frontmatter schema, that every cited file path
  still resolves, and that pinned value claims (coverage gate, console scripts, env flags, spec
  statuses) still match their sources. Use this whenever you add or edit a skill, agent, or the
  charter, before pushing changes under .claude/, or when you suspect an orientation doc has drifted
  from the code it describes. Pure filesystem + regex — no network, no LLM, same tree in, same
  verdict out.
---

# Validate Context Docs

The orientation docs under `.claude/` make concrete, checkable claims — file paths, the coverage
gate, console-script names, env-var flags. Those drift silently as code moves (an independent review
once caught the primer citing a factory module that had moved). This makes the check mechanical, so
drift fails fast instead of surviving until someone notices.

```bash
# Full check — exits 1 on any failure (a path that moved, a value claim that no longer matches source)
validate-context-docs                    # console script; or: python -m src.tools.context_docs
python -m src.tools.context_docs --debug  # per-token trace when diagnosing a verdict; --json for CI

# The same checks as part of the unit suite — this is what gates CI
pytest tests/unit/tools/test_context_docs.py -v
```

What it verifies (engine in `src/tools/context_docs.py`, thin `scripts/validate_context_docs.py` shim,
wrapped by `tests/unit/tools/test_context_docs.py`):

1. **Frontmatter schema** — every skill/agent doc has `name` + `description` (agents also `tools`),
   and `name` matches the file/dir. Root governance docs (`GOVERNANCE_DOCS`, currently `CHARTER.md`)
   carry no frontmatter, so they are path-checked only — but a *missing* one is a failure, so the
   charter's existence is itself gated.
2. **Path existence** — every backticked repo path a doc cites resolves on disk. Brace groups such
   as `src/adapters/llm/{base,resilience}.py` expand to each file; a bare filename resolves against
   the nearest directory cited on the same line.
3. **Pinned value claims** — the coverage `fail_under` (from `pyproject.toml`), the `benchmark` /
   `harness` / `policy-lift` console scripts, the opt-in env flags, and the spec statuses
   (`SPEC_STATUSES` in `src/framework/harness/intent/spec_validator.py`) all still match source.

Notes:

- **Adding or editing a skill/agent?** Run this before committing — it catches a mistyped or moved
  path immediately.
- To cite a path that *intentionally* no longer exists (to explain drift), write it **without** a
  `src/`-style root prefix, or add it to `INTENTIONALLY_ABSENT` in the validator. Only
  fully-qualified, rooted paths are checked, so an unprefixed mention reads as prose. An allowlisted
  path that later *reappears* fails the check, so the list can't quietly go stale.
- Existence catches "the path moved," not "you forgot to mention X" — completeness stays a review
  (or `strategos-guide`) job.
- Sits beside `/validate-specs` (spec schema) and `/quality-gate` (lint/type/test); this one guards
  the context docs specifically.
