---
goal: Hold the >=85% branch-coverage gate by filling the genuinely under-tested modules
phase: "2"
milestone: M3
status: active
---

# Goal

Bring every module that the Phase 0 baseline flags below the threshold up to the gate, targeting real
gaps rather than re-testing already-covered code.

# Acceptance Criteria

- Every module the Phase 0 `docs/STATUS.md` table flagged below 85% branch coverage is now at or above it.
- `src/integrations/google_adk/` reaches >=85% branch coverage by adding tests only for the specific
  uncovered branches (the five agents already have unit + integration tests — no greenfield re-test).
- The RAG path in `src/api/framework_service.py` (the coverage-bearing module; `rest_server.py` is omitted
  from coverage) is tested for `rag_available=True/False` and degraded-LLM surfacing, with all I/O mocked.
- The overall branch-coverage gate stays green at `fail_under = 85.0`.

# Constraints

- No real network or API calls in unit tests; mock all I/O; use the existing pytest markers.
- Any decision to un-omit `rest_server.py` from coverage is called out explicitly, not relied on silently.
- Backward compatible; no hardcoded values; full local gate green before push.
