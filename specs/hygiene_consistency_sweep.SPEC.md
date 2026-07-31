---
id: hygiene_consistency_sweep
goal: Structured logging everywhere, no library print(), deduped small utils, import hygiene, test-suite hygiene
module: src/
status: draft
---

# Goal

Five mechanical consistency tracks over disjoint file sets: benchmark logging, library print
calls, sanitize dedupe, import hygiene (guarded optional deps, lazy re-exports), and
test-suite dedup/relocation.

# Acceptance Criteria

- AC-1: Zero bare logging.getLogger in src/benchmark/ (project logger with correlation IDs; module APIs untouched).
- AC-2: Zero print( in library code (CLIs exempt).
- AC-3: One sanitize implementation (observability's sanitize_dict); the retry-decorator resolution from the LLM phase is finalized.
- AC-4: feature_extractor guards its optional dependency and reads settings, not raw env; utils/__init__ lazy-loads personality_response.
- AC-5: Test files matching the _ext\d*.py suffix rule (explicit list; the naive *_ext* glob over-matches) are merged with pytest --collect-only count parity per pair and per-module coverage non-decrease; duplicate basenames renamed; loose root tests relocated.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
