---
id: hygiene_framework_service
goal: Decompose FrameworkService; project logging; no mock in production code
module: src/api/
status: draft
---

# Goal

framework_service.py carries a third logging abstraction, a 172-line initialize and 184-line
process_query, a protocol-violating MockLLMClient in production code, and an inline parallel
framework implementation.

# Acceptance Criteria

- AC-1: FlexibleLogger is gone; the module uses the project logger with correlation IDs; initialize/process_query are decomposed into phase methods (max function length 60 lines) with phase-boundary logging.
- AC-2: The production MockLLMClient is removed; mock providers are injected only in tests via the client factory.
- AC-3: LightweightFramework moves to its own module as a documented fallback.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
