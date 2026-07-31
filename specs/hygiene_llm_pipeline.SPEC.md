---
id: hygiene_llm_pipeline
goal: Replace StdlibLLMClient with injected adapters on an async-native pipeline; bound application-level retries
module: src/api/
status: draft
---

# Goal

The live comparison path uses a hand-rolled urllib LLM client with no retry/breaker, driven
synchronously beneath an async FastAPI route. Make the pipeline async-native (no asyncio.run
inside the running loop) and inject the real client via LLMClientFactory.

# Acceptance Criteria

- AC-1: The llm_mcts pipeline and ComparisonService are async-native; the /compare route's behavior parity is asserted by existing tests; sync entry survives only at CLI/demo top level.
- AC-2: StdlibLLMClient remains one cycle as a deprecated shim (tracking artifact for removal exists); the generate_sync protocol collapses onto the async path.
- AC-3: Application-level retries in benchmark scorer/harness are KEPT with settings-bounded total attempts; the unused observability retry decorator is adopted or deleted (recorded in CHANGELOG).
- AC-4: llm_mcts keeps DEFAULT_* and MockLLMClient importable (re-exports) so comparison_service's import block is unaffected.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
