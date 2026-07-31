---
id: hygiene_train_extract
goal: Migrate the root training/ keepers into src/ (checkpoint loader, synthetic generator, RAG metrics, embedders, knowledge-graph model)
module: src/
status: draft
---

# Goal

Four clusters of root training/ code are worth keeping and must be rewritten to project
conventions (settings config, project logger, seeding utility, adapters-only LLM access)
before the fork is deleted.

# Acceptance Criteria

- AC-1: training/utils/checkpoint_loader.py lives at src/utils/checkpoint_loader.py and is adopted at all open-coded torch.load sites: zero open-coded torch.load in src/ outside it.
- AC-2: synthetic_knowledge_generator -> src/benchmark/synthetic/; benchmark_suite -> src/benchmark/rag_metrics.py; the embedder abstraction is adapters-backed (no raw OpenAI); the knowledge-graph model half is extracted with a typed-exception LLM half.
- AC-3: Each migrated original becomes a re-import shim emitting DeprecationWarning in the same PR (no divergence window); the six CI test files' imports migrate atomically; MockPineconeClient replaces the sys.modules clobber.

# Constraints

- Runs after hygiene_llm_base so migrated code targets the unified adapter API.
- Placement avoids src/training/ (module claimed by open approved specs).
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
