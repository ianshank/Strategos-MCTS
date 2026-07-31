---
id: hygiene_delete_llm_guided
goal: Delete the parked llm_guided successor stack and its dead settings fields
module: src/framework/mcts/llm_guided/
status: draft
---

# Goal

src/framework/mcts/llm_guided/ (~10.5k LOC) has zero production importers (two comment
mentions only). Per program decision it is deleted with a recoverability tag; its removal
also retires the third RAG path and third LLM client protocol.

# Acceptance Criteria

- AC-1: The subtree, its tests, and both comment mentions are gone; annotated tag pre-llm-guided-removal exists with a restore recipe in MIGRATION_NOTES noting the RAG-context overlap with roadmap phase 2.x.
- AC-2: Settings fields MCTS_GENERATOR_MODEL / MCTS_REFLECTOR_MODEL / MCTS_EXECUTION_TIMEOUT / MCTS_MAX_MEMORY_MB are removed; the pydantic extra policy is verified so stale .env entries are ignored, not fatal; retired names are on the warn-on-presence list.
- AC-3: Local coverage dry-run at or above 85% pasted into the PR.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
