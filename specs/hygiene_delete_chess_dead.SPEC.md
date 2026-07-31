---
id: hygiene_delete_chess_dead
goal: Delete the dead chess verification/ and observability/ subtrees
module: src/games/chess/
status: draft
---

# Goal

src/games/chess/verification/ (~3.0k LOC) and src/games/chess/observability/ (~1.0k) are
reachable from no entry point; chess engines/ and the ensemble chain are explicitly kept.

# Acceptance Criteria

- AC-1: verification/ and observability/ plus their tests are deleted after repo-wide reachability re-verification; the chess logger-name strings in src/observability/logging.py are cleaned in the same PR.
- AC-2: engines/stockfish_adapter.py, ensemble_agent.py, meta_controller.py, continuous_learning.py, and ui.py are NOT touched (kept per program decision to wire the UI).
- AC-3: Same-PR cleanup rule applied; CHANGELOG Removed; rollback tag pre-hygiene-delete-chess-dead; coverage dry-run pasted in PR.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
