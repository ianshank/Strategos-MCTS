---
id: hygiene_chess_consolidation
goal: One RoutingDecision, one phase classifier, one piece-value source; MCP de-coupling; wire the chess UI
module: src/games/chess/
status: draft
---

# Goal

The chess package carries two same-named RoutingDecision dataclasses, three divergent phase
classifiers, four piece-value tables on different scales, private-member coupling from the
MCP tools, a chess import hard-coded in the generic MCP server, and an orphaned Gradio UI.

# Acceptance Criteria

- AC-1: RoutingDecision from games/chess/meta_controller.py is canonical (llm_chess_engine re-exports it); one phase classifier lives in state.py with settings-backed thresholds; the PR body enumerates every FEN whose classification changed (old -> new).
- AC-2: All piece-value consumers use constants.get_piece_values with an explicit scale parameter; the divergent tables are gone.
- AC-3: mcp_chess_tools uses a public facade (no private-member reaches); the generic MCP server has zero chess imports (register_tools seam); the MCP tool name+schema set is unchanged (parity test).
- AC-4: ui.py is wired as the chess-ui console script and split into render/controller/learning/app modules with injected session state; examples/chess_demo/ and tests/chess_demo/ are deleted with a rollback tag.

# Constraints

- Re-read open specs' module claims first.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
