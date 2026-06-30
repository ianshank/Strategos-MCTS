---
goal: Expose existing streaming, graph-visualization, comparison, and MCTS early-termination capabilities via testable services, REST endpoints, and the Gradio UI
phase: "4"
milestone: M4
status: active
---

# Goal

Make the already-implemented framework capabilities user-facing: MCTS early termination (wired through the
graph), LangGraph streaming, graph visualization, and MCTS-vs-single-shot comparison — surfaced through
coverage-bearing service modules, thin REST endpoints, and the existing Gradio app. All additive and
backward compatible.

# Acceptance Criteria

- MCTS early termination is config-driven through the graph builder path, gated by an
  `enable_early_termination` flag that defaults to current behavior; a test asserts the termination
  reason/early-stop stats propagate when enabled.
- Streaming, graph-visualization, and comparison logic live in coverage-bearing service modules
  (`src/api/streaming.py`, `src/api/graph_service.py`, comparison service) with unit tests that mock all
  I/O; REST endpoints are thin adapters delegating to them.
- The single source of truth for early-termination thresholds remains `MCTSConfig` (no duplicated
  settings values).
- The Gradio app (`app.py`) exposes comparison + streaming + tree visualization via those services; a new
  `[ui]` extra declares `gradio`; the module imports without the extra installed.
- New behavior is opt-in via settings flags; existing `/query`, demo, and default config are unchanged.

# Constraints

- Backward compatible; no hardcoded values (thresholds via `MCTSConfig`, flags via settings); structured
  logging where relevant.
- No real network/API calls in unit tests; UI tests remain e2e-gated and do not count toward the gate.
- Full local gate (black/ruff/mypy/pytest ≥85%) green before push.
