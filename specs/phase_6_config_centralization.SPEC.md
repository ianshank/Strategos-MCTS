---
goal: Centralize every hardcoded configuration value into the shared config infrastructure
phase: "6"
milestone: M5
status: active
---

# Goal

Eliminate scattered hardcoded values by sourcing them from `src/config/constants.py` (Final-typed defaults
and bounds) and `src/config/settings.py` (user-overridable Pydantic fields), reusing the fallback accessor
pattern from `src/games/chess/constants.py` (`def get_x(settings: Settings | None = None)`). Public names are
preserved so the change is backward compatible.

# Acceptance Criteria

- Scattered LLM temperature defaults are defined once in `constants.py` and referenced by
  `src/framework/mcts/llm_mcts.py`, `src/framework/agents/llm_hrm.py`, `src/framework/agents/llm_trm.py`,
  and `src/games/chess/llm_chess_engine.py`.
- Import-time `os.getenv()` reads in `src/framework/mcts/llm_guided/constants.py` are replaced by lazy
  Settings lookups (new fields `MCTS_GENERATOR_MODEL`, `MCTS_REFLECTOR_MODEL`, `MCTS_EXECUTION_TIMEOUT`,
  `MCTS_MAX_MEMORY_MB` in `settings.py`); every existing import site keeps working.
- Chess routing weights / phase thresholds / Elo default, assembly-router confidence scores, Google ADK
  defaults, and the re-hardcoded LMStudio URL (`src/adapters/llm/__init__.py`) all resolve from
  constants/settings rather than inline literals.
- Each centralized value has a unit test asserting the module reads the constant and that a settings
  override propagates.
- Intentional backward-compat code (gated pickle migration, model aliases, the CircuitBreaker re-export)
  is retained and annotated as intentional so future scans do not re-flag it.

# Constraints

- No public API/signature changes; modules keep their existing exported names.
- Audit every import site with `git grep` before converting a module-level constant to a settings lookup.
- Backward compatible; no hardcoded values; full local gate green before push.
