---
id: hygiene_config_consolidation
goal: One settings system: shared base config, sub-settings shims, chess shadow config folded in
module: src/config/
status: draft
---

# Goal

Four-plus independent settings singletons duplicate env_file boilerplate; chess config.py
reads the CHESS_ env prefix via raw os.getenv beside the Pydantic fields that own it, with
assert-based validation.

# Acceptance Criteria

- AC-1: One shared model_config base removes the five env_file boilerplate copies; per-module get_*_settings() remain as delegating shims.
- AC-2: Chess shadow config fields move to the Pydantic CHESS_ section with lazy accessors (constants.py pattern); assert validation becomes pydantic validators.
- AC-3: Dead settings fields are removed after a final grep; a permanent guard test asserts no duplicate env-var names across all Settings classes; env-precedence tests use the settings_override fixture.

# Constraints

- Re-read open specs' module claims first; no mass SCREAMING/snake case rename.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
