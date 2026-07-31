---
id: hygiene_rest_split
goal: Split rest_server.py into an app-factory package with a compatibility shim
module: src/api/
status: draft
---

# Goal

rest_server.py mixes import-time capability probes, Pydantic models, lifespan wiring,
FastAPI+CORS construction at import time, auth, and nine routes; tests must reload the module
to change settings.

# Acceptance Criteria

- AC-1: src/api/rest/ package provides models/deps/lifespan/routes and create_app(settings=None); module-level get_settings() at import time is gone.
- AC-2: rest_server.py remains as a shim (app = create_app() plus re-exports) so uvicorn targets and imports keep working.
- AC-3: Route parity: the (method, path) set and the OpenAPI schema snapshot are unchanged; the collection-gated server suites now pass green against create_app().

# Constraints

- /security-review runs on this PR (auth dependency moves).
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
