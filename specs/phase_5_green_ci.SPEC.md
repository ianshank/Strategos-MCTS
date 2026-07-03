---
id: phase_5_green_ci
goal: Restore a fully green CI pipeline by fixing the two failing jobs
module: .github/workflows/
phase: "5"
milestone: M5
status: implemented
---

# Goal

Bring the CI Pipeline back to green. Unit tests, lint, mypy, and the coverage gate already pass; two jobs
fail and cascade into the `CI Summary` gate: the ADK integration test and the Docker Trivy SARIF upload.

# Acceptance Criteria

- AC-1: `tests/integration/google_adk/test_adk_base.py::test_factory_creates_registered_agent` passes: its local
  `FactoryTestAgent` accepts the `agent_name` keyword the factory supplies (`ADKAgentFactory.create` at
  `src/integrations/google_adk/base.py`), matching the real `ADKAgentAdapter.__init__(config, agent_name)`
  contract. No production source signature changes.
- AC-2: Any other stale `__init__(self, config)` subclass in `tests/integration/google_adk/` is fixed identically.
- AC-3: The Docker job's "Upload Trivy scan results" step no longer fails the pipeline: the job declares
  `permissions: security-events: write` so SARIF upload succeeds, or falls back to `continue-on-error: true`
  if code scanning is unavailable. The scan stays advisory and its results remain visible as an artifact.
- AC-4: The `CI Summary` gate reports success on the branch after these fixes.

# Constraints

- Do not weaken any real quality gate (lint, mypy, unit tests, coverage) — only the advisory SARIF upload
  becomes non-blocking, and only if the permission fix is insufficient.
- The `integration-test` job runs on `main` only, so verify the test fix locally
  (`pytest tests/integration/google_adk/test_adk_base.py`) before relying on post-merge CI.
- Backward compatible; no hardcoded values; full local gate green before push.
