---
id: hygiene_llm_base
goal: Extract the shared LLM client base; LMStudio subclasses OpenAI; one mock shape; protocol conformance suite
module: src/adapters/llm/
status: draft
---

# Goal

The three LLM clients duplicate transport construction, close(), the error-status ladder,
retry decoration, and circuit-breaker wrapping; LMStudio re-implements the OpenAI wire format
with degraded error mapping; three incompatible mock clients exist.

# Acceptance Criteria

- AC-1: BaseLLMClient hosts _get_client/close/_handle_error_response/retry decorator/circuit-breaker wrap; retry knobs reconcile with the existing HTTP_MAX_RETRIES settings field and constants (no duplicate semantics; aliases where renamed).
- AC-2: LMStudioClient subclasses OpenAIClient and regains 401/429/404 error mapping; the byte-identical SSE loop exists once.
- AC-3: A parametrized protocol-conformance suite (keyword-only async generate returning LLMResponse) runs against all clients (httpx MockTransport) and all mocks, permanently.
- AC-4: Structured retry/breaker-transition logs with correlation IDs.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
