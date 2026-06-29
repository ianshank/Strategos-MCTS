"""Tests for the shared LLM resilience primitives.

`CircuitBreaker` was extracted from ``openai_client`` into ``resilience`` so
every adapter shares one implementation. These tests cover the new module's
public location and the backward-compatible re-export that existing imports
(``openai_client.CircuitBreaker``, ``anthropic_client``) still rely on.
"""

from src.adapters.llm.resilience import CircuitBreaker


def test_circuit_breaker_importable_from_resilience():
    cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.0)
    assert cb.state == "closed"
    assert cb.can_execute() is True


def test_circuit_breaker_reexported_from_openai_client_is_same_class():
    """Back-compat: openai_client must re-export the exact same class object."""
    from src.adapters.llm.openai_client import CircuitBreaker as ReexportedFromOpenAI

    assert ReexportedFromOpenAI is CircuitBreaker


def test_circuit_breaker_used_by_anthropic_client_is_same_class():
    """The Anthropic client must use the shared implementation, not a copy."""
    import src.adapters.llm.anthropic_client as anthropic_client

    assert anthropic_client.CircuitBreaker is CircuitBreaker


def test_get_reset_time_zero_when_not_open():
    cb = CircuitBreaker()
    assert cb.get_reset_time() == 0.0


def test_get_reset_time_positive_when_open():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
    cb.record_failure()
    assert cb.state == "open"
    assert cb.get_reset_time() > 0.0
