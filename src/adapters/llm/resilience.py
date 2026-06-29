"""
Resilience primitives shared across LLM client adapters.

Currently houses the :class:`CircuitBreaker` used by the OpenAI- and
Anthropic-compatible clients to prevent cascading failures. Kept in a
provider-agnostic module so every adapter imports a single implementation.
"""

import time


class CircuitBreaker:
    """Simple circuit breaker implementation for resilience."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if reset timeout has passed
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.state = "half-open"
                # Count this transition call as the first half-open trial so
                # half_open_max_calls is actually enforced from the start.
                self.half_open_calls = 1
                return True
            return False

        if self.state == "half-open":
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def record_success(self) -> None:
        """Record successful request."""
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
        elif self.state == "closed":
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "half-open" or self.failure_count >= self.failure_threshold:
            self.state = "open"

    def get_reset_time(self) -> float:
        """Get time until circuit resets."""
        if self.state != "open":
            return 0.0
        elapsed = time.time() - self.last_failure_time
        return max(0, self.reset_timeout - elapsed)
