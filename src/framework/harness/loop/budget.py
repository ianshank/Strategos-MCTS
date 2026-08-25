"""Token, time, and iteration budgets for the harness control loop.

Budgets are *typed* and *monotonic*: every consumer holds a reference to a
shared budget object, calls :meth:`consume`, and the runner enforces the
limit. There are no global counters and no implicit clocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Protocol


class Clock(Protocol):
    """Minimal clock interface; satisfied by ``time.monotonic``."""

    def __call__(self) -> float: ...


@dataclass
class Budget:
    """Monotonically-consumed numeric budget."""

    limit: float
    consumed: float = 0.0
    name: str = "budget"

    def remaining(self) -> float:
        """How much is left."""
        return max(0.0, self.limit - self.consumed)

    def exhausted(self) -> bool:
        """``True`` once ``consumed >= limit``."""
        return self.consumed >= self.limit

    def consume(self, amount: float) -> None:
        """Consume ``amount`` units. Negative amounts are rejected."""
        if amount < 0:
            raise ValueError(f"{self.name} budget cannot consume negative amount {amount}")
        self.consumed += amount


@dataclass
class TimeBudget:
    """Wall-clock budget anchored on a monotonic clock."""

    limit_seconds: float
    started_at: float = field(default_factory=time.monotonic)
    clock: Clock = field(default=time.monotonic)
    name: str = "time"

    def elapsed(self) -> float:
        """Seconds since the budget was created (or last reset)."""
        return self.clock() - self.started_at

    def remaining(self) -> float:
        """Seconds left."""
        return max(0.0, self.limit_seconds - self.elapsed())

    def exhausted(self) -> bool:
        """``True`` if elapsed exceeds ``limit_seconds``."""
        return self.elapsed() >= self.limit_seconds

    def reset(self) -> None:
        """Restart the timer (used by Ralph loop on each outer cycle)."""
        self.started_at = self.clock()


@dataclass
class IterationBudget:
    """Counts iterations of the control loop."""

    limit: int
    consumed: int = 0
    name: str = "iterations"

    def consume(self, amount: int = 1) -> None:
        """Increment the counter by ``amount`` (default 1)."""
        if amount < 0:
            raise ValueError("iteration budget cannot consume negative amount")
        self.consumed += amount

    def exhausted(self) -> bool:
        """``True`` once ``consumed >= limit``."""
        return self.consumed >= self.limit

    def remaining(self) -> int:
        """Iterations left."""
        return max(0, self.limit - self.consumed)


@dataclass
class BudgetBundle:
    """Convenience aggregate; the runner holds one of these per task."""

    tokens: Budget
    time: TimeBudget
    iterations: IterationBudget

    def first_exhausted(self) -> str | None:
        """Return the name of the first exhausted budget, or ``None``."""
        for budget in (self.tokens, self.time, self.iterations):
            if budget.exhausted():
                return budget.name
        return None


__all__ = ["Budget", "BudgetBundle", "Clock", "IterationBudget", "TimeBudget"]
