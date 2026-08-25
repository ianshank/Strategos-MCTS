"""Recording / deterministic clock used to funnel all nondeterminism.

Every phase function and every tool that needs the wall clock, a UUID, or a
random number obtains it through a :class:`RecordingClock`. This single point
of contact makes deterministic replay achievable: in record mode the clock
emits real values; in replay mode it deals out values previously captured.
"""

from __future__ import annotations

from datetime import UTC, datetime
import random
import time
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4


@runtime_checkable
class RecordingClock(Protocol):
    """Protocol for clocks providing wall time, UUIDs, and randomness."""

    def now(self) -> datetime: ...
    def monotonic(self) -> float: ...
    def uuid(self) -> str: ...
    def random(self) -> float: ...


class SystemClock:
    """Real clock — wraps stdlib primitives."""

    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def uuid(self) -> str:
        return str(uuid4())

    def random(self) -> float:
        return random.random()


class DeterministicClock:
    """Reproducible clock — frozen time, counter UUIDs, seeded RNG.

    Used by replay tests and by the harness when ``HARNESS_DETERMINISTIC_CLOCK``
    is enabled.
    """

    _NAMESPACE = UUID("00000000-0000-4000-8000-000000000000")

    def __init__(
        self,
        seed: int = 0,
        start: datetime | None = None,
        tick_seconds: float = 0.001,
    ) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._start = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._tick = tick_seconds
        self._counter = 0
        self._monotonic = 0.0

    def _step(self) -> None:
        self._counter += 1
        self._monotonic += self._tick

    def now(self) -> datetime:
        self._step()
        from datetime import timedelta

        return self._start + timedelta(seconds=self._monotonic)

    def monotonic(self) -> float:
        self._step()
        return self._monotonic

    def uuid(self) -> str:
        self._step()
        # UUIDv5-style derivation from a counter — stable across runs with the
        # same seed without colliding across distinct counters.
        from uuid import uuid5

        return str(uuid5(self._NAMESPACE, f"{self._seed}:{self._counter}"))

    def random(self) -> float:
        self._step()
        return self._rng.random()


__all__ = ["DeterministicClock", "RecordingClock", "SystemClock"]
