"""Hook chain — ordered, cost-class-prioritised guardrail pipeline.

Hooks are executed cheap → expensive (stable insertion order within each
cost class). When a hook returns a :class:`HookViolation`, the chain may
short-circuit (the default) or accumulate — the behaviour is per-hook via
``short_circuit`` so individual policies can opt out.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import logging

from src.framework.harness.outcomes import HookViolation
from src.framework.harness.protocols import HookCost, hook_cost_rank
from src.framework.harness.state import HarnessState
from src.observability.logging import get_logger


@dataclass
class HookOutcome:
    """Aggregate result of running a chain end-to-end."""

    violations: list[HookViolation] = field(default_factory=list)
    short_circuited: bool = False

    @property
    def passed(self) -> bool:
        """``True`` if no violation was reported."""
        return not self.violations


class BaseHook:
    """Ergonomic base implementation for :class:`Hook`.

    Subclasses override :meth:`check`. The class enforces ``cost_class`` and
    ``short_circuit`` as instance attributes so callers can introspect them
    without touching the protocol's methods.
    """

    def __init__(
        self,
        name: str,
        *,
        cost_class: HookCost = HookCost.MEDIUM,
        short_circuit: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.name = name
        self.cost_class = cost_class
        self.short_circuit = short_circuit
        self._logger = logger or get_logger(__name__)

    async def __call__(self, state: HarnessState) -> HookViolation | None:
        return await self.check(state)

    async def check(self, state: HarnessState) -> HookViolation | None:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class HookChain:
    """Ordered collection of hooks executed cheapest-first."""

    hooks: list[BaseHook] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: get_logger(__name__))

    def __post_init__(self) -> None:
        self._sort_stable()

    def _sort_stable(self) -> None:
        # Stable sort by cost rank preserves registration order within a class.
        self.hooks.sort(key=lambda h: hook_cost_rank(h.cost_class))

    def add(self, hook: BaseHook) -> None:
        """Register a hook; chain is re-sorted in stable order."""
        self.hooks.append(hook)
        self._sort_stable()

    def names(self) -> list[str]:
        return [h.name for h in self.hooks]

    async def run(self, state: HarnessState) -> HookOutcome:
        """Execute the chain against ``state``. Honours ``short_circuit``."""
        outcome = HookOutcome()
        for hook in self.hooks:
            try:
                violation = await hook(state)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "hook %s raised %s; treating as violation: %s",
                    hook.name,
                    type(exc).__name__,
                    exc,
                )
                violation = HookViolation(
                    hook_name=hook.name,
                    detail=f"hook raised {type(exc).__name__}: {exc}",
                )
            if violation is None:
                continue
            outcome.violations.append(violation)
            if hook.short_circuit:
                outcome.short_circuited = True
                return outcome
        return outcome

    @classmethod
    def of(cls, *hooks: BaseHook) -> HookChain:
        """Construct a chain from positional arguments."""
        chain = cls(list(hooks))
        return chain

    def __iter__(self) -> Iterator[BaseHook]:
        return iter(self.hooks)


__all__ = ["BaseHook", "HookChain", "HookOutcome"]
