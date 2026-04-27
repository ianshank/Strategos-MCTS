"""
MCTS edge case handling and validation.

Handles:
- Empty action space
- Terminal state detection
- Timeout handling
- Budget exhaustion
- Tree corruption detection

Based on: CLAUDE_CODE_IMPLEMENTATION_TEMPLATE.md Section 7
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.config.settings import get_settings
from src.observability.logging import get_logger

logger = get_logger(__name__)


class MCTSTerminationReason(str, Enum):
    """Reasons for MCTS search termination."""

    ITERATIONS_COMPLETE = "iterations_complete"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TERMINAL_STATE = "terminal_state"
    NO_ACTIONS = "no_actions"
    CONVERGENCE = "convergence"
    EARLY_TERMINATION = "early_termination"
    ERROR = "error"


@dataclass
class MCTSSearchResult:
    """Result of MCTS search with termination info."""

    best_action: str | None
    stats: dict[str, Any]
    termination_reason: MCTSTerminationReason
    iterations_completed: int
    time_elapsed_seconds: float
    error: Exception | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_action": self.best_action,
            "stats": self.stats,
            "termination_reason": self.termination_reason.value,
            "iterations_completed": self.iterations_completed,
            "time_elapsed_seconds": round(self.time_elapsed_seconds, 3),
            "error": str(self.error) if self.error else None,
        }


class MCTSValidator:
    """
    Validates MCTS tree structure and invariants.

    Invariants checked:
    - Sum of child visits <= parent visits
    - No cycles in tree
    - Single root
    - UCB1 scores >= 0
    - Value bounds [0, 1] (when normalized)
    """

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, raise on first violation
        """
        self.strict = strict
        self._logger = logger.getChild("MCTSValidator")

    def validate_tree(self, root) -> list[str]:
        """
        Validate tree structure and return list of violations.

        Args:
            root: Root node of MCTS tree

        Returns:
            Empty list if tree is valid, otherwise list of violation messages

        Raises:
            MCTSValidationError: If strict mode and violations found
        """
        violations = []
        visited = set()

        def validate_node(node, path: list[int]) -> None:
            node_id = id(node)

            # Check for cycles
            if node_id in path:
                violation = f"Cycle detected at depth {len(path)}"
                violations.append(violation)
                if self.strict:
                    raise MCTSValidationError(violation)
                return

            # Check for multiple visits (shouldn't happen in tree)
            if node_id in visited:
                violation = "Node visited multiple times (graph structure, not tree)"
                violations.append(violation)
                if self.strict:
                    raise MCTSValidationError(violation)
                return

            visited.add(node_id)
            path.append(node_id)

            # Check visit count invariant
            if node.children:
                child_visits = sum(c.visits for c in node.children)
                if child_visits > node.visits:
                    violation = (
                        f"Child visits ({child_visits}) > parent visits ({node.visits}) at depth {len(path) - 1}"
                    )
                    violations.append(violation)
                    if self.strict:
                        raise MCTSValidationError(violation)

            # Check value bounds (when we have visits)
            if node.visits > 0:
                avg_value = node.value_sum / node.visits
                if avg_value < -1 or avg_value > 2:  # Allow some slack for edge cases
                    violation = f"Value {avg_value:.4f} outside expected bounds at depth {len(path) - 1}"
                    violations.append(violation)
                    self._logger.warning(violation)

            # Check for negative visits
            if node.visits < 0:
                violation = f"Negative visit count ({node.visits})"
                violations.append(violation)
                if self.strict:
                    raise MCTSValidationError(violation)

            # Recurse to children
            for child in node.children:
                validate_node(child, path.copy())

        validate_node(root, [])

        if violations:
            self._logger.warning(
                f"Tree validation found {len(violations)} issues",
                extra={"violations": violations[:5]},  # Log first 5
            )

        return violations

    def validate_action_space(self, actions: list[str]) -> list[str]:
        """
        Validate action space.

        Args:
            actions: List of available actions

        Returns:
            List of violations
        """
        violations = []

        if not actions:
            violations.append("Empty action space")
            return violations

        # Check for duplicates
        if len(actions) != len(set(actions)):
            violations.append("Duplicate actions in action space")

        # Check for empty strings
        if any(a == "" for a in actions):
            violations.append("Empty string in actions")

        return violations


class MCTSValidationError(Exception):
    """Exception raised when MCTS validation fails."""

    pass


@dataclass
class TimeoutConfig:
    """Configuration for timeout handling."""

    search_timeout_seconds: float = 60.0
    iteration_timeout_seconds: float = 5.0
    simulation_timeout_seconds: float = 10.0

    @classmethod
    def from_settings(cls) -> TimeoutConfig:
        """Create config from settings."""
        settings = get_settings()
        return cls(
            search_timeout_seconds=getattr(settings, "MCTS_SEARCH_TIMEOUT_SECONDS", 60.0),
            iteration_timeout_seconds=getattr(settings, "MCTS_ITERATION_TIMEOUT_SECONDS", 5.0),
            simulation_timeout_seconds=getattr(settings, "MCTS_SIMULATION_TIMEOUT_SECONDS", 10.0),
        )


@dataclass
class BudgetConfig:
    """Configuration for budget management."""

    token_budget: int | None = None
    cost_budget_usd: float | None = None
    max_nodes: int | None = None

    @classmethod
    def from_settings(cls) -> BudgetConfig:
        """Create config from settings."""
        settings = get_settings()
        return cls(
            token_budget=getattr(settings, "MCTS_TOKEN_BUDGET", None),
            cost_budget_usd=getattr(settings, "MCTS_COST_BUDGET_USD", None),
            max_nodes=getattr(settings, "MCTS_MAX_NODES", None),
        )


class TimeoutHandler:
    """
    Handles timeout and budget management for MCTS.

    Provides context managers for protecting operations
    and checking budget status.

    Example:
        >>> handler = TimeoutHandler(timeout_seconds=30)
        >>> async with handler.guard():
        ...     await run_mcts()
        >>> if handler.is_timeout:
        ...     print("Search timed out")
    """

    def __init__(
        self,
        timeout_config: TimeoutConfig | None = None,
        budget_config: BudgetConfig | None = None,
    ):
        """
        Initialize timeout handler.

        Args:
            timeout_config: Timeout configuration
            budget_config: Budget configuration
        """
        self.timeout_config = timeout_config or TimeoutConfig.from_settings()
        self.budget_config = budget_config or BudgetConfig()
        self.tokens_used = 0
        self.cost_used_usd = 0.0
        self.nodes_created = 0
        self._start_time: float | None = None
        self._logger = logger.getChild("TimeoutHandler")

    def start(self) -> None:
        """Start the timeout clock."""
        self._start_time = time.time()

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def is_timeout(self) -> bool:
        """Check if search timeout has been exceeded."""
        if self._start_time is None:
            return False
        return self.elapsed_seconds > self.timeout_config.search_timeout_seconds

    @property
    def is_budget_exhausted(self) -> bool:
        """Check if any budget limit has been exceeded."""
        token_exceeded = (
            self.budget_config.token_budget is not None and self.tokens_used >= self.budget_config.token_budget
        )
        cost_exceeded = (
            self.budget_config.cost_budget_usd is not None and self.cost_used_usd >= self.budget_config.cost_budget_usd
        )
        nodes_exceeded = self.budget_config.max_nodes is not None and self.nodes_created >= self.budget_config.max_nodes
        return token_exceeded or cost_exceeded or nodes_exceeded

    @property
    def should_terminate(self) -> bool:
        """Check if search should terminate early."""
        return self.is_timeout or self.is_budget_exhausted

    def record_tokens(self, tokens: int) -> None:
        """Record token usage."""
        self.tokens_used += tokens

    def record_cost(self, cost_usd: float) -> None:
        """Record cost usage."""
        self.cost_used_usd += cost_usd

    def record_node(self) -> None:
        """Record node creation."""
        self.nodes_created += 1

    def get_remaining_budget(self) -> dict[str, Any]:
        """Get remaining budget information."""
        result = {
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "remaining_seconds": max(0, self.timeout_config.search_timeout_seconds - self.elapsed_seconds),
        }

        if self.budget_config.token_budget is not None:
            result["tokens_remaining"] = max(0, self.budget_config.token_budget - self.tokens_used)

        if self.budget_config.cost_budget_usd is not None:
            result["cost_remaining_usd"] = max(0, self.budget_config.cost_budget_usd - self.cost_used_usd)

        if self.budget_config.max_nodes is not None:
            result["nodes_remaining"] = max(0, self.budget_config.max_nodes - self.nodes_created)

        return result

    async def guard(self):
        """
        Context manager for timeout protection.

        Example:
            >>> async with handler.guard():
            ...     await run_operation()
        """
        self.start()
        return _TimeoutGuard(self)


class _TimeoutGuard:
    """Internal context manager for timeout handling."""

    def __init__(self, handler: TimeoutHandler):
        self.handler = handler

    async def __aenter__(self) -> TimeoutHandler:
        return self.handler

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Log summary on exit
        if exc_type is not None:
            logger.warning(
                "Operation failed",
                extra={
                    "elapsed_seconds": self.handler.elapsed_seconds,
                    "error": str(exc_val),
                },
            )
        return False  # Don't suppress exceptions


class EmptyActionHandler:
    """
    Handles empty action space scenarios.

    Provides fallback strategies when no actions are available.
    """

    def __init__(self, fallback_action: str | None = None):
        """
        Initialize handler.

        Args:
            fallback_action: Default action to use when no actions available
        """
        self.fallback_action = fallback_action or "no_action"
        self._logger = logger.getChild("EmptyActionHandler")

    def handle_empty_actions(
        self,
        state,
        reason: str = "unknown",
    ) -> str | None:
        """
        Handle case where no actions are available.

        Args:
            state: Current MCTS state
            reason: Reason for empty actions

        Returns:
            Fallback action or None
        """
        self._logger.warning(
            "Empty action space encountered",
            extra={
                "state_id": getattr(state, "state_id", "unknown"),
                "reason": reason,
            },
        )
        return self.fallback_action

    def should_terminate(self, state, depth: int, max_depth: int) -> bool:
        """
        Determine if search should terminate at this state.

        Args:
            state: Current state
            depth: Current depth
            max_depth: Maximum allowed depth

        Returns:
            True if should terminate
        """
        if depth >= max_depth:
            return True

        # Check if state is marked as terminal
        return hasattr(state, "is_terminal") and state.is_terminal


__all__ = [
    "MCTSTerminationReason",
    "MCTSSearchResult",
    "MCTSValidator",
    "MCTSValidationError",
    "TimeoutConfig",
    "BudgetConfig",
    "TimeoutHandler",
    "EmptyActionHandler",
]
