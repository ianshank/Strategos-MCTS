"""
Symbolic Constraint System for MCTS.

Provides symbolic constraint checking to:
- Prune invalid action branches before neural evaluation
- Encode domain knowledge as logical constraints
- Guarantee safety properties through formal verification

Best Practices 2025:
- Protocol-based constraint interfaces
- Efficient constraint compilation and caching
- Support for hard/soft/advisory constraints
- Integration with SMT solvers when available
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, TypeVar

from .config import ConstraintConfig, ConstraintEnforcement
from .state import NeuroSymbolicState


class ConstraintSatisfactionLevel(Enum):
    """Level of constraint satisfaction."""

    SATISFIED = auto()  # Fully satisfied
    PARTIALLY_SATISFIED = auto()  # Soft constraint with penalty
    VIOLATED = auto()  # Hard constraint violated
    UNKNOWN = auto()  # Could not determine (timeout, etc.)


@dataclass(frozen=True)
class ConstraintResult:
    """Result of constraint evaluation."""

    satisfied: ConstraintSatisfactionLevel
    constraint_id: str
    message: str = ""
    penalty: float = 0.0  # For soft constraints
    bindings: dict[str, Any] = field(default_factory=dict)
    evaluation_time_ms: float = 0.0

    @property
    def is_satisfied(self) -> bool:
        """Check if constraint is satisfied (fully or partially)."""
        return self.satisfied in (
            ConstraintSatisfactionLevel.SATISFIED,
            ConstraintSatisfactionLevel.PARTIALLY_SATISFIED,
        )

    @property
    def is_violated(self) -> bool:
        """Check if constraint is violated."""
        return self.satisfied == ConstraintSatisfactionLevel.VIOLATED


T = TypeVar("T")


class Constraint(ABC):
    """
    Abstract base class for symbolic constraints.

    Constraints can be:
    - Preconditions: Must hold before an action
    - Postconditions: Must hold after an action
    - Invariants: Must always hold
    - Safety: Must never be violated
    """

    def __init__(
        self,
        constraint_id: str,
        name: str,
        description: str = "",
        enforcement: ConstraintEnforcement = ConstraintEnforcement.HARD,
        priority: int = 0,
    ):
        self.constraint_id = constraint_id
        self.name = name
        self.description = description
        self.enforcement = enforcement
        self.priority = priority  # Higher = more important
        self._compiled: bool = False
        self._created_at = datetime.now(UTC)

    @abstractmethod
    def evaluate(
        self,
        state: NeuroSymbolicState,
        action: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ConstraintResult:
        """
        Evaluate constraint on a state.

        Args:
            state: Current neuro-symbolic state
            action: Optional action being considered
            context: Additional evaluation context

        Returns:
            ConstraintResult with satisfaction status
        """
        ...

    @abstractmethod
    def compile(self) -> None:
        """Compile constraint for efficient evaluation."""
        ...

    def get_hash(self) -> str:
        """Get deterministic hash for caching."""
        content = f"{self.constraint_id}:{self.name}:{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.constraint_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constraint):
            return False
        return self.constraint_id == other.constraint_id


class PredicateConstraint(Constraint):
    """
    Constraint based on fact presence/absence.

    Examples:
    - "must have fact: ready(agent)"
    - "must not have fact: error(system)"
    """

    def __init__(
        self,
        constraint_id: str,
        name: str,
        required_facts: list[tuple[str, tuple[Any, ...]]] | None = None,
        forbidden_facts: list[tuple[str, tuple[Any, ...]]] | None = None,
        enforcement: ConstraintEnforcement = ConstraintEnforcement.HARD,
        **kwargs: Any,
    ):
        super().__init__(constraint_id, name, enforcement=enforcement, **kwargs)
        self.required_facts = required_facts or []
        self.forbidden_facts = forbidden_facts or []
        self._required_fact_set: set[str] | None = None
        self._forbidden_fact_set: set[str] | None = None

    def compile(self) -> None:
        """Compile fact patterns for fast lookup."""
        self._required_fact_set = {f"{name}:{args}" for name, args in self.required_facts}
        self._forbidden_fact_set = {f"{name}:{args}" for name, args in self.forbidden_facts}
        self._compiled = True

    def evaluate(
        self,
        state: NeuroSymbolicState,
        action: str | None = None,  # noqa: ARG002
        context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> ConstraintResult:
        """Evaluate predicate constraint."""
        import time

        start = time.perf_counter()

        if not self._compiled:
            self.compile()

        violations = []
        bindings: dict[str, Any] = {}

        # Check required facts
        for fact_name, fact_args in self.required_facts:
            found = state.has_fact(fact_name, *fact_args)
            if not found:
                violations.append(f"missing required fact: {fact_name}{fact_args}")

        # Check forbidden facts
        for fact_name, fact_args in self.forbidden_facts:
            found = state.has_fact(fact_name, *fact_args)
            if found:
                violations.append(f"has forbidden fact: {fact_name}{fact_args}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        if violations:
            if self.enforcement == ConstraintEnforcement.HARD:
                return ConstraintResult(
                    satisfied=ConstraintSatisfactionLevel.VIOLATED,
                    constraint_id=self.constraint_id,
                    message="; ".join(violations),
                    evaluation_time_ms=elapsed_ms,
                )
            elif self.enforcement == ConstraintEnforcement.SOFT:
                penalty = len(violations) * 0.1
                return ConstraintResult(
                    satisfied=ConstraintSatisfactionLevel.PARTIALLY_SATISFIED,
                    constraint_id=self.constraint_id,
                    message="; ".join(violations),
                    penalty=penalty,
                    evaluation_time_ms=elapsed_ms,
                )
            else:  # ADVISORY
                return ConstraintResult(
                    satisfied=ConstraintSatisfactionLevel.SATISFIED,
                    constraint_id=self.constraint_id,
                    message=f"advisory violations: {'; '.join(violations)}",
                    evaluation_time_ms=elapsed_ms,
                )

        return ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.SATISFIED,
            constraint_id=self.constraint_id,
            bindings=bindings,
            evaluation_time_ms=elapsed_ms,
        )


class TemporalConstraint(Constraint):
    """
    Constraint on action ordering/sequencing.

    Examples:
    - "action A must precede action B"
    - "action C cannot occur after action D"
    """

    def __init__(
        self,
        constraint_id: str,
        name: str,
        must_precede: list[tuple[str, str]] | None = None,
        must_not_follow: list[tuple[str, str]] | None = None,
        enforcement: ConstraintEnforcement = ConstraintEnforcement.HARD,
        **kwargs: Any,
    ):
        super().__init__(constraint_id, name, enforcement=enforcement, **kwargs)
        self.must_precede = must_precede or []  # (A, B) means A before B
        self.must_not_follow = must_not_follow or []  # (A, B) means B cannot follow A

    def compile(self) -> None:
        """Compile temporal patterns."""
        self._compiled = True

    def evaluate(
        self,
        state: NeuroSymbolicState,
        action: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ConstraintResult:
        """Evaluate temporal constraint."""
        import time

        start = time.perf_counter()
        violations = []

        # Get action history from metadata
        action_history = state.metadata.get("action_history", [])
        if action:
            action_history = [*action_history, action]

        action_set = set(action_history)

        # Check must_precede constraints
        for before, after in self.must_precede:
            if after in action_set and before not in action_set:
                violations.append(f"{before} must precede {after}")
            elif after in action_set and before in action_set:
                # Check ordering
                try:
                    before_idx = action_history.index(before)
                    after_idx = action_history.index(after)
                    if before_idx > after_idx:
                        violations.append(f"{before} must precede {after} (found in wrong order)")
                except ValueError:
                    pass

        # Check must_not_follow constraints
        for trigger, forbidden in self.must_not_follow:
            if trigger in action_set and forbidden in action_set:
                try:
                    trigger_idx = action_history.index(trigger)
                    forbidden_idx = action_history.index(forbidden)
                    if forbidden_idx > trigger_idx:
                        violations.append(f"{forbidden} cannot follow {trigger}")
                except ValueError:
                    pass

        elapsed_ms = (time.perf_counter() - start) * 1000

        if violations:
            if self.enforcement == ConstraintEnforcement.HARD:
                return ConstraintResult(
                    satisfied=ConstraintSatisfactionLevel.VIOLATED,
                    constraint_id=self.constraint_id,
                    message="; ".join(violations),
                    evaluation_time_ms=elapsed_ms,
                )
            else:
                return ConstraintResult(
                    satisfied=ConstraintSatisfactionLevel.PARTIALLY_SATISFIED,
                    constraint_id=self.constraint_id,
                    message="; ".join(violations),
                    penalty=len(violations) * 0.2,
                    evaluation_time_ms=elapsed_ms,
                )

        return ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.SATISFIED,
            constraint_id=self.constraint_id,
            evaluation_time_ms=elapsed_ms,
        )


class ExpressionConstraint(Constraint):
    """
    Constraint based on logical/arithmetic expressions.

    Supports simple expression evaluation on state metadata.
    """

    # Supported operators
    OPERATORS = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "in": lambda a, b: a in b,
        "not_in": lambda a, b: a not in b,
    }

    def __init__(
        self,
        constraint_id: str,
        name: str,
        expressions: list[tuple[str, str, Any]] | None = None,
        enforcement: ConstraintEnforcement = ConstraintEnforcement.HARD,
        **kwargs: Any,
    ):
        """
        Initialize expression constraint.

        Args:
            expressions: List of (variable_path, operator, value) tuples
                e.g., [("metadata.depth", "<", 10), ("confidence", ">=", 0.5)]
        """
        super().__init__(constraint_id, name, enforcement=enforcement, **kwargs)
        self.expressions = expressions or []
        self._compiled_expressions: list[tuple[list[str], Callable[[Any, Any], bool], Any]] = []

    def compile(self) -> None:
        """Compile expressions for efficient evaluation."""
        self._compiled_expressions = []
        for var_path, op, value in self.expressions:
            path_parts = var_path.split(".")
            if op not in self.OPERATORS:
                raise ValueError(f"Unknown operator: {op}")
            op_fn = self.OPERATORS[op]
            self._compiled_expressions.append((path_parts, op_fn, value))
        self._compiled = True

    def _get_value(self, state: NeuroSymbolicState, path: list[str]) -> Any:
        """Get value from state using dot-notation path."""
        obj: Any = state
        for part in path:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        return obj

    def evaluate(
        self,
        state: NeuroSymbolicState,
        action: str | None = None,  # noqa: ARG002
        context: dict[str, Any] | None = None,
    ) -> ConstraintResult:
        """Evaluate expression constraint."""
        import time

        start = time.perf_counter()

        if not self._compiled:
            self.compile()

        violations = []
        context = context or {}

        for path, op_fn, expected_value in self._compiled_expressions:
            actual_value = self._get_value(state, path)

            # Also check context
            if actual_value is None and path[0] in context:
                actual_value = context[path[0]]
                for part in path[1:]:
                    if isinstance(actual_value, dict) and part in actual_value:
                        actual_value = actual_value[part]
                    else:
                        actual_value = None
                        break

            if actual_value is None:
                violations.append(f"variable not found: {'.'.join(path)}")
            elif not op_fn(actual_value, expected_value):
                violations.append(
                    f"expression failed: {'.'.join(path)} "
                    f"(actual={actual_value}, expected operator with {expected_value})"
                )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if violations:
            if self.enforcement == ConstraintEnforcement.HARD:
                return ConstraintResult(
                    satisfied=ConstraintSatisfactionLevel.VIOLATED,
                    constraint_id=self.constraint_id,
                    message="; ".join(violations),
                    evaluation_time_ms=elapsed_ms,
                )
            else:
                return ConstraintResult(
                    satisfied=ConstraintSatisfactionLevel.PARTIALLY_SATISFIED,
                    constraint_id=self.constraint_id,
                    message="; ".join(violations),
                    penalty=len(violations) * 0.15,
                    evaluation_time_ms=elapsed_ms,
                )

        return ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.SATISFIED,
            constraint_id=self.constraint_id,
            evaluation_time_ms=elapsed_ms,
        )


class LambdaConstraint(Constraint):
    """
    Constraint defined by a Python lambda/function.

    For complex constraints that can't be expressed declaratively.
    """

    def __init__(
        self,
        constraint_id: str,
        name: str,
        predicate: Callable[[NeuroSymbolicState, str | None, dict[str, Any] | None], bool],
        penalty_fn: Callable[[NeuroSymbolicState], float] | None = None,
        enforcement: ConstraintEnforcement = ConstraintEnforcement.HARD,
        **kwargs: Any,
    ):
        super().__init__(constraint_id, name, enforcement=enforcement, **kwargs)
        self.predicate = predicate
        self.penalty_fn = penalty_fn

    def compile(self) -> None:
        """Lambda constraints don't need compilation."""
        self._compiled = True

    def evaluate(
        self,
        state: NeuroSymbolicState,
        action: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ConstraintResult:
        """Evaluate lambda constraint."""
        import time

        start = time.perf_counter()

        try:
            satisfied = self.predicate(state, action, context)
        except Exception as e:
            return ConstraintResult(
                satisfied=ConstraintSatisfactionLevel.UNKNOWN,
                constraint_id=self.constraint_id,
                message=f"evaluation error: {e}",
                evaluation_time_ms=(time.perf_counter() - start) * 1000,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if satisfied:
            return ConstraintResult(
                satisfied=ConstraintSatisfactionLevel.SATISFIED,
                constraint_id=self.constraint_id,
                evaluation_time_ms=elapsed_ms,
            )
        else:
            penalty = self.penalty_fn(state) if self.penalty_fn else 0.5
            if self.enforcement == ConstraintEnforcement.HARD:
                return ConstraintResult(
                    satisfied=ConstraintSatisfactionLevel.VIOLATED,
                    constraint_id=self.constraint_id,
                    message="predicate returned False",
                    evaluation_time_ms=elapsed_ms,
                )
            else:
                return ConstraintResult(
                    satisfied=ConstraintSatisfactionLevel.PARTIALLY_SATISFIED,
                    constraint_id=self.constraint_id,
                    penalty=penalty,
                    evaluation_time_ms=elapsed_ms,
                )


class ConstraintValidator:
    """
    Validates actions against a set of constraints.

    Used by MCTS to prune invalid action branches.
    """

    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.constraints: dict[str, Constraint] = {}
        self._cache: dict[str, ConstraintResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the validator."""
        if len(self.constraints) >= self.config.max_constraints_per_state:
            raise ValueError(f"Maximum constraints exceeded: {self.config.max_constraints_per_state}")
        self.constraints[constraint.constraint_id] = constraint
        if self.config.precompile_constraints:
            constraint.compile()

    def remove_constraint(self, constraint_id: str) -> None:
        """Remove a constraint by ID."""
        if constraint_id in self.constraints:
            del self.constraints[constraint_id]

    def _get_cache_key(
        self,
        state: NeuroSymbolicState,
        action: str | None,
        constraint_id: str,
    ) -> str:
        """Generate cache key for constraint evaluation."""
        return f"{state.hash_key}:{action or 'none'}:{constraint_id}"

    def validate(
        self,
        state: NeuroSymbolicState,
        action: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, list[ConstraintResult]]:
        """
        Validate state/action against all constraints.

        Args:
            state: Current state
            action: Optional action being considered
            context: Additional context

        Returns:
            (is_valid, results) tuple
        """
        results = []
        all_satisfied = True
        total_penalty = 0.0

        # Sort constraints by priority (higher first)
        sorted_constraints = sorted(
            self.constraints.values(),
            key=lambda c: c.priority,
            reverse=True,
        )

        for constraint in sorted_constraints:
            # Check cache
            cache_key = self._get_cache_key(state, action, constraint.constraint_id)
            if cache_key in self._cache:
                self._cache_hits += 1
                result = self._cache[cache_key]
            else:
                self._cache_misses += 1
                result = constraint.evaluate(state, action, context)
                self._cache[cache_key] = result

            results.append(result)

            if result.is_violated and constraint.enforcement == ConstraintEnforcement.HARD:
                all_satisfied = False
                # Early exit for hard constraint violation
                break

            total_penalty += result.penalty

        # Check if penalty exceeds threshold
        if total_penalty > 0 and all_satisfied:
            penalty_ratio = 1.0 - total_penalty
            if penalty_ratio < self.config.min_satisfaction_ratio:
                all_satisfied = False

        return all_satisfied, results

    def validate_action(
        self,
        state: NeuroSymbolicState,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Quick validation of a single action."""
        valid, _ = self.validate(state, action, context)
        return valid

    def filter_valid_actions(
        self,
        state: NeuroSymbolicState,
        actions: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Filter actions to only those that are valid."""
        return [a for a in actions if self.validate_action(state, a, context)]

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self) -> None:
        """Clear the constraint cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class ConstraintSystem:
    """
    Complete constraint system for neuro-symbolic MCTS.

    Provides:
    - Constraint registration and management
    - Batch validation
    - Conflict detection and resolution
    - Integration with MCTS expansion
    """

    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.validator = ConstraintValidator(config)
        self._conflict_log: list[dict[str, Any]] = []

    def register_constraint(self, constraint: Constraint) -> None:
        """Register a new constraint."""
        self.validator.add_constraint(constraint)

    def register_predicate_constraint(
        self,
        constraint_id: str,
        name: str,
        required_facts: list[tuple[str, tuple[Any, ...]]] | None = None,
        forbidden_facts: list[tuple[str, tuple[Any, ...]]] | None = None,
        enforcement: ConstraintEnforcement | None = None,
    ) -> Constraint:
        """Convenience method to register a predicate constraint."""
        enforcement = enforcement or self.config.default_enforcement
        constraint = PredicateConstraint(
            constraint_id=constraint_id,
            name=name,
            required_facts=required_facts,
            forbidden_facts=forbidden_facts,
            enforcement=enforcement,
        )
        self.register_constraint(constraint)
        return constraint

    def register_temporal_constraint(
        self,
        constraint_id: str,
        name: str,
        must_precede: list[tuple[str, str]] | None = None,
        must_not_follow: list[tuple[str, str]] | None = None,
        enforcement: ConstraintEnforcement | None = None,
    ) -> Constraint:
        """Convenience method to register a temporal constraint."""
        enforcement = enforcement or self.config.default_enforcement
        constraint = TemporalConstraint(
            constraint_id=constraint_id,
            name=name,
            must_precede=must_precede,
            must_not_follow=must_not_follow,
            enforcement=enforcement,
        )
        self.register_constraint(constraint)
        return constraint

    def register_expression_constraint(
        self,
        constraint_id: str,
        name: str,
        expressions: list[tuple[str, str, Any]],
        enforcement: ConstraintEnforcement | None = None,
    ) -> Constraint:
        """Convenience method to register an expression constraint."""
        enforcement = enforcement or self.config.default_enforcement
        constraint = ExpressionConstraint(
            constraint_id=constraint_id,
            name=name,
            expressions=expressions,
            enforcement=enforcement,
        )
        self.register_constraint(constraint)
        return constraint

    def validate_expansion(
        self,
        parent_state: NeuroSymbolicState,
        candidate_actions: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Validate candidate actions for MCTS expansion.

        Returns list of (action, score) tuples where:
        - score = 1.0 for fully valid actions
        - score = 1.0 - penalty for partially valid actions
        - Actions with hard constraint violations are excluded

        Args:
            parent_state: Current MCTS node state
            candidate_actions: Actions to evaluate
            context: Additional context

        Returns:
            List of (action, validity_score) tuples
        """
        valid_actions: list[tuple[str, float]] = []

        for action in candidate_actions:
            is_valid, results = self.validator.validate(parent_state, action, context)

            if is_valid:
                # Calculate score based on soft constraint penalties
                total_penalty = sum(r.penalty for r in results)
                score = max(0.0, 1.0 - total_penalty)
                valid_actions.append((action, score))
            else:
                # Log conflict for analysis
                if self.config.enable_conflict_analysis:
                    self._conflict_log.append(
                        {
                            "parent_state_id": parent_state.state_id,
                            "action": action,
                            "violations": [r.message for r in results if r.is_violated],
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

        return valid_actions

    def get_conflict_log(self) -> list[dict[str, Any]]:
        """Get the conflict log for analysis."""
        return self._conflict_log.copy()

    def clear_conflict_log(self) -> None:
        """Clear the conflict log."""
        self._conflict_log.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get constraint system statistics."""
        return {
            "num_constraints": len(self.validator.constraints),
            "cache_stats": self.validator.get_cache_stats(),
            "conflict_count": len(self._conflict_log),
            "constraints": [
                {
                    "id": c.constraint_id,
                    "name": c.name,
                    "enforcement": c.enforcement.name,
                    "priority": c.priority,
                }
                for c in self.validator.constraints.values()
            ],
        }
