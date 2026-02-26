"""
Unit tests for symbolic constraint system.

Tests:
- Constraint creation and evaluation
- Constraint validation
- Constraint system operations
- MCTS integration scenarios

Best Practices 2025:
- Comprehensive constraint type coverage
- Performance benchmarks
- Edge case handling
"""

import pytest

from src.neuro_symbolic.config import ConstraintConfig, ConstraintEnforcement
from src.neuro_symbolic.constraints import (
    ConstraintResult,
    ConstraintSatisfactionLevel,
    ConstraintSystem,
    ConstraintValidator,
    ExpressionConstraint,
    LambdaConstraint,
    PredicateConstraint,
    TemporalConstraint,
)
from src.neuro_symbolic.state import Fact, NeuroSymbolicState


class TestConstraintResult:
    """Tests for ConstraintResult dataclass."""

    def test_result_satisfied(self):
        """Test satisfied result properties."""
        result = ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.SATISFIED,
            constraint_id="c1",
        )

        assert result.is_satisfied
        assert not result.is_violated
        assert result.penalty == 0.0

    def test_result_violated(self):
        """Test violated result properties."""
        result = ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.VIOLATED,
            constraint_id="c1",
            message="constraint failed",
        )

        assert not result.is_satisfied
        assert result.is_violated

    def test_result_partially_satisfied(self):
        """Test partially satisfied result."""
        result = ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.PARTIALLY_SATISFIED,
            constraint_id="c1",
            penalty=0.3,
        )

        assert result.is_satisfied
        assert not result.is_violated
        assert result.penalty == 0.3


class TestPredicateConstraint:
    """Tests for PredicateConstraint."""

    def test_required_fact_satisfied(self):
        """Test constraint satisfied when required fact present."""
        fact = Fact(name="ready", arguments=("agent",))
        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([fact]),
        )

        constraint = PredicateConstraint(
            constraint_id="c1",
            name="require_ready",
            required_facts=[("ready", ("agent",))],
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED

    def test_required_fact_missing(self):
        """Test constraint violated when required fact missing."""
        state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        constraint = PredicateConstraint(
            constraint_id="c1",
            name="require_ready",
            required_facts=[("ready", ("agent",))],
            enforcement=ConstraintEnforcement.HARD,
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.VIOLATED
        assert "missing required fact" in result.message

    def test_forbidden_fact_absent_satisfied(self):
        """Test constraint satisfied when forbidden fact absent."""
        state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        constraint = PredicateConstraint(
            constraint_id="c1",
            name="no_error",
            forbidden_facts=[("error", ("system",))],
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED

    def test_forbidden_fact_present_violated(self):
        """Test constraint violated when forbidden fact present."""
        fact = Fact(name="error", arguments=("system",))
        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([fact]),
        )

        constraint = PredicateConstraint(
            constraint_id="c1",
            name="no_error",
            forbidden_facts=[("error", ("system",))],
            enforcement=ConstraintEnforcement.HARD,
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.VIOLATED
        assert "forbidden fact" in result.message

    def test_soft_constraint_partial_satisfaction(self):
        """Test soft constraint returns partial satisfaction."""
        state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        constraint = PredicateConstraint(
            constraint_id="c1",
            name="prefer_ready",
            required_facts=[("ready", ("agent",))],
            enforcement=ConstraintEnforcement.SOFT,
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.PARTIALLY_SATISFIED
        assert result.penalty > 0

    def test_advisory_constraint_always_satisfied(self):
        """Test advisory constraint reports satisfaction with message."""
        state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        constraint = PredicateConstraint(
            constraint_id="c1",
            name="advisory_ready",
            required_facts=[("ready", ("agent",))],
            enforcement=ConstraintEnforcement.ADVISORY,
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED
        assert "advisory" in result.message

    def test_constraint_compilation(self):
        """Test constraint compilation for efficiency."""
        constraint = PredicateConstraint(
            constraint_id="c1",
            name="test",
            required_facts=[("a", (1,)), ("b", (2,))],
            forbidden_facts=[("c", (3,))],
        )

        assert not constraint._compiled
        constraint.compile()
        assert constraint._compiled


class TestTemporalConstraint:
    """Tests for TemporalConstraint."""

    def test_must_precede_satisfied(self):
        """Test must_precede satisfied when order correct."""
        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"action_history": ["init", "prepare", "execute"]},
        )

        constraint = TemporalConstraint(
            constraint_id="c1",
            name="prep_before_exec",
            must_precede=[("prepare", "execute")],
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED

    def test_must_precede_violated(self):
        """Test must_precede violated when order wrong."""
        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"action_history": ["init", "execute", "prepare"]},
        )

        constraint = TemporalConstraint(
            constraint_id="c1",
            name="prep_before_exec",
            must_precede=[("prepare", "execute")],
            enforcement=ConstraintEnforcement.HARD,
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.VIOLATED

    def test_must_not_follow_satisfied(self):
        """Test must_not_follow satisfied when condition met."""
        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"action_history": ["error", "recover"]},
        )

        constraint = TemporalConstraint(
            constraint_id="c1",
            name="no_proceed_after_error",
            must_not_follow=[("error", "proceed")],  # proceed doesn't follow error
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED

    def test_must_not_follow_violated(self):
        """Test must_not_follow violated when forbidden sequence occurs."""
        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"action_history": ["error", "proceed"]},
        )

        constraint = TemporalConstraint(
            constraint_id="c1",
            name="no_proceed_after_error",
            must_not_follow=[("error", "proceed")],
            enforcement=ConstraintEnforcement.HARD,
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.VIOLATED

    def test_with_action_parameter(self):
        """Test evaluation with proposed action."""
        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"action_history": ["init"]},
        )

        constraint = TemporalConstraint(
            constraint_id="c1",
            name="require_init_before_start",
            must_precede=[("init", "start")],
        )

        # Propose "start" action - should be satisfied since init already done
        result = constraint.evaluate(state, action="start")

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED


class TestExpressionConstraint:
    """Tests for ExpressionConstraint."""

    def test_comparison_satisfied(self):
        """Test comparison expression satisfied."""
        state = NeuroSymbolicState(
            state_id="s1",
            confidence=0.8,
            metadata={"depth": 5},
        )

        constraint = ExpressionConstraint(
            constraint_id="c1",
            name="depth_limit",
            expressions=[("metadata.depth", "<", 10)],
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED

    def test_comparison_violated(self):
        """Test comparison expression violated."""
        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"depth": 15},
        )

        constraint = ExpressionConstraint(
            constraint_id="c1",
            name="depth_limit",
            expressions=[("metadata.depth", "<", 10)],
            enforcement=ConstraintEnforcement.HARD,
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.VIOLATED

    def test_equality_check(self):
        """Test equality expression."""
        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"status": "ready"},
        )

        constraint = ExpressionConstraint(
            constraint_id="c1",
            name="status_check",
            expressions=[("metadata.status", "==", "ready")],
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED

    def test_state_attribute_access(self):
        """Test accessing state attributes directly."""
        state = NeuroSymbolicState(
            state_id="s1",
            confidence=0.9,
        )

        constraint = ExpressionConstraint(
            constraint_id="c1",
            name="confidence_check",
            expressions=[("confidence", ">=", 0.8)],
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED

    @pytest.mark.parametrize(
        "op,value,expected_satisfied",
        [
            ("==", 10, True),
            ("!=", 5, True),
            ("<", 20, True),
            ("<=", 10, True),
            (">", 5, True),
            (">=", 10, True),
            ("in", [5, 10, 15], True),
            ("not_in", [1, 2, 3], True),
        ],
    )
    def test_all_operators(self, op, value, expected_satisfied):
        """Test all supported operators."""
        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"x": 10},
        )

        constraint = ExpressionConstraint(
            constraint_id="c1",
            name="test",
            expressions=[("metadata.x", op, value)],
        )

        result = constraint.evaluate(state)

        if expected_satisfied:
            assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED
        else:
            assert result.satisfied != ConstraintSatisfactionLevel.SATISFIED


class TestLambdaConstraint:
    """Tests for LambdaConstraint."""

    def test_lambda_satisfied(self):
        """Test lambda constraint satisfied."""
        constraint = LambdaConstraint(
            constraint_id="c1",
            name="custom",
            predicate=lambda state, action, ctx: len(state.facts) > 0,
        )

        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([Fact(name="a", arguments=(1,))]),
        )

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.SATISFIED

    def test_lambda_violated(self):
        """Test lambda constraint violated."""
        constraint = LambdaConstraint(
            constraint_id="c1",
            name="custom",
            predicate=lambda state, action, ctx: len(state.facts) > 0,
            enforcement=ConstraintEnforcement.HARD,
        )

        state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.VIOLATED

    def test_lambda_with_penalty_function(self):
        """Test lambda constraint with custom penalty."""
        constraint = LambdaConstraint(
            constraint_id="c1",
            name="custom",
            predicate=lambda state, action, ctx: False,
            penalty_fn=lambda state: 0.75,
            enforcement=ConstraintEnforcement.SOFT,
        )

        state = NeuroSymbolicState(state_id="s1")

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.PARTIALLY_SATISFIED
        assert result.penalty == 0.75

    def test_lambda_exception_handling(self):
        """Test lambda constraint handles exceptions."""
        constraint = LambdaConstraint(
            constraint_id="c1",
            name="failing",
            predicate=lambda state, action, ctx: 1 / 0,  # Raises ZeroDivisionError
        )

        state = NeuroSymbolicState(state_id="s1")

        result = constraint.evaluate(state)

        assert result.satisfied == ConstraintSatisfactionLevel.UNKNOWN
        assert "evaluation error" in result.message


class TestConstraintValidator:
    """Tests for ConstraintValidator."""

    def test_validator_add_constraint(self):
        """Test adding constraints to validator."""
        config = ConstraintConfig()
        validator = ConstraintValidator(config)

        constraint = PredicateConstraint(
            constraint_id="c1",
            name="test",
            required_facts=[],
        )
        validator.add_constraint(constraint)

        assert "c1" in validator.constraints

    def test_validator_remove_constraint(self):
        """Test removing constraints from validator."""
        config = ConstraintConfig()
        validator = ConstraintValidator(config)

        constraint = PredicateConstraint(
            constraint_id="c1",
            name="test",
            required_facts=[],
        )
        validator.add_constraint(constraint)
        validator.remove_constraint("c1")

        assert "c1" not in validator.constraints

    def test_validator_max_constraints_limit(self):
        """Test validator respects max constraints limit."""
        config = ConstraintConfig(max_constraints_per_state=2)
        validator = ConstraintValidator(config)

        validator.add_constraint(PredicateConstraint(constraint_id="c1", name="t1", required_facts=[]))
        validator.add_constraint(PredicateConstraint(constraint_id="c2", name="t2", required_facts=[]))

        with pytest.raises(ValueError, match="Maximum constraints exceeded"):
            validator.add_constraint(PredicateConstraint(constraint_id="c3", name="t3", required_facts=[]))

    def test_validate_all_satisfied(self):
        """Test validation when all constraints satisfied."""
        config = ConstraintConfig()
        validator = ConstraintValidator(config)

        fact = Fact(name="ready", arguments=())
        state = NeuroSymbolicState(state_id="s1", facts=frozenset([fact]))

        validator.add_constraint(
            PredicateConstraint(
                constraint_id="c1",
                name="require_ready",
                required_facts=[("ready", ())],
            )
        )

        is_valid, results = validator.validate(state)

        assert is_valid
        assert len(results) == 1

    def test_validate_hard_constraint_violation(self):
        """Test validation fails on hard constraint violation."""
        config = ConstraintConfig()
        validator = ConstraintValidator(config)

        state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        validator.add_constraint(
            PredicateConstraint(
                constraint_id="c1",
                name="require_ready",
                required_facts=[("ready", ())],
                enforcement=ConstraintEnforcement.HARD,
            )
        )

        is_valid, results = validator.validate(state)

        assert not is_valid

    def test_validate_action(self):
        """Test quick action validation."""
        config = ConstraintConfig()
        validator = ConstraintValidator(config)

        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"action_history": ["init"]},
        )

        validator.add_constraint(
            TemporalConstraint(
                constraint_id="c1",
                name="order",
                must_precede=[("init", "start")],
            )
        )

        assert validator.validate_action(state, "start")

    def test_filter_valid_actions(self):
        """Test filtering valid actions."""
        config = ConstraintConfig()
        validator = ConstraintValidator(config)

        state = NeuroSymbolicState(
            state_id="s1",
            metadata={"action_history": []},
        )

        validator.add_constraint(
            TemporalConstraint(
                constraint_id="c1",
                name="require_init",
                must_precede=[("init", "start"), ("init", "run")],
                enforcement=ConstraintEnforcement.HARD,
            )
        )

        actions = ["init", "start", "run", "stop"]
        valid = validator.filter_valid_actions(state, actions)

        # Only "init" and "stop" should be valid (start/run require init first)
        assert "init" in valid
        assert "stop" in valid

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        config = ConstraintConfig()
        validator = ConstraintValidator(config)

        state = NeuroSymbolicState(state_id="s1")
        validator.add_constraint(PredicateConstraint(constraint_id="c1", name="test", required_facts=[]))

        # First evaluation - cache miss
        validator.validate(state)
        stats1 = validator.get_cache_stats()
        assert stats1["cache_misses"] == 1

        # Second evaluation - cache hit
        validator.validate(state)
        stats2 = validator.get_cache_stats()
        assert stats2["cache_hits"] == 1

    def test_clear_cache(self):
        """Test cache clearing."""
        config = ConstraintConfig()
        validator = ConstraintValidator(config)

        state = NeuroSymbolicState(state_id="s1")
        validator.add_constraint(PredicateConstraint(constraint_id="c1", name="test", required_facts=[]))

        validator.validate(state)
        validator.clear_cache()

        stats = validator.get_cache_stats()
        assert stats["cache_size"] == 0
        assert stats["cache_hits"] == 0


class TestConstraintSystem:
    """Tests for ConstraintSystem."""

    def test_register_predicate_constraint(self):
        """Test registering predicate constraint."""
        config = ConstraintConfig()
        system = ConstraintSystem(config)

        constraint = system.register_predicate_constraint(
            constraint_id="c1",
            name="test",
            required_facts=[("ready", ())],
        )

        assert constraint.constraint_id == "c1"
        assert "c1" in system.validator.constraints

    def test_register_temporal_constraint(self):
        """Test registering temporal constraint."""
        config = ConstraintConfig()
        system = ConstraintSystem(config)

        constraint = system.register_temporal_constraint(
            constraint_id="c1",
            name="order",
            must_precede=[("a", "b")],
        )

        assert constraint.constraint_id == "c1"

    def test_register_expression_constraint(self):
        """Test registering expression constraint."""
        config = ConstraintConfig()
        system = ConstraintSystem(config)

        constraint = system.register_expression_constraint(
            constraint_id="c1",
            name="limit",
            expressions=[("metadata.depth", "<", 10)],
        )

        assert constraint.constraint_id == "c1"

    def test_validate_expansion(self):
        """Test MCTS expansion validation."""
        config = ConstraintConfig()
        system = ConstraintSystem(config)

        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([Fact(name="ready", arguments=())]),
        )

        system.register_predicate_constraint(
            constraint_id="c1",
            name="require_ready",
            required_facts=[("ready", ())],
        )

        actions = ["action_a", "action_b", "action_c"]
        valid_actions = system.validate_expansion(state, actions)

        # All actions should be valid (state satisfies constraint)
        assert len(valid_actions) == 3
        for _action, score in valid_actions:
            assert score == 1.0

    def test_conflict_logging(self):
        """Test conflict logging for analysis."""
        config = ConstraintConfig(enable_conflict_analysis=True)
        system = ConstraintSystem(config)

        state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        system.register_predicate_constraint(
            constraint_id="c1",
            name="require_ready",
            required_facts=[("ready", ())],
            enforcement=ConstraintEnforcement.HARD,
        )

        # Validation should fail and log conflict
        system.validate_expansion(state, ["action"])

        conflicts = system.get_conflict_log()
        assert len(conflicts) == 1
        assert conflicts[0]["action"] == "action"

    def test_get_statistics(self):
        """Test statistics retrieval."""
        config = ConstraintConfig()
        system = ConstraintSystem(config)

        system.register_predicate_constraint(
            constraint_id="c1",
            name="test",
            required_facts=[],
        )

        stats = system.get_statistics()

        assert stats["num_constraints"] == 1
        assert "cache_stats" in stats
        assert "constraints" in stats
