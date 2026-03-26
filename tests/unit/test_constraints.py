"""
Tests for neuro-symbolic constraint system.

Tests ConstraintResult, PredicateConstraint, TemporalConstraint,
ExpressionConstraint, LambdaConstraint, ConstraintValidator,
and ConstraintSystem.
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


def _make_state(facts=None, metadata=None, state_id="test"):
    return NeuroSymbolicState(
        state_id=state_id,
        facts=frozenset(facts or []),
        metadata=metadata or {},
    )


def _make_config(**overrides):
    defaults = dict(
        max_constraints_per_state=100,
        precompile_constraints=True,
        enable_conflict_analysis=True,
        min_satisfaction_ratio=0.8,
    )
    defaults.update(overrides)
    return ConstraintConfig(**defaults)


@pytest.mark.unit
class TestConstraintResult:
    """Tests for ConstraintResult dataclass."""

    def test_satisfied(self):
        r = ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.SATISFIED,
            constraint_id="c1",
        )
        assert r.is_satisfied
        assert not r.is_violated

    def test_partially_satisfied(self):
        r = ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.PARTIALLY_SATISFIED,
            constraint_id="c1",
            penalty=0.1,
        )
        assert r.is_satisfied
        assert not r.is_violated

    def test_violated(self):
        r = ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.VIOLATED,
            constraint_id="c1",
            message="bad",
        )
        assert not r.is_satisfied
        assert r.is_violated

    def test_unknown(self):
        r = ConstraintResult(
            satisfied=ConstraintSatisfactionLevel.UNKNOWN,
            constraint_id="c1",
        )
        assert not r.is_satisfied
        assert not r.is_violated


@pytest.mark.unit
class TestPredicateConstraint:
    """Tests for PredicateConstraint."""

    def test_required_fact_satisfied(self):
        fact = Fact(name="ready", arguments=("agent",))
        state = _make_state(facts=[fact])
        c = PredicateConstraint("p1", "ready check", required_facts=[("ready", ("agent",))])
        result = c.evaluate(state)
        assert result.is_satisfied

    def test_required_fact_missing_hard(self):
        state = _make_state()
        c = PredicateConstraint(
            "p1", "ready check",
            required_facts=[("ready", ("agent",))],
            enforcement=ConstraintEnforcement.HARD,
        )
        result = c.evaluate(state)
        assert result.is_violated

    def test_required_fact_missing_soft(self):
        state = _make_state()
        c = PredicateConstraint(
            "p1", "ready check",
            required_facts=[("ready", ("agent",))],
            enforcement=ConstraintEnforcement.SOFT,
        )
        result = c.evaluate(state)
        assert result.satisfied == ConstraintSatisfactionLevel.PARTIALLY_SATISFIED
        assert result.penalty > 0

    def test_required_fact_missing_advisory(self):
        state = _make_state()
        c = PredicateConstraint(
            "p1", "ready check",
            required_facts=[("ready", ("agent",))],
            enforcement=ConstraintEnforcement.ADVISORY,
        )
        result = c.evaluate(state)
        assert result.is_satisfied
        assert "advisory" in result.message

    def test_forbidden_fact_present(self):
        fact = Fact(name="error", arguments=("system",))
        state = _make_state(facts=[fact])
        c = PredicateConstraint(
            "p2", "no error",
            forbidden_facts=[("error", ("system",))],
            enforcement=ConstraintEnforcement.HARD,
        )
        result = c.evaluate(state)
        assert result.is_violated

    def test_forbidden_fact_absent(self):
        state = _make_state()
        c = PredicateConstraint(
            "p2", "no error",
            forbidden_facts=[("error", ("system",))],
        )
        result = c.evaluate(state)
        assert result.is_satisfied

    def test_compile(self):
        c = PredicateConstraint(
            "p3", "test",
            required_facts=[("a", (1,))],
            forbidden_facts=[("b", (2,))],
        )
        c.compile()
        assert c._compiled
        assert c._required_fact_set is not None
        assert c._forbidden_fact_set is not None

    def test_evaluation_time_recorded(self):
        state = _make_state()
        c = PredicateConstraint("p4", "time test")
        result = c.evaluate(state)
        assert result.evaluation_time_ms >= 0

    def test_hash_and_eq(self):
        c1 = PredicateConstraint("p1", "test1")
        c2 = PredicateConstraint("p1", "test2")
        c3 = PredicateConstraint("p2", "test1")
        assert c1 == c2  # Same constraint_id
        assert c1 != c3
        assert hash(c1) == hash(c2)

    def test_get_hash(self):
        c = PredicateConstraint("p1", "test")
        h = c.get_hash()
        assert isinstance(h, str)
        assert len(h) == 16


@pytest.mark.unit
class TestTemporalConstraint:
    """Tests for TemporalConstraint."""

    def test_must_precede_satisfied(self):
        state = _make_state(metadata={"action_history": ["init", "process"]})
        c = TemporalConstraint(
            "t1", "init before process",
            must_precede=[("init", "process")],
        )
        result = c.evaluate(state)
        assert result.is_satisfied

    def test_must_precede_violated_wrong_order(self):
        state = _make_state(metadata={"action_history": ["process", "init"]})
        c = TemporalConstraint(
            "t1", "init before process",
            must_precede=[("init", "process")],
            enforcement=ConstraintEnforcement.HARD,
        )
        result = c.evaluate(state)
        assert result.is_violated

    def test_must_precede_violated_missing(self):
        state = _make_state(metadata={"action_history": ["process"]})
        c = TemporalConstraint(
            "t1", "init before process",
            must_precede=[("init", "process")],
            enforcement=ConstraintEnforcement.HARD,
        )
        result = c.evaluate(state)
        assert result.is_violated

    def test_must_not_follow_satisfied(self):
        state = _make_state(metadata={"action_history": ["a", "b"]})
        c = TemporalConstraint(
            "t2", "c cannot follow a",
            must_not_follow=[("a", "c")],
        )
        result = c.evaluate(state)
        assert result.is_satisfied

    def test_must_not_follow_violated(self):
        state = _make_state(metadata={"action_history": ["a", "c"]})
        c = TemporalConstraint(
            "t2", "c cannot follow a",
            must_not_follow=[("a", "c")],
            enforcement=ConstraintEnforcement.HARD,
        )
        result = c.evaluate(state)
        assert result.is_violated

    def test_with_action_parameter(self):
        state = _make_state(metadata={"action_history": ["init"]})
        c = TemporalConstraint(
            "t3", "init before process",
            must_precede=[("init", "process")],
        )
        result = c.evaluate(state, action="process")
        assert result.is_satisfied

    def test_soft_enforcement(self):
        state = _make_state(metadata={"action_history": ["process"]})
        c = TemporalConstraint(
            "t4", "init before process",
            must_precede=[("init", "process")],
            enforcement=ConstraintEnforcement.SOFT,
        )
        result = c.evaluate(state)
        assert result.satisfied == ConstraintSatisfactionLevel.PARTIALLY_SATISFIED
        assert result.penalty > 0

    def test_compile(self):
        c = TemporalConstraint("t5", "test")
        c.compile()
        assert c._compiled


@pytest.mark.unit
class TestExpressionConstraint:
    """Tests for ExpressionConstraint."""

    def test_simple_expression_satisfied(self):
        state = _make_state(metadata={"depth": 5})
        c = ExpressionConstraint(
            "e1", "depth check",
            expressions=[("metadata.depth", "<", 10)],
        )
        result = c.evaluate(state)
        assert result.is_satisfied

    def test_simple_expression_violated(self):
        state = _make_state(metadata={"depth": 15})
        c = ExpressionConstraint(
            "e1", "depth check",
            expressions=[("metadata.depth", "<", 10)],
            enforcement=ConstraintEnforcement.HARD,
        )
        result = c.evaluate(state)
        assert result.is_violated

    def test_variable_not_found(self):
        state = _make_state()
        c = ExpressionConstraint(
            "e2", "missing var",
            expressions=[("metadata.nonexistent", "==", 1)],
            enforcement=ConstraintEnforcement.HARD,
        )
        result = c.evaluate(state)
        assert result.is_violated
        assert "variable not found" in result.message

    def test_multiple_expressions(self):
        state = _make_state(metadata={"depth": 5, "score": 0.8})
        c = ExpressionConstraint(
            "e3", "multi check",
            expressions=[
                ("metadata.depth", "<", 10),
                ("metadata.score", ">=", 0.5),
            ],
        )
        result = c.evaluate(state)
        assert result.is_satisfied

    def test_context_lookup(self):
        state = _make_state()
        c = ExpressionConstraint(
            "e4", "context check",
            expressions=[("level", "==", "high")],
        )
        result = c.evaluate(state, context={"level": "high"})
        assert result.is_satisfied

    def test_in_operator(self):
        state = _make_state(metadata={"role": "admin"})
        c = ExpressionConstraint(
            "e5", "role check",
            expressions=[("metadata.role", "in", ["admin", "moderator"])],
        )
        result = c.evaluate(state)
        assert result.is_satisfied

    def test_unknown_operator(self):
        c = ExpressionConstraint(
            "e6", "bad op",
            expressions=[("x", "~=", 1)],
        )
        with pytest.raises(ValueError, match="Unknown operator"):
            c.compile()

    def test_soft_enforcement(self):
        state = _make_state(metadata={"depth": 15})
        c = ExpressionConstraint(
            "e7", "depth check",
            expressions=[("metadata.depth", "<", 10)],
            enforcement=ConstraintEnforcement.SOFT,
        )
        result = c.evaluate(state)
        assert result.satisfied == ConstraintSatisfactionLevel.PARTIALLY_SATISFIED


@pytest.mark.unit
class TestLambdaConstraint:
    """Tests for LambdaConstraint."""

    def test_satisfied(self):
        c = LambdaConstraint(
            "l1", "always true",
            predicate=lambda s, a, ctx: True,
        )
        state = _make_state()
        result = c.evaluate(state)
        assert result.is_satisfied

    def test_violated_hard(self):
        c = LambdaConstraint(
            "l2", "always false",
            predicate=lambda s, a, ctx: False,
            enforcement=ConstraintEnforcement.HARD,
        )
        state = _make_state()
        result = c.evaluate(state)
        assert result.is_violated

    def test_violated_soft_with_penalty_fn(self):
        c = LambdaConstraint(
            "l3", "soft false",
            predicate=lambda s, a, ctx: False,
            penalty_fn=lambda s: 0.3,
            enforcement=ConstraintEnforcement.SOFT,
        )
        state = _make_state()
        result = c.evaluate(state)
        assert result.satisfied == ConstraintSatisfactionLevel.PARTIALLY_SATISFIED
        assert result.penalty == 0.3

    def test_exception_returns_unknown(self):
        c = LambdaConstraint(
            "l4", "error",
            predicate=lambda s, a, ctx: 1 / 0,
        )
        state = _make_state()
        result = c.evaluate(state)
        assert result.satisfied == ConstraintSatisfactionLevel.UNKNOWN
        assert "evaluation error" in result.message

    def test_compile(self):
        c = LambdaConstraint("l5", "test", predicate=lambda s, a, ctx: True)
        c.compile()
        assert c._compiled


@pytest.mark.unit
class TestConstraintValidator:
    """Tests for ConstraintValidator."""

    def test_add_and_validate(self):
        config = _make_config()
        validator = ConstraintValidator(config)
        c = PredicateConstraint("p1", "test")
        validator.add_constraint(c)
        assert "p1" in validator.constraints

    def test_remove_constraint(self):
        config = _make_config()
        validator = ConstraintValidator(config)
        c = PredicateConstraint("p1", "test")
        validator.add_constraint(c)
        validator.remove_constraint("p1")
        assert "p1" not in validator.constraints

    def test_max_constraints_exceeded(self):
        config = _make_config(max_constraints_per_state=1)
        validator = ConstraintValidator(config)
        validator.add_constraint(PredicateConstraint("p1", "test1"))
        with pytest.raises(ValueError, match="Maximum constraints"):
            validator.add_constraint(PredicateConstraint("p2", "test2"))

    def test_validate_all_satisfied(self):
        config = _make_config()
        validator = ConstraintValidator(config)
        validator.add_constraint(PredicateConstraint("p1", "no reqs"))
        state = _make_state()
        valid, results = validator.validate(state)
        assert valid
        assert len(results) == 1

    def test_validate_hard_violation(self):
        config = _make_config()
        validator = ConstraintValidator(config)
        validator.add_constraint(PredicateConstraint(
            "p1", "need ready",
            required_facts=[("ready", ("x",))],
            enforcement=ConstraintEnforcement.HARD,
        ))
        state = _make_state()
        valid, results = validator.validate(state)
        assert not valid

    def test_cache_hit(self):
        config = _make_config()
        validator = ConstraintValidator(config)
        validator.add_constraint(PredicateConstraint("p1", "no reqs"))
        state = _make_state()
        validator.validate(state)
        validator.validate(state)
        stats = validator.get_cache_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1

    def test_clear_cache(self):
        config = _make_config()
        validator = ConstraintValidator(config)
        validator.add_constraint(PredicateConstraint("p1", "no reqs"))
        state = _make_state()
        validator.validate(state)
        validator.clear_cache()
        stats = validator.get_cache_stats()
        assert stats["cache_size"] == 0
        assert stats["cache_hits"] == 0

    def test_validate_action(self):
        config = _make_config()
        validator = ConstraintValidator(config)
        validator.add_constraint(PredicateConstraint("p1", "no reqs"))
        state = _make_state()
        assert validator.validate_action(state, "do_something")

    def test_filter_valid_actions(self):
        config = _make_config()
        validator = ConstraintValidator(config)
        validator.add_constraint(PredicateConstraint("p1", "no reqs"))
        state = _make_state()
        valid = validator.filter_valid_actions(state, ["a", "b", "c"])
        assert valid == ["a", "b", "c"]

    def test_priority_ordering(self):
        config = _make_config()
        validator = ConstraintValidator(config)
        # Add high priority hard constraint that fails
        validator.add_constraint(PredicateConstraint(
            "p_high", "high priority",
            required_facts=[("missing", ("x",))],
            enforcement=ConstraintEnforcement.HARD,
            priority=10,
        ))
        validator.add_constraint(PredicateConstraint(
            "p_low", "low priority",
            priority=1,
        ))
        state = _make_state()
        valid, results = validator.validate(state)
        assert not valid
        # Should stop early at high priority constraint
        assert len(results) == 1

    def test_penalty_threshold(self):
        config = _make_config(min_satisfaction_ratio=0.8)
        validator = ConstraintValidator(config)
        # Add soft constraint with high penalty
        validator.add_constraint(LambdaConstraint(
            "l1", "penalize",
            predicate=lambda s, a, ctx: False,
            penalty_fn=lambda s: 0.5,
            enforcement=ConstraintEnforcement.SOFT,
        ))
        state = _make_state()
        valid, _ = validator.validate(state)
        # 1.0 - 0.5 = 0.5 < 0.8 threshold
        assert not valid


@pytest.mark.unit
class TestConstraintSystem:
    """Tests for ConstraintSystem."""

    def test_register_predicate_constraint(self):
        config = _make_config()
        system = ConstraintSystem(config)
        c = system.register_predicate_constraint(
            "pc1", "test",
            required_facts=[("ready", ("x",))],
        )
        assert c.constraint_id == "pc1"
        assert "pc1" in system.validator.constraints

    def test_register_temporal_constraint(self):
        config = _make_config()
        system = ConstraintSystem(config)
        c = system.register_temporal_constraint(
            "tc1", "ordering",
            must_precede=[("a", "b")],
        )
        assert c.constraint_id == "tc1"

    def test_register_expression_constraint(self):
        config = _make_config()
        system = ConstraintSystem(config)
        c = system.register_expression_constraint(
            "ec1", "depth check",
            expressions=[("metadata.depth", "<", 10)],
        )
        assert c.constraint_id == "ec1"

    def test_validate_expansion(self):
        config = _make_config()
        system = ConstraintSystem(config)
        system.register_predicate_constraint("pc1", "no reqs")
        state = _make_state()
        results = system.validate_expansion(state, ["a", "b", "c"])
        assert len(results) == 3
        assert all(score == 1.0 for _, score in results)

    def test_validate_expansion_filters_invalid(self):
        config = _make_config()
        system = ConstraintSystem(config)
        system.register_predicate_constraint(
            "pc1", "need ready",
            required_facts=[("ready", ("x",))],
            enforcement=ConstraintEnforcement.HARD,
        )
        state = _make_state()
        results = system.validate_expansion(state, ["a", "b"])
        assert len(results) == 0

    def test_conflict_log(self):
        config = _make_config(enable_conflict_analysis=True)
        system = ConstraintSystem(config)
        system.register_predicate_constraint(
            "pc1", "need ready",
            required_facts=[("ready", ("x",))],
            enforcement=ConstraintEnforcement.HARD,
        )
        state = _make_state()
        system.validate_expansion(state, ["action1"])
        log = system.get_conflict_log()
        assert len(log) == 1
        assert log[0]["action"] == "action1"

    def test_clear_conflict_log(self):
        config = _make_config()
        system = ConstraintSystem(config)
        system._conflict_log.append({"test": True})
        system.clear_conflict_log()
        assert len(system.get_conflict_log()) == 0

    def test_get_statistics(self):
        config = _make_config()
        system = ConstraintSystem(config)
        system.register_predicate_constraint("pc1", "test")
        stats = system.get_statistics()
        assert stats["num_constraints"] == 1
        assert "cache_stats" in stats
        assert len(stats["constraints"]) == 1
        assert stats["constraints"][0]["id"] == "pc1"

    def test_soft_penalty_reduces_score(self):
        # Use a low min_satisfaction_ratio so penalty doesn't exclude
        config = _make_config(min_satisfaction_ratio=0.1)
        system = ConstraintSystem(config)
        system.register_constraint(LambdaConstraint(
            "l1", "penalize",
            predicate=lambda s, a, ctx: False,
            penalty_fn=lambda s: 0.3,
            enforcement=ConstraintEnforcement.SOFT,
        ))
        state = _make_state()
        results = system.validate_expansion(state, ["action1"])
        assert len(results) == 1
        action, score = results[0]
        assert score < 1.0
        assert score == pytest.approx(0.7)
