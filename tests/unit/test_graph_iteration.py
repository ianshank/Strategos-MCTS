"""Tests for iteration counter and consensus checking in GraphBuilder.

Validates:
- _evaluate_consensus_node: iteration incrementing, auto-consensus, confidence-based consensus
- _check_consensus: routing decisions based on consensus state and iteration limits
- Integration: iteration loop termination behavior
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

# ---------------------------------------------------------------------------
# Descriptive constants -- avoid hardcoded magic numbers in assertions
# ---------------------------------------------------------------------------
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_CONSENSUS_THRESHOLD = 0.7

HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.3
MODERATE_CONFIDENCE = 0.6

PERFECT_CONSENSUS_SCORE = 1.0
INITIAL_ITERATION = 0
SECOND_ITERATION = 1

ROUTE_SYNTHESIZE = "synthesize"
ROUTE_ITERATE = "iterate"

INTEGRATION_LOOP_ROUNDS = 3
SAFETY_LOOP_LIMIT = 20

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_graph_builder(
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
):
    """Construct a GraphBuilder with fully mocked dependencies.

    Uses the real constructor so that all instance attributes are set the
    same way production code sets them.  MCTSEngine is patched out to
    avoid heavy initialisation.
    """
    import src.framework.graph as graph_module

    hrm_agent = MagicMock(name="hrm_agent")
    trm_agent = MagicMock(name="trm_agent")
    model_adapter = MagicMock(name="model_adapter")
    logger = MagicMock(name="logger")

    with patch.object(graph_module, "MCTSEngine", MagicMock()):
        builder = graph_module.GraphBuilder(
            hrm_agent=hrm_agent,
            trm_agent=trm_agent,
            model_adapter=model_adapter,
            logger=logger,
            max_iterations=max_iterations,
            consensus_threshold=consensus_threshold,
        )
    return builder


def _agent_output(agent_name: str, confidence: float) -> dict:
    """Create a minimal agent-output dict."""
    return {
        "agent": agent_name,
        "confidence": confidence,
        "response": f"{agent_name} response",
    }


def _make_state(
    *,
    agent_outputs: list[dict] | None = None,
    iteration: int = INITIAL_ITERATION,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    consensus_reached: bool = False,
    consensus_score: float = 0.0,
    include_max_iterations: bool = True,
) -> dict:
    """Build a minimal AgentState-compatible dict."""
    state: dict = {
        "query": "test query",
        "use_mcts": False,
        "use_rag": False,
        "agent_outputs": agent_outputs if agent_outputs is not None else [],
        "iteration": iteration,
        "consensus_reached": consensus_reached,
        "consensus_score": consensus_score,
    }
    if include_max_iterations:
        state["max_iterations"] = max_iterations
    return state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_builder():
    """GraphBuilder with default test parameters."""
    return _build_graph_builder()


# ===========================================================================
# _evaluate_consensus_node tests
# ===========================================================================


class TestEvaluateConsensusNode:
    """Tests for GraphBuilder._evaluate_consensus_node."""

    def test_evaluate_consensus_increments_iteration(self, graph_builder):
        """Iteration counter must advance by exactly one from a non-zero start."""
        starting_iteration = 3
        state = _make_state(
            agent_outputs=[
                _agent_output("hrm", HIGH_CONFIDENCE),
                _agent_output("trm", HIGH_CONFIDENCE),
            ],
            iteration=starting_iteration,
        )

        result = graph_builder._evaluate_consensus_node(state)

        expected_next = starting_iteration + 1
        assert result["iteration"] == expected_next

    def test_evaluate_consensus_increments_from_zero(self, graph_builder):
        """First evaluation should move iteration from 0 to 1."""
        state = _make_state(
            agent_outputs=[
                _agent_output("hrm", HIGH_CONFIDENCE),
                _agent_output("trm", HIGH_CONFIDENCE),
            ],
            iteration=INITIAL_ITERATION,
        )

        result = graph_builder._evaluate_consensus_node(state)

        assert result["iteration"] == SECOND_ITERATION

    def test_evaluate_consensus_single_agent_auto_consensus(self, graph_builder):
        """With fewer than 2 agent outputs, consensus is granted automatically."""
        state = _make_state(
            agent_outputs=[_agent_output("hrm", LOW_CONFIDENCE)],
            iteration=INITIAL_ITERATION,
        )

        result = graph_builder._evaluate_consensus_node(state)

        assert result["consensus_reached"] is True
        assert result["consensus_score"] == PERFECT_CONSENSUS_SCORE
        assert result["iteration"] == SECOND_ITERATION

    def test_evaluate_consensus_empty_outputs_auto_consensus(self, graph_builder):
        """Zero agent outputs should also trigger auto-consensus."""
        state = _make_state(agent_outputs=[], iteration=INITIAL_ITERATION)

        result = graph_builder._evaluate_consensus_node(state)

        assert result["consensus_reached"] is True
        assert result["consensus_score"] == PERFECT_CONSENSUS_SCORE
        assert result["iteration"] == SECOND_ITERATION

    def test_evaluate_consensus_high_confidence_reaches_consensus(self, graph_builder):
        """Average confidence above threshold -> consensus_reached is True."""
        state = _make_state(
            agent_outputs=[
                _agent_output("hrm", HIGH_CONFIDENCE),
                _agent_output("trm", HIGH_CONFIDENCE),
            ],
        )

        result = graph_builder._evaluate_consensus_node(state)

        assert result["consensus_reached"] is True
        assert result["consensus_score"] == pytest.approx(HIGH_CONFIDENCE)

    def test_evaluate_consensus_low_confidence_no_consensus(self, graph_builder):
        """Average confidence below threshold -> consensus_reached is False."""
        state = _make_state(
            agent_outputs=[
                _agent_output("hrm", LOW_CONFIDENCE),
                _agent_output("trm", LOW_CONFIDENCE),
            ],
        )

        result = graph_builder._evaluate_consensus_node(state)

        assert result["consensus_reached"] is False
        assert result["consensus_score"] == pytest.approx(LOW_CONFIDENCE)

    def test_evaluate_consensus_at_exact_threshold(self, graph_builder):
        """Average confidence exactly equal to threshold -> consensus (>= comparison)."""
        threshold = DEFAULT_CONSENSUS_THRESHOLD
        state = _make_state(
            agent_outputs=[
                _agent_output("hrm", threshold),
                _agent_output("trm", threshold),
            ],
        )

        result = graph_builder._evaluate_consensus_node(state)

        assert result["consensus_reached"] is True
        assert result["consensus_score"] == pytest.approx(threshold)

    def test_evaluate_consensus_just_below_threshold(self, graph_builder):
        """Average confidence just below threshold -> no consensus."""
        just_below = DEFAULT_CONSENSUS_THRESHOLD - 0.01
        state = _make_state(
            agent_outputs=[
                _agent_output("hrm", just_below),
                _agent_output("trm", just_below),
            ],
        )

        result = graph_builder._evaluate_consensus_node(state)

        assert result["consensus_reached"] is False

    @pytest.mark.parametrize(
        "confidences, expected_consensus",
        [
            ([HIGH_CONFIDENCE, HIGH_CONFIDENCE, HIGH_CONFIDENCE], True),
            ([LOW_CONFIDENCE, LOW_CONFIDENCE, LOW_CONFIDENCE], False),
            # avg of HIGH_CONFIDENCE and LOW_CONFIDENCE = 0.6, below 0.7
            ([HIGH_CONFIDENCE, LOW_CONFIDENCE], False),
            ([MODERATE_CONFIDENCE, HIGH_CONFIDENCE], True),  # avg = 0.75, above 0.7
        ],
        ids=[
            "all_high",
            "all_low",
            "mixed_below_threshold",
            "mixed_above_threshold",
        ],
    )
    def test_evaluate_consensus_parametrized(
        self, graph_builder, confidences, expected_consensus
    ):
        """Parametrized consensus evaluation across confidence combinations."""
        outputs = [_agent_output(f"agent_{i}", c) for i, c in enumerate(confidences)]
        state = _make_state(agent_outputs=outputs)

        result = graph_builder._evaluate_consensus_node(state)

        assert result["consensus_reached"] is expected_consensus

    def test_evaluate_consensus_handles_missing_iteration_key(self, graph_builder):
        """When state lacks an 'iteration' key, default of 0 is used."""
        state = _make_state(
            agent_outputs=[_agent_output("hrm", HIGH_CONFIDENCE)],
        )
        del state["iteration"]

        result = graph_builder._evaluate_consensus_node(state)

        assert result["iteration"] == SECOND_ITERATION  # 0 + 1


# ===========================================================================
# _check_consensus tests
# ===========================================================================


class TestCheckConsensus:
    """Tests for GraphBuilder._check_consensus."""

    def test_check_consensus_returns_synthesize_on_consensus(self, graph_builder):
        """When consensus_reached is True, route to synthesize."""
        state = _make_state(consensus_reached=True, iteration=SECOND_ITERATION)

        result = graph_builder._check_consensus(state)

        assert result == ROUTE_SYNTHESIZE

    def test_check_consensus_returns_synthesize_on_max_iterations(self, graph_builder):
        """When iteration == max_iterations without consensus, still synthesize."""
        state = _make_state(
            consensus_reached=False,
            iteration=DEFAULT_MAX_ITERATIONS,
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )

        result = graph_builder._check_consensus(state)

        assert result == ROUTE_SYNTHESIZE

    def test_check_consensus_returns_synthesize_when_exceeding_max(self, graph_builder):
        """When iteration > max_iterations, route to synthesize."""
        exceeded_iteration = DEFAULT_MAX_ITERATIONS + 2
        state = _make_state(
            consensus_reached=False,
            iteration=exceeded_iteration,
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )

        result = graph_builder._check_consensus(state)

        assert result == ROUTE_SYNTHESIZE

    def test_check_consensus_returns_iterate_when_below_max(self, graph_builder):
        """When no consensus and iterations remain, route to iterate."""
        state = _make_state(
            consensus_reached=False,
            iteration=SECOND_ITERATION,
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )

        result = graph_builder._check_consensus(state)

        assert result == ROUTE_ITERATE

    def test_check_consensus_returns_iterate_at_zero(self, graph_builder):
        """At iteration 0 with no consensus, should iterate."""
        state = _make_state(
            consensus_reached=False,
            iteration=INITIAL_ITERATION,
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )

        result = graph_builder._check_consensus(state)

        assert result == ROUTE_ITERATE

    def test_check_consensus_max_iterations_from_state(self, graph_builder):
        """max_iterations in state overrides the builder's default."""
        state_specific_max = 2
        state = _make_state(
            consensus_reached=False,
            iteration=state_specific_max,
            max_iterations=state_specific_max,
        )

        # Builder default is DEFAULT_MAX_ITERATIONS (5), but state says 2
        result = graph_builder._check_consensus(state)

        assert result == ROUTE_SYNTHESIZE

    def test_check_consensus_falls_back_to_builder_default(self, graph_builder):
        """When state lacks max_iterations, builder.max_iterations is used."""
        state = _make_state(
            consensus_reached=False,
            iteration=DEFAULT_MAX_ITERATIONS,
            include_max_iterations=False,
        )

        result = graph_builder._check_consensus(state)

        assert result == ROUTE_SYNTHESIZE

    def test_check_consensus_falls_back_to_builder_default_iterate(self, graph_builder):
        """Below builder default with missing state key -> iterate."""
        state = _make_state(
            consensus_reached=False,
            iteration=SECOND_ITERATION,
            include_max_iterations=False,
        )

        result = graph_builder._check_consensus(state)

        assert result == ROUTE_ITERATE

    def test_check_consensus_consensus_takes_priority(self, graph_builder):
        """consensus_reached is checked before max_iterations."""
        state = _make_state(
            consensus_reached=True,
            iteration=DEFAULT_MAX_ITERATIONS,
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )

        result = graph_builder._check_consensus(state)

        assert result == ROUTE_SYNTHESIZE

    def test_check_consensus_zero_max_iterations_terminates(self, graph_builder):
        """max_iterations=0 means immediate synthesis even at iteration 0."""
        state = _make_state(
            consensus_reached=False,
            iteration=INITIAL_ITERATION,
            max_iterations=0,
        )

        result = graph_builder._check_consensus(state)

        assert result == ROUTE_SYNTHESIZE


# ===========================================================================
# Integration: iteration loop termination
# ===========================================================================


class TestIterationLoopTermination:
    """Simulate the evaluate -> check loop to verify termination."""

    def test_iteration_loop_terminates(self, graph_builder):
        """Simulate 3 iterations where confidence ramps up to reach consensus.

        Round 0: low confidence  -> iterate
        Round 1: low confidence  -> iterate
        Round 2: high confidence -> synthesize (consensus reached)
        """
        confidence_schedule = [
            LOW_CONFIDENCE,
            LOW_CONFIDENCE,
            HIGH_CONFIDENCE,
        ]

        iteration = INITIAL_ITERATION
        decision = ROUTE_ITERATE
        rounds_executed = 0

        for round_idx in range(SAFETY_LOOP_LIMIT):
            conf = confidence_schedule[min(round_idx, len(confidence_schedule) - 1)]
            state = _make_state(
                agent_outputs=[
                    _agent_output("hrm", conf),
                    _agent_output("trm", conf),
                ],
                iteration=iteration,
                max_iterations=DEFAULT_MAX_ITERATIONS,
            )

            eval_result = graph_builder._evaluate_consensus_node(state)
            iteration = eval_result["iteration"]

            check_state = _make_state(
                consensus_reached=eval_result["consensus_reached"],
                consensus_score=eval_result["consensus_score"],
                iteration=iteration,
                max_iterations=DEFAULT_MAX_ITERATIONS,
            )
            decision = graph_builder._check_consensus(check_state)
            rounds_executed += 1

            if decision == ROUTE_SYNTHESIZE:
                break

        assert decision == ROUTE_SYNTHESIZE
        assert eval_result["consensus_reached"] is True
        assert rounds_executed == INTEGRATION_LOOP_ROUNDS
        assert iteration == INTEGRATION_LOOP_ROUNDS

    def test_iteration_loop_terminates_at_max_without_consensus(self):
        """Without consensus ever being reached, loop stops at max_iterations."""
        custom_max = 4
        builder = _build_graph_builder(max_iterations=custom_max)

        iteration = INITIAL_ITERATION
        decisions: list[str] = []

        for _ in range(SAFETY_LOOP_LIMIT):
            state = _make_state(
                agent_outputs=[
                    _agent_output("hrm", LOW_CONFIDENCE),
                    _agent_output("trm", LOW_CONFIDENCE),
                ],
                iteration=iteration,
                max_iterations=custom_max,
            )

            eval_result = builder._evaluate_consensus_node(state)
            iteration = eval_result["iteration"]

            check_state = _make_state(
                consensus_reached=eval_result["consensus_reached"],
                iteration=iteration,
                max_iterations=custom_max,
            )
            decision = builder._check_consensus(check_state)
            decisions.append(decision)

            if decision == ROUTE_SYNTHESIZE:
                break

        assert decisions[-1] == ROUTE_SYNTHESIZE
        assert all(d == ROUTE_ITERATE for d in decisions[:-1])
        assert eval_result["consensus_reached"] is False
        assert iteration == custom_max

    def test_single_agent_loop_terminates_immediately(self, graph_builder):
        """With a single agent output, auto-consensus ends the loop on round 1."""
        state = _make_state(
            agent_outputs=[_agent_output("hrm", LOW_CONFIDENCE)],
            iteration=INITIAL_ITERATION,
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )

        eval_result = graph_builder._evaluate_consensus_node(state)

        check_state = _make_state(
            consensus_reached=eval_result["consensus_reached"],
            iteration=eval_result["iteration"],
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )
        decision = graph_builder._check_consensus(check_state)

        assert decision == ROUTE_SYNTHESIZE
        assert eval_result["consensus_reached"] is True
        assert eval_result["iteration"] == SECOND_ITERATION
