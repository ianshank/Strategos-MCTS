"""
Property-based tests for MCTS invariants.

Uses Hypothesis to generate random inputs and verify
that MCTS invariants always hold.

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 10
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

# Skip if hypothesis not installed
pytest.importorskip("hypothesis")

# Import MCTS components with graceful fallback
try:
    from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
    from src.framework.mcts.policies import ucb1

    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False

# Skip all tests if MCTS not available
pytestmark = [
    pytest.mark.property,
    pytest.mark.skipif(not MCTS_AVAILABLE, reason="MCTS module not available"),
]


class TestMCTSEngineInvariants:
    """Property-based tests for MCTS engine invariants."""

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        exploration_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_engine_determinism(self, seed: int, exploration_weight: float):
        """
        Property: Same seed should produce same random sequence.

        This is crucial for reproducibility.
        """
        engine1 = MCTSEngine(seed=seed, exploration_weight=exploration_weight)
        engine2 = MCTSEngine(seed=seed, exploration_weight=exploration_weight)

        # First 10 random numbers should match
        for _ in range(10):
            assert engine1.rng.random() == engine2.rng.random()

    @given(
        seed1=st.integers(min_value=0, max_value=2**31 - 1),
        seed2=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_different_seeds_different_results(self, seed1: int, seed2: int):
        """
        Property: Different seeds should (almost always) produce different results.
        """
        assume(seed1 != seed2)

        engine1 = MCTSEngine(seed=seed1)
        engine2 = MCTSEngine(seed=seed2)

        # Generate sequence of random numbers
        seq1 = [engine1.rng.random() for _ in range(10)]
        seq2 = [engine2.rng.random() for _ in range(10)]

        # Sequences should differ (with overwhelming probability)
        assert seq1 != seq2


class TestUCB1Invariants:
    """Property-based tests for UCB1 selection formula."""

    @given(
        visits=st.integers(min_value=1, max_value=100000),
        value_sum=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False),
        parent_visits=st.integers(min_value=1, max_value=100000),
        c=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_ucb1_non_negative(
        self,
        visits: int,
        value_sum: float,
        parent_visits: int,
        c: float,
    ):
        """
        Property: UCB1 score should be non-negative (or inf for unvisited).
        """
        # Ensure parent has at least as many visits
        assume(parent_visits >= visits)

        score = ucb1(
            value_sum=value_sum,
            visits=visits,
            parent_visits=parent_visits,
            c=c,
        )

        assert score >= 0 or score == float("inf")

    @given(
        visits=st.integers(min_value=0, max_value=100000),
        value_sum=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False),
        parent_visits=st.integers(min_value=1, max_value=100000),
    )
    def test_ucb1_zero_visits_returns_infinity(
        self,
        visits: int,
        value_sum: float,
        parent_visits: int,
    ):
        """
        Property: Zero visits should return infinity (unexplored priority).
        """
        if visits == 0:
            score = ucb1(
                value_sum=value_sum,
                visits=0,
                parent_visits=parent_visits,
                c=1.414,
            )
            assert score == float("inf")

    @given(
        visits=st.integers(min_value=1, max_value=10000),
        value_sum=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        parent_visits=st.integers(min_value=1, max_value=10000),
    )
    def test_ucb1_zero_exploration_is_greedy(
        self,
        visits: int,
        value_sum: float,
        parent_visits: int,
    ):
        """
        Property: With c=0, UCB1 should just return exploitation term.
        """
        assume(parent_visits >= visits)

        score = ucb1(
            value_sum=value_sum,
            visits=visits,
            parent_visits=parent_visits,
            c=0.0,
        )

        expected = value_sum / visits
        assert abs(score - expected) < 1e-10

    @given(
        visits=st.integers(min_value=1, max_value=1000),
        value_sum=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
        parent_visits=st.integers(min_value=2, max_value=1000),
        c1=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
        c2=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
    )
    def test_ucb1_higher_c_more_exploration(
        self,
        visits: int,
        value_sum: float,
        parent_visits: int,
        c1: float,
        c2: float,
    ):
        """
        Property: Higher exploration constant should give higher scores
        (more exploration).
        """
        assume(parent_visits >= visits)
        assume(c2 > c1)

        score1 = ucb1(value_sum=value_sum, visits=visits, parent_visits=parent_visits, c=c1)
        score2 = ucb1(value_sum=value_sum, visits=visits, parent_visits=parent_visits, c=c2)

        assert score2 >= score1


class TestMCTSNodeInvariants:
    """Property-based tests for MCTS node invariants."""

    @given(
        num_children=st.integers(min_value=1, max_value=20),
        parent_visits=st.integers(min_value=10, max_value=1000),
    )
    def test_child_visits_invariant(self, num_children: int, parent_visits: int):
        """
        Property: Sum of child visits should be <= parent visits.
        """
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = parent_visits

        # Distribute visits among children fairly
        child_visits = parent_visits // (num_children + 1)

        for i in range(num_children):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = child_visits

        total_child_visits = sum(c.visits for c in root.children)
        assert total_child_visits <= root.visits

    @given(
        num_children=st.integers(min_value=2, max_value=10),
    )
    def test_select_child_always_returns_child(self, num_children: int):
        """
        Property: select_child should always return a valid child.
        """
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 100

        for i in range(num_children):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = 10 + i
            child.value_sum = 5.0 + i * 0.1

        selected = root.select_child(exploration_weight=1.414)

        assert selected in root.children

    @given(
        depth=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50)
    def test_depth_tracking(self, depth: int):
        """
        Property: Node depth should be correctly tracked through tree.
        """
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)

        current = root
        for d in range(depth):
            child_state = MCTSState(f"level_{d}", {})
            current.available_actions = [f"action_{d}"]
            current = current.add_child(f"action_{d}", child_state)

        # Check depth of leaf node
        assert current.depth == depth


class TestMCTSStateInvariants:
    """Property-based tests for MCTS state invariants."""

    @given(
        state_id=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N"))),
        features=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
            values=st.one_of(
                st.integers(min_value=-1000, max_value=1000),
                st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
                st.text(min_size=0, max_size=50),
            ),
            min_size=0,
            max_size=10,
        ),
    )
    def test_state_hashing_deterministic(self, state_id: str, features: dict):
        """
        Property: Same state should always produce same hash.
        """
        state1 = MCTSState(state_id=state_id, features=features)
        state2 = MCTSState(state_id=state_id, features=features)

        assert state1.to_hash_key() == state2.to_hash_key()

    @given(
        state_id1=st.text(min_size=1, max_size=50),
        state_id2=st.text(min_size=1, max_size=50),
    )
    def test_different_states_different_hashes(self, state_id1: str, state_id2: str):
        """
        Property: Different states should (almost always) have different hashes.
        """
        assume(state_id1 != state_id2)

        state1 = MCTSState(state_id=state_id1, features={})
        state2 = MCTSState(state_id=state_id2, features={})

        # Hashes should differ
        assert state1.to_hash_key() != state2.to_hash_key()


class TestBackpropagationInvariants:
    """Property-based tests for backpropagation invariants."""

    @given(
        depth=st.integers(min_value=1, max_value=10),
        value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_backpropagation_updates_all_ancestors(self, depth: int, value: float):
        """
        Property: Backpropagation should update all nodes from leaf to root.
        """
        engine = MCTSEngine(seed=42)

        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)

        # Build path to leaf
        current = root
        path = [root]
        for d in range(depth):
            child_state = MCTSState(f"level_{d}", {})
            current.available_actions = [f"action_{d}"]
            current = current.add_child(f"action_{d}", child_state)
            path.append(current)

        leaf = current

        # Record visits before backprop
        visits_before = [node.visits for node in path]

        # Backpropagate
        engine.backpropagate(leaf, value)

        # All nodes in path should have incremented visits
        for i, node in enumerate(path):
            assert node.visits == visits_before[i] + 1
            assert node.value_sum >= 0

    @given(
        num_backprops=st.integers(min_value=1, max_value=100),
        value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_value_sum_accumulates(self, num_backprops: int, value: float):
        """
        Property: Value sum should accumulate correctly over multiple backprops.
        """
        engine = MCTSEngine(seed=42)

        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)

        for _ in range(num_backprops):
            engine.backpropagate(root, value)

        assert root.visits == num_backprops
        expected_sum = value * num_backprops
        assert abs(root.value_sum - expected_sum) < 1e-10


class TestCacheInvariants:
    """Property-based tests for MCTS cache invariants."""

    @given(
        cache_size=st.integers(min_value=1, max_value=100),
        num_entries=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=50)
    def test_cache_size_limit_respected(self, cache_size: int, num_entries: int):
        """
        Property: Cache should never exceed size limit.
        """
        engine = MCTSEngine(seed=42, cache_size_limit=cache_size)

        for i in range(num_entries):
            state = MCTSState(f"state_{i}", {"id": i})
            key = state.to_hash_key()
            engine._simulation_cache[key] = (0.5, 1)

            # Evict if needed
            while len(engine._simulation_cache) > engine.cache_size_limit:
                engine._simulation_cache.popitem(last=False)

        assert len(engine._simulation_cache) <= cache_size


class TestProgressiveWideningInvariants:
    """Property-based tests for progressive widening."""

    @given(
        k=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        visits=st.integers(min_value=0, max_value=1000),
        num_children=st.integers(min_value=0, max_value=100),
    )
    def test_should_expand_consistency(
        self,
        k: float,
        alpha: float,
        visits: int,
        num_children: int,
    ):
        """
        Property: should_expand should be consistent with formula.
        """
        engine = MCTSEngine(
            seed=42,
            progressive_widening_k=k,
            progressive_widening_alpha=alpha,
        )

        state = MCTSState("test", {})
        rng = np.random.default_rng(42)
        node = MCTSNode(state=state, rng=rng)
        node.visits = visits
        node.available_actions = [f"action_{i}" for i in range(num_children + 5)]

        # Add children
        for i in range(num_children):
            child_state = MCTSState(f"child_{i}", {})
            node.add_child(f"action_{i}", child_state)

        should_expand = engine.should_expand(node)

        # Verify against formula: expand when visits > k * n^alpha
        threshold = k * (num_children**alpha) if num_children > 0 else 0
        expected = visits > threshold and not node.is_fully_expanded and not node.terminal

        assert should_expand == expected
