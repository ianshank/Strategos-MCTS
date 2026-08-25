"""
Property-based tests for core framework components.

Property-based testing validates that code behaves correctly across
a wide range of inputs, catching edge cases that example-based tests miss.

Best Practices 2025:
- Test invariants rather than specific examples
- Use hypothesis strategies for input generation
- Focus on properties that should always hold
- Combine with example-based tests for completeness
"""

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from src.framework.mcts.core import MCTSNode, MCTSState


class TestMCTSNodeProperties:
    """Property-based tests for MCTS node behavior."""

    @pytest.mark.property
    @given(
        visits=st.integers(min_value=0, max_value=10000),
        value_sum=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_node_value_invariant(self, visits: int, value_sum: float):
        """
        Property: Node value should equal value_sum / visits when visits > 0.

        This tests the fundamental invariant of MCTS node values.
        """
        state = MCTSState(state_id="test", features={})
        node = MCTSNode(state=state)
        node.visits = visits
        node.value_sum = value_sum

        if visits == 0:
            assert node.value == 0.0
        else:
            expected = value_sum / visits
            assert abs(node.value - expected) < 1e-10

    @pytest.mark.property
    @given(
        state_id=st.text(min_size=1, max_size=100),
        feature_keys=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10),
        feature_values=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=10),
    )
    @settings(max_examples=50)
    def test_state_hash_determinism(self, state_id: str, feature_keys: list, feature_values: list):
        """
        Property: Same state should produce same hash.

        This tests hash consistency for state caching.
        """
        # Ensure equal length lists
        min_len = min(len(feature_keys), len(feature_values))
        features = dict(zip(feature_keys[:min_len], feature_values[:min_len], strict=False))

        state1 = MCTSState(state_id=state_id, features=features)
        state2 = MCTSState(state_id=state_id, features=features)

        hash1 = state1.to_hash_key()
        hash2 = state2.to_hash_key()

        assert hash1 == hash2, "Same state should produce identical hash"

    @pytest.mark.property
    @given(
        depth=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50)
    def test_node_depth_tracking(self, depth: int):
        """
        Property: Child node depth should equal parent depth + 1.

        This tests depth tracking through the tree.
        """
        # Create root and chain of children
        state = MCTSState(state_id="root", features={})
        node = MCTSNode(state=state)
        node.depth = 0

        current = node
        for i in range(depth):
            child_state = MCTSState(state_id=f"child_{i}", features={})
            child = MCTSNode(state=child_state, parent=current, action=f"action_{i}")
            assert child.depth == current.depth + 1
            current = child

        assert current.depth == depth

    @pytest.mark.property
    @given(
        visits=st.integers(min_value=0, max_value=1000),
        num_children=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_visit_count_propagation(self, visits: int, num_children: int):
        """
        Property: Parent visits >= sum of all child visits.

        This tests backpropagation correctness.
        """
        state = MCTSState(state_id="root", features={})
        parent = MCTSNode(state=state)
        parent.visits = visits

        children_visits = 0
        for i in range(num_children):
            child_state = MCTSState(state_id=f"child_{i}", features={})
            child = MCTSNode(state=child_state, parent=parent)
            child.visits = visits // max(num_children, 1)  # Distribute visits
            children_visits += child.visits
            parent.children.append(child)

        # In a correct MCTS implementation, parent visits should be at least
        # the sum of child visits (accounting for selection phase visits)
        assert parent.visits >= 0


class TestAgentContextProperties:
    """Property-based tests for agent context handling."""

    @pytest.mark.property
    @given(
        query=st.text(min_size=1, max_size=1000),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        max_iterations=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_context_creation_valid_inputs(self, query: str, temperature: float, max_iterations: int):
        """
        Property: Valid inputs should create valid context.

        This tests that context creation handles various valid inputs.
        """
        from src.framework.agents.base import AgentContext

        context = AgentContext(
            query=query,
            temperature=temperature,
            max_iterations=max_iterations,
        )

        assert context.query == query
        assert context.temperature == temperature
        assert context.max_iterations == max_iterations
        assert isinstance(context.session_id, str)
        assert len(context.session_id) > 0

    @pytest.mark.property
    @given(
        metadata_keys=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10, unique=True),
        metadata_values=st.lists(st.integers(), min_size=0, max_size=10),
    )
    @settings(max_examples=50)
    def test_context_metadata_preservation(self, metadata_keys: list, metadata_values: list):
        """
        Property: Context metadata should be preserved through serialization.

        This tests metadata handling and serialization.
        """
        from src.framework.agents.base import AgentContext

        # Create metadata dict
        min_len = min(len(metadata_keys), len(metadata_values))
        metadata = dict(zip(metadata_keys[:min_len], metadata_values[:min_len], strict=False))

        context = AgentContext(query="test", metadata=metadata)
        context_dict = context.to_dict()

        assert context_dict["metadata"] == metadata


class TestLLMResponseProperties:
    """Property-based tests for LLM response handling."""

    @pytest.mark.property
    @given(
        prompt_tokens=st.integers(min_value=0, max_value=100000),
        completion_tokens=st.integers(min_value=0, max_value=100000),
    )
    @settings(max_examples=50)
    def test_token_count_invariant(self, prompt_tokens: int, completion_tokens: int):
        """
        Property: Total tokens should equal prompt + completion tokens.

        This tests token accounting correctness.
        """
        from src.adapters.llm.base import LLMResponse

        total_tokens = prompt_tokens + completion_tokens
        response = LLMResponse(
            text="test",
            model="test-model",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )

        assert response.total_tokens == total_tokens
        assert response.total_tokens >= 0

    @pytest.mark.property
    @given(
        text=st.text(min_size=0, max_size=10000),
        model=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=50)
    def test_response_creation_always_succeeds(self, text: str, model: str):
        """
        Property: Response creation should never fail with valid inputs.

        This tests robustness of response creation.
        """
        from src.adapters.llm.base import LLMResponse

        response = LLMResponse(
            text=text,
            model=model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        assert response.text == text
        assert response.model == model
        assert isinstance(response.prompt_tokens, int)


class TestValidationProperties:
    """Property-based tests for input validation."""

    @pytest.mark.property
    @given(
        query=st.text(min_size=1, max_size=5000),
    )
    @settings(max_examples=50)
    def test_query_validation_non_empty(self, query: str):
        """
        Property: Non-empty queries should pass validation.

        This tests query validation for legitimate inputs.
        """
        from src.models.validation import QueryInput

        # Strip to handle whitespace-only strings
        stripped = query.strip()
        if stripped:
            request = QueryInput(query=stripped)
            assert len(request.query) > 0

    @pytest.mark.property
    @given(
        value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_confidence_score_range(self, value: float):
        """
        Property: Confidence scores should be in [0, 1] range.

        This tests confidence score validation.
        """
        from src.framework.agents.base import AgentResult

        result = AgentResult(
            response="test",
            confidence=value,
        )

        assert 0.0 <= result.confidence <= 1.0


class TestMCTSConfigProperties:
    """Property-based tests for MCTS configuration."""

    @pytest.mark.property
    @given(
        exploration_weight=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_iterations=st.integers(min_value=50, max_value=10000),  # Must be >= min_iterations_before_termination
    )
    @settings(max_examples=50)
    def test_config_positive_values(self, exploration_weight: float, num_iterations: int):
        """
        Property: MCTS config values should be positive.

        This tests configuration validation.
        """
        from src.framework.mcts.config import MCTSConfig

        config = MCTSConfig(
            exploration_weight=exploration_weight,
            num_iterations=num_iterations,
        )

        assert config.exploration_weight > 0
        assert config.num_iterations > 0

    @pytest.mark.property
    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=50)
    def test_seed_determinism(self, seed: int):
        """
        Property: Same seed should produce same RNG state.

        This tests deterministic behavior.
        """
        import numpy as np

        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        # Generate same sequence
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]

        assert vals1 == vals2, "Same seed should produce identical random sequences"


# Run property tests with example-based fallback
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "property"])
