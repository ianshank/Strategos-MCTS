"""
Unit tests for MockLLMClient prompt-prefix response routing.

Pins the contract the e2e agent-flow fixtures rely on: an explicit agent
prefix (e.g. "TRM:") selects the matching canned response no matter what the
rest of the prompt says, with fallback to the sequential response queue.
"""

from tests.mocks.mock_external_services import create_mock_llm


class TestPromptPrefixRouting:
    """Tests for MockLLMClient.set_prompt_responses routing in generate()."""

    async def test_prefix_selects_response_regardless_of_prompt_wording(self):
        """The agent prefix wins even when the query wording matches another agent."""
        client = create_mock_llm()
        client.set_prompt_responses({"HRM": "hierarchical analysis", "TRM": "refinement cycle"})

        response = await client.generate("TRM: hierarchical objectives for northern sector")

        assert response.content == "refinement cycle"

    async def test_longest_matching_prefix_wins(self):
        client = create_mock_llm()
        client.set_prompt_responses({"TRM": "generic refinement", "TRM Iteration": "iteration refinement"})

        response = await client.generate("TRM Iteration 2: refine positions")

        assert response.content == "iteration refinement"

    async def test_unmatched_prompt_falls_back_to_sequential_responses(self):
        client = create_mock_llm(responses=["queued response"])
        client.set_prompt_responses({"HRM": "hierarchical analysis"})

        response = await client.generate("MCTS: simulate actions")

        assert response.content == "queued response"

    async def test_unmatched_prompt_without_queue_uses_default(self):
        client = create_mock_llm()
        client.set_prompt_responses({"HRM": "hierarchical analysis"})

        response = await client.generate("MCTS: simulate actions")

        assert response.content.startswith("Mock response for:")

    async def test_prefix_routed_calls_are_recorded_in_history(self):
        client = create_mock_llm()
        client.set_prompt_responses({"HRM": "hierarchical analysis"})

        await client.generate("HRM: analyze sector")

        assert client.get_call_count() == 1
        assert client.get_last_call()["prompt"] == "HRM: analyze sector"

    async def test_reset_clears_prompt_responses(self):
        client = create_mock_llm()
        client.set_prompt_responses({"HRM": "hierarchical analysis"})
        client.reset()

        response = await client.generate("HRM: analyze sector")

        assert response.content.startswith("Mock response for:")
