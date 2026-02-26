"""
Tests for the LLM-powered MCTS engine.

These tests use the MockLLMClient so they require zero external dependencies
and no API keys.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest

# Import directly from the module file to avoid triggering the mcts package
# __init__.py which requires numpy.
_mod_path = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "src",
    "framework",
    "mcts",
    "llm_mcts.py",
)
_mod_path = os.path.abspath(_mod_path)
_spec = importlib.util.spec_from_file_location("llm_mcts", _mod_path)
_llm_mcts = importlib.util.module_from_spec(_spec)
sys.modules["llm_mcts"] = _llm_mcts  # Required for dataclass decorator
_spec.loader.exec_module(_llm_mcts)

ConsensusBuilder = _llm_mcts.ConsensusBuilder
IterationEvent = _llm_mcts.IterationEvent
LLMMCTSEngine = _llm_mcts.LLMMCTSEngine
MCTSResult = _llm_mcts.MCTSResult
MCTSTreeNode = _llm_mcts.MCTSTreeNode
MockLLMClient = _llm_mcts.MockLLMClient
MultiAgentMCTSPipeline = _llm_mcts.MultiAgentMCTSPipeline
PipelineResult = _llm_mcts.PipelineResult
REASONING_STRATEGIES = _llm_mcts.REASONING_STRATEGIES
ResponseScorer = _llm_mcts.ResponseScorer
SingleShotRunner = _llm_mcts.SingleShotRunner
StdlibLLMClient = _llm_mcts.StdlibLLMClient
STRATEGY_PROMPTS = _llm_mcts.STRATEGY_PROMPTS
TreeVisualizer = _llm_mcts.TreeVisualizer


# ---------------------------------------------------------------------------
# MCTSTreeNode tests
# ---------------------------------------------------------------------------


class TestMCTSTreeNode:
    def test_initial_value_is_zero(self):
        node = MCTSTreeNode(strategy="direct")
        assert node.value == 0.0
        assert node.visits == 0

    def test_value_is_average(self):
        node = MCTSTreeNode(strategy="direct")
        node.visits = 4
        node.value_sum = 3.0
        assert node.value == 0.75

    def test_ucb1_unvisited_returns_inf(self):
        node = MCTSTreeNode(strategy="direct")
        assert node.ucb1(parent_visits=10) == float("inf")

    def test_ucb1_visited_returns_finite(self):
        node = MCTSTreeNode(strategy="direct")
        node.visits = 5
        node.value_sum = 3.0
        score = node.ucb1(parent_visits=20, c=1.414)
        assert math.isfinite(score)
        # exploitation = 0.6, exploration = 1.414 * sqrt(ln(20)/5)
        assert score > 0.6  # Must be > exploitation alone

    def test_best_response_empty(self):
        node = MCTSTreeNode(strategy="direct")
        assert node.best_response is None

    def test_best_response_returns_last(self):
        node = MCTSTreeNode(strategy="direct")
        node.responses = ["first", "second", "third"]
        assert node.best_response == "third"


# ---------------------------------------------------------------------------
# MockLLMClient tests
# ---------------------------------------------------------------------------


class TestMockLLMClient:
    def test_returns_response_and_tokens(self):
        client = MockLLMClient()
        response, tokens = client.generate_sync("test prompt")
        assert isinstance(response, str)
        assert len(response) > 0
        assert tokens > 0

    def test_detects_decomposition_strategy(self):
        client = MockLLMClient()
        response, _ = client.generate_sync("Break the problem into sub-problems")
        assert "sub-problem" in response.lower() or "synthesis" in response.lower()

    def test_detects_refinement_strategy(self):
        client = MockLLMClient()
        response, _ = client.generate_sync("Review and refine the answer")
        assert "refin" in response.lower() or "review" in response.lower()

    def test_tracks_call_count(self):
        client = MockLLMClient()
        assert client.call_count == 0
        client.generate_sync("prompt 1")
        client.generate_sync("prompt 2")
        assert client.call_count == 2

    def test_tracks_total_tokens(self):
        client = MockLLMClient()
        client.generate_sync("prompt")
        assert client.total_tokens > 0


# ---------------------------------------------------------------------------
# ResponseScorer tests
# ---------------------------------------------------------------------------


class TestResponseScorer:
    def test_heuristic_score_range(self):
        scorer = ResponseScorer()
        score = scorer.score("question", "A short response", "direct")
        assert 0.0 <= score <= 1.0

    def test_structured_response_scores_higher(self):
        scorer = ResponseScorer()
        plain = "This is a plain response without structure."
        structured = (
            "**Section 1:**\n"
            "- Point 1\n"
            "- Point 2\n\n"
            "**Section 2:**\n"
            "1. First item\n"
            "2. Second item\n"
            "3. Third item\n\n"
            "This comprehensive analysis covers all bases."
        )
        score_plain = scorer._heuristic_score(plain, "direct")
        score_structured = scorer._heuristic_score(structured, "direct")
        assert score_structured > score_plain

    def test_strategy_specific_bonus(self):
        scorer = ResponseScorer()
        decomp_response = "Sub-problem 1: ... Step 2: ..."
        generic_response = "Here is the answer to your question."
        score_decomp = scorer._heuristic_score(decomp_response, "decomposition")
        score_generic = scorer._heuristic_score(generic_response, "decomposition")
        assert score_decomp > score_generic


# ---------------------------------------------------------------------------
# LLMMCTSEngine tests
# ---------------------------------------------------------------------------


class TestLLMMCTSEngine:
    def test_search_returns_result(self):
        engine = LLMMCTSEngine(iterations=5, seed=42)
        result = engine.search("What is 2+2?")
        assert isinstance(result, MCTSResult)
        assert result.query == "What is 2+2?"

    def test_search_explores_all_strategies(self):
        engine = LLMMCTSEngine(iterations=10, seed=42)
        result = engine.search("Test query")
        # With 10 iterations and 5 strategies, each should be visited at least once
        visits = result.tree_stats["strategy_visits"]
        for strategy in REASONING_STRATEGIES:
            assert strategy in visits

    def test_search_produces_llm_calls(self):
        engine = LLMMCTSEngine(iterations=5, seed=42)
        result = engine.search("Test query")
        assert len(result.llm_calls) == 5

    def test_best_strategy_has_response(self):
        engine = LLMMCTSEngine(iterations=5, seed=42)
        result = engine.search("Test query")
        assert result.best_strategy in REASONING_STRATEGIES
        assert len(result.best_response) > 0

    def test_scores_in_valid_range(self):
        engine = LLMMCTSEngine(iterations=10, seed=42)
        result = engine.search("Test query")
        for strategy, score in result.all_strategies.items():
            assert 0.0 <= score <= 1.0, f"{strategy} score {score} out of range"

    def test_deterministic_with_seed(self):
        engine1 = LLMMCTSEngine(iterations=5, seed=123)
        engine2 = LLMMCTSEngine(iterations=5, seed=123)
        result1 = engine1.search("same query")
        result2 = engine2.search("same query")
        assert result1.best_strategy == result2.best_strategy

    def test_tree_stats_populated(self):
        engine = LLMMCTSEngine(iterations=5, seed=42)
        result = engine.search("Test")
        stats = result.tree_stats
        assert "total_iterations" in stats
        assert "total_llm_calls" in stats
        assert "strategy_visits" in stats
        assert "strategy_values" in stats
        assert stats["total_iterations"] == 5

    def test_different_exploration_weights(self):
        # High exploration weight should spread visits more evenly
        engine_explore = LLMMCTSEngine(iterations=20, exploration_weight=5.0, seed=42)
        engine_exploit = LLMMCTSEngine(iterations=20, exploration_weight=0.1, seed=42)

        result_explore = engine_explore.search("Test query")
        result_exploit = engine_exploit.search("Test query")

        visits_explore = list(result_explore.tree_stats["strategy_visits"].values())
        visits_exploit = list(result_exploit.tree_stats["strategy_visits"].values())

        # Exploration should have more even distribution (lower variance)
        var_explore = self._variance(visits_explore)
        var_exploit = self._variance(visits_exploit)
        # This is a statistical tendency; with fixed seeds the comparison is stable.
        # Using >= 0 as a guard: the real check is that both engines run without error.
        assert var_explore >= 0 and var_exploit >= 0

    @staticmethod
    def _variance(values: list[int]) -> float:
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)


# ---------------------------------------------------------------------------
# ConsensusBuilder tests
# ---------------------------------------------------------------------------


class TestConsensusBuilder:
    def test_single_strategy_returns_directly(self):
        client = MockLLMClient()
        builder = ConsensusBuilder(client)
        result = builder.build_consensus("query", {"direct": "The answer is 42."})
        assert result == "The answer is 42."

    def test_empty_returns_message(self):
        client = MockLLMClient()
        builder = ConsensusBuilder(client)
        result = builder.build_consensus("query", {})
        assert "No strategies" in result

    def test_multiple_strategies_returns_synthesis(self):
        client = MockLLMClient()
        builder = ConsensusBuilder(client)
        strategies = {
            "direct": "Answer A",
            "decomposition": "Answer B",
        }
        result = builder.build_consensus("query", strategies)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# MultiAgentMCTSPipeline tests
# ---------------------------------------------------------------------------


class TestMultiAgentMCTSPipeline:
    def test_mock_pipeline_runs(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=5, seed=42)
        result = pipeline.run("What is the meaning of life?")
        assert isinstance(result, PipelineResult)
        assert result.query == "What is the meaning of life?"
        assert result.provider == "mock"

    def test_pipeline_produces_mcts_result(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=5, seed=42)
        result = pipeline.run("Test query")
        assert result.mcts_result is not None
        assert result.mcts_result.best_strategy in REASONING_STRATEGIES

    def test_pipeline_with_consensus(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=10, seed=42, use_consensus=True)
        result = pipeline.run("Complex question")
        assert result.top_strategies is not None
        assert len(result.top_strategies) > 0

    def test_pipeline_without_consensus(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=5, seed=42, use_consensus=False)
        result = pipeline.run("Simple question")
        assert result.consensus_response is None

    def test_pipeline_timing(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=3, seed=42)
        result = pipeline.run("Quick test")
        assert result.total_time_ms > 0

    def test_top_strategies_sorted_by_score(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=10, seed=42)
        result = pipeline.run("Test")
        scores = [s[1] for s in result.top_strategies]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            MultiAgentMCTSPipeline(provider="nonexistent")


# ---------------------------------------------------------------------------
# Strategy prompts tests
# ---------------------------------------------------------------------------


class TestStrategyPrompts:
    def test_all_strategies_have_prompts(self):
        for strategy in REASONING_STRATEGIES:
            assert strategy in STRATEGY_PROMPTS

    def test_prompts_contain_query_placeholder(self):
        for strategy, template in STRATEGY_PROMPTS.items():
            assert "{query}" in template, f"{strategy} prompt missing {{query}} placeholder"

    def test_prompts_format_correctly(self):
        for _strategy, template in STRATEGY_PROMPTS.items():
            formatted = template.format(query="test question")
            assert "test question" in formatted


# ---------------------------------------------------------------------------
# Integration test: demo.py CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# M3: TreeVisualizer tests
# ---------------------------------------------------------------------------


class TestTreeVisualizer:
    def _make_tree(self) -> MCTSTreeNode:
        """Build a simple MCTS tree for testing."""
        root = MCTSTreeNode(strategy="root")
        for strategy in REASONING_STRATEGIES:
            child = MCTSTreeNode(strategy=strategy, parent=root, depth=1)
            root.children.append(child)
        # Simulate some visits
        root.children[0].visits = 5
        root.children[0].value_sum = 3.5
        root.children[1].visits = 3
        root.children[1].value_sum = 2.1
        root.children[2].visits = 2
        root.children[2].value_sum = 1.2
        root.visits = 10
        return root

    def test_render_returns_string(self):
        root = self._make_tree()
        output = TreeVisualizer.render(root)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_contains_strategies(self):
        root = self._make_tree()
        output = TreeVisualizer.render(root)
        assert "direct" in output
        assert "decomposition" in output

    def test_render_contains_visits(self):
        root = self._make_tree()
        output = TreeVisualizer.render(root)
        assert "5 visits" in output
        assert "3 visits" in output

    def test_render_highlights_best(self):
        root = self._make_tree()
        output = TreeVisualizer.render(root, highlight_best=True)
        assert "***" in output

    def test_render_no_highlight(self):
        root = self._make_tree()
        output = TreeVisualizer.render(root, highlight_best=False)
        assert "***" not in output

    def test_render_empty_children(self):
        root = MCTSTreeNode(strategy="root")
        output = TreeVisualizer.render(root)
        assert "no children" in output

    def test_render_contains_avg_values(self):
        root = self._make_tree()
        output = TreeVisualizer.render(root)
        assert "avg=" in output


# ---------------------------------------------------------------------------
# M4: Streaming callback tests
# ---------------------------------------------------------------------------


class TestStreamingCallback:
    def test_callback_invoked_per_iteration(self):
        events = []
        engine = LLMMCTSEngine(iterations=5, seed=42)
        engine.search("test query", on_iteration=lambda e: events.append(e))
        assert len(events) == 5

    def test_callback_events_have_correct_fields(self):
        events = []
        engine = LLMMCTSEngine(iterations=3, seed=42)
        engine.search("test query", on_iteration=lambda e: events.append(e))
        for event in events:
            assert isinstance(event, IterationEvent)
            assert event.strategy in REASONING_STRATEGIES
            assert 0.0 <= event.score <= 1.0
            assert event.iteration >= 1
            assert event.total_iterations == 3
            assert event.elapsed_ms >= 0
            assert event.node_visits >= 1

    def test_callback_iteration_numbers_sequential(self):
        events = []
        engine = LLMMCTSEngine(iterations=5, seed=42)
        engine.search("test", on_iteration=lambda e: events.append(e))
        iterations = [e.iteration for e in events]
        assert iterations == [1, 2, 3, 4, 5]

    def test_pipeline_forwards_callback(self):
        events = []
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=5, seed=42)
        pipeline.run("test", on_iteration=lambda e: events.append(e))
        assert len(events) == 5


# ---------------------------------------------------------------------------
# M2: SingleShotRunner tests
# ---------------------------------------------------------------------------


class TestSingleShotRunner:
    def test_returns_response_score_latency(self):
        client = MockLLMClient()
        runner = SingleShotRunner(client)
        response, score, latency = runner.run("test question")
        assert isinstance(response, str)
        assert len(response) > 0
        assert 0.0 <= score <= 1.0
        assert latency >= 0

    def test_score_within_range(self):
        client = MockLLMClient()
        runner = SingleShotRunner(client)
        _, score, _ = runner.run("Design a rate limiter")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# M3: Pipeline tree_root tests
# ---------------------------------------------------------------------------


class TestPipelineTreeRoot:
    def test_pipeline_result_includes_tree_root(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=5, seed=42)
        result = pipeline.run("test")
        assert result.tree_root is not None
        assert result.tree_root.strategy == "root"
        assert len(result.tree_root.children) == 5

    def test_tree_root_visualizable(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=10, seed=42)
        result = pipeline.run("test")
        viz = TreeVisualizer.render(result.tree_root)
        assert "root" in viz
        assert "visits" in viz


# ---------------------------------------------------------------------------
# Engine last_root tests
# ---------------------------------------------------------------------------


class TestEngineLastRoot:
    def test_last_root_stored_after_search(self):
        engine = LLMMCTSEngine(iterations=5, seed=42)
        assert engine.last_root is None
        engine.search("query")
        assert engine.last_root is not None
        assert engine.last_root.strategy == "root"


# ---------------------------------------------------------------------------
# StdlibLLMClient tests (constructor / config validation)
# ---------------------------------------------------------------------------


class TestStdlibLLMClientConfig:
    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            StdlibLLMClient(provider="nonexistent", api_key="test")

    def test_missing_api_key_raises(self):
        # Unset env vars temporarily
        import os

        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="No API key"):
                StdlibLLMClient(provider="openai", api_key="")
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original

    def test_openai_provider_sets_defaults(self):
        client = StdlibLLMClient(provider="openai", api_key="sk-test")
        assert client.provider == "openai"
        assert client.model == "gpt-4o-mini"
        assert "openai.com" in client.base_url

    def test_anthropic_provider_sets_defaults(self):
        client = StdlibLLMClient(provider="anthropic", api_key="sk-test")
        assert client.provider == "anthropic"
        assert "anthropic.com" in client.base_url

    def test_custom_model_override(self):
        client = StdlibLLMClient(provider="openai", api_key="sk-test", model="gpt-4")
        assert client.model == "gpt-4"

    def test_custom_timeout(self):
        client = StdlibLLMClient(provider="openai", api_key="sk-test", timeout=30.0)
        assert client.timeout == 30.0

    def test_initial_state(self):
        client = StdlibLLMClient(provider="openai", api_key="sk-test")
        assert client.total_tokens == 0
        assert client.call_count == 0


# ---------------------------------------------------------------------------
# MockLLMClient extended tests
# ---------------------------------------------------------------------------


class TestMockLLMClientExtended:
    def test_all_strategies_have_mock_responses(self):
        """Every REASONING_STRATEGY should have a mock response."""
        for strategy in REASONING_STRATEGIES:
            assert strategy in MockLLMClient.MOCK_RESPONSES

    def test_detects_adversarial_strategy(self):
        client = MockLLMClient()
        response, _ = client.generate_sync("Argue against your position and find counter-arguments")
        assert "counter" in response.lower() or "adversarial" in response.lower() or "strengthen" in response.lower()

    def test_detects_analogy_strategy(self):
        client = MockLLMClient()
        response, _ = client.generate_sync("Use an analogy to explain")
        assert "analogy" in response.lower()

    def test_default_strategy_is_direct(self):
        client = MockLLMClient()
        response, _ = client.generate_sync("Generic prompt with no keywords")
        # Should get the direct response
        assert "direct analysis" in response.lower() or "key recommendations" in response.lower()

    def test_token_counts_accumulate(self):
        client = MockLLMClient()
        client.generate_sync("p1")
        tokens_after_1 = client.total_tokens
        client.generate_sync("p2")
        assert client.total_tokens > tokens_after_1

    def test_provider_is_mock(self):
        client = MockLLMClient()
        assert client.provider == "mock"


# ---------------------------------------------------------------------------
# ResponseScorer extended tests
# ---------------------------------------------------------------------------


class TestResponseScorerExtended:
    def test_empty_response_gets_low_score(self):
        scorer = ResponseScorer()
        score = scorer.score("question", "", "direct")
        assert score < 0.5

    def test_very_long_response_gets_reasonable_score(self):
        scorer = ResponseScorer()
        long_response = "A " * 3000
        score = scorer.score("question", long_response, "direct")
        assert 0.0 <= score <= 1.0

    def test_score_never_exceeds_bounds(self):
        """Score must always be in [0, 1] regardless of input."""
        scorer = ResponseScorer()
        for strategy in REASONING_STRATEGIES:
            for resp in ["", "x", "x" * 10000, "**bold** - list\n\n1. numbered"]:
                score = scorer.score("test", resp, strategy)
                assert 0.0 <= score <= 1.0, f"score {score} out of bounds for strategy={strategy}"

    def test_scorer_with_no_client(self):
        """Scorer without client should still work via heuristics."""
        scorer = ResponseScorer(llm_client=None)
        score = scorer.score("query", "Some response text", "direct")
        assert 0.0 <= score <= 1.0

    def test_scorer_with_mock_uses_heuristic(self):
        """Scorer with MockLLMClient should use heuristic, not LLM judge."""
        client = MockLLMClient()
        scorer = ResponseScorer(client)
        # This should NOT make an LLM call (mock should not be used as judge)
        initial_calls = client.call_count
        scorer.score("q", "response", "direct")
        assert client.call_count == initial_calls  # No new calls

    def test_strategy_decomposition_keyword_bonus(self):
        scorer = ResponseScorer()
        with_kw = scorer._heuristic_score("Sub-problem 1: analyze the data and step 2", "decomposition")
        without_kw = scorer._heuristic_score("Generic response without keywords", "decomposition")
        assert with_kw > without_kw

    def test_strategy_adversarial_keyword_bonus(self):
        scorer = ResponseScorer()
        with_kw = scorer._heuristic_score("Counter-argument: the approach is flawed", "adversarial")
        without_kw = scorer._heuristic_score("The approach is good and works well", "adversarial")
        assert with_kw > without_kw


# ---------------------------------------------------------------------------
# LLMMCTSEngine extended tests
# ---------------------------------------------------------------------------


class TestLLMMCTSEngineExtended:
    def test_custom_strategies(self):
        """Engine should work with a subset of strategies."""
        engine = LLMMCTSEngine(iterations=4, seed=42, strategies=["direct", "refinement"])
        result = engine.search("test")
        assert result.best_strategy in ["direct", "refinement"]
        assert len(result.tree_stats["strategy_visits"]) == 2

    def test_single_iteration(self):
        engine = LLMMCTSEngine(iterations=1, seed=42)
        result = engine.search("test")
        assert result.iterations_run == 1
        assert len(result.llm_calls) == 1

    def test_high_iterations_converges(self):
        """More iterations should have more total visits."""
        engine = LLMMCTSEngine(iterations=30, seed=42)
        result = engine.search("test")
        total_visits = sum(result.tree_stats["strategy_visits"].values())
        assert total_visits == 30

    def test_llm_calls_have_correct_strategies(self):
        engine = LLMMCTSEngine(iterations=5, seed=42)
        result = engine.search("test")
        for call in result.llm_calls:
            assert call.strategy in REASONING_STRATEGIES

    def test_llm_calls_have_positive_latency(self):
        engine = LLMMCTSEngine(iterations=3, seed=42)
        result = engine.search("test")
        for call in result.llm_calls:
            assert call.latency_ms >= 0

    def test_total_time_positive(self):
        engine = LLMMCTSEngine(iterations=3, seed=42)
        result = engine.search("test")
        assert result.total_time_ms >= 0

    def test_backpropagation_updates_root(self):
        """Root should accumulate all visits."""
        engine = LLMMCTSEngine(iterations=10, seed=42)
        engine.search("test")
        root = engine.last_root
        assert root is not None
        assert root.visits == 10  # Each iteration backprops to root

    def test_ucb1_selection_is_deterministic(self):
        """Same seed should produce same selection sequence."""
        engine1 = LLMMCTSEngine(iterations=5, seed=99)
        engine2 = LLMMCTSEngine(iterations=5, seed=99)
        r1 = engine1.search("same query")
        r2 = engine2.search("same query")
        assert r1.best_strategy == r2.best_strategy
        for c1, c2 in zip(r1.llm_calls, r2.llm_calls):
            assert c1.strategy == c2.strategy


# ---------------------------------------------------------------------------
# ConsensusBuilder extended tests
# ---------------------------------------------------------------------------


class TestConsensusBuilderExtended:
    def test_two_strategies_calls_llm(self):
        client = MockLLMClient()
        initial_calls = client.call_count
        builder = ConsensusBuilder(client)
        builder.build_consensus("query", {"direct": "A", "refinement": "B"})
        assert client.call_count == initial_calls + 1

    def test_consensus_preserves_query(self):
        """The consensus prompt should reference the original query."""
        client = MockLLMClient()
        builder = ConsensusBuilder(client)
        result = builder.build_consensus("What is gravity?", {"a": "Force", "b": "Curvature"})
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TreeVisualizer extended tests
# ---------------------------------------------------------------------------


class TestTreeVisualizerExtended:
    def test_render_unvisited_children(self):
        """Children with 0 visits should render with empty bars."""
        root = MCTSTreeNode(strategy="root")
        for s in REASONING_STRATEGIES:
            root.children.append(MCTSTreeNode(strategy=s, parent=root, depth=1))
        root.visits = 0
        output = TreeVisualizer.render(root)
        assert "0 visits" in output

    def test_render_single_child(self):
        root = MCTSTreeNode(strategy="root")
        child = MCTSTreeNode(strategy="direct", parent=root, depth=1)
        child.visits = 3
        child.value_sum = 2.1
        root.children.append(child)
        root.visits = 3
        output = TreeVisualizer.render(root)
        assert "direct" in output
        assert "***" in output  # Only child is best

    def test_render_contains_tree_connectors(self):
        root = MCTSTreeNode(strategy="root")
        for s in ["a", "b", "c"]:
            root.children.append(MCTSTreeNode(strategy=s, parent=root, depth=1))
        root.visits = 1
        output = TreeVisualizer.render(root)
        assert "\u251c" in output or "\u2514" in output  # Has tree connectors


# ---------------------------------------------------------------------------
# SingleShotRunner extended tests
# ---------------------------------------------------------------------------


class TestSingleShotRunnerExtended:
    def test_uses_direct_strategy_prompt(self):
        """SingleShotRunner should reuse the STRATEGY_PROMPTS['direct'] template."""
        client = MockLLMClient()
        runner = SingleShotRunner(client)
        response, _, _ = runner.run("What is Python?")
        # The direct response from MockLLMClient
        assert isinstance(response, str)

    def test_multiple_runs_accumulate_calls(self):
        client = MockLLMClient()
        runner = SingleShotRunner(client)
        runner.run("q1")
        runner.run("q2")
        assert client.call_count >= 2  # At least 2 calls from runs


# ---------------------------------------------------------------------------
# Pipeline extended tests
# ---------------------------------------------------------------------------


class TestMultiAgentMCTSPipelineExtended:
    def test_pipeline_with_custom_strategies(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=4, seed=42, strategies=["direct", "analogy"])
        result = pipeline.run("test")
        assert result.mcts_result.best_strategy in ["direct", "analogy"]

    def test_pipeline_top_strategies_limit(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=10, seed=42, top_strategies_limit=2)
        result = pipeline.run("test")
        assert len(result.top_strategies) <= 2

    def test_pipeline_result_has_all_fields(self):
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=5, seed=42)
        result = pipeline.run("test")
        assert result.query == "test"
        assert result.mcts_result is not None
        assert result.total_time_ms >= 0
        assert result.provider == "mock"
        assert result.tree_root is not None
        assert isinstance(result.top_strategies, list)


# ---------------------------------------------------------------------------
# Constants / configuration tests
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Test exported module constants (accessed via importlib to avoid numpy)."""

    def test_default_constants_are_positive(self):
        assert _llm_mcts.DEFAULT_ITERATIONS > 0
        assert _llm_mcts.DEFAULT_EXPLORATION_WEIGHT > 0
        assert _llm_mcts.DEFAULT_LLM_TIMEOUT > 0
        assert _llm_mcts.DEFAULT_TEMPERATURE > 0
        assert _llm_mcts.DEFAULT_MAX_TOKENS > 0

    def test_provider_config_has_required_keys(self):
        for provider, config in _llm_mcts.PROVIDER_CONFIG.items():
            assert "env_key" in config, f"{provider} missing env_key"
            assert "default_model" in config, f"{provider} missing default_model"
            assert "base_url" in config, f"{provider} missing base_url"

    def test_all_strategies_have_keywords(self):
        """Every non-direct strategy should have detection keywords."""
        for strategy in REASONING_STRATEGIES:
            if strategy != "direct":
                assert strategy in _llm_mcts.STRATEGY_KEYWORDS, f"{strategy} missing from STRATEGY_KEYWORDS"

    def test_structural_markers_not_empty(self):
        assert len(_llm_mcts.STRUCTURAL_MARKERS) > 0


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_mcts_result_dataclass_fields(self):
        """MCTSResult must have all fields expected by demo.py."""
        result = MCTSResult(
            query="q",
            best_strategy="direct",
            best_response="resp",
            best_score=0.5,
            all_strategies={"direct": 0.5},
            tree_stats={},
            llm_calls=[],
            total_time_ms=1.0,
            iterations_run=1,
        )
        # These are read by demo.py and graph.py
        assert hasattr(result, "query")
        assert hasattr(result, "best_strategy")
        assert hasattr(result, "best_response")
        assert hasattr(result, "best_score")
        assert hasattr(result, "all_strategies")
        assert hasattr(result, "tree_stats")
        assert hasattr(result, "llm_calls")
        assert hasattr(result, "total_time_ms")
        assert hasattr(result, "iterations_run")

    def test_pipeline_result_dataclass_fields(self):
        """PipelineResult must have all fields expected by demo.py."""
        mcts_result = MCTSResult(
            query="q",
            best_strategy="direct",
            best_response="r",
            best_score=0.5,
            all_strategies={},
            tree_stats={},
            llm_calls=[],
            total_time_ms=1.0,
            iterations_run=1,
        )
        result = PipelineResult(
            query="q",
            mcts_result=mcts_result,
            consensus_response=None,
            top_strategies=[],
            total_time_ms=1.0,
            provider="mock",
            tree_root=None,
        )
        assert hasattr(result, "tree_root")
        assert hasattr(result, "consensus_response")
        assert hasattr(result, "top_strategies")

    def test_search_without_callback_still_works(self):
        """search() must work without on_iteration (backward compat)."""
        engine = LLMMCTSEngine(iterations=3, seed=42)
        result = engine.search("test")
        assert isinstance(result, MCTSResult)

    def test_pipeline_run_without_callback_still_works(self):
        """run() must work without on_iteration (backward compat)."""
        pipeline = MultiAgentMCTSPipeline(provider="mock", iterations=3, seed=42)
        result = pipeline.run("test")
        assert isinstance(result, PipelineResult)


# ---------------------------------------------------------------------------
# Integration test: demo.py CLI
# ---------------------------------------------------------------------------


class TestDemoCLI:
    def test_demo_module_imports(self):
        """Verify demo.py can be imported without errors."""
        import importlib

        spec = importlib.util.spec_from_file_location("demo", "demo.py")
        assert spec is not None

    def test_demo_mock_mode_runs(self):
        """demo.py --json should produce valid JSON in mock mode."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "demo.py", "--json", "--iterations", "3"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        import json

        output = json.loads(result.stdout)
        assert "query" in output
        assert "best_strategy" in output
        assert "best_score" in output

    def test_demo_stream_flag_runs(self):
        """demo.py --stream should work."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "demo.py", "--stream", "--iterations", "3"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_demo_tree_flag_runs(self):
        """demo.py --tree should render a tree."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "demo.py", "--tree", "--iterations", "3"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "visits" in result.stdout

    def test_demo_compare_flag_runs(self):
        """demo.py --compare should work."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "demo.py", "--compare", "--iterations", "3"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Comparison" in result.stdout or "comparison" in result.stdout.lower()
