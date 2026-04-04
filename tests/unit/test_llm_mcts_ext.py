"""
Extended tests for LLM-powered MCTS engine to increase coverage on missed lines.

Targets:
- StdlibLLMClient.generate_sync (OpenAI/Anthropic dispatch)
- StdlibLLMClient._call_openai / _call_anthropic
- StdlibLLMClient._http_post error handling
- ResponseScorer._llm_judge_score (LLM-as-judge path)
- ConsensusBuilder.build_consensus exception fallback
- MultiAgentMCTSPipeline consensus with strategy_responses > 1
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

# Import via importlib to avoid triggering the mcts package __init__.py
# which may require numpy.
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
assert _spec is not None and _spec.loader is not None
_llm_mcts = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("llm_mcts", _llm_mcts)
_spec.loader.exec_module(_llm_mcts)

ConsensusBuilder = _llm_mcts.ConsensusBuilder
LLMMCTSEngine = _llm_mcts.LLMMCTSEngine
MCTSResult = _llm_mcts.MCTSResult
MCTSTreeNode = _llm_mcts.MCTSTreeNode
MockLLMClient = _llm_mcts.MockLLMClient
MultiAgentMCTSPipeline = _llm_mcts.MultiAgentMCTSPipeline
ResponseScorer = _llm_mcts.ResponseScorer
StdlibLLMClient = _llm_mcts.StdlibLLMClient
REASONING_STRATEGIES = _llm_mcts.REASONING_STRATEGIES
LLMCall = _llm_mcts.LLMCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_client(**kwargs) -> StdlibLLMClient:
    """Create an OpenAI StdlibLLMClient without needing a real env key."""
    return StdlibLLMClient(provider="openai", api_key="sk-test-fake", **kwargs)


def _make_anthropic_client(**kwargs) -> StdlibLLMClient:
    """Create an Anthropic StdlibLLMClient without needing a real env key."""
    return StdlibLLMClient(provider="anthropic", api_key="sk-ant-test-fake", **kwargs)


# ---------------------------------------------------------------------------
# StdlibLLMClient.generate_sync – OpenAI dispatch (lines 296-304)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStdlibGenerateSyncOpenAI:
    """Cover generate_sync dispatching to _call_openai."""

    def test_generate_sync_calls_openai(self):
        client = _make_openai_client()
        fake_response = {
            "choices": [{"message": {"content": "Hello from OpenAI"}}],
            "usage": {"total_tokens": 42},
        }
        with patch.object(client, "_http_post", return_value=fake_response):
            text, tokens = client.generate_sync("test prompt")
        assert text == "Hello from OpenAI"
        assert tokens == 42
        assert client.call_count == 1
        assert client.total_tokens == 42

    def test_generate_sync_openai_no_usage(self):
        """usage may be absent; tokens should default to 0."""
        client = _make_openai_client()
        fake_response = {
            "choices": [{"message": {"content": "response"}}],
        }
        with patch.object(client, "_http_post", return_value=fake_response):
            text, tokens = client.generate_sync("prompt")
        assert text == "response"
        assert tokens == 0

    def test_generate_sync_openai_custom_params(self):
        client = _make_openai_client()
        fake_response = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 10},
        }
        with patch.object(client, "_http_post", return_value=fake_response) as mock_post:
            client.generate_sync("p", temperature=0.2, max_tokens=50)
        payload = mock_post.call_args[0][1]
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 50


# ---------------------------------------------------------------------------
# StdlibLLMClient._call_anthropic (lines 325-346)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStdlibGenerateSyncAnthropic:
    """Cover generate_sync dispatching to _call_anthropic."""

    def test_generate_sync_calls_anthropic(self):
        client = _make_anthropic_client()
        fake_response = {
            "content": [{"type": "text", "text": "Hello from Anthropic"}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        with patch.object(client, "_http_post", return_value=fake_response):
            text, tokens = client.generate_sync("test prompt")
        assert text == "Hello from Anthropic"
        assert tokens == 30
        assert client.call_count == 1
        assert client.total_tokens == 30

    def test_anthropic_multiple_content_blocks(self):
        client = _make_anthropic_client()
        fake_response = {
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 15},
        }
        with patch.object(client, "_http_post", return_value=fake_response):
            text, tokens = client.generate_sync("prompt")
        assert text == "Part 1\nPart 2"
        assert tokens == 20

    def test_anthropic_non_text_blocks_ignored(self):
        client = _make_anthropic_client()
        fake_response = {
            "content": [
                {"type": "image", "data": "..."},
                {"type": "text", "text": "Only text"},
            ],
            "usage": {"input_tokens": 3, "output_tokens": 7},
        }
        with patch.object(client, "_http_post", return_value=fake_response):
            text, tokens = client.generate_sync("prompt")
        assert text == "Only text"

    def test_anthropic_empty_content(self):
        client = _make_anthropic_client()
        fake_response = {
            "content": [],
            "usage": {"input_tokens": 2, "output_tokens": 0},
        }
        with patch.object(client, "_http_post", return_value=fake_response):
            text, tokens = client.generate_sync("prompt")
        assert text == ""

    def test_anthropic_no_usage(self):
        client = _make_anthropic_client()
        fake_response = {
            "content": [{"type": "text", "text": "resp"}],
        }
        with patch.object(client, "_http_post", return_value=fake_response):
            text, tokens = client.generate_sync("prompt")
        assert text == "resp"
        assert tokens == 0

    def test_anthropic_temperature_clamped(self):
        """Anthropic temperature is clamped to max 1.0."""
        client = _make_anthropic_client()
        fake_response = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        with patch.object(client, "_http_post", return_value=fake_response) as mock_post:
            client.generate_sync("prompt", temperature=1.5)
        payload = mock_post.call_args[0][1]
        assert payload["temperature"] == 1.0

    def test_anthropic_headers_include_version(self):
        client = _make_anthropic_client()
        fake_response = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        with patch.object(client, "_http_post", return_value=fake_response) as mock_post:
            client.generate_sync("prompt")
        headers = mock_post.call_args[0][2]
        assert "anthropic-version" in headers
        assert "x-api-key" in headers


# ---------------------------------------------------------------------------
# StdlibLLMClient._http_post error handling (lines 350-364)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStdlibHttpPostErrors:
    """Cover _http_post HTTPError and URLError handling."""

    def test_http_error_raises_runtime_error(self):
        client = _make_openai_client()
        mock_error = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b'{"error": "rate limited"}')),
        )
        with patch("urllib.request.urlopen", side_effect=mock_error):
            with pytest.raises(RuntimeError, match="HTTP 429"):
                client._http_post("https://api.openai.com/v1/chat/completions", {}, {})

    def test_url_error_raises_runtime_error(self):
        client = _make_openai_client()
        mock_error = urllib.error.URLError(reason="Connection refused")
        with patch("urllib.request.urlopen", side_effect=mock_error):
            with pytest.raises(RuntimeError, match="Connection error"):
                client._http_post("https://api.openai.com/v1/chat/completions", {}, {})

    def test_http_error_includes_provider(self):
        client = _make_anthropic_client()
        mock_error = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b"server error")),
        )
        with patch("urllib.request.urlopen", side_effect=mock_error):
            with pytest.raises(RuntimeError, match="anthropic"):
                client._http_post("https://api.anthropic.com/v1/messages", {}, {})

    def test_url_error_includes_provider(self):
        client = _make_anthropic_client()
        mock_error = urllib.error.URLError(reason="DNS resolution failed")
        with patch("urllib.request.urlopen", side_effect=mock_error):
            with pytest.raises(RuntimeError, match="anthropic"):
                client._http_post("https://api.anthropic.com/v1/messages", {}, {})

    def test_http_post_success(self):
        """Covers the happy path through _http_post with urlopen."""
        client = _make_openai_client()
        fake_json = json.dumps({"result": "ok"}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_json
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = client._http_post("https://example.com", {"key": "val"}, {"H": "V"})
        assert result == {"result": "ok"}


# ---------------------------------------------------------------------------
# ResponseScorer._llm_judge_score (lines 511, 516-533)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResponseScorerLLMJudge:
    """Cover the LLM-as-judge scoring path."""

    def _make_real_scorer(self, generate_return):
        """Create a scorer with a non-Mock LLM client that returns a fixed value."""
        client = _make_openai_client()
        with patch.object(client, "_http_post"):
            pass  # just to validate it's constructed
        # Patch generate_sync directly
        client.generate_sync = MagicMock(return_value=generate_return)
        scorer = ResponseScorer(llm_client=client)
        return scorer

    def test_llm_judge_valid_json_score(self):
        """LLM returns valid JSON with a score."""
        scorer = self._make_real_scorer(('{"score": 0.85, "reason": "good answer"}', 50))
        score = scorer.score("What is Python?", "Python is a programming language.", "direct")
        assert score == 0.85

    def test_llm_judge_score_clamped_high(self):
        """Score > 1.0 should be clamped to 1.0."""
        scorer = self._make_real_scorer(('{"score": 1.5, "reason": "amazing"}', 50))
        score = scorer.score("q", "response", "direct")
        assert score == 1.0

    def test_llm_judge_score_clamped_low(self):
        """Score < 0.0 should be clamped to 0.0."""
        scorer = self._make_real_scorer(('{"score": -0.5, "reason": "terrible"}', 50))
        score = scorer.score("q", "response", "direct")
        assert score == 0.0

    def test_llm_judge_no_score_key_defaults(self):
        """Missing 'score' key should default to 0.5."""
        scorer = self._make_real_scorer(('{"reason": "no score key"}', 50))
        score = scorer.score("q", "response", "direct")
        assert score == 0.5

    def test_llm_judge_json_with_surrounding_text(self):
        """LLM may return text around the JSON; should still parse."""
        scorer = self._make_real_scorer(('Here is my rating: {"score": 0.7, "reason": "decent"} done.', 50))
        score = scorer.score("q", "response", "direct")
        assert score == 0.7

    def test_llm_judge_no_json_falls_back_to_heuristic(self):
        """If no JSON found, fall back to heuristic scoring."""
        scorer = self._make_real_scorer(("I rate this a 7 out of 10", 50))
        score = scorer.score("q", "response", "direct")
        # Should be a heuristic score (typically 0.3-0.8 range)
        assert 0.0 <= score <= 1.0

    def test_llm_judge_exception_falls_back_to_heuristic(self):
        """If generate_sync raises, fall back to heuristic scoring."""
        client = _make_openai_client()
        client.generate_sync = MagicMock(side_effect=RuntimeError("API error"))
        scorer = ResponseScorer(llm_client=client)
        score = scorer.score("q", "response", "direct")
        assert 0.0 <= score <= 1.0

    def test_llm_judge_invalid_json_falls_back(self):
        """If JSON is malformed, fall back to heuristic scoring."""
        scorer = self._make_real_scorer(('{"score": not_a_number}', 50))
        score = scorer.score("q", "response", "direct")
        assert 0.0 <= score <= 1.0

    def test_llm_judge_uses_correct_temperature(self):
        """Judge calls should use low temperature (0.1)."""
        client = _make_openai_client()
        client.generate_sync = MagicMock(return_value=('{"score": 0.5}', 10))
        scorer = ResponseScorer(llm_client=client)
        scorer.score("q", "response", "direct")
        call_kwargs = client.generate_sync.call_args
        assert call_kwargs[1]["temperature"] == 0.1 or call_kwargs[0][1] == 0.1


# ---------------------------------------------------------------------------
# ConsensusBuilder exception fallback (lines 814-816)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConsensusBuilderExceptionFallback:
    """Cover the exception path in build_consensus."""

    def test_exception_returns_first_response(self):
        """When LLM call fails, should return first strategy response."""
        client = MockLLMClient()
        client.generate_sync = MagicMock(side_effect=RuntimeError("API failure"))
        builder = ConsensusBuilder(client)
        result = builder.build_consensus(
            "query",
            {"direct": "Fallback answer A", "refinement": "Answer B"},
        )
        assert result == "Fallback answer A"

    def test_exception_with_different_ordering(self):
        """Verify it returns the first item from the dict."""
        client = MockLLMClient()
        client.generate_sync = MagicMock(side_effect=Exception("boom"))
        builder = ConsensusBuilder(client)
        responses = {"analogy": "Analogy response", "adversarial": "Adversarial response"}
        result = builder.build_consensus("q", responses)
        assert result == "Analogy response"


# ---------------------------------------------------------------------------
# MultiAgentMCTSPipeline – consensus with strategy_responses (line 999)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineConsensusIntegration:
    """Cover the pipeline consensus branch where strategy_responses > 1."""

    def test_consensus_invoked_when_multiple_top_strategies(self):
        """Pipeline should call consensus when there are multiple top strategies with score > 0."""
        pipeline = MultiAgentMCTSPipeline(
            provider="mock",
            iterations=20,
            seed=42,
            use_consensus=True,
        )
        result = pipeline.run("Explain MCTS in detail")
        # With 20 iterations and mock, we should get multiple strategies with score > 0
        assert result.mcts_result is not None
        strategies_with_score = [s for s, v in result.top_strategies if v > 0]
        if len(strategies_with_score) > 1:
            # Consensus should have been built
            assert result.consensus_response is not None

    def test_pipeline_consensus_none_when_disabled(self):
        pipeline = MultiAgentMCTSPipeline(
            provider="mock",
            iterations=20,
            seed=42,
            use_consensus=False,
        )
        result = pipeline.run("test query")
        assert result.consensus_response is None

    def test_pipeline_consensus_none_when_single_strategy(self):
        """With only one strategy, consensus should be None."""
        pipeline = MultiAgentMCTSPipeline(
            provider="mock",
            iterations=5,
            seed=42,
            use_consensus=True,
            strategies=["direct"],
        )
        result = pipeline.run("test")
        # Only one strategy, so no consensus
        assert result.consensus_response is None

    def test_pipeline_consensus_skipped_when_strategy_responses_too_few(self):
        """Cover branch where top_strategies > 1 but strategy_responses <= 1.

        This happens when the llm_calls don't contain matching strategies for
        all top strategies (e.g. if call.strategy is not in dict(top_strategies)).
        We achieve this by patching the engine search to return crafted results.
        """
        pipeline = MultiAgentMCTSPipeline(
            provider="mock",
            iterations=5,
            seed=42,
            use_consensus=True,
        )
        # Create a fake MCTSResult where top strategies exist with score > 0
        # but llm_calls only have one matching strategy
        fake_mcts = MCTSResult(
            query="q",
            best_strategy="direct",
            best_response="response",
            best_score=0.8,
            all_strategies={"direct": 0.8, "refinement": 0.7, "analogy": 0.6},
            tree_stats={
                "total_iterations": 5,
                "total_llm_calls": 5,
                "total_tokens": 100,
                "strategy_visits": {},
                "strategy_values": {},
                "provider": "mock",
            },
            llm_calls=[
                # Only "direct" calls - no "refinement" or "analogy" calls
                LLMCall(strategy="direct", prompt="p", response="r", score=0.8, latency_ms=1.0, tokens_used=10),
            ],
            total_time_ms=10.0,
            iterations_run=5,
        )
        with patch.object(pipeline._engine, "search", return_value=fake_mcts):
            result = pipeline.run("q")
        # top_strategies has 3 items (all > 0), but only 1 strategy_response
        # so consensus should be None
        assert result.consensus_response is None


# ---------------------------------------------------------------------------
# StdlibLLMClient – generate_sync full round trip (mocked _http_post)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStdlibClientRoundTrip:
    """Full round-trip tests using mocked _http_post to cover call paths end-to-end."""

    def test_openai_round_trip_accumulates_tokens(self):
        client = _make_openai_client()
        fake = {
            "choices": [{"message": {"content": "resp1"}}],
            "usage": {"total_tokens": 100},
        }
        with patch.object(client, "_http_post", return_value=fake):
            client.generate_sync("p1")
            client.generate_sync("p2")
        assert client.call_count == 2
        assert client.total_tokens == 200

    def test_anthropic_round_trip_accumulates_tokens(self):
        client = _make_anthropic_client()
        fake = {
            "content": [{"type": "text", "text": "resp"}],
            "usage": {"input_tokens": 10, "output_tokens": 40},
        }
        with patch.object(client, "_http_post", return_value=fake):
            client.generate_sync("p1")
            client.generate_sync("p2")
        assert client.call_count == 2
        assert client.total_tokens == 100  # 50 * 2

    def test_openai_generate_sync_sends_correct_payload(self):
        client = _make_openai_client(model="gpt-4")
        fake = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 5},
        }
        with patch.object(client, "_http_post", return_value=fake) as mock_post:
            client.generate_sync("hello world", temperature=0.5, max_tokens=256)
        url, payload, headers = mock_post.call_args[0]
        assert payload["model"] == "gpt-4"
        assert payload["messages"][0]["content"] == "hello world"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 256
        assert "Bearer" in headers["Authorization"]


# ---------------------------------------------------------------------------
# Edge cases for LLMMCTSEngine with real (mocked) LLM client
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEngineWithRealClient:
    """Test LLMMCTSEngine when wired to a StdlibLLMClient (mocked HTTP)."""

    def test_engine_with_stdlib_client_runs(self):
        client = _make_openai_client()
        fake_response = {
            "choices": [{"message": {"content": "LLM response text with some detail"}}],
            "usage": {"total_tokens": 50},
        }
        with patch.object(client, "_http_post", return_value=fake_response):
            # The scorer will also call generate_sync for LLM-as-judge
            engine = LLMMCTSEngine(llm_client=client, iterations=3, seed=42)
            # Mock the judge response too
            judge_response = {
                "choices": [{"message": {"content": '{"score": 0.7, "reason": "good"}'}}],
                "usage": {"total_tokens": 20},
            }

        # Need to handle alternating responses (simulate + judge)
        responses = []
        for _ in range(3):
            responses.append(fake_response)  # simulate call
            responses.append(judge_response)  # judge call

        with patch.object(client, "_http_post", side_effect=responses):
            result = engine.search("test query")

        assert isinstance(result, MCTSResult)
        assert result.iterations_run == 3
        assert len(result.llm_calls) == 3


# ---------------------------------------------------------------------------
# LLMCall dataclass (ensuring it's constructable and stores values)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMCallDataclass:
    def test_llm_call_defaults(self):
        call = LLMCall(
            strategy="direct",
            prompt="p",
            response="r",
            score=0.5,
            latency_ms=10.0,
        )
        assert call.tokens_used == 0

    def test_llm_call_with_tokens(self):
        call = LLMCall(
            strategy="refinement",
            prompt="prompt text",
            response="response text",
            score=0.8,
            latency_ms=25.3,
            tokens_used=150,
        )
        assert call.tokens_used == 150
        assert call.strategy == "refinement"
