"""
LLM-Powered MCTS Engine - Monte Carlo Tree Search driving real LLM reasoning.

This module bridges the gap between the mechanical MCTS algorithm and actual
LLM-powered reasoning. Instead of simulating abstract states, each MCTS node
represents a reasoning strategy, and rollouts are real LLM calls that explore
different approaches to answering a query.

Zero external dependencies beyond the Python standard library.

Strategies explored by MCTS:
- Direct: Straightforward answer
- Decomposition: Break into sub-problems (HRM-style)
- Refinement: Iterative improvement (TRM-style)
- Analogy: Reason by analogy
- Adversarial: Consider counterarguments
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

REASONING_STRATEGIES = [
    "direct",
    "decomposition",
    "refinement",
    "analogy",
    "adversarial",
]

STRATEGY_PROMPTS: dict[str, str] = {
    "direct": (
        "Answer the following question directly and concisely.\n\n"
        "Question: {query}\n\n"
        "Provide a clear, well-structured answer."
    ),
    "decomposition": (
        "Break the following question into sub-problems, solve each, "
        "then synthesize a final answer.\n\n"
        "Question: {query}\n\n"
        "Step 1: Identify sub-problems\n"
        "Step 2: Solve each sub-problem\n"
        "Step 3: Synthesize final answer"
    ),
    "refinement": (
        "Answer the following question, then critically review your answer "
        "and refine it for accuracy and completeness.\n\n"
        "Question: {query}\n\n"
        "Initial answer:\n"
        "Critical review:\n"
        "Refined answer:"
    ),
    "analogy": (
        "Answer the following question by first finding a useful analogy, "
        "then reasoning through the analogy to reach your answer.\n\n"
        "Question: {query}\n\n"
        "Analogy:\n"
        "Reasoning through analogy:\n"
        "Final answer:"
    ),
    "adversarial": (
        "Answer the following question, then argue against your own answer "
        "to find weaknesses, and produce a stronger final answer.\n\n"
        "Question: {query}\n\n"
        "Initial answer:\n"
        "Counter-arguments:\n"
        "Strengthened answer:"
    ),
}


@dataclass
class LLMCall:
    """Record of a single LLM API call."""

    strategy: str
    prompt: str
    response: str
    score: float
    latency_ms: float
    tokens_used: int = 0


@dataclass
class MCTSTreeNode:
    """A node in the LLM-MCTS reasoning tree."""

    strategy: str
    parent: MCTSTreeNode | None = None
    children: list[MCTSTreeNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    responses: list[str] = field(default_factory=list)
    depth: int = 0

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb1(self, parent_visits: int, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    @property
    def best_response(self) -> str | None:
        if not self.responses:
            return None
        return self.responses[-1]


@dataclass
class MCTSResult:
    """Result of an LLM-MCTS search."""

    query: str
    best_strategy: str
    best_response: str
    best_score: float
    all_strategies: dict[str, float]
    tree_stats: dict[str, Any]
    llm_calls: list[LLMCall]
    total_time_ms: float
    iterations_run: int


# ---------------------------------------------------------------------------
# LLM Client (stdlib-only, supports OpenAI and Anthropic)
# ---------------------------------------------------------------------------


class StdlibLLMClient:
    """
    Lightweight LLM client using only Python standard library.

    Supports OpenAI and Anthropic APIs via urllib.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        import os

        self.provider = provider.lower()
        self.timeout = timeout
        self.total_tokens = 0
        self.call_count = 0

        if self.provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.model = model or "gpt-4o-mini"
            self.base_url = "https://api.openai.com/v1/chat/completions"
        elif self.provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self.model = model or "claude-sonnet-4-20250514"
            self.base_url = "https://api.anthropic.com/v1/messages"
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'.")

        if not self.api_key:
            raise ValueError(
                f"No API key for {provider}. Set {'OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY'} "
                "environment variable or pass api_key parameter."
            )

    def generate_sync(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> tuple[str, int]:
        """
        Synchronous LLM call. Returns (response_text, tokens_used).
        """
        if self.provider == "openai":
            return self._call_openai(prompt, temperature, max_tokens)
        else:
            return self._call_anthropic(prompt, temperature, max_tokens)

    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = self._http_post(self.base_url, payload, headers)
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        self.total_tokens += tokens
        self.call_count += 1
        return text, tokens

    def _call_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": min(temperature, 1.0),
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = self._http_post(self.base_url, payload, headers)
        text_parts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
        text = "\n".join(text_parts)
        tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
        self.total_tokens += tokens
        self.call_count += 1
        return text, tokens

    def _http_post(self, url: str, payload: dict, headers: dict) -> dict:
        """Make an HTTP POST request using urllib."""
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        # Create SSL context that works in most environments
        ctx = ssl.create_default_context()

        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {self.provider}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error to {self.provider}: {e.reason}") from e


# ---------------------------------------------------------------------------
# Mock LLM Client for demos without API keys
# ---------------------------------------------------------------------------


class MockLLMClient:
    """
    Mock LLM client that returns plausible responses without API calls.

    Used for demos, testing, and environments without API access.
    """

    MOCK_RESPONSES: dict[str, str] = {
        "direct": (
            "Based on the key considerations, here is a direct analysis:\n\n"
            "The primary factors involve balancing trade-offs between scalability, "
            "maintainability, and performance. A pragmatic approach considers both "
            "short-term delivery needs and long-term architectural sustainability.\n\n"
            "Key recommendations:\n"
            "1. Start with the simplest approach that meets current requirements\n"
            "2. Design interfaces that allow future evolution\n"
            "3. Measure before optimizing"
        ),
        "decomposition": (
            "**Sub-problem 1: Core Requirements Analysis**\n"
            "Identifying the fundamental constraints and goals.\n\n"
            "**Sub-problem 2: Technical Feasibility**\n"
            "Evaluating which approaches are viable given the constraints.\n\n"
            "**Sub-problem 3: Trade-off Evaluation**\n"
            "Comparing approaches across dimensions of cost, complexity, and risk.\n\n"
            "**Synthesis:**\n"
            "By decomposing the problem, we can see that the optimal path forward "
            "combines elements from multiple approaches: use a modular architecture "
            "that allows independent scaling of components while maintaining simplicity "
            "in the overall system design."
        ),
        "refinement": (
            "**Initial Answer:**\n"
            "The best approach prioritizes simplicity and iterative improvement.\n\n"
            "**Critical Review:**\n"
            "The initial answer overlooks edge cases around distributed state management "
            "and doesn't address failure recovery. It also assumes uniform load patterns.\n\n"
            "**Refined Answer:**\n"
            "A robust solution must account for non-uniform load, partial failures, and "
            "state consistency. The recommended approach uses event-driven architecture "
            "with idempotent operations, circuit breakers for fault isolation, and "
            "eventual consistency where strong consistency isn't required."
        ),
        "analogy": (
            "**Analogy: City Traffic Management**\n"
            "Just as a city manages traffic through a combination of traffic lights "
            "(deterministic rules), roundabouts (self-organizing), and highway ramps "
            "(throttling), a well-designed system uses multiple coordination strategies.\n\n"
            "**Reasoning Through Analogy:**\n"
            "Traffic lights = rate limiters and load balancers\n"
            "Roundabouts = eventual consistency and self-healing\n"
            "Highway ramps = backpressure and admission control\n\n"
            "**Answer:**\n"
            "Layer multiple coordination mechanisms: deterministic routing for predictable "
            "loads, self-organizing components for dynamic adaptation, and admission "
            "control for overload scenarios."
        ),
        "adversarial": (
            "**Initial Position:**\n"
            "A microservices architecture provides the best scalability and team autonomy.\n\n"
            "**Counter-arguments:**\n"
            "1. Distributed systems add network latency and partial failure modes\n"
            "2. Data consistency across services requires complex saga patterns\n"
            "3. Operational overhead of many services exceeds benefits for small teams\n"
            "4. Premature decomposition creates coupling through shared data models\n\n"
            "**Strengthened Answer:**\n"
            "Start with a well-modularized monolith that enforces service boundaries "
            "internally. Extract services only when a specific module has genuinely "
            "different scaling, deployment, or team ownership requirements. This avoids "
            "distributed systems complexity while preserving the option to decompose later."
        ),
    }

    def __init__(self):
        self.total_tokens = 0
        self.call_count = 0
        self.provider = "mock"

    def generate_sync(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> tuple[str, int]:
        """Return a mock response based on the detected strategy."""
        strategy = self._detect_strategy(prompt)
        response = self.MOCK_RESPONSES.get(strategy, self.MOCK_RESPONSES["direct"])

        # Simulate slight variation
        mock_tokens = random.randint(150, 400)
        self.total_tokens += mock_tokens
        self.call_count += 1

        return response, mock_tokens

    def _detect_strategy(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "break" in prompt_lower or "sub-problem" in prompt_lower:
            return "decomposition"
        elif "refine" in prompt_lower or "review" in prompt_lower:
            return "refinement"
        elif "analogy" in prompt_lower:
            return "analogy"
        elif "argue against" in prompt_lower or "counter" in prompt_lower:
            return "adversarial"
        return "direct"


# ---------------------------------------------------------------------------
# Self-evaluation scorer
# ---------------------------------------------------------------------------


class ResponseScorer:
    """
    Scores LLM responses for quality.

    If an LLM client is available, uses LLM-as-judge.
    Otherwise, uses heuristic scoring.
    """

    JUDGE_PROMPT = (
        "You are evaluating the quality of an answer to a question.\n\n"
        "Question: {query}\n\n"
        "Answer: {response}\n\n"
        "Rate the answer on a scale of 0.0 to 1.0 based on:\n"
        "- Accuracy and correctness\n"
        "- Completeness and depth\n"
        "- Clarity and structure\n"
        "- Practical usefulness\n\n"
        "Respond with ONLY a JSON object: {{\"score\": <float>, \"reason\": \"<brief reason>\"}}"
    )

    def __init__(self, llm_client: StdlibLLMClient | MockLLMClient | None = None):
        self._llm = llm_client

    def score(self, query: str, response: str, strategy: str) -> float:
        """Score a response. Returns float in [0, 1]."""
        if self._llm is not None and not isinstance(self._llm, MockLLMClient):
            return self._llm_judge_score(query, response)
        return self._heuristic_score(response, strategy)

    def _llm_judge_score(self, query: str, response: str) -> float:
        """Use the LLM as a judge to score the response."""
        prompt = self.JUDGE_PROMPT.format(query=query, response=response[:2000])
        try:
            text, _ = self._llm.generate_sync(prompt, temperature=0.1, max_tokens=100)
            # Parse the JSON score
            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                score = float(data.get("score", 0.5))
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM judge scoring failed: {e}, falling back to heuristic")
        return self._heuristic_score(response, "direct")

    def _heuristic_score(self, response: str, strategy: str) -> float:
        """
        Heuristic scoring based on response characteristics.

        Not a substitute for LLM-as-judge, but useful for mock/offline demos.
        """
        score = 0.3  # baseline

        # Length bonus (prefer substantive answers, penalize too short/long)
        length = len(response)
        if 200 <= length <= 2000:
            score += 0.2
        elif length > 100:
            score += 0.1

        # Structure bonus (markdown headers, bullet points, numbered lists)
        structural_markers = ["**", "##", "- ", "1.", "2.", "3.", "\n\n"]
        structure_count = sum(1 for m in structural_markers if m in response)
        score += min(0.2, structure_count * 0.03)

        # Strategy-specific bonuses
        if strategy == "decomposition" and ("sub-problem" in response.lower() or "step" in response.lower()):
            score += 0.15
        elif strategy == "refinement" and "refin" in response.lower():
            score += 0.15
        elif strategy == "adversarial" and "counter" in response.lower():
            score += 0.15
        elif strategy == "analogy" and "analogy" in response.lower():
            score += 0.15

        # Add small random noise for variety
        score += random.uniform(-0.05, 0.05)

        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# LLM-MCTS Engine
# ---------------------------------------------------------------------------


class LLMMCTSEngine:
    """
    LLM-powered Monte Carlo Tree Search engine.

    Each node in the tree represents a reasoning strategy.
    Rollouts are actual LLM calls that generate responses using that strategy.
    Backpropagation uses response quality scores to guide future exploration.

    This demonstrates the core value proposition: MCTS systematically explores
    different reasoning approaches and converges on the best one, rather than
    relying on a single LLM prompt.
    """

    def __init__(
        self,
        llm_client: StdlibLLMClient | MockLLMClient | None = None,
        iterations: int = 10,
        exploration_weight: float = 1.414,
        seed: int | None = 42,
    ):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.rng = random.Random(seed)
        self.llm_calls: list[LLMCall] = []

        # Use mock client if none provided
        if llm_client is None:
            self._llm = MockLLMClient()
        else:
            self._llm = llm_client

        self._scorer = ResponseScorer(self._llm)

    def search(self, query: str) -> MCTSResult:
        """
        Run MCTS search over reasoning strategies for a query.

        Args:
            query: The question/task to reason about

        Returns:
            MCTSResult with the best strategy, response, and tree statistics
        """
        start_time = time.perf_counter()
        self.llm_calls = []

        # Build the root with strategy children
        root = MCTSTreeNode(strategy="root")
        for strategy in REASONING_STRATEGIES:
            child = MCTSTreeNode(strategy=strategy, parent=root, depth=1)
            root.children.append(child)

        # MCTS iterations
        for i in range(self.iterations):
            logger.debug(f"MCTS iteration {i + 1}/{self.iterations}")

            # 1. Selection: choose the most promising node via UCB1
            node = self._select(root)

            # 2. Simulation: run an LLM call with this strategy
            score, response, llm_call = self._simulate(query, node.strategy)

            # 3. Backpropagation: update the tree with the result
            self._backpropagate(node, score)

            # Store the response
            node.responses.append(response)
            self.llm_calls.append(llm_call)

        # Find the best strategy (most visited = most robust)
        best_child = max(root.children, key=lambda c: c.visits)

        # Collect strategy scores
        strategy_scores = {}
        for child in root.children:
            strategy_scores[child.strategy] = round(child.value, 3)

        # Tree statistics
        total_visits = sum(c.visits for c in root.children)
        tree_stats = {
            "total_iterations": self.iterations,
            "total_llm_calls": len(self.llm_calls),
            "total_tokens": self._llm.total_tokens,
            "strategy_visits": {c.strategy: c.visits for c in root.children},
            "strategy_values": strategy_scores,
            "provider": self._llm.provider,
        }

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MCTSResult(
            query=query,
            best_strategy=best_child.strategy,
            best_response=best_child.best_response or "",
            best_score=round(best_child.value, 3),
            all_strategies=strategy_scores,
            tree_stats=tree_stats,
            llm_calls=self.llm_calls,
            total_time_ms=round(elapsed_ms, 1),
            iterations_run=self.iterations,
        )

    def _select(self, root: MCTSTreeNode) -> MCTSTreeNode:
        """Select a child node using UCB1."""
        if root.visits == 0:
            # First iteration: pick randomly
            return self.rng.choice(root.children)

        # UCB1 selection
        best_ucb = -float("inf")
        best_node = root.children[0]
        for child in root.children:
            ucb = child.ucb1(root.visits, self.exploration_weight)
            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child

        return best_node

    def _simulate(self, query: str, strategy: str) -> tuple[float, str, LLMCall]:
        """Run an LLM call with the given strategy and score the result."""
        prompt_template = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["direct"])
        prompt = prompt_template.format(query=query)

        start = time.perf_counter()
        response, tokens = self._llm.generate_sync(prompt, temperature=0.7)
        latency_ms = (time.perf_counter() - start) * 1000

        score = self._scorer.score(query, response, strategy)

        llm_call = LLMCall(
            strategy=strategy,
            prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
            response=response[:500] + "..." if len(response) > 500 else response,
            score=score,
            latency_ms=round(latency_ms, 1),
            tokens_used=tokens,
        )

        return score, response, llm_call

    def _backpropagate(self, node: MCTSTreeNode, score: float) -> None:
        """Backpropagate the score up the tree."""
        current: MCTSTreeNode | None = node
        while current is not None:
            current.visits += 1
            current.value_sum += score
            current = current.parent


# ---------------------------------------------------------------------------
# Consensus builder
# ---------------------------------------------------------------------------


class ConsensusBuilder:
    """
    Builds consensus across multiple MCTS strategy results.

    Takes the top strategies and synthesizes a final answer.
    """

    CONSENSUS_PROMPT = (
        "You have received answers to the same question from multiple reasoning strategies. "
        "Synthesize the best elements into a single, comprehensive answer.\n\n"
        "Question: {query}\n\n"
        "{strategy_answers}\n"
        "Provide a synthesized answer that combines the strongest insights from each approach."
    )

    def __init__(self, llm_client: StdlibLLMClient | MockLLMClient):
        self._llm = llm_client

    def build_consensus(self, query: str, strategy_responses: dict[str, str]) -> str:
        """
        Build a consensus response from multiple strategy outputs.

        Args:
            query: Original query
            strategy_responses: Map of strategy name → response text

        Returns:
            Synthesized consensus response
        """
        if not strategy_responses:
            return "No strategies produced responses."

        # If only one strategy, return it directly
        if len(strategy_responses) == 1:
            return next(iter(strategy_responses.values()))

        # Build the synthesis prompt
        parts = []
        for i, (strategy, response) in enumerate(strategy_responses.items(), 1):
            parts.append(f"**Strategy {i} ({strategy}):**\n{response[:800]}\n")

        strategy_text = "\n".join(parts)
        prompt = self.CONSENSUS_PROMPT.format(query=query, strategy_answers=strategy_text)

        try:
            response, _ = self._llm.generate_sync(prompt, temperature=0.3, max_tokens=1500)
            return response
        except Exception as e:
            logger.warning(f"Consensus synthesis failed: {e}")
            # Fallback: return the best individual response
            return next(iter(strategy_responses.values()))


# ---------------------------------------------------------------------------
# Multi-Agent MCTS Pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Full pipeline result combining MCTS search and consensus."""

    query: str
    mcts_result: MCTSResult
    consensus_response: str | None
    top_strategies: list[tuple[str, float]]
    total_time_ms: float
    provider: str


class MultiAgentMCTSPipeline:
    """
    End-to-end pipeline: Query → MCTS exploration → Consensus → Answer.

    This is the main entry point for the MVP demo. It:
    1. Takes a user query
    2. Uses MCTS to explore multiple reasoning strategies via LLM calls
    3. Identifies the best strategies
    4. Optionally synthesizes a consensus answer from the top strategies
    5. Returns the full result with tree statistics

    Works with real LLM APIs or in mock mode for offline demos.
    """

    def __init__(
        self,
        provider: str = "mock",
        api_key: str | None = None,
        model: str | None = None,
        iterations: int = 10,
        exploration_weight: float = 1.414,
        seed: int | None = 42,
        use_consensus: bool = True,
    ):
        # Create the LLM client
        if provider == "mock":
            self._llm: StdlibLLMClient | MockLLMClient = MockLLMClient()
        else:
            self._llm = StdlibLLMClient(
                provider=provider,
                api_key=api_key,
                model=model,
            )

        self._engine = LLMMCTSEngine(
            llm_client=self._llm,
            iterations=iterations,
            exploration_weight=exploration_weight,
            seed=seed,
        )
        self._consensus = ConsensusBuilder(self._llm) if use_consensus else None
        self._use_consensus = use_consensus

    def run(self, query: str) -> PipelineResult:
        """
        Run the full multi-agent MCTS pipeline on a query.

        Args:
            query: The question or task to process

        Returns:
            PipelineResult with all details
        """
        start = time.perf_counter()

        # Step 1: MCTS search
        mcts_result = self._engine.search(query)

        # Step 2: Get top strategies (those with above-median scores)
        sorted_strategies = sorted(
            mcts_result.all_strategies.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        top_strategies = [(s, v) for s, v in sorted_strategies if v > 0][:3]

        # Step 3: Consensus synthesis
        consensus = None
        if self._use_consensus and len(top_strategies) > 1:
            # Get the best response from each top strategy
            strategy_responses = {}
            for call in mcts_result.llm_calls:
                if call.strategy in dict(top_strategies) and call.strategy not in strategy_responses:
                    strategy_responses[call.strategy] = call.response
            if len(strategy_responses) > 1:
                consensus = self._consensus.build_consensus(query, strategy_responses)

        elapsed = (time.perf_counter() - start) * 1000

        return PipelineResult(
            query=query,
            mcts_result=mcts_result,
            consensus_response=consensus,
            top_strategies=top_strategies,
            total_time_ms=round(elapsed, 1),
            provider=self._llm.provider,
        )
