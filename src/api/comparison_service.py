"""
Single-shot vs MCTS comparison service.

Extracts the core A/B logic from ``demo.py``'s ``run_comparison`` into a
coverage-bearing, presentation-free service. ``demo.py`` is refactored to call
this service so the comparison logic lives in exactly one place.

Reuses :class:`SingleShotRunner`, :class:`MultiAgentMCTSPipeline` and
:class:`TreeVisualizer` from ``src.framework.mcts.llm_mcts``.

Design constraints:
- No FastAPI import; logic is coverage-bearing here, the REST adapter stays thin.
- Dependency-injection friendly: an LLM client and/or pipeline may be injected
  (so tests can drive it with ``MockLLMClient`` and a deterministic pipeline).
- Configuration-driven: gated on ``settings.ENABLE_DEMO_COMPARISON``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.settings import Settings, get_settings
from src.framework.mcts.llm_mcts import (
    DEFAULT_EXPLORATION_WEIGHT,
    DEFAULT_ITERATIONS,
    DEFAULT_SEED,
    MockLLMClient,
    MultiAgentMCTSPipeline,
    SingleShotRunner,
    StdlibLLMClient,
    TreeVisualizer,
)
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Provider identifier for the no-API-key mock path. Centralized to avoid a magic
# literal that must stay in lockstep with the demo CLI.
MOCK_PROVIDER = "mock"


@dataclass
class SingleShotResult:
    """Result of the single-shot (direct prompt) arm of the comparison."""

    response: str
    score: float
    latency_ms: float


@dataclass
class MctsArmResult:
    """Result of the MCTS arm of the comparison."""

    best_strategy: str
    best_response: str
    best_score: float
    total_time_ms: float
    llm_calls: int
    all_strategies: dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Structured single-shot vs MCTS comparison outcome."""

    query: str
    provider: str
    single_shot: SingleShotResult
    mcts: MctsArmResult
    delta: float
    improvement_pct: float
    tree: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON/REST responses."""
        return {
            "query": self.query,
            "provider": self.provider,
            "single_shot": {
                "response": self.single_shot.response,
                "score": self.single_shot.score,
                "latency_ms": self.single_shot.latency_ms,
            },
            "mcts": {
                "best_strategy": self.mcts.best_strategy,
                "best_response": self.mcts.best_response,
                "best_score": self.mcts.best_score,
                "total_time_ms": self.mcts.total_time_ms,
                "llm_calls": self.mcts.llm_calls,
                "all_strategies": self.mcts.all_strategies,
            },
            "delta": self.delta,
            "improvement_pct": self.improvement_pct,
            "tree": self.tree,
        }


class ComparisonDisabledError(RuntimeError):
    """Raised when comparison is requested while disabled by configuration."""


class ComparisonService:
    """Run single-shot vs MCTS and return a structured comparison."""

    def __init__(
        self,
        pipeline: MultiAgentMCTSPipeline | None = None,
        llm_client: StdlibLLMClient | MockLLMClient | None = None,
        provider: str = MOCK_PROVIDER,
        api_key: str | None = None,
        model: str | None = None,
        iterations: int = DEFAULT_ITERATIONS,
        exploration_weight: float = DEFAULT_EXPLORATION_WEIGHT,
        seed: int | None = DEFAULT_SEED,
        use_consensus: bool = True,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider

        # Pipeline: injected (preferred for tests) or constructed from params.
        if pipeline is not None:
            self._pipeline = pipeline
        else:
            self._pipeline = MultiAgentMCTSPipeline(
                provider=provider,
                api_key=api_key,
                model=model,
                iterations=iterations,
                exploration_weight=exploration_weight,
                seed=seed,
                use_consensus=use_consensus,
            )

        # LLM client for the single-shot arm: injected, or built to match the
        # pipeline's provider so both arms use the same backend.
        if llm_client is not None:
            self._llm = llm_client
        elif provider == MOCK_PROVIDER:
            self._llm = MockLLMClient()
        else:
            self._llm = StdlibLLMClient(provider=provider, api_key=api_key, model=model)

    @property
    def enabled(self) -> bool:
        """Whether the comparison feature is enabled by configuration."""
        return bool(self._settings.ENABLE_DEMO_COMPARISON)

    @staticmethod
    def _improvement_pct(single_shot_score: float, delta: float) -> float:
        """Percentage improvement of MCTS over single-shot (0.0 if base is 0)."""
        if single_shot_score > 0:
            return round(delta / single_shot_score * 100, 1)
        return 0.0

    def compare(
        self,
        query: str,
        on_iteration: Any | None = None,
        include_tree: bool = True,
    ) -> ComparisonResult:
        """
        Run both arms and return the structured comparison.

        Args:
            query: The question to evaluate.
            on_iteration: Optional MCTS iteration callback (streaming progress).
            include_tree: Render the ASCII MCTS tree into the result.

        Raises:
            ComparisonDisabledError: When disabled by configuration.
        """
        if not self.enabled:
            logger.warning("Comparison requested but ENABLE_DEMO_COMPARISON is False")
            raise ComparisonDisabledError("Comparison is disabled (ENABLE_DEMO_COMPARISON=False)")

        logger.info("Running comparison: provider=%s, query_len=%d", self._provider, len(query))

        # --- Single-shot arm ---
        runner = SingleShotRunner(self._llm)
        ss_response, ss_score, ss_latency = runner.run(query)
        single_shot = SingleShotResult(
            response=ss_response,
            score=round(ss_score, 3),
            latency_ms=round(ss_latency, 1),
        )

        # --- MCTS arm ---
        pipeline_result = self._pipeline.run(query, on_iteration=on_iteration)
        mcts_res = pipeline_result.mcts_result
        mcts_arm = MctsArmResult(
            best_strategy=mcts_res.best_strategy,
            best_response=mcts_res.best_response,
            best_score=round(mcts_res.best_score, 3),
            total_time_ms=pipeline_result.total_time_ms,
            llm_calls=len(mcts_res.llm_calls),
            all_strategies=dict(mcts_res.all_strategies),
        )

        # --- Delta ---
        delta = round(mcts_arm.best_score - single_shot.score, 3)
        improvement_pct = self._improvement_pct(single_shot.score, delta)

        tree = None
        if include_tree and pipeline_result.tree_root is not None:
            tree = TreeVisualizer.render(pipeline_result.tree_root)

        logger.info(
            "Comparison complete: single_shot=%.3f, mcts=%.3f, delta=%.3f (%.1f%%)",
            single_shot.score,
            mcts_arm.best_score,
            delta,
            improvement_pct,
        )

        return ComparisonResult(
            query=query,
            provider=self._provider,
            single_shot=single_shot,
            mcts=mcts_arm,
            delta=delta,
            improvement_pct=improvement_pct,
            tree=tree,
        )


__all__ = [
    "ComparisonService",
    "ComparisonResult",
    "SingleShotResult",
    "MctsArmResult",
    "ComparisonDisabledError",
    "MOCK_PROVIDER",
]
