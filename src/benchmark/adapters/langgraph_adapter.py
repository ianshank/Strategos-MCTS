"""
LangGraph MCTS benchmark adapter.

Wraps the existing LangGraph multi-agent MCTS framework to conform
to the BenchmarkSystemProtocol for uniform evaluation.

Supports two execution modes:
1. Full framework mode: Uses IntegratedFramework.process() for real graph execution
2. Direct LLM mode: Fallback using raw LLM client for simpler invocations
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.benchmark.config.benchmark_settings import BenchmarkSettings, get_benchmark_settings
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.tasks.models import BenchmarkTask
from src.observability.logging import get_correlation_id

logger = logging.getLogger(__name__)


class LangGraphBenchmarkAdapter:
    """
    Adapter for benchmarking the LangGraph MCTS multi-agent system.

    Wraps the existing framework components (GraphBuilder, agents, MCTS)
    to execute benchmark tasks and collect performance metrics.

    Uses dependency injection for all external dependencies.
    """

    def __init__(
        self,
        settings: BenchmarkSettings | None = None,
        graph_builder: Any | None = None,
        llm_client: Any | None = None,
        framework: Any | None = None,
    ) -> None:
        """
        Initialize the adapter.

        Args:
            settings: Benchmark settings (uses global if not provided)
            graph_builder: Pre-configured GraphBuilder (created lazily if not provided)
            llm_client: LLM client for agent operations
            framework: IntegratedFramework instance for full graph execution.
                      If provided, takes precedence over graph_builder.
        """
        self._settings = settings or get_benchmark_settings()
        self._graph_builder = graph_builder
        self._llm_client = llm_client
        self._framework = framework
        self._graph: Any | None = None
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def name(self) -> str:
        return "langgraph_mcts"

    @property
    def is_available(self) -> bool:
        """Check if LangGraph dependencies are available."""
        if not self._settings.langgraph.enabled:
            return False
        try:
            import langgraph  # noqa: F401

            return True
        except ImportError:
            self._logger.warning("LangGraph not installed, adapter unavailable")
            return False

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        """
        Execute a benchmark task through the LangGraph MCTS pipeline.

        Execution priority:
        1. IntegratedFramework.process() if framework is injected
        2. GraphBuilder graph if graph_builder is injected
        3. Direct LLM call if only llm_client is available
        4. Error response if nothing is configured

        Args:
            task: Benchmark task to execute

        Returns:
            BenchmarkResult with timing and response data
        """
        correlation_id = get_correlation_id()
        self._logger.info(
            "Executing task %s via LangGraph MCTS",
            task.task_id,
            extra={"correlation_id": correlation_id, "task_id": task.task_id},
        )

        result = BenchmarkResult(
            task_id=task.task_id,
            system=self.name,
            task_description=task.description,
        )

        start_time = time.perf_counter()

        try:
            # Choose execution path
            if self._framework is not None:
                response_data = await self._run_framework(task)
            else:
                response_data = await self._run_graph(task)

            end_time = time.perf_counter()
            result.total_latency_ms = (end_time - start_time) * 1000

            result.raw_response = response_data.get("final_response", "")
            result.num_agent_calls = response_data.get("agent_call_count", 0)
            result.num_tool_calls = response_data.get("tool_call_count", 0)
            result.input_tokens = response_data.get("input_tokens", 0)
            result.output_tokens = response_data.get("output_tokens", 0)
            result.agent_trace = response_data.get("trace", [])

            self._logger.info(
                "Task %s completed in %.1fms",
                task.task_id,
                result.total_latency_ms,
                extra={"correlation_id": correlation_id},
            )

        except Exception as e:
            end_time = time.perf_counter()
            result.total_latency_ms = (end_time - start_time) * 1000
            result.raw_response = f"Error: {type(e).__name__}: {e}"
            self._logger.error(
                "Task %s failed: %s",
                task.task_id,
                e,
                extra={"correlation_id": correlation_id},
                exc_info=True,
            )

        return result

    async def health_check(self) -> bool:
        """Verify LangGraph system is operational."""
        try:
            if not self.is_available:
                return False
            # Verify core imports are functional
            from src.framework.mcts.core import MCTSEngine  # noqa: F401

            return True
        except Exception as e:
            self._logger.warning("Health check failed: %s", e)
            return False

    async def _run_framework(self, task: BenchmarkTask) -> dict[str, Any]:
        """
        Run task through the full IntegratedFramework.

        Uses IntegratedFramework.process() which orchestrates
        the LangGraph state machine with HRM/TRM agents and MCTS.

        Args:
            task: Benchmark task to execute

        Returns:
            Normalized response dictionary
        """
        lg_config = self._settings.langgraph
        use_mcts = lg_config.mcts_iterations > 0

        self._logger.debug(
            "Running task %s via IntegratedFramework (use_mcts=%s)",
            task.task_id,
            use_mcts,
        )

        result = await self._framework.process(
            query=task.input_data,
            use_rag=False,
            use_mcts=use_mcts,
        )

        # Normalize IntegratedFramework response to adapter format
        response_text = result.get("response", "")
        metadata = result.get("metadata", {})
        state = result.get("state", {})
        agent_outputs = state.get("agent_outputs", [])

        return {
            "final_response": response_text,
            "agent_call_count": len(agent_outputs),
            "tool_call_count": metadata.get("tool_calls", 0),
            "input_tokens": metadata.get("input_tokens", 0),
            "output_tokens": metadata.get("output_tokens", 0),
            "trace": agent_outputs,
        }

    async def _run_graph(self, task: BenchmarkTask) -> dict[str, Any]:
        """
        Run the LangGraph state machine for a benchmark task.

        This method builds the graph if needed and invokes it with the task input.
        """
        graph = self._get_or_build_graph()

        if graph is None:
            return await self._run_direct(task)

        # Build input state
        input_state = {
            "query": task.input_data,
            "use_mcts": True,
            "use_rag": False,
            "iteration": 0,
            "max_iterations": self._settings.langgraph.max_graph_iterations,
            "agent_outputs": [],
        }

        # Execute graph
        final_state = await graph.ainvoke(input_state)

        return {
            "final_response": final_state.get("final_response", ""),
            "agent_call_count": len(final_state.get("agent_outputs", [])),
            "tool_call_count": final_state.get("metadata", {}).get("tool_calls", 0),
            "input_tokens": final_state.get("metadata", {}).get("input_tokens", 0),
            "output_tokens": final_state.get("metadata", {}).get("output_tokens", 0),
            "trace": final_state.get("agent_outputs", []),
        }

    async def _run_direct(self, task: BenchmarkTask) -> dict[str, Any]:
        """
        Fallback: Run task directly through LLM client without full graph.

        Used when the graph cannot be built (e.g., missing components).
        """
        if self._llm_client is None:
            return {
                "final_response": "LLM client not configured for direct execution",
                "agent_call_count": 0,
                "tool_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "trace": [],
            }

        response = await self._llm_client.generate(
            messages=[
                {"role": "system", "content": "You are a multi-agent reasoning system."},
                {"role": "user", "content": task.input_data},
            ],
        )

        return {
            "final_response": response.text,
            "agent_call_count": 1,
            "tool_call_count": 0,
            "input_tokens": response.usage.get("prompt_tokens", 0),
            "output_tokens": response.usage.get("completion_tokens", 0),
            "trace": [{"agent": "direct_llm", "response": response.text}],
        }

    def _get_or_build_graph(self) -> Any:
        """Get or lazily build the LangGraph state machine."""
        if self._graph is not None:
            return self._graph

        if self._graph_builder is not None:
            try:
                self._graph = self._graph_builder.build_graph().compile()
                return self._graph
            except Exception as e:
                self._logger.warning("Failed to build graph: %s", e)
                return None

        return None
