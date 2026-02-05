"""
Google ADK benchmark adapter.

Wraps Google's Agent Development Kit (ADK) to conform to the
BenchmarkSystemProtocol for uniform evaluation against LangGraph MCTS.

Requires optional dependency: pip install -e ".[google-adk]"
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


def _check_adk_available() -> bool:
    """Check if Google ADK dependencies are installed."""
    try:
        from google.adk.agents import LlmAgent  # noqa: F401

        return True
    except ImportError:
        return False


class ADKBenchmarkAdapter:
    """
    Adapter for benchmarking Google's ADK Agent Builder.

    Constructs ADK agent hierarchies (coordinator + sub-agents) and
    executes benchmark tasks through ADK's orchestration layer.

    Uses dependency injection for all external dependencies.
    """

    def __init__(
        self,
        settings: BenchmarkSettings | None = None,
        coordinator_agent: Any | None = None,
    ) -> None:
        """
        Initialize the adapter.

        Args:
            settings: Benchmark settings (uses global if not provided)
            coordinator_agent: Pre-configured ADK coordinator agent
        """
        self._settings = settings or get_benchmark_settings()
        self._coordinator = coordinator_agent
        self._runner: Any | None = None
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def name(self) -> str:
        return "vertex_adk"

    @property
    def is_available(self) -> bool:
        """Check if ADK dependencies and configuration are available."""
        if not self._settings.adk.enabled:
            return False
        if not _check_adk_available():
            self._logger.warning("Google ADK not installed, adapter unavailable")
            return False
        if self._settings.adk.google_api_key is None and self._settings.adk.google_project_id is None:
            self._logger.warning("No Google API key or project ID configured")
            return False
        return True

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        """
        Execute a benchmark task through the ADK agent hierarchy.

        Args:
            task: Benchmark task to execute

        Returns:
            BenchmarkResult with timing and response data
        """
        correlation_id = get_correlation_id()
        self._logger.info(
            "Executing task %s via ADK",
            task.task_id,
            extra={"correlation_id": correlation_id, "task_id": task.task_id},
        )

        result = BenchmarkResult(
            task_id=task.task_id,
            system=self.name,
            task_description=task.description,
        )

        start_time = time.perf_counter()
        first_token_time: float | None = None

        try:
            coordinator = self._get_or_build_coordinator()
            runner = self._get_or_build_runner(coordinator)

            response_parts: list[str] = []
            trace_events: list[dict[str, Any]] = []
            tool_call_count = 0

            async for event in self._stream_execution(runner, task):
                if first_token_time is None and event.get("content"):
                    first_token_time = time.perf_counter()

                trace_events.append(event)

                # Extract text content
                content = event.get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                for part in parts:
                    if isinstance(part, dict):
                        if "text" in part:
                            response_parts.append(part["text"])
                        if "function_call" in part:
                            tool_call_count += 1

            end_time = time.perf_counter()
            result.total_latency_ms = (end_time - start_time) * 1000
            if first_token_time is not None:
                result.time_to_first_token_ms = (first_token_time - start_time) * 1000

            result.raw_response = "\n".join(response_parts)
            result.num_tool_calls = tool_call_count
            result.num_agent_calls = self._count_agent_delegations(trace_events)
            result.agent_trace = trace_events

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
        """Verify ADK system is operational."""
        try:
            if not self.is_available:
                return False
            self._get_or_build_coordinator()
            return True
        except Exception as e:
            self._logger.warning("Health check failed: %s", e)
            return False

    def _get_or_build_coordinator(self) -> Any:
        """Get or build the ADK coordinator agent."""
        if self._coordinator is not None:
            return self._coordinator

        if not _check_adk_available():
            raise RuntimeError("Google ADK not installed")

        from google.adk.agents import LlmAgent

        adk_config = self._settings.adk

        # Build sub-agents
        code_reviewer = LlmAgent(
            name="code_reviewer",
            model=adk_config.sub_agent_model,
            instruction=(
                "You are an expert code reviewer. Analyze code for:\n"
                "- Bugs and logic errors\n"
                "- Security vulnerabilities\n"
                "- Performance issues\n"
                "- Code style and maintainability\n"
                "Provide specific line references and severity ratings."
            ),
            description="Reviews code for bugs, security, and quality issues",
        )

        test_strategist = LlmAgent(
            name="test_strategist",
            model=adk_config.sub_agent_model,
            instruction=(
                "You are a test strategy expert. Given requirements or code:\n"
                "- Identify testable scenarios\n"
                "- Classify by test type (unit, integration, e2e, performance)\n"
                "- Prioritize by risk\n"
                "- Estimate effort"
            ),
            description="Designs test strategies and plans",
        )

        compliance_analyst = LlmAgent(
            name="compliance_analyst",
            model=adk_config.sub_agent_model,
            instruction=(
                "You are a regulatory compliance analyst. You:\n"
                "- Extract requirements from regulatory documents\n"
                "- Map to technical controls\n"
                "- Identify compliance gaps\n"
                "- Recommend remediations with priority"
            ),
            description="Analyzes regulatory compliance requirements and gaps",
        )

        risk_assessor = LlmAgent(
            name="risk_assessor",
            model=adk_config.sub_agent_model,
            instruction=(
                "You are a risk assessment specialist. You:\n"
                "- Evaluate risks across technical, operational, and strategic dimensions\n"
                "- Score probability and impact\n"
                "- Identify mitigations\n"
                "- Model scenarios and their outcomes"
            ),
            description="Assesses and models risks across multiple dimensions",
        )

        # Build coordinator
        self._coordinator = LlmAgent(
            name="qe_coordinator",
            model=adk_config.coordinator_model,
            instruction=(
                "You are the Quality Engineering Coordinator. You orchestrate "
                "a team of specialist agents to solve complex challenges.\n\n"
                "Your team:\n"
                "- code_reviewer: Analyzes code for bugs and security issues\n"
                "- test_strategist: Designs test strategies and plans\n"
                "- compliance_analyst: Handles regulatory compliance analysis\n"
                "- risk_assessor: Evaluates and models risks\n\n"
                "For each task:\n"
                "1. Break it into subtasks appropriate for each specialist\n"
                "2. Delegate to the right agents\n"
                "3. Synthesize their findings into a cohesive response\n"
                "4. Identify conflicts or gaps between agent outputs\n"
                "5. Provide a final recommendation with confidence level\n\n"
                "Always explain your delegation reasoning."
            ),
            description="Coordinates multi-agent quality engineering workflows",
            sub_agents=[code_reviewer, test_strategist, compliance_analyst, risk_assessor],
        )

        self._logger.info("Built ADK coordinator with %d sub-agents", 4)
        return self._coordinator

    def _get_or_build_runner(self, coordinator: Any) -> Any:
        """Get or build the ADK Runner."""
        if self._runner is not None:
            return self._runner

        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        session_service = InMemorySessionService()
        self._runner = Runner(
            agent=coordinator,
            app_name="benchmark",
            session_service=session_service,
        )

        return self._runner

    async def _stream_execution(self, runner: Any, task: BenchmarkTask) -> Any:
        """
        Stream execution events from ADK runner.

        Yields event dictionaries from the ADK agent execution.
        """
        from google.genai import types

        content = types.Content(
            role="user",
            parts=[types.Part(text=task.input_data)],
        )

        session_id = f"benchmark_{task.task_id}_{int(time.time())}"

        async for event in runner.run_async(
            user_id="benchmark",
            session_id=session_id,
            new_message=content,
        ):
            yield self._normalize_event(event)

    def _normalize_event(self, event: Any) -> dict[str, Any]:
        """Normalize ADK event to dictionary format."""
        try:
            return {
                "author": getattr(event, "author", "unknown"),
                "content": {
                    "parts": [
                        (
                            {"text": part.text}
                            if hasattr(part, "text") and part.text
                            else {"function_call": getattr(part, "function_call", None)}
                        )
                        for part in (event.content.parts if event.content and event.content.parts else [])
                    ]
                },
                "is_final": getattr(event, "is_final_response", lambda: False)(),
            }
        except (AttributeError, TypeError):
            return {"author": "unknown", "content": {}, "is_final": False}

    def _count_agent_delegations(self, events: list[dict[str, Any]]) -> int:
        """Count unique agent delegations from trace events."""
        agents_seen: set[str] = set()
        for event in events:
            author = event.get("author", "")
            if author and author != "unknown":
                agents_seen.add(author)
        return len(agents_seen)
