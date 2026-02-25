"""
Base async agent class for Multi-Agent MCTS Framework.

Provides common patterns for all agents with hook points for metrics,
logging, and extensibility.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, cast

from src.adapters.llm.base import LLMClient, LLMResponse


@dataclass
class AgentContext:
    """
    Context passed to agent during processing.

    Contains all information needed for the agent to process a request.
    """

    query: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rag_context: str | None = None
    metadata: dict = field(default_factory=dict)
    conversation_history: list[dict] = field(default_factory=list)
    max_iterations: int = 5
    temperature: float = 0.7
    additional_context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert context to dictionary."""
        return {
            "query": self.query,
            "session_id": self.session_id,
            "rag_context": self.rag_context,
            "metadata": self.metadata,
            "conversation_history": self.conversation_history,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "additional_context": self.additional_context,
        }


@dataclass
class AgentResult:
    """
    Result from agent processing.

    Standardized result format for all agents.
    """

    response: str
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    agent_name: str = ""
    processing_time_ms: float = 0.0
    token_usage: dict = field(default_factory=dict)
    intermediate_steps: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    error: str | None = None
    success: bool = True

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "response": self.response,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "agent_name": self.agent_name,
            "processing_time_ms": self.processing_time_ms,
            "token_usage": self.token_usage,
            "intermediate_steps": self.intermediate_steps,
            "created_at": self.created_at.isoformat(),
            "error": self.error,
            "success": self.success,
        }


class MetricsCollector(Protocol):
    """Protocol for metrics collection."""

    def record_latency(self, agent_name: str, latency_ms: float) -> None: ...
    def record_tokens(self, agent_name: str, tokens: int) -> None: ...
    def record_error(self, agent_name: str, error_type: str) -> None: ...
    def record_success(self, agent_name: str) -> None: ...


class NoOpMetricsCollector:
    """Default no-op metrics collector."""

    def record_latency(self, agent_name: str, latency_ms: float) -> None:
        pass

    def record_tokens(self, agent_name: str, tokens: int) -> None:
        pass

    def record_error(self, agent_name: str, error_type: str) -> None:
        pass

    def record_success(self, agent_name: str) -> None:
        pass


class AsyncAgentBase(ABC):
    """
    Base class for async agents in the Multi-Agent MCTS Framework.

    Features:
    - Async processing by default
    - Hook points for metrics/logging
    - Lifecycle management
    - Error handling patterns
    - Backward compatibility with existing framework
    """

    def __init__(
        self,
        model_adapter: LLMClient,
        logger: Any = None,
        name: str | None = None,
        metrics_collector: MetricsCollector | None = None,
        **config: Any,
    ):
        """
        Initialize async agent.

        Args:
            model_adapter: LLM client for generating responses
            logger: Logger instance (uses standard logging if None)
            name: Agent name (uses class name if None)
            metrics_collector: Optional metrics collector
            **config: Additional configuration parameters
        """
        self.model_adapter = model_adapter
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.name = name or self.__class__.__name__
        self.metrics = metrics_collector or NoOpMetricsCollector()
        self.config = config

        # Runtime state
        self._request_count = 0
        self._total_processing_time = 0.0
        self._error_count = 0
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize agent resources.

        Override this method to perform async initialization tasks
        like loading prompts, setting up connections, etc.
        """
        self._initialized = True
        self.logger.info(f"Agent {self.name} initialized")

    async def shutdown(self) -> None:
        """
        Clean up agent resources.

        Override this method to perform cleanup tasks.
        """
        self._initialized = False
        self.logger.info(f"Agent {self.name} shutdown")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

    # Hook points for subclasses
    async def pre_process(self, context: AgentContext) -> AgentContext:
        """
        Hook called before processing.

        Override to modify context or perform pre-processing.

        Args:
            context: Agent context

        Returns:
            Potentially modified context
        """
        return context

    async def post_process(self, _context: AgentContext, result: AgentResult) -> AgentResult:
        """
        Hook called after processing.

        Override to modify result or perform post-processing.

        Args:
            context: Agent context
            result: Agent result

        Returns:
            Potentially modified result
        """
        return result

    async def on_error(self, _context: AgentContext, error: Exception) -> AgentResult:
        """
        Hook called when processing fails.

        Override to customize error handling.

        Args:
            context: Agent context
            error: The exception that occurred

        Returns:
            Error result
        """
        self.logger.error(f"Agent {self.name} error: {error}")
        self._error_count += 1
        self.metrics.record_error(self.name, type(error).__name__)

        return AgentResult(
            response="",
            confidence=0.0,
            agent_name=self.name,
            error=str(error),
            success=False,
        )

    @abstractmethod
    async def _process_impl(self, context: AgentContext) -> AgentResult:
        """
        Core processing logic to be implemented by subclasses.

        Args:
            context: Agent context with all necessary information

        Returns:
            AgentResult with response and metadata
        """
        pass

    async def process(
        self,
        query: str | None = None,
        context: AgentContext | None = None,
        *,
        rag_context: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Process a query and return structured response.

        This method provides backward compatibility with the existing
        LangGraphMultiAgentFramework while using the new async patterns.

        Args:
            query: Query string (if not using context object)
            context: Full context object (if not using query string)
            rag_context: RAG context (used if query provided)
            **kwargs: Additional parameters merged into context

        Returns:
            Dictionary with 'response' and 'metadata' keys for backward compatibility
        """
        # Build context if not provided
        if context is None:
            if query is None:
                raise ValueError("Either 'query' or 'context' must be provided")
            context = AgentContext(
                query=query,
                rag_context=rag_context,
                additional_context=kwargs,
            )

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Track timing
        start_time = time.perf_counter()

        try:
            # Pre-processing hook
            context = await self.pre_process(context)

            # Core processing
            result = await self._process_impl(context)

            # Calculate timing
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result.processing_time_ms = elapsed_ms
            result.agent_name = self.name

            # Update stats
            self._request_count += 1
            self._total_processing_time += elapsed_ms
            self.metrics.record_latency(self.name, elapsed_ms)
            if result.token_usage:
                self.metrics.record_tokens(self.name, result.token_usage.get("total_tokens", 0))
            self.metrics.record_success(self.name)

            # Post-processing hook
            result = await self.post_process(context, result)

            self.logger.info(f"Agent {self.name} processed query in {elapsed_ms:.2f}ms")

        except Exception as e:
            result = await self.on_error(context, e)
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Return backward-compatible format
        return {
            "response": result.response,
            "metadata": {
                **result.metadata,
                "agent_name": result.agent_name,
                "confidence": result.confidence,
                "processing_time_ms": result.processing_time_ms,
                "token_usage": result.token_usage,
                "success": result.success,
                "error": result.error,
            },
        }

    @property
    def stats(self) -> dict:
        """Get agent statistics."""
        return {
            "name": self.name,
            "request_count": self._request_count,
            "total_processing_time_ms": self._total_processing_time,
            "error_count": self._error_count,
            "average_processing_time_ms": (
                self._total_processing_time / self._request_count if self._request_count > 0 else 0.0
            ),
            "initialized": self._initialized,
        }

    async def generate_llm_response(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Convenience method to generate LLM response with error handling.

        Args:
            prompt: Simple string prompt
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse from the model adapter
        """
        response = await self.model_adapter.generate(
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return cast(LLMResponse, response)


class CompositeAgent(AsyncAgentBase):
    """
    Agent that combines multiple sub-agents.

    Useful for creating complex agents from simpler building blocks.
    """

    def __init__(
        self,
        model_adapter: LLMClient,
        logger: Any = None,
        name: str = "CompositeAgent",
        sub_agents: list[AsyncAgentBase] | None = None,
        **config: Any,
    ):
        super().__init__(model_adapter, logger, name, **config)
        self.sub_agents = sub_agents or []

    def add_agent(self, agent: AsyncAgentBase) -> None:
        """Add a sub-agent."""
        self.sub_agents.append(agent)

    async def initialize(self) -> None:
        """Initialize all sub-agents."""
        await super().initialize()
        for agent in self.sub_agents:
            await agent.initialize()

    async def shutdown(self) -> None:
        """Shutdown all sub-agents."""
        for agent in self.sub_agents:
            await agent.shutdown()
        await super().shutdown()


class ParallelAgent(CompositeAgent):
    """
    Execute multiple agents in parallel and aggregate results.
    """

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        """Execute all sub-agents in parallel."""
        if not self.sub_agents:
            return AgentResult(
                response="No sub-agents configured",
                confidence=0.0,
                agent_name=self.name,
            )

        # Run all agents concurrently
        tasks = [agent.process(context=context) for agent in self.sub_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        successful_results = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"{self.sub_agents[i].name}: {str(result)}")
            elif isinstance(result, dict) and result.get("metadata", {}).get("success", True):
                successful_results.append(result)
            else:
                if isinstance(result, dict):
                    errors.append(
                        f"{self.sub_agents[i].name}: {result.get('metadata', {}).get('error', 'Unknown error')}"
                    )

        if not successful_results:
            return AgentResult(
                response=f"All sub-agents failed: {'; '.join(errors)}",
                confidence=0.0,
                agent_name=self.name,
                success=False,
                error="All sub-agents failed",
            )

        # Aggregate: highest confidence wins (simple strategy)
        best_result = max(successful_results, key=lambda r: r.get("metadata", {}).get("confidence", 0.0))

        return AgentResult(
            response=best_result["response"],
            confidence=best_result.get("metadata", {}).get("confidence", 0.0),
            metadata={
                "aggregation_method": "highest_confidence",
                "sub_agent_results": successful_results,
                "errors": errors,
            },
            agent_name=self.name,
        )


class SequentialAgent(CompositeAgent):
    """
    Execute multiple agents sequentially, passing context through each.
    """

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        """Execute sub-agents in sequence."""
        if not self.sub_agents:
            return AgentResult(
                response="No sub-agents configured",
                confidence=0.0,
                agent_name=self.name,
            )

        current_context = context
        intermediate_results: list[dict[str, Any]] = []

        for agent in self.sub_agents:
            result = await agent.process(context=current_context)

            intermediate_results.append(
                {
                    "agent": agent.name,
                    "result": result,
                }
            )

            # Check for failure
            if not result.get("metadata", {}).get("success", True):
                return AgentResult(
                    response=result["response"],
                    confidence=result.get("metadata", {}).get("confidence", 0.0),
                    metadata={
                        "failed_at": agent.name,
                        "intermediate_results": intermediate_results,
                    },
                    agent_name=self.name,
                    success=False,
                    error=result.get("metadata", {}).get("error"),
                )

            # Update context for next agent
            current_context = AgentContext(
                query=current_context.query,
                session_id=current_context.session_id,
                rag_context=result["response"],  # Previous output becomes context
                metadata={
                    **current_context.metadata,
                    f"{agent.name}_result": result["response"],
                },
                additional_context=current_context.additional_context,
            )

        # Final result from last agent
        final_result = intermediate_results[-1]["result"]

        return AgentResult(
            response=final_result["response"],
            confidence=final_result.get("metadata", {}).get("confidence", 0.0),
            metadata={
                "pipeline": [r["agent"] for r in intermediate_results],
                "intermediate_results": intermediate_results,
            },
            agent_name=self.name,
        )
