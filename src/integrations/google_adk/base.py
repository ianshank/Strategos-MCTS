"""
Base adapter for integrating Google ADK agents with LangGraph framework.

This module provides the foundation for wrapping Google ADK agents to work
seamlessly within the LangGraph multi-agent MCTS framework.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ADKBackend(str, Enum):
    """Supported ADK execution backends."""

    ML_DEV = "ml_dev"  # Google ML Dev (local development)
    VERTEX_AI = "vertex_ai"  # Google Cloud Vertex AI
    LOCAL = "local"  # Local execution without Google Cloud


@dataclass
class ADKConfig:
    """
    Configuration for Google ADK agent integration.

    Attributes:
        project_id: Google Cloud project ID
        location: Google Cloud region (e.g., 'us-central1')
        model_name: Gemini model identifier (e.g., 'gemini-2.0-flash-001')
        backend: Execution backend (ml_dev, vertex_ai, or local)
        workspace_dir: Directory for agent artifacts and outputs
        enable_tracing: Enable OpenTelemetry tracing
        enable_search: Enable web search capabilities
        timeout: Maximum execution time in seconds
        max_iterations: Maximum agent iteration steps
        temperature: LLM sampling temperature
        env_vars: Additional environment variables
    """

    project_id: str | None = None
    location: str = "us-central1"
    model_name: str = "gemini-2.0-flash-001"
    backend: ADKBackend = ADKBackend.LOCAL
    workspace_dir: str = "./workspace/adk"
    enable_tracing: bool = True
    enable_search: bool = True
    timeout: int = 300
    max_iterations: int = 10
    temperature: float = 0.7
    env_vars: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> ADKConfig:
        """Create configuration from environment variables."""
        backend_str = os.getenv("ADK_BACKEND", "local")
        backend = ADKBackend(backend_str)

        return cls(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            model_name=os.getenv("ROOT_AGENT_MODEL", "gemini-2.0-flash-001"),
            backend=backend,
            workspace_dir=os.getenv("ADK_WORKSPACE_DIR", "./workspace/adk"),
            enable_tracing=os.getenv("ADK_ENABLE_TRACING", "true").lower() == "true",
            enable_search=os.getenv("ADK_ENABLE_SEARCH", "true").lower() == "true",
            timeout=int(os.getenv("ADK_TIMEOUT", "300")),
            max_iterations=int(os.getenv("ADK_MAX_ITERATIONS", "10")),
            temperature=float(os.getenv("ADK_TEMPERATURE", "0.7")),
        )

    def validate(self) -> None:
        """Validate configuration."""
        if self.backend in (ADKBackend.ML_DEV, ADKBackend.VERTEX_AI) and not self.project_id:
            raise ValueError(f"{self.backend} backend requires GOOGLE_CLOUD_PROJECT to be set")


class ADKAgentRequest(BaseModel):
    """Request structure for ADK agent invocation."""

    query: str = Field(description="User query or task description")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context for the agent")
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Agent-specific parameters")


class ADKAgentResponse(BaseModel):
    """Response structure from ADK agent execution."""

    result: str = Field(description="Agent's response or output")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    artifacts: list[str] = Field(default_factory=list, description="Generated artifacts (files, reports, etc.)")
    status: str = Field(default="success", description="Execution status")
    error: str | None = Field(default=None, description="Error message if status is 'error'")
    session_id: str | None = Field(default=None, description="Session ID for tracking")


class ADKAgentAdapter(ABC):
    """
    Base adapter for Google ADK agents.

    This abstract class provides the interface for integrating Google ADK agents
    with the LangGraph framework. Concrete implementations wrap specific ADK agents
    and handle the translation between LangGraph's execution model and ADK's.
    """

    def __init__(self, config: ADKConfig, agent_name: str):
        """
        Initialize the ADK adapter.

        Args:
            config: ADK configuration
            agent_name: Name identifier for this agent
        """
        self.config = config
        self.agent_name = agent_name
        self._initialized = False
        self._session_state: dict[str, Any] = {}

        # Validate configuration
        self.config.validate()

        # Setup workspace
        os.makedirs(self.config.workspace_dir, exist_ok=True)

    async def initialize(self) -> None:
        """
        Initialize the ADK agent.

        This method should be called before invoking the agent. It handles
        authentication, model loading, and any required setup.
        """
        if self._initialized:
            return

        # Set environment variables
        self._setup_environment()

        # Perform agent-specific initialization
        await self._agent_initialize()

        self._initialized = True

    def _setup_environment(self) -> None:
        """Setup environment variables for Google ADK."""
        if self.config.backend == ADKBackend.VERTEX_AI:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        elif self.config.backend == ADKBackend.ML_DEV:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

        if self.config.project_id:
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.config.project_id

        os.environ["GOOGLE_CLOUD_LOCATION"] = self.config.location
        os.environ["ROOT_AGENT_MODEL"] = self.config.model_name

        # Apply custom environment variables
        for key, value in self.config.env_vars.items():
            os.environ[key] = value

    @abstractmethod
    async def _agent_initialize(self) -> None:
        """
        Agent-specific initialization logic.

        Implement this method in subclasses to perform agent-specific setup.
        """
        pass

    @abstractmethod
    async def _agent_invoke(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """
        Agent-specific invocation logic.

        Implement this method in subclasses to handle the actual agent execution.

        Args:
            request: Agent request

        Returns:
            Agent response
        """
        pass

    async def invoke(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """
        Invoke the ADK agent with a request.

        Args:
            request: Agent request

        Returns:
            Agent response

        Raises:
            RuntimeError: If agent is not initialized
            TimeoutError: If execution exceeds timeout
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Execute with timeout
            response = await asyncio.wait_for(
                self._agent_invoke(request),
                timeout=self.config.timeout,
            )
            return response

        except TimeoutError:
            return ADKAgentResponse(
                result="",
                status="error",
                error=f"Agent execution exceeded timeout of {self.config.timeout}s",
            )

        except Exception as e:
            return ADKAgentResponse(
                result="",
                status="error",
                error=f"Agent execution failed: {str(e)}",
            )

    async def invoke_streaming(self, request: ADKAgentRequest):
        """
        Invoke the ADK agent with streaming response.

        Args:
            request: Agent request

        Yields:
            Partial responses as they become available
        """
        if not self._initialized:
            await self.initialize()

        # Default implementation: single response
        # Override in subclasses for true streaming
        response = await self.invoke(request)
        yield response

    def get_capabilities(self) -> dict[str, Any]:
        """
        Get agent capabilities and metadata.

        Returns:
            Dictionary describing agent capabilities
        """
        return {
            "name": self.agent_name,
            "backend": self.config.backend.value,
            "model": self.config.model_name,
            "supports_streaming": False,
            "supports_search": self.config.enable_search,
            "max_iterations": self.config.max_iterations,
        }

    async def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        self._initialized = False
        self._session_state.clear()


class ADKAgentFactory:
    """Factory for creating ADK agent instances."""

    _registry: dict[str, type[ADKAgentAdapter]] = {}

    @classmethod
    def register(cls, agent_type: str, agent_class: type[ADKAgentAdapter]) -> None:
        """
        Register an ADK agent type.

        Args:
            agent_type: Agent type identifier
            agent_class: Agent adapter class
        """
        cls._registry[agent_type] = agent_class

    @classmethod
    def create(cls, agent_type: str, config: ADKConfig) -> ADKAgentAdapter:
        """
        Create an ADK agent instance.

        Args:
            agent_type: Agent type identifier
            config: ADK configuration

        Returns:
            ADK agent instance

        Raises:
            ValueError: If agent type is not registered
        """
        if agent_type not in cls._registry:
            raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(cls._registry.keys())}")

        agent_class = cls._registry[agent_type]
        return agent_class(config, agent_name=agent_type)

    @classmethod
    def list_agent_types(cls) -> list[str]:
        """List all registered agent types."""
        return list(cls._registry.keys())
