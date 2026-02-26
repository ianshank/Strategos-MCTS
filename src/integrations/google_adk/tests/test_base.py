"""
Tests for Google ADK base adapter functionality.
"""

import os

import pytest

from src.integrations.google_adk.base import (
    ADKAgentAdapter,
    ADKAgentRequest,
    ADKAgentResponse,
    ADKBackend,
    ADKConfig,
)


class TestADKConfig:
    """Tests for ADKConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ADKConfig()

        assert config.backend == ADKBackend.LOCAL
        assert config.location == "us-central1"
        assert config.model_name == "gemini-2.0-flash-001"
        assert config.workspace_dir == "./workspace/adk"
        assert config.enable_tracing is True
        assert config.enable_search is True

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.setenv("ADK_BACKEND", "vertex_ai")
        monkeypatch.setenv("ROOT_AGENT_MODEL", "gemini-2.5-pro")
        monkeypatch.setenv("ADK_WORKSPACE_DIR", "/tmp/adk")

        config = ADKConfig.from_env()

        assert config.project_id == "test-project"
        assert config.backend == ADKBackend.VERTEX_AI
        assert config.model_name == "gemini-2.5-pro"
        assert config.workspace_dir == "/tmp/adk"

    def test_config_validation_vertex_ai(self):
        """Test validation for Vertex AI backend."""
        config = ADKConfig(
            backend=ADKBackend.VERTEX_AI,
            project_id=None,
        )

        with pytest.raises(ValueError, match="requires GOOGLE_CLOUD_PROJECT"):
            config.validate()

    def test_config_validation_success(self):
        """Test successful validation."""
        config = ADKConfig(
            backend=ADKBackend.VERTEX_AI,
            project_id="test-project",
        )

        config.validate()  # Should not raise


class TestADKAgentRequest:
    """Tests for ADKAgentRequest."""

    def test_basic_request(self):
        """Test basic request creation."""
        request = ADKAgentRequest(
            query="Test query",
        )

        assert request.query == "Test query"
        assert request.context == {}
        assert request.session_id is None
        assert request.parameters == {}

    def test_request_with_all_fields(self):
        """Test request with all fields."""
        request = ADKAgentRequest(
            query="Test query",
            context={"key": "value"},
            session_id="session-123",
            parameters={"param": "value"},
        )

        assert request.query == "Test query"
        assert request.context == {"key": "value"}
        assert request.session_id == "session-123"
        assert request.parameters == {"param": "value"}


class TestADKAgentResponse:
    """Tests for ADKAgentResponse."""

    def test_success_response(self):
        """Test successful response."""
        response = ADKAgentResponse(
            result="Test result",
            metadata={"key": "value"},
            artifacts=["file1.txt"],
            status="success",
        )

        assert response.result == "Test result"
        assert response.metadata == {"key": "value"}
        assert response.artifacts == ["file1.txt"]
        assert response.status == "success"
        assert response.error is None

    def test_error_response(self):
        """Test error response."""
        response = ADKAgentResponse(
            result="",
            status="error",
            error="Something went wrong",
        )

        assert response.result == ""
        assert response.status == "error"
        assert response.error == "Something went wrong"


class MockADKAgent(ADKAgentAdapter):
    """Mock ADK agent for testing."""

    async def _agent_initialize(self) -> None:
        """Mock initialization."""
        self.initialized = True

    async def _agent_invoke(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """Mock invocation."""
        return ADKAgentResponse(
            result=f"Processed: {request.query}",
            status="success",
        )


class TestADKAgentAdapter:
    """Tests for ADKAgentAdapter."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ADKConfig(
            backend=ADKBackend.LOCAL,
            workspace_dir="/tmp/adk_test",
        )

    @pytest.fixture
    def agent(self, config):
        """Create test agent."""
        return MockADKAgent(config, agent_name="test_agent")

    def test_agent_creation(self, agent, config):
        """Test agent creation."""
        assert agent.agent_name == "test_agent"
        assert agent.config == config
        assert agent._initialized is False

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert not agent._initialized

        await agent.initialize()

        assert agent._initialized

    @pytest.mark.asyncio
    async def test_agent_invoke(self, agent):
        """Test agent invocation."""
        await agent.initialize()

        request = ADKAgentRequest(query="Test query")
        response = await agent.invoke(request)

        assert response.status == "success"
        assert response.result == "Processed: Test query"

    @pytest.mark.asyncio
    async def test_agent_invoke_auto_init(self, agent):
        """Test agent auto-initializes on invoke."""
        assert not agent._initialized

        request = ADKAgentRequest(query="Test query")
        response = await agent.invoke(request)

        assert agent._initialized
        assert response.status == "success"

    @pytest.mark.asyncio
    async def test_agent_timeout(self, agent):
        """Test agent timeout handling."""
        # Create agent with very short timeout
        config = ADKConfig(backend=ADKBackend.LOCAL, timeout=0)
        agent = MockADKAgent(config, "timeout_test")

        # Mock _agent_invoke to take longer than timeout
        async def slow_invoke(request):
            import asyncio

            await asyncio.sleep(1)
            return ADKAgentResponse(result="Done", status="success")

        agent._agent_invoke = slow_invoke

        request = ADKAgentRequest(query="Test")
        response = await agent.invoke(request)

        assert response.status == "error"
        assert "timeout" in response.error.lower()

    @pytest.mark.asyncio
    async def test_agent_cleanup(self, agent):
        """Test agent cleanup."""
        await agent.initialize()
        assert agent._initialized

        await agent.cleanup()

        assert not agent._initialized
        assert len(agent._session_state) == 0

    def test_get_capabilities(self, agent):
        """Test get capabilities."""
        capabilities = agent.get_capabilities()

        assert capabilities["name"] == "test_agent"
        assert capabilities["backend"] == "local"
        assert capabilities["model"] == "gemini-2.0-flash-001"
        assert "supports_streaming" in capabilities

    def test_setup_environment(self, agent, monkeypatch):
        """Test environment setup."""
        # Clear any existing env vars
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        agent._setup_environment()

        # Local backend shouldn't set VERTEXAI
        assert os.getenv("GOOGLE_GENAI_USE_VERTEXAI") != "true"

        # Should set location and model
        assert os.getenv("GOOGLE_CLOUD_LOCATION") == "us-central1"
        assert os.getenv("ROOT_AGENT_MODEL") == "gemini-2.0-flash-001"

    def test_setup_environment_vertex_ai(self, monkeypatch):
        """Test environment setup for Vertex AI backend."""
        config = ADKConfig(
            backend=ADKBackend.VERTEX_AI,
            project_id="test-project",
        )
        agent = MockADKAgent(config, "test")

        agent._setup_environment()

        assert os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "true"
        assert os.getenv("GOOGLE_CLOUD_PROJECT") == "test-project"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
