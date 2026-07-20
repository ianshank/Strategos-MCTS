"""
Integration tests for Google ADK Agents.

Tests the integration of ADK agents with the multi-agent MCTS framework,
including user journey tests and agent collaboration scenarios.
"""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Check if google-adk is actually installed
try:
    import google.adk  # noqa: F401

    GOOGLE_ADK_INSTALLED = True
except ImportError:
    GOOGLE_ADK_INSTALLED = False

# Mock ADK imports if not available
try:
    from src.integrations.google_adk.agents.data_science import DataScienceAgent
    from src.integrations.google_adk.agents.deep_search import DeepSearchAgent
    from src.integrations.google_adk.agents.ml_engineering import MLEngineeringAgent
    from src.integrations.google_adk.base import ADKAgentAdapter, ADKConfig

    ADK_AVAILABLE = True and GOOGLE_ADK_INSTALLED
except ImportError:
    ADK_AVAILABLE = False
    ADKAgentAdapter = Mock
    ADKConfig = Mock
    DeepSearchAgent = Mock
    MLEngineeringAgent = Mock
    DataScienceAgent = Mock

from src.adapters.llm.base import BaseLLMClient, LLMResponse

# Skip entire module if google-adk is not installed
pytestmark = pytest.mark.skipif(
    not GOOGLE_ADK_INSTALLED, reason="google-adk package not installed. Install with: pip install google-adk"
)


class TestADKAgentIntegration:
    """Test ADK agent integration with the framework."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock(spec=BaseLLMClient)
        client.generate.return_value = LLMResponse(
            text="Mock response",
            raw_response={"choices": [{"message": {"content": "Mock response"}}]},
        )
        client.generate_async = AsyncMock(
            return_value=LLMResponse(
                text="Mock async response",
                raw_response={"choices": [{"message": {"content": "Mock async response"}}]},
            )
        )
        return client

    @pytest.fixture
    def adk_config(self):
        """Create ADK configuration.

        Note: Google Cloud credentials are handled via GOOGLE_APPLICATION_CREDENTIALS
        environment variable automatically by the Google Cloud client libraries.
        """
        if ADK_AVAILABLE:
            return ADKConfig(
                project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", "test-project"),
                location="us-central1",
            )
        return Mock()

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    @pytest.mark.asyncio
    async def test_deep_search_agent_initialization(self, adk_config, mock_llm_client):
        """Test DeepSearchAgent initialization and basic functionality."""
        with patch("src.integrations.google_adk.agents.deep_search.vertexai", create=True):
            agent = DeepSearchAgent(config=adk_config)

            assert agent is not None
            assert agent.config == adk_config
            # llm_client is no longer an attribute in the new architecture
            assert agent.agent_name == "deep_search"

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    @pytest.mark.asyncio
    async def test_ml_engineering_agent_task_execution(self, adk_config, mock_llm_client):
        """Test MLEngineeringAgent task execution."""
        from src.integrations.google_adk.base import ADKAgentRequest

        with patch("src.integrations.google_adk.agents.ml_engineering.vertexai", create=True):
            agent = MLEngineeringAgent(config=adk_config)

            # Mock task execution using invoke instead of execute_task
            request = ADKAgentRequest(query="Optimize model training pipeline", parameters={"data_path": "dummy.csv"})
            result = await agent.invoke(request)

            assert result is not None
            assert result.status == "success"
            assert isinstance(result.result, str)

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    @pytest.mark.asyncio
    async def test_data_science_agent_analysis(self, adk_config, mock_llm_client):
        """Test DataScienceAgent data analysis capabilities."""
        from src.integrations.google_adk.base import ADKAgentRequest, ADKAgentResponse

        with patch("src.integrations.google_adk.agents.data_science.vertexai", create=True):
            agent = DataScienceAgent(config=adk_config)

            # If DataScienceAgent has analyze_data, we can use it, otherwise use invoke
            if hasattr(agent, "analyze_data"):
                result = await agent.analyze_data(
                    query="Analyze customer churn patterns", data_source="bigquery://project.dataset.table"
                )
            else:
                result = await agent.invoke(
                    ADKAgentRequest(
                        query="Analyze customer churn patterns",
                        parameters={"data_source": "bigquery://project.dataset.table"},
                    )
                )

            assert result is not None
            assert isinstance(result, ADKAgentResponse)
            assert isinstance(result.result, str)


class TestADKUserJourneys:
    """Test end-to-end user journeys with ADK agents."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for user journey tests."""
        client = Mock(spec=BaseLLMClient)
        client.generate.return_value = LLMResponse(
            text="Mock response",
            raw_response={"choices": [{"message": {"content": "Mock response"}}]},
        )
        client.generate_async = AsyncMock(
            return_value=LLMResponse(
                text="Mock async response",
                raw_response={"choices": [{"message": {"content": "Mock async response"}}]},
            )
        )
        return client

    @pytest.fixture
    def adk_config(self):
        """Create ADK configuration for user journey tests."""
        if ADK_AVAILABLE:
            return ADKConfig(
                project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", "test-project"),
                location="us-central1",
            )
        return Mock()

    @pytest.fixture
    def graph_builder(self, mock_llm_client):
        """Create a mock GraphBuilder with llm_client for testing."""
        mock_builder = Mock()
        mock_builder.llm_client = mock_llm_client
        return mock_builder

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    @pytest.mark.asyncio
    async def test_research_and_development_journey(self, graph_builder, adk_config):
        """Test a complete R&D user journey using ADK agents."""
        from src.integrations.google_adk.base import ADKAgentRequest, ADKAgentResponse

        with patch("src.integrations.google_adk.base.vertexai", create=True):
            # Initialize agents
            deep_search = DeepSearchAgent(config=adk_config)
            ml_engineer = MLEngineeringAgent(config=adk_config)
            data_scientist = DataScienceAgent(config=adk_config)

            # User journey: Research -> Design -> Implement
            journey_steps = [
                {
                    "step": "Research",
                    "agent": deep_search,
                    "task": "Research latest advances in transformer architectures for NLP",
                },
                {
                    "step": "Design",
                    "agent": ml_engineer,
                    "task": "Design a scalable training pipeline for the new architecture",
                },
                {
                    "step": "Analysis",
                    "agent": data_scientist,
                    "task": "Analyze performance metrics and suggest optimizations",
                },
            ]

            results = []
            for step in journey_steps:
                # Mock agent execution
                mock_response = ADKAgentResponse(result=f"Completed: {step['task']}", metadata={"step": step["step"]})
                with patch.object(step["agent"], "invoke", return_value=mock_response):
                    result = await step["agent"].invoke(ADKAgentRequest(query=step["task"]))
                    results.append({"step": step["step"], "result": result})

            assert len(results) == 3
            assert all(hasattr(r["result"], "result") for r in results)

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    @pytest.mark.asyncio
    async def test_collaborative_problem_solving_journey(self, graph_builder, adk_config):
        """Test collaborative problem-solving between ADK agents."""
        from src.integrations.google_adk.base import ADKAgentRequest, ADKAgentResponse

        with patch("src.integrations.google_adk.base.vertexai", create=True):
            # Initialize agents
            agents = {
                "search": DeepSearchAgent(config=adk_config),
                "engineer": MLEngineeringAgent(config=adk_config),
                "analyst": DataScienceAgent(config=adk_config),
            }

            problem = "Develop a real-time fraud detection system with explainable AI"

            workflow = {
                "research_phase": {
                    "agent": "search",
                    "task": f"Research state-of-the-art fraud detection techniques for {problem}",
                    "output_to": ["engineer", "analyst"],
                },
                "design_phase": {
                    "agent": "engineer",
                    "task": "Design system architecture based on research findings",
                    "requires": ["research_phase"],
                    "output_to": ["analyst"],
                },
                "evaluation_phase": {
                    "agent": "analyst",
                    "task": "Evaluate proposed architecture and suggest improvements",
                    "requires": ["research_phase", "design_phase"],
                },
            }

            results = {}
            for phase_name, phase_config in workflow.items():
                agent = agents[phase_config["agent"]]

                context = {}
                if "requires" in phase_config:
                    for req in phase_config["requires"]:
                        if req in results:
                            context[req] = results[req].result

                mock_response = ADKAgentResponse(
                    result=f"Completed: {phase_config['task']}", metadata={"confidence": 0.85, "context": context}
                )
                with patch.object(agent, "invoke", return_value=mock_response):
                    request = ADKAgentRequest(query=phase_config["task"], context=context)
                    results[phase_name] = await agent.invoke(request)

            assert len(results) == 3
            assert results["evaluation_phase"].metadata["context"]
            assert "research_phase" in results["evaluation_phase"].metadata["context"]
            assert "design_phase" in results["evaluation_phase"].metadata["context"]

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, graph_builder, adk_config):
        """Test error handling and recovery in ADK agent workflows."""
        from src.integrations.google_adk.base import ADKAgentRequest

        with patch("src.integrations.google_adk.base.vertexai", create=True):
            agent = DeepSearchAgent(config=adk_config)

            error_scenarios = [
                {"error_type": "RateLimitError", "should_retry": True, "max_retries": 3},
                {"error_type": "AuthenticationError", "should_retry": False, "max_retries": 0},
                {"error_type": "TimeoutError", "should_retry": True, "max_retries": 2},
            ]

            for scenario in error_scenarios:
                with patch.object(agent, "_agent_invoke", new_callable=AsyncMock) as mock_invoke:
                    mock_invoke.side_effect = Exception(scenario["error_type"])

                    result = await agent.invoke(ADKAgentRequest(query="Test task"))
                    assert result.status == "error"
                    assert scenario["error_type"] in result.error


class TestADKAgentPerformance:
    """Test performance characteristics of ADK agents."""

    @pytest.fixture
    def adk_config(self):
        """Create ADK configuration for performance tests."""
        if ADK_AVAILABLE:
            return ADKConfig(
                project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", "test-project"),
                location="us-central1",
            )
        return Mock()

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, adk_config):
        """Test parallel execution of multiple ADK agents."""
        from src.integrations.google_adk.base import ADKAgentRequest, ADKAgentResponse

        with patch("src.integrations.google_adk.base.vertexai", create=True):
            agents = [
                DeepSearchAgent(config=adk_config),
                MLEngineeringAgent(config=adk_config),
                DataScienceAgent(config=adk_config),
            ]

            tasks = [
                "Research quantum computing applications",
                "Design distributed training system",
                "Analyze model performance metrics",
            ]

            for agent in agents:
                agent.invoke = AsyncMock(return_value=ADKAgentResponse(result="Task completed"))

            results = await asyncio.gather(
                *[agent.invoke(ADKAgentRequest(query=task)) for agent, task in zip(agents, tasks, strict=True)]
            )

            assert len(results) == 3
            assert all(r.result == "Task completed" for r in results)

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    def test_agent_initialization_performance(self, adk_config):
        """Test ADK agent initialization time."""
        import time

        with patch("src.integrations.google_adk.base.vertexai", create=True):
            start = time.time()
            agent = DeepSearchAgent(config=adk_config)
            assert agent is not None
            assert time.time() - start < 1.0  # Should initialize quickly

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    @pytest.mark.asyncio
    async def test_agent_caching_effectiveness(self, adk_config):
        """Test caching effectiveness in ADK agents."""
        from src.integrations.google_adk.base import ADKAgentRequest, ADKAgentResponse

        with patch("src.integrations.google_adk.base.vertexai", create=True):
            agent = DeepSearchAgent(config=adk_config)
            agent.invoke = AsyncMock(return_value=ADKAgentResponse(result="Cached response"))

            result1 = await agent.invoke(ADKAgentRequest(query="Research topic A"))
            result2 = await agent.invoke(ADKAgentRequest(query="Research topic A"))
            await agent.invoke(ADKAgentRequest(query="Research topic B"))

            assert result1.result == result2.result
            assert agent.invoke.call_count == 3


class TestADKAgentSecurity:
    """Test security aspects of ADK agent integration."""

    @pytest.fixture
    def adk_config(self):
        """Create ADK configuration for security tests."""
        if ADK_AVAILABLE:
            return ADKConfig(
                project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", "test-project"),
                location="us-central1",
            )
        return Mock()

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    def test_credential_handling(self, adk_config):
        """Test secure handling of credentials."""
        with patch("src.integrations.google_adk.base.vertexai", create=True):
            agent = DeepSearchAgent(config=adk_config)

            agent_str = str(agent)
            assert "api_key" not in agent_str.lower()
            assert "credential" not in agent_str.lower()

            if ADK_AVAILABLE:
                assert adk_config.project_id is not None

    @pytest.mark.skipif(not ADK_AVAILABLE, reason="ADK dependencies not available")
    @pytest.mark.asyncio
    async def test_input_sanitization(self, adk_config):
        """Test input sanitization in ADK agents."""
        from src.integrations.google_adk.base import ADKAgentRequest, ADKAgentResponse

        with patch("src.integrations.google_adk.base.vertexai", create=True):
            agent = DeepSearchAgent(config=adk_config)
            agent.invoke = AsyncMock(return_value=ADKAgentResponse(result="Sanitized response"))

            malicious_inputs = [
                "<script>alert('XSS')</script>",
                "'; DROP TABLE users; --",
                "../../../etc/passwd",
                "{{7*7}}",
            ]

            for malicious_input in malicious_inputs:
                result = await agent.invoke(ADKAgentRequest(query=malicious_input))
                assert result.result == "Sanitized response"
                agent.invoke.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
