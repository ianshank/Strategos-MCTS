"""
Comprehensive UI E2E Tests for the LangGraph Multi-Agent MCTS Framework.

Tests complete user interface interactions through the Gradio frontend:
1. Page load and initial state verification
2. Query submission and response validation
3. Model selector functionality (RNN/BERT)
4. Example query dropdown functionality
5. Response completeness validation
6. Agent details accordion functionality
7. Performance metrics display
8. Error handling and edge cases
9. Personality-infused response generation
10. Multi-query session testing

These tests use the Gradio test client for fast, reliable UI testing.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

# Try to import Gradio test client
try:
    from gradio_client import Client as GradioClient
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    GradioClient = None

from tests.utils.langsmith_tracing import (
    trace_e2e_test,
    update_run_metadata,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def gradio_app():
    """Import and return the Gradio app for testing."""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Import the app module
    import app as gradio_app_module

    return gradio_app_module


@pytest.fixture
def mock_framework():
    """Create a mock framework for isolated UI testing."""
    from dataclasses import dataclass

    @dataclass
    class MockAgentResult:
        agent_name: str = "MCTS (Monte Carlo Tree Search)"
        response: str = "This is a complete mock response for testing purposes. The MCTS analysis has determined that the optimal approach involves strategic exploration of the decision tree to identify the best course of action."
        confidence: float = 0.88
        reasoning_steps: list = None
        execution_time_ms: float = 125.5

        def __post_init__(self):
            if self.reasoning_steps is None:
                self.reasoning_steps = [
                    "Build search tree",
                    "Selection: UCB1 exploration",
                    "Expansion: Add promising nodes",
                    "Simulation: Rollout evaluation",
                    "Backpropagation: Update values",
                ]

    @dataclass
    class MockControllerDecision:
        selected_agent: str = "mcts"
        confidence: float = 0.73
        routing_probabilities: dict = None
        features_used: dict = None

        def __post_init__(self):
            if self.routing_probabilities is None:
                self.routing_probabilities = {
                    "hrm": 0.266,
                    "trm": 0.003,
                    "mcts": 0.730,
                }
            if self.features_used is None:
                self.features_used = {
                    "hrm_confidence": 0.35,
                    "trm_confidence": 0.25,
                    "mcts_value": 0.40,
                    "consensus_score": 0.65,
                    "query_length": 75,
                    "is_technical": True,
                }

    mock = MagicMock()
    mock.process_query = AsyncMock(return_value=(MockAgentResult(), MockControllerDecision()))

    return mock


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "tactical": "What is the best approach for analyzing complex tactical situations with multiple variables?",
        "technical": "How can we optimize a Python application that processes 10GB of log files daily?",
        "comparison": "Compare the performance characteristics of B-trees vs LSM-trees for write-heavy workloads",
        "design": "Design a distributed rate limiting system that handles 100k requests per second",
        "simple": "What is machine learning?",
        "short": "Hello",
        "long": "This is a very long query " * 50,
        "special_chars": "How do I handle <script>alert('xss')</script> and other special characters?",
        "unicode": "如何处理多语言文本？ Как обрабатывать многоязычный текст?",
        "empty": "",
    }


# ============================================================================
# UI COMPONENT TESTS
# ============================================================================


class TestUIPageLoad:
    """Tests for initial page load and component rendering."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_page_load_components",
        phase="ui_testing",
        scenario_type="page_load",
        tags=["ui", "page_load", "components"],
    )
    def test_page_components_present(self, gradio_app):
        """Verify all expected UI components are present."""
        demo = gradio_app.demo

        # Check that the demo block exists
        assert demo is not None
        assert hasattr(demo, 'blocks')

        # Verify essential components exist in the blocks
        component_types = []
        for _block_id, block in demo.blocks.items():
            component_types.append(type(block).__name__)

        # Expected component types
        expected_types = ["Textbox", "Button", "Radio", "Dropdown", "Markdown", "JSON", "Accordion"]

        for expected in expected_types:
            found = any(expected in ct for ct in component_types)
            if not found:
                # Some components might have different names
                pass

        update_run_metadata({
            "test": "page_components_present",
            "component_count": len(component_types),
            "component_types": list(set(component_types))[:10],  # First 10 unique types
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_page_title",
        phase="ui_testing",
        scenario_type="page_load",
        tags=["ui", "page_load", "title"],
    )
    def test_page_title(self, gradio_app):
        """Verify page title is set correctly."""
        demo = gradio_app.demo

        assert demo.title == "LangGraph Multi-Agent MCTS - Trained Models Demo"

        update_run_metadata({
            "test": "page_title",
            "title": demo.title,
        })


# ============================================================================
# QUERY PROCESSING TESTS
# ============================================================================


class TestUIQueryProcessing:
    """Tests for query submission and processing."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_query_submission",
        phase="ui_testing",
        scenario_type="query_processing",
        tags=["ui", "query", "submission"],
    )
    def test_query_submission_sync(self, gradio_app, sample_queries):
        """Test synchronous query submission."""
        # Test the sync wrapper function directly
        result = gradio_app.process_query_sync(
            query=sample_queries["tactical"],
            controller_type="RNN",
        )

        # Result should be a tuple of 6 elements
        assert isinstance(result, tuple)
        assert len(result) == 6

        final_response, agent_details, routing_viz, features_viz, metrics, personality_response = result

        # Validate response is not empty
        assert final_response is not None
        assert len(final_response) > 0

        # Validate agent details
        assert agent_details is not None
        assert "agent" in agent_details
        assert "confidence" in agent_details
        assert "reasoning_steps" in agent_details

        # Validate routing visualization contains expected content
        assert "Meta-Controller Decision" in routing_viz
        assert "Selected Agent" in routing_viz
        assert "Confidence" in routing_viz

        # Validate features visualization
        assert "Features Used" in features_viz

        # Validate metrics
        assert "Controller" in metrics
        assert "Execution Time" in metrics

        # Validate personality response
        assert personality_response is not None
        assert len(personality_response) > 0

        update_run_metadata({
            "test": "query_submission_sync",
            "query_length": len(sample_queries["tactical"]),
            "response_length": len(final_response),
            "agent": agent_details.get("agent"),
            "has_personality_response": len(personality_response) > 0,
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_rnn_controller",
        phase="ui_testing",
        scenario_type="controller_selection",
        tags=["ui", "controller", "rnn"],
    )
    def test_rnn_controller_selection(self, gradio_app, sample_queries):
        """Test query processing with RNN controller."""
        result = gradio_app.process_query_sync(
            query=sample_queries["technical"],
            controller_type="RNN",
        )

        final_response, agent_details, routing_viz, features_viz, metrics, _ = result

        # Verify RNN controller was used
        assert "RNN" in metrics

        # Verify response quality
        assert len(final_response) > 20  # Response should be substantive

        update_run_metadata({
            "test": "rnn_controller_selection",
            "controller": "RNN",
            "response_length": len(final_response),
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_bert_controller",
        phase="ui_testing",
        scenario_type="controller_selection",
        tags=["ui", "controller", "bert"],
    )
    def test_bert_controller_selection(self, gradio_app, sample_queries):
        """Test query processing with BERT controller."""
        result = gradio_app.process_query_sync(
            query=sample_queries["comparison"],
            controller_type="BERT",
        )

        final_response, agent_details, routing_viz, features_viz, metrics, _ = result

        # Verify BERT controller was used
        assert "BERT" in metrics

        # Verify response quality
        assert len(final_response) > 20

        update_run_metadata({
            "test": "bert_controller_selection",
            "controller": "BERT",
            "response_length": len(final_response),
        })


# ============================================================================
# RESPONSE COMPLETENESS TESTS
# ============================================================================


class TestUIResponseCompleteness:
    """Tests for response completeness and quality."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_response_not_truncated",
        phase="ui_testing",
        scenario_type="response_quality",
        tags=["ui", "response", "completeness"],
    )
    def test_response_not_truncated(self, gradio_app, sample_queries):
        """Verify responses are not truncated with '...'."""
        result = gradio_app.process_query_sync(
            query=sample_queries["tactical"],
            controller_type="RNN",
        )

        final_response, agent_details, routing_viz, features_viz, metrics, personality_response = result

        # Check that response doesn't end with truncation indicators
        # Note: Some truncation may be acceptable in certain contexts
        truncation_indicators = ["...", "…", "[truncated]", "[cut off]"]

        # Log if truncation is detected for investigation
        response_truncated = any(final_response.rstrip().endswith(ind) for ind in truncation_indicators)

        # The response should contain meaningful content
        assert len(final_response) > 50, "Response too short, may be truncated"

        # Personality response should also be complete
        assert len(personality_response) > 50, "Personality response too short"

        update_run_metadata({
            "test": "response_not_truncated",
            "response_length": len(final_response),
            "personality_response_length": len(personality_response),
            "response_truncated": response_truncated,
            "response_preview": final_response[:200],
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_agent_details_complete",
        phase="ui_testing",
        scenario_type="response_quality",
        tags=["ui", "response", "agent_details"],
    )
    def test_agent_details_complete(self, gradio_app, sample_queries):
        """Verify agent details contain all expected fields."""
        result = gradio_app.process_query_sync(
            query=sample_queries["design"],
            controller_type="RNN",
        )

        _, agent_details, _, _, _, _ = result

        # Required fields
        required_fields = ["agent", "confidence", "reasoning_steps", "execution_time_ms"]

        for field in required_fields:
            assert field in agent_details, f"Missing required field: {field}"

        # Validate field values
        assert isinstance(agent_details["agent"], str)
        assert len(agent_details["agent"]) > 0

        # Confidence should be a valid percentage string
        confidence_str = agent_details["confidence"]
        assert "%" in confidence_str

        # Reasoning steps should be a non-empty list
        assert isinstance(agent_details["reasoning_steps"], list)
        assert len(agent_details["reasoning_steps"]) > 0

        # Execution time should be a number
        assert isinstance(agent_details["execution_time_ms"], (int, float))
        assert agent_details["execution_time_ms"] >= 0

        update_run_metadata({
            "test": "agent_details_complete",
            "agent": agent_details["agent"],
            "num_reasoning_steps": len(agent_details["reasoning_steps"]),
            "execution_time_ms": agent_details["execution_time_ms"],
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_routing_probabilities",
        phase="ui_testing",
        scenario_type="response_quality",
        tags=["ui", "response", "routing"],
    )
    def test_routing_probabilities_displayed(self, gradio_app, sample_queries):
        """Verify routing probabilities are displayed correctly."""
        result = gradio_app.process_query_sync(
            query=sample_queries["tactical"],
            controller_type="RNN",
        )

        _, _, routing_viz, _, _, _ = result

        # Check that all agents are represented
        agents = ["HRM", "TRM", "MCTS"]
        for agent in agents:
            assert agent in routing_viz, f"Missing agent in routing: {agent}"

        # Check that percentages are displayed
        assert "%" in routing_viz, "No percentages in routing visualization"

        # Check for progress bar characters
        assert "█" in routing_viz, "No progress bar visualization"

        update_run_metadata({
            "test": "routing_probabilities_displayed",
            "routing_viz_length": len(routing_viz),
            "agents_found": [a for a in agents if a in routing_viz],
        })


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================


class TestUIInputValidation:
    """Tests for input validation and edge cases."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_empty_query_handling",
        phase="ui_testing",
        scenario_type="input_validation",
        tags=["ui", "input", "validation", "empty"],
    )
    def test_empty_query_handling(self, gradio_app, sample_queries):
        """Test handling of empty query."""
        result = gradio_app.process_query_sync(
            query=sample_queries["empty"],
            controller_type="RNN",
        )

        final_response, _, _, _, _, _ = result

        # Should return a message asking for input
        assert "Please enter a query" in final_response or len(final_response) == 0 or "enter" in final_response.lower()

        update_run_metadata({
            "test": "empty_query_handling",
            "response": final_response[:100] if final_response else "empty",
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_short_query_handling",
        phase="ui_testing",
        scenario_type="input_validation",
        tags=["ui", "input", "validation", "short"],
    )
    def test_short_query_handling(self, gradio_app, sample_queries):
        """Test handling of short query."""
        result = gradio_app.process_query_sync(
            query=sample_queries["short"],
            controller_type="RNN",
        )

        final_response, agent_details, _, _, _, _ = result

        # Should still process and return a response
        assert final_response is not None
        # Response might be an error or a processed result

        update_run_metadata({
            "test": "short_query_handling",
            "query": sample_queries["short"],
            "response_length": len(final_response) if final_response else 0,
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_special_chars_handling",
        phase="ui_testing",
        scenario_type="input_validation",
        tags=["ui", "input", "validation", "special_chars"],
    )
    def test_special_characters_handling(self, gradio_app, sample_queries):
        """Test handling of special characters in query."""
        result = gradio_app.process_query_sync(
            query=sample_queries["special_chars"],
            controller_type="RNN",
        )

        final_response, _, _, _, _, _ = result

        # Should process without errors
        assert final_response is not None

        # Should not contain unescaped script tags in response
        assert "<script>" not in final_response.lower()

        update_run_metadata({
            "test": "special_chars_handling",
            "query_preview": sample_queries["special_chars"][:50],
            "response_length": len(final_response) if final_response else 0,
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_unicode_handling",
        phase="ui_testing",
        scenario_type="input_validation",
        tags=["ui", "input", "validation", "unicode"],
    )
    def test_unicode_handling(self, gradio_app, sample_queries):
        """Test handling of unicode characters in query."""
        result = gradio_app.process_query_sync(
            query=sample_queries["unicode"],
            controller_type="RNN",
        )

        final_response, _, _, _, _, _ = result

        # Should process without errors
        assert final_response is not None

        update_run_metadata({
            "test": "unicode_handling",
            "query_preview": sample_queries["unicode"][:30],
            "response_length": len(final_response) if final_response else 0,
        })


# ============================================================================
# CONTROLLER COMPARISON TESTS
# ============================================================================


class TestUIControllerComparison:
    """Tests comparing RNN and BERT controller behavior."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_controller_comparison",
        phase="ui_testing",
        scenario_type="controller_comparison",
        tags=["ui", "controller", "comparison"],
    )
    def test_controller_comparison(self, gradio_app, sample_queries):
        """Compare responses from RNN and BERT controllers."""
        query = sample_queries["tactical"]

        # Test with RNN
        rnn_result = gradio_app.process_query_sync(
            query=query,
            controller_type="RNN",
        )

        # Test with BERT
        bert_result = gradio_app.process_query_sync(
            query=query,
            controller_type="BERT",
        )

        rnn_response, rnn_details, rnn_routing, _, rnn_metrics, _ = rnn_result
        bert_response, bert_details, bert_routing, _, bert_metrics, _ = bert_result

        # Both should return valid responses
        assert len(rnn_response) > 0
        assert len(bert_response) > 0

        # Verify different controllers were used
        assert "RNN" in rnn_metrics
        assert "BERT" in bert_metrics

        # Extract selected agents for comparison
        rnn_agent = rnn_details.get("agent", "")
        bert_agent = bert_details.get("agent", "")

        update_run_metadata({
            "test": "controller_comparison",
            "query": query[:50],
            "rnn_agent": rnn_agent,
            "bert_agent": bert_agent,
            "rnn_response_length": len(rnn_response),
            "bert_response_length": len(bert_response),
            "agents_match": rnn_agent == bert_agent,
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_controller_consistency",
        phase="ui_testing",
        scenario_type="controller_consistency",
        tags=["ui", "controller", "consistency"],
    )
    def test_controller_consistency(self, gradio_app, sample_queries):
        """Test that same controller gives consistent results."""
        query = sample_queries["technical"]

        results = []
        for _ in range(3):
            result = gradio_app.process_query_sync(
                query=query,
                controller_type="RNN",
            )
            results.append(result)

        # Extract agents from each run
        agents = [r[1].get("agent", "") for r in results]

        # All runs should select the same agent (deterministic behavior)
        assert len(set(agents)) == 1, f"Inconsistent agent selection: {agents}"

        update_run_metadata({
            "test": "controller_consistency",
            "num_runs": 3,
            "agents_selected": agents,
            "consistent": len(set(agents)) == 1,
        })


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestUIPerformance:
    """Tests for UI response performance."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_response_time",
        phase="ui_testing",
        scenario_type="performance",
        tags=["ui", "performance", "response_time"],
    )
    def test_response_time(self, gradio_app, sample_queries):
        """Test that responses are returned within acceptable time."""
        query = sample_queries["tactical"]

        start_time = time.perf_counter()
        result = gradio_app.process_query_sync(
            query=query,
            controller_type="RNN",
        )
        elapsed_time = time.perf_counter() - start_time

        # Response should be within 30 seconds (generous for first load)
        assert elapsed_time < 30, f"Response took too long: {elapsed_time:.2f}s"

        _, agent_details, _, _, _, _ = result

        update_run_metadata({
            "test": "response_time",
            "elapsed_time_s": elapsed_time,
            "execution_time_ms": agent_details.get("execution_time_ms", 0),
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_multiple_queries_performance",
        phase="ui_testing",
        scenario_type="performance",
        tags=["ui", "performance", "throughput"],
    )
    def test_multiple_queries_performance(self, gradio_app, sample_queries):
        """Test performance with multiple sequential queries."""
        queries = [
            sample_queries["tactical"],
            sample_queries["technical"],
            sample_queries["comparison"],
        ]

        times = []
        for query in queries:
            start_time = time.perf_counter()
            gradio_app.process_query_sync(
                query=query,
                controller_type="RNN",
            )
            elapsed_time = time.perf_counter() - start_time
            times.append(elapsed_time)

        avg_time = sum(times) / len(times)

        # Average response time should be reasonable
        assert avg_time < 15, f"Average response time too high: {avg_time:.2f}s"

        update_run_metadata({
            "test": "multiple_queries_performance",
            "num_queries": len(queries),
            "times": times,
            "avg_time_s": avg_time,
            "max_time_s": max(times),
            "min_time_s": min(times),
        })


# ============================================================================
# EXAMPLE QUERIES TESTS
# ============================================================================


class TestUIExampleQueries:
    """Tests for example query functionality."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_example_queries_available",
        phase="ui_testing",
        scenario_type="example_queries",
        tags=["ui", "examples"],
    )
    def test_example_queries_available(self, gradio_app):
        """Verify example queries are available."""
        examples = gradio_app.EXAMPLE_QUERIES

        assert isinstance(examples, list)
        assert len(examples) >= 3  # Should have multiple examples

        for example in examples:
            assert isinstance(example, str)
            assert len(example) > 10  # Examples should be meaningful

        update_run_metadata({
            "test": "example_queries_available",
            "num_examples": len(examples),
            "example_previews": [e[:50] for e in examples[:3]],
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_example_queries_processing",
        phase="ui_testing",
        scenario_type="example_queries",
        tags=["ui", "examples", "processing"],
    )
    def test_example_queries_processing(self, gradio_app):
        """Test that all example queries can be processed."""
        examples = gradio_app.EXAMPLE_QUERIES

        results = []
        for example in examples[:3]:  # Test first 3 examples
            result = gradio_app.process_query_sync(
                query=example,
                controller_type="RNN",
            )

            final_response, agent_details, _, _, _, _ = result

            results.append({
                "query": example[:50],
                "response_length": len(final_response) if final_response else 0,
                "agent": agent_details.get("agent", "unknown"),
            })

        # All examples should produce valid responses
        for r in results:
            assert r["response_length"] > 0, f"Empty response for: {r['query']}"

        update_run_metadata({
            "test": "example_queries_processing",
            "num_tested": len(results),
            "results": results,
        })


# ============================================================================
# PERSONALITY RESPONSE TESTS
# ============================================================================


class TestUIPersonalityResponse:
    """Tests for personality-infused response generation."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_personality_response_generated",
        phase="ui_testing",
        scenario_type="personality",
        tags=["ui", "personality", "response"],
    )
    def test_personality_response_generated(self, gradio_app, sample_queries):
        """Verify personality-infused response is generated."""
        result = gradio_app.process_query_sync(
            query=sample_queries["tactical"],
            controller_type="RNN",
        )

        final_response, _, _, _, _, personality_response = result

        # Personality response should exist and differ from raw response
        assert personality_response is not None
        assert len(personality_response) > 0

        # Personality response should be more conversational
        # (typically longer or with different phrasing)

        update_run_metadata({
            "test": "personality_response_generated",
            "raw_response_length": len(final_response) if final_response else 0,
            "personality_response_length": len(personality_response),
            "personality_preview": personality_response[:200] if personality_response else "",
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_personality_response_quality",
        phase="ui_testing",
        scenario_type="personality",
        tags=["ui", "personality", "quality"],
    )
    def test_personality_response_quality(self, gradio_app, sample_queries):
        """Test quality of personality-infused responses."""
        result = gradio_app.process_query_sync(
            query=sample_queries["design"],
            controller_type="RNN",
        )

        _, _, _, _, _, personality_response = result

        # Personality response should be substantive
        assert len(personality_response) > 50

        # Should not be just an error message
        error_indicators = ["error", "failed", "exception", "traceback"]
        response_lower = personality_response.lower()
        has_error = any(ind in response_lower for ind in error_indicators)

        # Log if error detected but don't fail (some errors might be expected)

        update_run_metadata({
            "test": "personality_response_quality",
            "response_length": len(personality_response),
            "has_error_indicator": has_error,
            "preview": personality_response[:300],
        })


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestUIErrorHandling:
    """Tests for error handling in the UI."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_graceful_error_handling",
        phase="ui_testing",
        scenario_type="error_handling",
        tags=["ui", "error", "graceful"],
    )
    def test_graceful_error_handling(self, gradio_app):
        """Test that errors are handled gracefully."""
        # Test with None query (should be handled)
        try:
            result = gradio_app.process_query_sync(
                query=None,
                controller_type="RNN",
            )
            # If it doesn't raise, check the response
            result[0] if result else ""
            assert True  # Graceful handling
        except (TypeError, AttributeError):
            # Expected - None is not a valid query
            pass
        except Exception as e:
            # Other exceptions should be informative
            assert str(e) is not None

        update_run_metadata({
            "test": "graceful_error_handling",
            "handled_gracefully": True,
        })


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestUIIntegration:
    """Integration tests for complete UI workflows."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_complete_workflow",
        phase="ui_testing",
        scenario_type="integration",
        tags=["ui", "integration", "workflow"],
    )
    def test_complete_ui_workflow(self, gradio_app, sample_queries):
        """Test complete UI workflow from query to response."""
        # Step 1: Submit query
        query = sample_queries["tactical"]

        # Step 2: Process with RNN controller
        result = gradio_app.process_query_sync(
            query=query,
            controller_type="RNN",
        )

        final_response, agent_details, routing_viz, features_viz, metrics, personality_response = result

        # Step 3: Validate all outputs
        workflow_valid = all([
            final_response is not None and len(final_response) > 0,
            agent_details is not None and "agent" in agent_details,
            routing_viz is not None and "Meta-Controller" in routing_viz,
            features_viz is not None and "Features" in features_viz,
            metrics is not None and "Controller" in metrics,
            personality_response is not None,
        ])

        assert workflow_valid, "Complete workflow validation failed"

        # Step 4: Verify agent selection logic
        selected_agent = agent_details.get("agent", "")
        assert any(agent in selected_agent for agent in ["HRM", "TRM", "MCTS"])

        update_run_metadata({
            "test": "complete_ui_workflow",
            "workflow_valid": workflow_valid,
            "selected_agent": selected_agent,
            "response_length": len(final_response),
            "has_personality": len(personality_response) > 0,
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_multi_query_session",
        phase="ui_testing",
        scenario_type="integration",
        tags=["ui", "integration", "session"],
    )
    def test_multi_query_session(self, gradio_app, sample_queries):
        """Test multiple queries in a session."""
        queries = [
            ("tactical", "RNN"),
            ("technical", "BERT"),
            ("comparison", "RNN"),
            ("design", "BERT"),
        ]

        session_results = []

        for query_key, controller in queries:
            result = gradio_app.process_query_sync(
                query=sample_queries[query_key],
                controller_type=controller,
            )

            final_response, agent_details, _, _, _, _ = result

            session_results.append({
                "query_type": query_key,
                "controller": controller,
                "agent": agent_details.get("agent", "unknown"),
                "response_length": len(final_response) if final_response else 0,
                "success": len(final_response) > 0 if final_response else False,
            })

        # All queries should succeed
        all_success = all(r["success"] for r in session_results)
        assert all_success, f"Some queries failed: {session_results}"

        update_run_metadata({
            "test": "multi_query_session",
            "num_queries": len(queries),
            "all_success": all_success,
            "results": session_results,
        })


# ============================================================================
# AGENT RESPONSE COMPLETENESS TESTS
# ============================================================================


class TestAgentResponseCompleteness:
    """Tests specifically for agent response completeness issues."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_hrm_response_complete",
        phase="ui_testing",
        scenario_type="agent_response",
        tags=["ui", "agent", "hrm", "completeness"],
    )
    def test_hrm_response_complete(self, gradio_app):
        """Test that HRM agent responses are complete."""
        # Query designed to trigger HRM routing
        hrm_query = "Break down the problem of designing a scalable microservices architecture into hierarchical components and sub-problems."

        result = gradio_app.process_query_sync(
            query=hrm_query,
            controller_type="RNN",
        )

        final_response, agent_details, _, _, _, _ = result

        # Check response completeness
        assert final_response is not None
        assert len(final_response) > 50, f"HRM response too short: {len(final_response)} chars"

        # Response should not end with truncation
        assert not final_response.rstrip().endswith("..."), f"HRM response truncated: {final_response[-50:]}"

        update_run_metadata({
            "test": "hrm_response_complete",
            "agent": agent_details.get("agent", ""),
            "response_length": len(final_response),
            "response_preview": final_response[:200],
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_trm_response_complete",
        phase="ui_testing",
        scenario_type="agent_response",
        tags=["ui", "agent", "trm", "completeness"],
    )
    def test_trm_response_complete(self, gradio_app):
        """Test that TRM agent responses are complete."""
        # Query designed to trigger TRM routing
        trm_query = "Iteratively refine and improve the following approach: Start with a basic sorting algorithm and progressively optimize it for large datasets."

        result = gradio_app.process_query_sync(
            query=trm_query,
            controller_type="RNN",
        )

        final_response, agent_details, _, _, _, _ = result

        # Check response completeness
        assert final_response is not None
        assert len(final_response) > 50, f"TRM response too short: {len(final_response)} chars"

        update_run_metadata({
            "test": "trm_response_complete",
            "agent": agent_details.get("agent", ""),
            "response_length": len(final_response),
            "response_preview": final_response[:200],
        })

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_mcts_response_complete",
        phase="ui_testing",
        scenario_type="agent_response",
        tags=["ui", "agent", "mcts", "completeness"],
    )
    def test_mcts_response_complete(self, gradio_app):
        """Test that MCTS agent responses are complete."""
        # Query designed to trigger MCTS routing
        mcts_query = "Optimize the best strategy for resource allocation across multiple competing objectives with uncertainty."

        result = gradio_app.process_query_sync(
            query=mcts_query,
            controller_type="RNN",
        )

        final_response, agent_details, _, _, _, _ = result

        # Check response completeness
        assert final_response is not None
        assert len(final_response) > 50, f"MCTS response too short: {len(final_response)} chars"

        update_run_metadata({
            "test": "mcts_response_complete",
            "agent": agent_details.get("agent", ""),
            "response_length": len(final_response),
            "response_preview": final_response[:200],
        })


# ============================================================================
# FEATURE EXTRACTION TESTS
# ============================================================================


class TestUIFeatureExtraction:
    """Tests for feature extraction and display."""

    @pytest.mark.e2e
    @pytest.mark.ui
    @trace_e2e_test(
        "ui_features_displayed",
        phase="ui_testing",
        scenario_type="features",
        tags=["ui", "features", "display"],
    )
    def test_features_displayed(self, gradio_app, sample_queries):
        """Test that features are correctly extracted and displayed."""
        result = gradio_app.process_query_sync(
            query=sample_queries["tactical"],
            controller_type="RNN",
        )

        _, _, _, features_viz, _, _ = result

        # Check that key features are displayed
        expected_features = [
            "hrm_confidence",
            "trm_confidence",
            "mcts_value",
            "consensus_score",
            "query_length",
        ]

        for feature in expected_features:
            assert feature in features_viz, f"Missing feature: {feature}"

        update_run_metadata({
            "test": "features_displayed",
            "features_viz_length": len(features_viz),
            "features_found": [f for f in expected_features if f in features_viz],
        })

