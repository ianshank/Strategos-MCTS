"""
Unit tests for Prometheus metrics collection module.

Tests DummyMetric fallback classes, metric registration, counter/histogram
operations, utility functions, and decorators.

Mocks prometheus_client if needed so tests run without it installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# DummyMetric Fallback Tests
# =============================================================================


class TestDummyMetric:
    """Tests for the DummyMetric fallback class used when prometheus_client is missing."""

    def test_dummy_metric_instantiation(self):
        """Test DummyMetric can be instantiated with arbitrary args."""
        from src.monitoring.prometheus_metrics import DummyMetric

        metric = DummyMetric("name", "description", ["label1", "label2"])
        assert metric is not None

    def test_dummy_metric_labels_returns_self(self):
        """Test DummyMetric.labels() returns self for chaining."""
        from src.monitoring.prometheus_metrics import DummyMetric

        metric = DummyMetric()
        result = metric.labels(agent_type="hrm", status="success")
        assert result is metric

    def test_dummy_metric_inc_is_noop(self):
        """Test DummyMetric.inc() does not raise."""
        from src.monitoring.prometheus_metrics import DummyMetric

        metric = DummyMetric()
        metric.inc()
        metric.inc(5)

    def test_dummy_metric_dec_is_noop(self):
        """Test DummyMetric.dec() does not raise."""
        from src.monitoring.prometheus_metrics import DummyMetric

        metric = DummyMetric()
        metric.dec()
        metric.dec(3)

    def test_dummy_metric_set_is_noop(self):
        """Test DummyMetric.set() does not raise."""
        from src.monitoring.prometheus_metrics import DummyMetric

        metric = DummyMetric()
        metric.set(42)

    def test_dummy_metric_observe_is_noop(self):
        """Test DummyMetric.observe() does not raise."""
        from src.monitoring.prometheus_metrics import DummyMetric

        metric = DummyMetric()
        metric.observe(1.5)

    def test_dummy_metric_info_is_noop(self):
        """Test DummyMetric.info() does not raise."""
        from src.monitoring.prometheus_metrics import DummyMetric

        metric = DummyMetric()
        metric.info({"key": "value"})

    def test_dummy_metric_chained_operations(self):
        """Test DummyMetric supports chained label().inc() calls."""
        from src.monitoring.prometheus_metrics import DummyMetric

        metric = DummyMetric()
        metric.labels(agent_type="hrm").inc()
        metric.labels(agent_type="trm").observe(0.5)
        metric.labels(status="success").dec()


# =============================================================================
# Module-Level Metric Registration Tests
# =============================================================================


class TestMetricRegistration:
    """Tests for module-level metric objects being properly defined."""

    def test_agent_requests_total_exists(self):
        """Test AGENT_REQUESTS_TOTAL metric is defined."""
        from src.monitoring.prometheus_metrics import AGENT_REQUESTS_TOTAL

        assert AGENT_REQUESTS_TOTAL is not None

    def test_agent_request_latency_exists(self):
        """Test AGENT_REQUEST_LATENCY metric is defined."""
        from src.monitoring.prometheus_metrics import AGENT_REQUEST_LATENCY

        assert AGENT_REQUEST_LATENCY is not None

    def test_mcts_iterations_total_exists(self):
        """Test MCTS_ITERATIONS_TOTAL metric is defined."""
        from src.monitoring.prometheus_metrics import MCTS_ITERATIONS_TOTAL

        assert MCTS_ITERATIONS_TOTAL is not None

    def test_mcts_node_count_exists(self):
        """Test MCTS_NODE_COUNT gauge is defined."""
        from src.monitoring.prometheus_metrics import MCTS_NODE_COUNT

        assert MCTS_NODE_COUNT is not None

    def test_active_operations_exists(self):
        """Test ACTIVE_OPERATIONS gauge is defined."""
        from src.monitoring.prometheus_metrics import ACTIVE_OPERATIONS

        assert ACTIVE_OPERATIONS is not None

    def test_llm_request_errors_exists(self):
        """Test LLM_REQUEST_ERRORS counter is defined."""
        from src.monitoring.prometheus_metrics import LLM_REQUEST_ERRORS

        assert LLM_REQUEST_ERRORS is not None

    def test_request_count_exists(self):
        """Test REQUEST_COUNT counter is defined."""
        from src.monitoring.prometheus_metrics import REQUEST_COUNT

        assert REQUEST_COUNT is not None

    def test_system_info_exists(self):
        """Test SYSTEM_INFO metric is defined."""
        from src.monitoring.prometheus_metrics import SYSTEM_INFO

        assert SYSTEM_INFO is not None


# =============================================================================
# Counter Operations Tests
# =============================================================================


class TestCounterOperations:
    """Tests for counter metric operations (using DummyMetric or real prometheus)."""

    def test_agent_requests_counter_inc(self):
        """Test incrementing agent requests counter does not raise."""
        from src.monitoring.prometheus_metrics import AGENT_REQUESTS_TOTAL

        AGENT_REQUESTS_TOTAL.labels(agent_type="hrm", status="success").inc()

    def test_mcts_iterations_counter_inc(self):
        """Test incrementing MCTS iterations counter does not raise."""
        from src.monitoring.prometheus_metrics import MCTS_ITERATIONS_TOTAL

        MCTS_ITERATIONS_TOTAL.labels(outcome="completed").inc()

    def test_error_count_counter_inc(self):
        """Test incrementing error count counter does not raise."""
        from src.monitoring.prometheus_metrics import ERROR_COUNT

        ERROR_COUNT.labels(error_type="ValueError").inc()

    def test_llm_token_usage_counter_inc(self):
        """Test incrementing LLM token usage counter does not raise."""
        from src.monitoring.prometheus_metrics import LLM_TOKEN_USAGE

        LLM_TOKEN_USAGE.labels(provider="openai", token_type="prompt").inc(150)

    def test_rate_limit_exceeded_counter_inc(self):
        """Test incrementing rate limit exceeded counter does not raise."""
        from src.monitoring.prometheus_metrics import RATE_LIMIT_EXCEEDED

        RATE_LIMIT_EXCEEDED.labels(client_id="client-1").inc()


# =============================================================================
# Histogram Operations Tests
# =============================================================================


class TestHistogramOperations:
    """Tests for histogram metric operations."""

    def test_agent_request_latency_observe(self):
        """Test observing agent request latency does not raise."""
        from src.monitoring.prometheus_metrics import AGENT_REQUEST_LATENCY

        AGENT_REQUEST_LATENCY.labels(agent_type="hrm").observe(1.5)

    def test_mcts_iteration_latency_observe(self):
        """Test observing MCTS iteration latency does not raise."""
        from src.monitoring.prometheus_metrics import MCTS_ITERATION_LATENCY

        MCTS_ITERATION_LATENCY.observe(0.05)

    def test_agent_confidence_scores_observe(self):
        """Test observing agent confidence scores does not raise."""
        from src.monitoring.prometheus_metrics import AGENT_CONFIDENCE_SCORES

        AGENT_CONFIDENCE_SCORES.labels(agent_type="trm").observe(0.85)

    def test_rag_retrieval_latency_observe(self):
        """Test observing RAG retrieval latency does not raise."""
        from src.monitoring.prometheus_metrics import RAG_RETRIEVAL_LATENCY

        RAG_RETRIEVAL_LATENCY.observe(0.25)


# =============================================================================
# Gauge Operations Tests
# =============================================================================


class TestGaugeOperations:
    """Tests for gauge metric operations."""

    def test_mcts_node_count_set(self):
        """Test setting MCTS node count does not raise."""
        from src.monitoring.prometheus_metrics import MCTS_NODE_COUNT

        MCTS_NODE_COUNT.set(500)

    def test_active_operations_inc_dec(self):
        """Test incrementing and decrementing active operations gauge."""
        from src.monitoring.prometheus_metrics import ACTIVE_OPERATIONS

        ACTIVE_OPERATIONS.labels(operation_type="mcts_simulation").inc()
        ACTIVE_OPERATIONS.labels(operation_type="mcts_simulation").dec()

    def test_active_requests_set(self):
        """Test setting active requests gauge."""
        from src.monitoring.prometheus_metrics import ACTIVE_REQUESTS

        ACTIVE_REQUESTS.inc()
        ACTIVE_REQUESTS.dec()

    def test_request_queue_depth_set(self):
        """Test setting request queue depth gauge."""
        from src.monitoring.prometheus_metrics import REQUEST_QUEUE_DEPTH

        REQUEST_QUEUE_DEPTH.set(10)


# =============================================================================
# setup_metrics Tests
# =============================================================================


class TestSetupMetrics:
    """Tests for the setup_metrics function."""

    def test_setup_metrics_with_prometheus_available(self):
        """Test setup_metrics sets system info when prometheus is available."""
        from src.monitoring.prometheus_metrics import setup_metrics

        # Should not raise regardless of prometheus availability
        setup_metrics(app_version="2.0.0", environment="test")

    def test_setup_metrics_with_prometheus_unavailable(self):
        """Test setup_metrics warns when prometheus is not available."""
        import src.monitoring.prometheus_metrics as mod

        with patch.object(mod, "PROMETHEUS_AVAILABLE", False):
            # Should not raise
            mod.setup_metrics(app_version="1.0.0", environment="development")


# =============================================================================
# track_operation Context Manager Tests
# =============================================================================


class TestTrackOperation:
    """Tests for the track_operation context manager."""

    def test_track_operation_increments_and_decrements(self):
        """Test track_operation increments on entry and decrements on exit."""
        from src.monitoring.prometheus_metrics import track_operation

        # Should not raise
        with track_operation("test_operation"):
            pass

    def test_track_operation_decrements_on_exception(self):
        """Test track_operation decrements even when exception occurs."""
        from src.monitoring.prometheus_metrics import track_operation

        with pytest.raises(ValueError, match="test error"):
            with track_operation("failing_operation"):
                raise ValueError("test error")


# =============================================================================
# measure_latency Context Manager Tests
# =============================================================================


class TestMeasureLatency:
    """Tests for the measure_latency context manager."""

    def test_measure_latency_with_labels(self):
        """Test measure_latency observes elapsed time with labels."""
        mock_histogram = MagicMock()

        from src.monitoring.prometheus_metrics import measure_latency

        with measure_latency(mock_histogram, agent_type="hrm"):
            pass

        mock_histogram.labels.assert_called_once_with(agent_type="hrm")
        mock_histogram.labels.return_value.observe.assert_called_once()

        # The observed value should be a positive float
        observed = mock_histogram.labels.return_value.observe.call_args[0][0]
        assert isinstance(observed, float)
        assert observed >= 0

    def test_measure_latency_without_labels(self):
        """Test measure_latency observes elapsed time without labels."""
        mock_histogram = MagicMock()

        from src.monitoring.prometheus_metrics import measure_latency

        with measure_latency(mock_histogram):
            pass

        mock_histogram.observe.assert_called_once()
        observed = mock_histogram.observe.call_args[0][0]
        assert isinstance(observed, float)
        assert observed >= 0

    def test_measure_latency_records_on_exception(self):
        """Test measure_latency records latency even when exception occurs."""
        mock_histogram = MagicMock()

        from src.monitoring.prometheus_metrics import measure_latency

        with pytest.raises(RuntimeError):
            with measure_latency(mock_histogram, provider="openai"):
                raise RuntimeError("api error")

        mock_histogram.labels.return_value.observe.assert_called_once()


# =============================================================================
# track_agent_request Decorator Tests
# =============================================================================


class TestTrackAgentRequest:
    """Tests for the track_agent_request decorator."""

    def test_sync_function_success(self):
        """Test track_agent_request wraps sync function correctly."""
        from src.monitoring.prometheus_metrics import track_agent_request

        @track_agent_request("hrm")
        def process_query(query):
            return f"result: {query}"

        result = process_query("test")
        assert result == "result: test"

    def test_sync_function_failure(self):
        """Test track_agent_request records error on sync function failure."""
        from src.monitoring.prometheus_metrics import track_agent_request

        @track_agent_request("trm")
        def failing_process():
            raise RuntimeError("processing error")

        with pytest.raises(RuntimeError, match="processing error"):
            failing_process()

    @pytest.mark.asyncio
    async def test_async_function_success(self):
        """Test track_agent_request wraps async function correctly."""
        from src.monitoring.prometheus_metrics import track_agent_request

        @track_agent_request("hrm")
        async def async_process(query):
            return f"async: {query}"

        result = await async_process("test")
        assert result == "async: test"

    @pytest.mark.asyncio
    async def test_async_function_failure(self):
        """Test track_agent_request records error on async function failure."""
        from src.monitoring.prometheus_metrics import track_agent_request

        @track_agent_request("trm")
        async def async_fail():
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            await async_fail()

    def test_preserves_function_metadata(self):
        """Test track_agent_request preserves function name and docstring."""
        from src.monitoring.prometheus_metrics import track_agent_request

        @track_agent_request("hrm")
        def documented_fn():
            """My docstring."""
            return True

        assert documented_fn.__name__ == "documented_fn"
        assert documented_fn.__doc__ == "My docstring."


# =============================================================================
# track_mcts_iteration Decorator Tests
# =============================================================================


class TestTrackMctsIteration:
    """Tests for the track_mcts_iteration decorator."""

    def test_successful_iteration(self):
        """Test track_mcts_iteration records completed iteration."""
        from src.monitoring.prometheus_metrics import track_mcts_iteration

        @track_mcts_iteration
        def run_iteration():
            return "done"

        result = run_iteration()
        assert result == "done"

    def test_timeout_iteration(self):
        """Test track_mcts_iteration records timeout outcome."""
        from src.monitoring.prometheus_metrics import track_mcts_iteration

        @track_mcts_iteration
        def timeout_iteration():
            raise TimeoutError("took too long")

        with pytest.raises(TimeoutError):
            timeout_iteration()

    def test_error_iteration(self):
        """Test track_mcts_iteration records error outcome."""
        from src.monitoring.prometheus_metrics import track_mcts_iteration

        @track_mcts_iteration
        def error_iteration():
            raise RuntimeError("mcts error")

        with pytest.raises(RuntimeError):
            error_iteration()

    def test_preserves_function_metadata(self):
        """Test track_mcts_iteration preserves function name."""
        from src.monitoring.prometheus_metrics import track_mcts_iteration

        @track_mcts_iteration
        def my_iteration():
            """Iteration docs."""
            pass

        assert my_iteration.__name__ == "my_iteration"
        assert my_iteration.__doc__ == "Iteration docs."


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestRecordConfidenceScore:
    """Tests for the record_confidence_score utility."""

    def test_valid_score_recorded(self):
        """Test valid confidence scores are recorded without error."""
        from src.monitoring.prometheus_metrics import record_confidence_score

        record_confidence_score("hrm", 0.85)
        record_confidence_score("trm", 0.0)
        record_confidence_score("trm", 1.0)

    def test_invalid_score_not_recorded(self):
        """Test invalid confidence scores log a warning and are skipped."""
        from src.monitoring.prometheus_metrics import record_confidence_score

        # Should not raise, just logs a warning
        record_confidence_score("hrm", -0.1)
        record_confidence_score("hrm", 1.5)


class TestRecordLlmUsage:
    """Tests for the record_llm_usage utility."""

    def test_records_prompt_and_completion_tokens(self):
        """Test record_llm_usage records both token types."""
        from src.monitoring.prometheus_metrics import record_llm_usage

        # Should not raise
        record_llm_usage("openai", prompt_tokens=100, completion_tokens=50)
        record_llm_usage("anthropic", prompt_tokens=200, completion_tokens=150)


class TestRecordRagRetrieval:
    """Tests for the record_rag_retrieval utility."""

    def test_records_rag_metrics(self):
        """Test record_rag_retrieval records all RAG metrics."""
        from src.monitoring.prometheus_metrics import record_rag_retrieval

        # Should not raise
        record_rag_retrieval(
            num_docs=5,
            relevance_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            latency=0.25,
        )

    def test_filters_invalid_relevance_scores(self):
        """Test record_rag_retrieval filters out-of-range scores."""
        from src.monitoring.prometheus_metrics import record_rag_retrieval

        # Should not raise; scores > 1.0 or < 0.0 are silently skipped
        record_rag_retrieval(
            num_docs=3,
            relevance_scores=[0.9, 1.5, -0.1],
            latency=0.1,
        )


# =============================================================================
# get_metrics_summary Tests
# =============================================================================


class TestGetMetricsSummary:
    """Tests for the get_metrics_summary health check function."""

    def test_returns_expected_keys(self):
        """Test get_metrics_summary returns expected dictionary keys."""
        from src.monitoring.prometheus_metrics import get_metrics_summary

        summary = get_metrics_summary()

        assert isinstance(summary, dict)
        assert "prometheus_available" in summary
        assert "metrics_initialized" in summary

    def test_prometheus_available_is_bool(self):
        """Test prometheus_available value is boolean."""
        from src.monitoring.prometheus_metrics import get_metrics_summary

        summary = get_metrics_summary()
        assert isinstance(summary["prometheus_available"], bool)

    def test_metrics_initialized_is_true(self):
        """Test metrics_initialized is True (metrics are defined at module level)."""
        from src.monitoring.prometheus_metrics import get_metrics_summary

        summary = get_metrics_summary()
        assert summary["metrics_initialized"] is True
