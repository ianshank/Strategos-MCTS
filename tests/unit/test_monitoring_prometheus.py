"""
Unit tests for src/monitoring/prometheus_metrics.py

Tests Prometheus metrics definitions, dummy fallbacks, helper functions, and decorators.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestPrometheusAvailability:
    """Tests for Prometheus availability detection."""

    def test_prometheus_available_flag(self):
        """PROMETHEUS_AVAILABLE reflects whether prometheus_client is installed."""
        from src.monitoring.prometheus_metrics import PROMETHEUS_AVAILABLE

        # The flag should be a boolean regardless of install status
        assert isinstance(PROMETHEUS_AVAILABLE, bool)


@pytest.mark.unit
class TestAgentRequestCounter:
    """Tests for AGENT_REQUESTS_TOTAL counter."""

    def test_agent_request_counter_labels(self):
        """AGENT_REQUESTS_TOTAL should accept agent_type and status labels."""
        from src.monitoring.prometheus_metrics import AGENT_REQUESTS_TOTAL

        # Should not raise when using expected labels
        labeled = AGENT_REQUESTS_TOTAL.labels(agent_type="hrm", status="success")
        assert labeled is not None

    def test_agent_request_counter_increment(self):
        """AGENT_REQUESTS_TOTAL should allow incrementing with labels."""
        from src.monitoring.prometheus_metrics import AGENT_REQUESTS_TOTAL

        # Should not raise
        AGENT_REQUESTS_TOTAL.labels(agent_type="trm", status="error").inc()


@pytest.mark.unit
class TestAgentLatencyHistogram:
    """Tests for AGENT_REQUEST_LATENCY histogram."""

    def test_agent_latency_histogram_buckets(self):
        """AGENT_REQUEST_LATENCY should accept observations with agent_type label."""
        from src.monitoring.prometheus_metrics import AGENT_REQUEST_LATENCY

        # Whether real Histogram or DummyMetric, observe should work without error
        AGENT_REQUEST_LATENCY.labels(agent_type="hrm").observe(1.5)

    def test_agent_latency_histogram_labels(self):
        """AGENT_REQUEST_LATENCY should accept agent_type label."""
        from src.monitoring.prometheus_metrics import AGENT_REQUEST_LATENCY

        labeled = AGENT_REQUEST_LATENCY.labels(agent_type="meta_controller")
        assert labeled is not None


@pytest.mark.unit
class TestAgentConfidenceHistogram:
    """Tests for AGENT_CONFIDENCE_SCORES histogram."""

    def test_confidence_histogram_accepts_observation(self):
        """AGENT_CONFIDENCE_SCORES should accept observations with agent_type label."""
        from src.monitoring.prometheus_metrics import AGENT_CONFIDENCE_SCORES

        AGENT_CONFIDENCE_SCORES.labels(agent_type="hrm").observe(0.85)

    def test_confidence_histogram_buckets_cover_zero_to_one(self):
        """AGENT_CONFIDENCE_SCORES should accept observations across 0.0-1.0 range."""
        from src.monitoring.prometheus_metrics import AGENT_CONFIDENCE_SCORES

        # Observe at boundary values without error
        AGENT_CONFIDENCE_SCORES.labels(agent_type="test").observe(0.0)
        AGENT_CONFIDENCE_SCORES.labels(agent_type="test").observe(0.5)
        AGENT_CONFIDENCE_SCORES.labels(agent_type="test").observe(1.0)


@pytest.mark.unit
class TestMCTSMetrics:
    """Tests for MCTS-specific metrics."""

    def test_mcts_iterations_total_labels(self):
        """MCTS_ITERATIONS_TOTAL should accept outcome label."""
        from src.monitoring.prometheus_metrics import MCTS_ITERATIONS_TOTAL

        MCTS_ITERATIONS_TOTAL.labels(outcome="completed").inc()

    def test_mcts_iteration_latency_observe(self):
        """MCTS_ITERATION_LATENCY should accept observations without labels."""
        from src.monitoring.prometheus_metrics import MCTS_ITERATION_LATENCY

        MCTS_ITERATION_LATENCY.observe(0.05)

    def test_mcts_simulation_depth_observe(self):
        """MCTS_SIMULATION_DEPTH should accept depth observations."""
        from src.monitoring.prometheus_metrics import MCTS_SIMULATION_DEPTH

        MCTS_SIMULATION_DEPTH.observe(10)

    def test_mcts_node_count_gauge(self):
        """MCTS_NODE_COUNT should support set/inc/dec as a Gauge."""
        from src.monitoring.prometheus_metrics import MCTS_NODE_COUNT

        MCTS_NODE_COUNT.set(42)
        MCTS_NODE_COUNT.inc()
        MCTS_NODE_COUNT.dec()

    def test_mcts_best_action_confidence(self):
        """MCTS_BEST_ACTION_CONFIDENCE should accept observations."""
        from src.monitoring.prometheus_metrics import MCTS_BEST_ACTION_CONFIDENCE

        MCTS_BEST_ACTION_CONFIDENCE.observe(0.95)


@pytest.mark.unit
class TestSystemHealthMetrics:
    """Tests for system health metrics."""

    def test_active_operations_gauge_labels(self):
        """ACTIVE_OPERATIONS should accept operation_type label."""
        from src.monitoring.prometheus_metrics import ACTIVE_OPERATIONS

        ACTIVE_OPERATIONS.labels(operation_type="mcts_simulation").set(3)

    def test_llm_request_errors_labels(self):
        """LLM_REQUEST_ERRORS should accept provider and error_type labels."""
        from src.monitoring.prometheus_metrics import LLM_REQUEST_ERRORS

        LLM_REQUEST_ERRORS.labels(provider="openai", error_type="timeout").inc()

    def test_llm_request_latency_labels(self):
        """LLM_REQUEST_LATENCY should accept provider label."""
        from src.monitoring.prometheus_metrics import LLM_REQUEST_LATENCY

        LLM_REQUEST_LATENCY.labels(provider="anthropic").observe(2.5)

    def test_llm_token_usage_labels(self):
        """LLM_TOKEN_USAGE should accept provider and token_type labels."""
        from src.monitoring.prometheus_metrics import LLM_TOKEN_USAGE

        LLM_TOKEN_USAGE.labels(provider="openai", token_type="prompt").inc(100)


@pytest.mark.unit
class TestRAGMetrics:
    """Tests for RAG-specific metrics."""

    def test_rag_queries_total_labels(self):
        """RAG_QUERIES_TOTAL should accept status label."""
        from src.monitoring.prometheus_metrics import RAG_QUERIES_TOTAL

        RAG_QUERIES_TOTAL.labels(status="success").inc()

    def test_rag_retrieval_latency(self):
        """RAG_RETRIEVAL_LATENCY should accept observations."""
        from src.monitoring.prometheus_metrics import RAG_RETRIEVAL_LATENCY

        RAG_RETRIEVAL_LATENCY.observe(0.25)

    def test_rag_documents_retrieved(self):
        """RAG_DOCUMENTS_RETRIEVED should accept observations."""
        from src.monitoring.prometheus_metrics import RAG_DOCUMENTS_RETRIEVED

        RAG_DOCUMENTS_RETRIEVED.observe(5)

    def test_rag_relevance_scores(self):
        """RAG_RELEVANCE_SCORES should accept observations."""
        from src.monitoring.prometheus_metrics import RAG_RELEVANCE_SCORES

        RAG_RELEVANCE_SCORES.observe(0.75)


@pytest.mark.unit
class TestRequestMetrics:
    """Tests for API-level request metrics."""

    def test_request_count_labels(self):
        """REQUEST_COUNT should accept method, endpoint, and status labels."""
        from src.monitoring.prometheus_metrics import REQUEST_COUNT

        REQUEST_COUNT.labels(method="GET", endpoint="/api/query", status="200").inc()

    def test_request_latency_labels(self):
        """REQUEST_LATENCY should accept method and endpoint labels."""
        from src.monitoring.prometheus_metrics import REQUEST_LATENCY

        REQUEST_LATENCY.labels(method="POST", endpoint="/api/solve").observe(1.2)

    def test_active_requests_gauge(self):
        """ACTIVE_REQUESTS should support inc/dec as a Gauge."""
        from src.monitoring.prometheus_metrics import ACTIVE_REQUESTS

        ACTIVE_REQUESTS.inc()
        ACTIVE_REQUESTS.dec()

    def test_error_count_labels(self):
        """ERROR_COUNT should accept error_type label."""
        from src.monitoring.prometheus_metrics import ERROR_COUNT

        ERROR_COUNT.labels(error_type="ValueError").inc()


@pytest.mark.unit
class TestRateLimitMetrics:
    """Tests for rate limiting metrics."""

    def test_rate_limit_exceeded_labels(self):
        """RATE_LIMIT_EXCEEDED should accept client_id label."""
        from src.monitoring.prometheus_metrics import RATE_LIMIT_EXCEEDED

        RATE_LIMIT_EXCEEDED.labels(client_id="test-client").inc()

    def test_request_queue_depth_gauge(self):
        """REQUEST_QUEUE_DEPTH should support set as a Gauge."""
        from src.monitoring.prometheus_metrics import REQUEST_QUEUE_DEPTH

        REQUEST_QUEUE_DEPTH.set(10)


@pytest.mark.unit
class TestDummyMetric:
    """Tests for DummyMetric fallback when prometheus_client is not installed."""

    def test_dummy_metric_noop(self):
        """DummyMetric should be a no-op for all metric operations."""
        from src.monitoring.prometheus_metrics import DummyMetric

        dummy = DummyMetric("test_metric", "A test metric")

        # All operations should be no-ops and not raise
        dummy.inc()
        dummy.dec()
        dummy.set(42)
        dummy.observe(1.5)
        dummy.info({"key": "value"})

        # labels() should return self for chaining
        labeled = dummy.labels(agent_type="hrm", status="success")
        assert labeled is dummy

        # Chained operations should also work
        dummy.labels(agent_type="hrm").inc()
        dummy.labels(agent_type="hrm").observe(0.5)


@pytest.mark.unit
class TestSetupMetrics:
    """Tests for setup_metrics function."""

    def test_setup_metrics_calls_system_info_when_available(self):
        """setup_metrics should set system info when Prometheus is available."""
        import src.monitoring.prometheus_metrics as mod

        original = mod.PROMETHEUS_AVAILABLE
        try:
            mod.PROMETHEUS_AVAILABLE = True
            with patch.object(mod.SYSTEM_INFO, "info") as mock_info:
                mod.setup_metrics(app_version="2.0.0", environment="testing")
                mock_info.assert_called_once_with(
                    {
                        "version": "2.0.0",
                        "environment": "testing",
                        "framework": "langgraph-mcts",
                    }
                )
        finally:
            mod.PROMETHEUS_AVAILABLE = original

    def test_setup_metrics_warns_when_unavailable(self):
        """setup_metrics should log warning when Prometheus is not available."""
        import src.monitoring.prometheus_metrics as mod

        original = mod.PROMETHEUS_AVAILABLE
        try:
            mod.PROMETHEUS_AVAILABLE = False
            with patch.object(mod.logger, "warning") as mock_warn:
                mod.setup_metrics()
                mock_warn.assert_called_once()
        finally:
            mod.PROMETHEUS_AVAILABLE = original


@pytest.mark.unit
class TestTrackOperation:
    """Tests for track_operation context manager."""

    def test_track_operation_increments_and_decrements(self):
        """track_operation should inc on entry and dec on exit."""
        from src.monitoring.prometheus_metrics import ACTIVE_OPERATIONS, track_operation

        with patch.object(ACTIVE_OPERATIONS, "labels") as mock_labels:
            mock_gauge = MagicMock()
            mock_labels.return_value = mock_gauge

            with track_operation("test_op"):
                pass

            # Should have been called with operation_type for both inc and dec
            assert mock_labels.call_count == 2
            mock_gauge.inc.assert_called_once()
            mock_gauge.dec.assert_called_once()

    def test_track_operation_decrements_on_exception(self):
        """track_operation should still dec even if the body raises."""
        from src.monitoring.prometheus_metrics import ACTIVE_OPERATIONS, track_operation

        with patch.object(ACTIVE_OPERATIONS, "labels") as mock_labels:
            mock_gauge = MagicMock()
            mock_labels.return_value = mock_gauge

            with pytest.raises(RuntimeError):
                with track_operation("failing_op"):
                    raise RuntimeError("boom")

            mock_gauge.dec.assert_called_once()


@pytest.mark.unit
class TestMeasureLatency:
    """Tests for measure_latency context manager."""

    def test_measure_latency_observes_elapsed_time(self):
        """measure_latency should observe the elapsed time on the metric."""
        from src.monitoring.prometheus_metrics import measure_latency

        mock_metric = MagicMock()
        mock_labeled = MagicMock()
        mock_metric.labels.return_value = mock_labeled

        with measure_latency(mock_metric, agent_type="hrm"):
            pass

        mock_metric.labels.assert_called_once_with(agent_type="hrm")
        mock_labeled.observe.assert_called_once()
        observed_value = mock_labeled.observe.call_args[0][0]
        assert observed_value >= 0

    def test_measure_latency_without_labels(self):
        """measure_latency without labels should observe directly on metric."""
        from src.monitoring.prometheus_metrics import measure_latency

        mock_metric = MagicMock()

        with measure_latency(mock_metric):
            pass

        mock_metric.observe.assert_called_once()
        observed_value = mock_metric.observe.call_args[0][0]
        assert observed_value >= 0


@pytest.mark.unit
class TestTrackAgentRequestDecorator:
    """Tests for track_agent_request decorator."""

    def test_track_agent_request_sync_success(self):
        """track_agent_request should record metrics for successful sync calls."""
        from src.monitoring.prometheus_metrics import (
            AGENT_REQUEST_LATENCY,
            AGENT_REQUESTS_TOTAL,
            track_agent_request,
        )

        @track_agent_request("hrm")
        def my_func():
            return "result"

        with (
            patch.object(AGENT_REQUESTS_TOTAL, "labels", return_value=MagicMock()) as mock_counter,
            patch.object(AGENT_REQUEST_LATENCY, "labels", return_value=MagicMock()) as mock_hist,
        ):
            result = my_func()
            assert result == "result"
            mock_counter.assert_called_with(agent_type="hrm", status="success")
            mock_hist.assert_called_with(agent_type="hrm")

    def test_track_agent_request_sync_error(self):
        """track_agent_request should record error status on exception."""
        from src.monitoring.prometheus_metrics import AGENT_REQUESTS_TOTAL, ERROR_COUNT, track_agent_request

        @track_agent_request("trm")
        def failing_func():
            raise ValueError("test error")

        with (
            patch.object(AGENT_REQUESTS_TOTAL, "labels", return_value=MagicMock()) as mock_counter,
            patch.object(ERROR_COUNT, "labels", return_value=MagicMock()) as mock_error,
        ):
            with pytest.raises(ValueError, match="test error"):
                failing_func()

            mock_counter.assert_called_with(agent_type="trm", status="error")
            mock_error.assert_called_with(error_type="ValueError")

    def test_track_agent_request_async_success(self):
        """track_agent_request should work with async functions."""
        from src.monitoring.prometheus_metrics import (
            AGENT_REQUEST_LATENCY,
            AGENT_REQUESTS_TOTAL,
            track_agent_request,
        )

        @track_agent_request("hybrid")
        async def my_async_func():
            return "async_result"

        with (
            patch.object(AGENT_REQUESTS_TOTAL, "labels", return_value=MagicMock()) as mock_counter,
            patch.object(AGENT_REQUEST_LATENCY, "labels", return_value=MagicMock()),
        ):
            result = asyncio.run(my_async_func())
            assert result == "async_result"
            mock_counter.assert_called_with(agent_type="hybrid", status="success")


@pytest.mark.unit
class TestTrackMCTSIterationDecorator:
    """Tests for track_mcts_iteration decorator."""

    def test_track_mcts_iteration_completed(self):
        """track_mcts_iteration should record completed outcome on success."""
        from src.monitoring.prometheus_metrics import MCTS_ITERATIONS_TOTAL, track_mcts_iteration

        @track_mcts_iteration
        def run_iteration():
            return "done"

        with patch.object(MCTS_ITERATIONS_TOTAL, "labels", return_value=MagicMock()) as mock_labels:
            result = run_iteration()
            assert result == "done"
            mock_labels.assert_called_with(outcome="completed")

    def test_track_mcts_iteration_timeout(self):
        """track_mcts_iteration should record timeout outcome on TimeoutError."""
        from src.monitoring.prometheus_metrics import MCTS_ITERATIONS_TOTAL, track_mcts_iteration

        @track_mcts_iteration
        def timeout_iteration():
            raise TimeoutError("timed out")

        with patch.object(MCTS_ITERATIONS_TOTAL, "labels", return_value=MagicMock()) as mock_labels:
            with pytest.raises(TimeoutError):
                timeout_iteration()
            mock_labels.assert_called_with(outcome="timeout")

    def test_track_mcts_iteration_error(self):
        """track_mcts_iteration should record error outcome on general exception."""
        from src.monitoring.prometheus_metrics import MCTS_ITERATIONS_TOTAL, track_mcts_iteration

        @track_mcts_iteration
        def error_iteration():
            raise RuntimeError("something broke")

        with patch.object(MCTS_ITERATIONS_TOTAL, "labels", return_value=MagicMock()) as mock_labels:
            with pytest.raises(RuntimeError):
                error_iteration()
            mock_labels.assert_called_with(outcome="error")


@pytest.mark.unit
class TestRecordConfidenceScore:
    """Tests for record_confidence_score helper."""

    def test_record_valid_confidence_score(self):
        """record_confidence_score should observe valid scores."""
        from src.monitoring.prometheus_metrics import AGENT_CONFIDENCE_SCORES, record_confidence_score

        with patch.object(AGENT_CONFIDENCE_SCORES, "labels", return_value=MagicMock()) as mock_labels:
            record_confidence_score("hrm", 0.85)
            mock_labels.assert_called_once_with(agent_type="hrm")
            mock_labels.return_value.observe.assert_called_once_with(0.85)

    def test_record_boundary_confidence_scores(self):
        """record_confidence_score should accept 0.0 and 1.0 boundary values."""
        from src.monitoring.prometheus_metrics import AGENT_CONFIDENCE_SCORES, record_confidence_score

        with patch.object(AGENT_CONFIDENCE_SCORES, "labels", return_value=MagicMock()) as mock_labels:
            record_confidence_score("hrm", 0.0)
            mock_labels.return_value.observe.assert_called_with(0.0)

            record_confidence_score("hrm", 1.0)
            mock_labels.return_value.observe.assert_called_with(1.0)

    def test_record_invalid_confidence_score(self):
        """record_confidence_score should log warning for out-of-range scores."""
        from src.monitoring.prometheus_metrics import AGENT_CONFIDENCE_SCORES, record_confidence_score

        with patch.object(AGENT_CONFIDENCE_SCORES, "labels", return_value=MagicMock()) as mock_labels:
            record_confidence_score("hrm", 1.5)
            mock_labels.return_value.observe.assert_not_called()

            record_confidence_score("hrm", -0.1)
            mock_labels.return_value.observe.assert_not_called()


@pytest.mark.unit
class TestRecordLLMUsage:
    """Tests for record_llm_usage helper."""

    def test_record_llm_usage_tracks_both_token_types(self):
        """record_llm_usage should increment both prompt and completion token counters."""
        from src.monitoring.prometheus_metrics import LLM_TOKEN_USAGE, record_llm_usage

        with patch.object(LLM_TOKEN_USAGE, "labels", return_value=MagicMock()) as mock_labels:
            record_llm_usage("openai", prompt_tokens=100, completion_tokens=50)

            calls = mock_labels.call_args_list
            assert any(c.kwargs.get("token_type") == "prompt" for c in calls)
            assert any(c.kwargs.get("token_type") == "completion" for c in calls)


@pytest.mark.unit
class TestRecordRAGRetrieval:
    """Tests for record_rag_retrieval helper."""

    def test_record_rag_retrieval_metrics(self):
        """record_rag_retrieval should record all RAG-related metrics."""
        from src.monitoring.prometheus_metrics import (
            RAG_DOCUMENTS_RETRIEVED,
            RAG_QUERIES_TOTAL,
            RAG_RELEVANCE_SCORES,
            RAG_RETRIEVAL_LATENCY,
            record_rag_retrieval,
        )

        with (
            patch.object(RAG_QUERIES_TOTAL, "labels", return_value=MagicMock()) as mock_queries,
            patch.object(RAG_DOCUMENTS_RETRIEVED, "observe") as mock_docs,
            patch.object(RAG_RETRIEVAL_LATENCY, "observe") as mock_latency,
            patch.object(RAG_RELEVANCE_SCORES, "observe") as mock_rel,
        ):
            record_rag_retrieval(num_docs=3, relevance_scores=[0.9, 0.7, 0.5], latency=0.25)

            mock_queries.assert_called_once_with(status="success")
            mock_docs.assert_called_once_with(3)
            mock_latency.assert_called_once_with(0.25)
            assert mock_rel.call_count == 3

    def test_record_rag_retrieval_skips_invalid_scores(self):
        """record_rag_retrieval should skip relevance scores outside 0.0-1.0."""
        from src.monitoring.prometheus_metrics import RAG_RELEVANCE_SCORES, record_rag_retrieval

        with patch.object(RAG_RELEVANCE_SCORES, "observe") as mock_rel:
            record_rag_retrieval(num_docs=2, relevance_scores=[0.5, 1.5, -0.1], latency=0.1)

            # Only 0.5 is valid
            mock_rel.assert_called_once_with(0.5)


@pytest.mark.unit
class TestGetMetricsSummary:
    """Tests for get_metrics_summary utility."""

    def test_get_metrics_summary_returns_expected_keys(self):
        """get_metrics_summary should return dict with prometheus_available and metrics_initialized."""
        from src.monitoring.prometheus_metrics import get_metrics_summary

        summary = get_metrics_summary()
        assert "prometheus_available" in summary
        assert "metrics_initialized" in summary
        assert isinstance(summary["prometheus_available"], bool)
        assert summary["metrics_initialized"] is True
