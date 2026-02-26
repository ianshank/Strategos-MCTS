"""
Unit tests for FlexibleLogger wrapper.

Tests graceful fallback between structured and standard logging.
"""

import logging
from unittest.mock import MagicMock

import pytest

pytest.importorskip("psutil", reason="psutil required for observability tests")

from src.api.framework_service import FlexibleLogger


class TestFlexibleLogger:
    """Tests for FlexibleLogger wrapper class."""

    @pytest.fixture
    def mock_standard_logger(self):
        """Create a mock standard Python logger."""
        logger = MagicMock(spec=logging.Logger)

        # Standard loggers raise TypeError for keyword arguments
        def log_side_effect(level, msg, **kwargs):
            if kwargs:
                raise TypeError("Logger._log() got an unexpected keyword argument")

        logger.log.side_effect = log_side_effect
        return logger

    @pytest.fixture
    def mock_structured_logger(self):
        """Create a mock structured logger that accepts kwargs."""
        logger = MagicMock(spec=logging.Logger)
        # Structured loggers accept keyword arguments
        logger.log.return_value = None
        return logger

    def test_initialization(self):
        """Test FlexibleLogger initialization."""
        base_logger = logging.getLogger("test")
        flexible = FlexibleLogger(base_logger)
        assert flexible._logger is base_logger

    def test_info_with_structured_logger(self, mock_structured_logger):
        """Test info logging with structured logger."""
        flexible = FlexibleLogger(mock_structured_logger)

        flexible.info(
            "Test message",
            correlation_id="123",
            extra_field="value",
        )

        mock_structured_logger.log.assert_called_once_with(
            logging.INFO,
            "Test message",
            correlation_id="123",
            extra_field="value",
        )

    def test_info_falls_back_with_standard_logger(self, mock_standard_logger):
        """Test info logging falls back with standard logger."""
        flexible = FlexibleLogger(mock_standard_logger)

        flexible.info(
            "Structured message",
            fallback_msg="Simple message",
            correlation_id="123",
        )

        # First call with structured args fails
        # Second call should use fallback message
        assert mock_standard_logger.log.call_count == 2
        # The second call should be the fallback
        last_call = mock_standard_logger.log.call_args_list[-1]
        assert last_call[0] == (logging.INFO, "Simple message")

    def test_info_uses_main_message_as_fallback_when_no_fallback(self, mock_standard_logger):
        """Test that main message is used when no fallback provided."""
        flexible = FlexibleLogger(mock_standard_logger)

        flexible.info(
            "Main message",
            correlation_id="123",
        )

        # Second call should use main message as fallback
        last_call = mock_standard_logger.log.call_args_list[-1]
        assert last_call[0] == (logging.INFO, "Main message")

    def test_debug_level(self, mock_structured_logger):
        """Test debug level logging."""
        flexible = FlexibleLogger(mock_structured_logger)

        flexible.debug("Debug message", key="value")

        mock_structured_logger.log.assert_called_with(
            logging.DEBUG,
            "Debug message",
            key="value",
        )

    def test_warning_level(self, mock_structured_logger):
        """Test warning level logging."""
        flexible = FlexibleLogger(mock_structured_logger)

        flexible.warning("Warning message", error="something")

        mock_structured_logger.log.assert_called_with(
            logging.WARNING,
            "Warning message",
            error="something",
        )

    def test_error_level(self, mock_structured_logger):
        """Test error level logging."""
        flexible = FlexibleLogger(mock_structured_logger)

        flexible.error("Error message", exception_type="ValueError")

        mock_structured_logger.log.assert_called_with(
            logging.ERROR,
            "Error message",
            exception_type="ValueError",
        )

    def test_fallback_message_format(self, mock_standard_logger):
        """Test that fallback messages can include dynamic content."""
        flexible = FlexibleLogger(mock_standard_logger)
        query = "What is machine learning?"

        flexible.info(
            "Processing query",
            fallback_msg=f"Processing query: {query[:20]}...",
            query_length=len(query),
            correlation_id="abc123",
        )

        last_call = mock_standard_logger.log.call_args_list[-1]
        assert last_call[0] == (logging.INFO, "Processing query: What is machine lear...")

    def test_no_kwargs_still_works(self, mock_standard_logger):
        """Test logging without kwargs works with both logger types."""
        flexible = FlexibleLogger(mock_standard_logger)

        # Reset side effect for this test - no kwargs should work
        mock_standard_logger.log.side_effect = None

        flexible.info("Simple message")

        mock_standard_logger.log.assert_called_once_with(
            logging.INFO,
            "Simple message",
        )


class TestFlexibleLoggerWithRealLogger:
    """Tests with real Python logger."""

    def test_with_real_standard_logger(self, caplog):
        """Test FlexibleLogger works with real standard logger."""
        real_logger = logging.getLogger("test_real")
        flexible = FlexibleLogger(real_logger)

        with caplog.at_level(logging.INFO):
            flexible.info(
                "Structured attempt",
                fallback_msg="Fallback message for real logger",
                some_key="some_value",
            )

        # Should have logged something (either structured or fallback)
        assert len(caplog.records) >= 1


class TestFlexibleLoggerIntegration:
    """Integration tests for FlexibleLogger in context."""

    def test_used_in_lightweight_framework(self):
        """Test FlexibleLogger is used correctly in LightweightFramework."""
        from unittest.mock import MagicMock

        from src.api.framework_service import FrameworkConfig, LightweightFramework

        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(text="Test response")

        config = FrameworkConfig(
            mcts_enabled=False,
            mcts_iterations=10,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=3,
            consensus_threshold=0.75,
            top_k_retrieval=5,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )

        # Should not raise even with standard logger
        framework = LightweightFramework(
            llm_client=mock_llm,
            config=config,
            logger=logging.getLogger("test"),
        )

        # Logger should be wrapped
        assert isinstance(framework._logger, FlexibleLogger)
