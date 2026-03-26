"""
Unit tests for chess observability modules.

Tests cover:
- ChessVerificationLogger (logger.py)
- ChessMetricsCollector, ChessVerificationMetrics, PhaseRoutingStats (metrics.py)
- Decorators: traced_move_selection, verified_game_play,
  with_verification_context, timed_verification (decorators.py)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.games.chess.observability.metrics import (
    ChessMetricsCollector,
    ChessVerificationMetrics,
    PhaseRoutingStats,
)


# ---------------------------------------------------------------------------
# ChessVerificationMetrics tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChessVerificationMetrics:
    """Test ChessVerificationMetrics dataclass."""

    def test_defaults(self) -> None:
        m = ChessVerificationMetrics()
        assert m.games_verified == 0
        assert m.moves_validated == 0
        assert m.positions_verified == 0
        assert m.invalid_moves_detected == 0
        assert m.encoding_errors == 0
        assert m.verification_errors == 0
        assert m.agent_agreement_sum == 0.0
        assert m.agent_agreement_count == 0
        assert m.total_verification_time_ms == 0.0
        assert m.total_move_validation_time_ms == 0.0
        assert m.edge_cases_tested == {}

    def test_avg_verification_time_zero(self) -> None:
        m = ChessVerificationMetrics()
        assert m.avg_verification_time_ms == 0.0

    def test_avg_verification_time(self) -> None:
        m = ChessVerificationMetrics(games_verified=4, total_verification_time_ms=200.0)
        assert m.avg_verification_time_ms == 50.0

    def test_avg_move_validation_time_zero(self) -> None:
        m = ChessVerificationMetrics()
        assert m.avg_move_validation_time_ms == 0.0

    def test_avg_move_validation_time(self) -> None:
        m = ChessVerificationMetrics(moves_validated=10, total_move_validation_time_ms=100.0)
        assert m.avg_move_validation_time_ms == 10.0

    def test_agent_agreement_rate_zero(self) -> None:
        m = ChessVerificationMetrics()
        assert m.agent_agreement_rate == 0.0

    def test_agent_agreement_rate(self) -> None:
        m = ChessVerificationMetrics(agent_agreement_sum=2.4, agent_agreement_count=3)
        assert abs(m.agent_agreement_rate - 0.8) < 0.001

    def test_error_rate_zero(self) -> None:
        m = ChessVerificationMetrics()
        assert m.error_rate == 0.0

    def test_error_rate(self) -> None:
        m = ChessVerificationMetrics(
            games_verified=5,
            moves_validated=15,
            invalid_moves_detected=2,
            encoding_errors=1,
            verification_errors=1,
        )
        # total = 20, errors = 4
        assert abs(m.error_rate - 0.2) < 0.001

    def test_to_dict(self) -> None:
        m = ChessVerificationMetrics(games_verified=1, moves_validated=10)
        d = m.to_dict()
        assert d["games_verified"] == 1
        assert d["moves_validated"] == 10
        assert "avg_verification_time_ms" in d
        assert "error_rate" in d
        assert "edge_cases_tested" in d


# ---------------------------------------------------------------------------
# PhaseRoutingStats tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPhaseRoutingStats:
    """Test PhaseRoutingStats dataclass."""

    def test_defaults(self) -> None:
        stats = PhaseRoutingStats()
        assert stats.total_routings == 0
        assert stats.agent_selections == {}
        assert stats.avg_confidence == 0.0

    def test_record_routing(self) -> None:
        stats = PhaseRoutingStats()
        stats.record_routing("hrm", 0.8)
        assert stats.total_routings == 1
        assert stats.agent_selections["hrm"] == 1
        assert abs(stats.avg_confidence - 0.8) < 0.001

    def test_record_multiple_routings(self) -> None:
        stats = PhaseRoutingStats()
        stats.record_routing("hrm", 0.8)
        stats.record_routing("mcts", 0.6)
        stats.record_routing("hrm", 0.9)
        assert stats.total_routings == 3
        assert stats.agent_selections["hrm"] == 2
        assert stats.agent_selections["mcts"] == 1
        expected_avg = (0.8 + 0.6 + 0.9) / 3
        assert abs(stats.avg_confidence - expected_avg) < 0.001

    def test_get_most_selected_agent_empty(self) -> None:
        stats = PhaseRoutingStats()
        assert stats.get_most_selected_agent() is None

    def test_get_most_selected_agent(self) -> None:
        stats = PhaseRoutingStats()
        stats.record_routing("hrm", 0.8)
        stats.record_routing("mcts", 0.7)
        stats.record_routing("hrm", 0.9)
        assert stats.get_most_selected_agent() == "hrm"


# ---------------------------------------------------------------------------
# ChessMetricsCollector tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChessMetricsCollector:
    """Test ChessMetricsCollector singleton and methods."""

    def setup_method(self) -> None:
        ChessMetricsCollector.reset_instance()

    def teardown_method(self) -> None:
        ChessMetricsCollector.reset_instance()

    def test_singleton(self) -> None:
        a = ChessMetricsCollector.get_instance()
        b = ChessMetricsCollector.get_instance()
        assert a is b

    def test_reset_instance(self) -> None:
        a = ChessMetricsCollector.get_instance()
        ChessMetricsCollector.reset_instance()
        b = ChessMetricsCollector.get_instance()
        assert a is not b

    def test_record_game_verification_success(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_game_verification("game1", True, 100.0, 40)
        m = collector.verification_metrics
        assert m.games_verified == 1
        assert m.total_verification_time_ms == 100.0
        assert m.verification_errors == 0

    def test_record_game_verification_failure(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_game_verification("game1", False, 50.0, 20)
        m = collector.verification_metrics
        assert m.games_verified == 1
        assert m.verification_errors == 1

    def test_record_move_validation(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_move_validation("e2e4", True, 5.0)
        m = collector.verification_metrics
        assert m.moves_validated == 1
        assert m.invalid_moves_detected == 0
        assert m.encoding_errors == 0

    def test_record_move_validation_invalid(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_move_validation("zzzz", False, 3.0, has_encoding_error=True)
        m = collector.verification_metrics
        assert m.moves_validated == 1
        assert m.invalid_moves_detected == 1
        assert m.encoding_errors == 1

    def test_record_position_verification(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_position_verification("some_fen", True)
        m = collector.verification_metrics
        assert m.positions_verified == 1

    def test_record_position_verification_invalid(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_position_verification("bad_fen", False)
        m = collector.verification_metrics
        assert m.positions_verified == 1
        assert m.verification_errors == 1

    def test_record_agent_routing(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_agent_routing("opening", "hrm", 0.9)
        stats = collector.get_phase_routing_stats("opening")
        assert stats is not None
        assert stats.total_routings == 1

    def test_record_agent_agreement(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_agent_agreement(0.8)
        collector.record_agent_agreement(0.6)
        m = collector.verification_metrics
        assert m.agent_agreement_count == 2
        assert abs(m.agent_agreement_rate - 0.7) < 0.001

    def test_record_edge_case(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_edge_case("castling")
        collector.record_edge_case("castling")
        collector.record_edge_case("en_passant")
        m = collector.verification_metrics
        assert m.edge_cases_tested["castling"] == 2
        assert m.edge_cases_tested["en_passant"] == 1

    def test_record_verification_error(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_verification_error("execution_error")
        m = collector.verification_metrics
        assert m.edge_cases_tested["error_execution_error"] == 1

    def test_get_verification_report(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_game_verification("g1", True, 100.0, 20)
        collector.record_agent_routing("opening", "hrm", 0.9)
        report = collector.get_verification_report()
        assert "verification_metrics" in report
        assert "phase_routing" in report
        assert "opening" in report["phase_routing"]

    def test_get_phase_routing_stats_none(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        assert collector.get_phase_routing_stats("nonexistent") is None

    def test_reset(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_game_verification("g1", True, 100.0, 20)
        collector.record_agent_routing("opening", "hrm", 0.9)
        collector.reset()
        assert collector.verification_metrics.games_verified == 0
        assert collector.get_phase_routing_stats("opening") is None

    def test_summary(self) -> None:
        collector = ChessMetricsCollector.get_instance()
        collector.record_game_verification("g1", True, 100.0, 20)
        summary = collector.summary()
        assert "Chess Verification Metrics" in summary
        assert "Games verified: 1" in summary


# ---------------------------------------------------------------------------
# ChessVerificationLogger tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChessVerificationLogger:
    """Test ChessVerificationLogger."""

    def test_init_default_name(self) -> None:
        from src.games.chess.observability.logger import ChessVerificationLogger

        logger = ChessVerificationLogger()
        assert logger._logger.name == "chess.verification"

    def test_init_custom_name(self) -> None:
        from src.games.chess.observability.logger import ChessVerificationLogger

        logger = ChessVerificationLogger("chess.test")
        assert logger._logger.name == "chess.test"

    def test_log_game_verification(self) -> None:
        chess = pytest.importorskip("chess", reason="python-chess required")
        from src.games.chess.observability.logger import ChessVerificationLogger
        from src.games.chess.verification.types import GameResult, GameVerificationResult

        logger = ChessVerificationLogger("chess.test")
        result = GameVerificationResult(
            is_valid=True,
            game_id="test_game",
            moves=["e2e4", "e7e5"],
            result=GameResult.IN_PROGRESS,
            total_moves=2,
        )
        # Should not raise
        logger.log_game_verification("test_game", result, 100.0)

    def test_log_move_validation(self) -> None:
        pytest.importorskip("chess", reason="python-chess required")
        from src.games.chess.observability.logger import ChessVerificationLogger
        from src.games.chess.verification.types import MoveType, MoveValidationResult

        logger = ChessVerificationLogger("chess.test")
        result = MoveValidationResult(
            is_valid=True,
            move_uci="e2e4",
            move_type=MoveType.NORMAL,
            encoded_index=42,
        )
        logger.log_move_validation("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", result)

    def test_log_verification_error(self) -> None:
        from src.games.chess.observability.logger import ChessVerificationLogger

        logger = ChessVerificationLogger("chess.test")
        error = ValueError("test error")
        logger.log_verification_error("test_op", error, {"key": "value"})

    def test_log_game_phase_transition(self) -> None:
        from src.games.chess.observability.logger import ChessVerificationLogger

        logger = ChessVerificationLogger("chess.test")
        logger.log_game_phase_transition("game1", "opening", "middlegame", 15)

    def test_log_encoding_roundtrip_success(self) -> None:
        from src.games.chess.observability.logger import ChessVerificationLogger

        logger = ChessVerificationLogger("chess.test")
        logger.log_encoding_roundtrip("e2e4", 42, "e2e4", True)

    def test_log_encoding_roundtrip_failure(self) -> None:
        from src.games.chess.observability.logger import ChessVerificationLogger

        logger = ChessVerificationLogger("chess.test")
        logger.log_encoding_roundtrip("e2e4", 42, "e2e5", False)

    def test_log_agent_divergence(self) -> None:
        from src.games.chess.observability.logger import ChessVerificationLogger

        logger = ChessVerificationLogger("chess.test")
        logger.log_agent_divergence(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            {"max_divergence": 0.5, "agents": ["hrm", "trm"]},
        )


@pytest.mark.unit
class TestGetChessLogger:
    """Test get_chess_logger factory function."""

    def test_prefix_added(self) -> None:
        from src.games.chess.observability.logger import get_chess_logger

        logger = get_chess_logger("verification")
        assert logger._logger.name == "chess.verification"

    def test_already_prefixed(self) -> None:
        from src.games.chess.observability.logger import get_chess_logger

        logger = get_chess_logger("chess.verification")
        assert logger._logger.name == "chess.verification"

    def test_default_name(self) -> None:
        from src.games.chess.observability.logger import get_chess_logger

        logger = get_chess_logger()
        assert logger._logger.name == "chess.chess"  # "chess" doesn't start with "chess."


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTracedMoveSelectionDecorator:
    """Test traced_move_selection decorator."""

    @pytest.mark.asyncio
    async def test_async_function_decorated(self) -> None:
        from src.games.chess.observability.decorators import traced_move_selection

        @traced_move_selection
        async def select_move(state: Any) -> str:
            return "e2e4"

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            result = await select_move("dummy_state")
            assert result == "e2e4"

    @pytest.mark.asyncio
    async def test_async_function_with_state_object(self) -> None:
        from src.games.chess.observability.decorators import traced_move_selection

        mock_state = MagicMock()
        mock_state.fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        mock_state.get_game_phase.return_value = MagicMock(value="opening")
        mock_state.get_legal_actions.return_value = ["e7e5", "d7d5"]

        @traced_move_selection
        async def select_move(state: Any) -> str:
            return "e7e5"

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            result = await select_move(mock_state)
            assert result == "e7e5"

    @pytest.mark.asyncio
    async def test_async_function_raises(self) -> None:
        from src.games.chess.observability.decorators import traced_move_selection

        @traced_move_selection
        async def failing_move(state: Any) -> str:
            raise ValueError("test error")

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            with pytest.raises(ValueError, match="test error"):
                await failing_move("state")

    def test_sync_function_decorated(self) -> None:
        from src.games.chess.observability.decorators import traced_move_selection

        @traced_move_selection
        def select_move_sync() -> str:
            return "d2d4"

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            result = select_move_sync()
            assert result == "d2d4"

    def test_sync_function_raises(self) -> None:
        from src.games.chess.observability.decorators import traced_move_selection

        @traced_move_selection
        def failing_sync() -> str:
            raise RuntimeError("sync error")

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            with pytest.raises(RuntimeError, match="sync error"):
                failing_sync()


@pytest.mark.unit
class TestTimedVerificationDecorator:
    """Test timed_verification decorator."""

    @pytest.mark.asyncio
    async def test_async_timed(self) -> None:
        from src.games.chess.observability.decorators import timed_verification

        @timed_verification("test_op")
        async def some_op() -> int:
            return 42

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            result = await some_op()
            assert result == 42

    def test_sync_timed(self) -> None:
        from src.games.chess.observability.decorators import timed_verification

        @timed_verification("test_sync_op")
        def some_sync_op() -> int:
            return 99

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            result = some_sync_op()
            assert result == 99

    def test_default_operation_name(self) -> None:
        from src.games.chess.observability.decorators import timed_verification

        @timed_verification()
        def my_func() -> str:
            return "ok"

        assert my_func.__name__ == "my_func"
        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            assert my_func() == "ok"


@pytest.mark.unit
class TestWithVerificationContextDecorator:
    """Test with_verification_context decorator."""

    @pytest.mark.asyncio
    async def test_async_with_metrics(self) -> None:
        from src.games.chess.observability.decorators import with_verification_context

        ChessMetricsCollector.reset_instance()

        @with_verification_context(collect_metrics=True)
        async def verify_something() -> MagicMock:
            result = MagicMock()
            result.game_id = "g1"
            result.is_valid = True
            result.total_moves = 10
            return result

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            result = await verify_something()
            assert result.is_valid is True

        ChessMetricsCollector.reset_instance()

    @pytest.mark.asyncio
    async def test_async_with_exception(self) -> None:
        from src.games.chess.observability.decorators import with_verification_context

        ChessMetricsCollector.reset_instance()

        @with_verification_context(collect_metrics=True)
        async def failing_verify() -> None:
            raise ValueError("fail")

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            with pytest.raises(ValueError, match="fail"):
                await failing_verify()

        ChessMetricsCollector.reset_instance()

    def test_sync_context(self) -> None:
        from src.games.chess.observability.decorators import with_verification_context

        @with_verification_context(collect_metrics=False)
        def sync_verify() -> str:
            return "done"

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            assert sync_verify() == "done"

    def test_sync_context_exception(self) -> None:
        from src.games.chess.observability.decorators import with_verification_context

        @with_verification_context(collect_metrics=False)
        def sync_failing() -> None:
            raise RuntimeError("sync fail")

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            with pytest.raises(RuntimeError, match="sync fail"):
                sync_failing()


@pytest.mark.unit
class TestVerifiedGamePlayDecorator:
    """Test verified_game_play decorator."""

    def test_sync_game_play(self) -> None:
        from src.games.chess.observability.decorators import verified_game_play

        @verified_game_play(verification_level="minimal")
        def play_game() -> str:
            return "game_result"

        with patch("src.games.chess.observability.decorators.get_chess_logger"):
            result = play_game()
            assert result == "game_result"

    @pytest.mark.asyncio
    async def test_async_game_play_minimal(self) -> None:
        pytest.importorskip("chess", reason="python-chess required")
        from src.games.chess.observability.decorators import verified_game_play

        @verified_game_play(verification_level="minimal")
        async def play_game_async() -> MagicMock:
            result = MagicMock()
            result.moves = ["e2e4", "e7e5"]
            return result

        mock_settings = MagicMock()
        mock_settings.LOG_LEVEL = MagicMock(value="INFO")
        with patch("src.games.chess.observability.decorators.get_chess_logger"), \
             patch("src.games.chess.verification.factory.get_settings", return_value=mock_settings):
            result = await play_game_async()
            assert result.moves == ["e2e4", "e7e5"]
