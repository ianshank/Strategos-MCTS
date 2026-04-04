"""
Unit tests for chess engines module (StockfishAdapter).

Tests cover StockfishConfig, StockfishAnalysis, EvaluationResult,
StockfishAdapter initialization, availability checking, and factory function.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.games.chess.engines.stockfish_adapter import (
    EvaluationResult,
    StockfishAdapter,
    StockfishAnalysis,
    StockfishConfig,
    create_stockfish_adapter,
)


@pytest.mark.unit
class TestStockfishConfig:
    """Test StockfishConfig dataclass."""

    def test_defaults(self) -> None:
        config = StockfishConfig()
        assert config.stockfish_path is None
        assert config.elo_limit is None
        assert config.hash_size_mb == 128
        assert config.threads == 1
        assert config.default_depth == 20
        assert config.comparison_depth == 15
        assert config.time_limit_ms == 1000
        assert config.multipv == 1
        assert config.skill_level is None

    def test_custom_values(self) -> None:
        config = StockfishConfig(
            stockfish_path="/usr/bin/stockfish",
            elo_limit=2000,
            hash_size_mb=256,
            threads=4,
            default_depth=25,
        )
        assert config.stockfish_path == "/usr/bin/stockfish"
        assert config.elo_limit == 2000
        assert config.hash_size_mb == 256
        assert config.threads == 4
        assert config.default_depth == 25

    def test_env_override_stockfish_path(self) -> None:
        with patch.dict("os.environ", {"STOCKFISH_PATH": "/custom/stockfish"}):
            config = StockfishConfig()
            assert config.stockfish_path == "/custom/stockfish"

    def test_explicit_path_not_overridden_by_env(self) -> None:
        with patch.dict("os.environ", {"STOCKFISH_PATH": "/custom/stockfish"}):
            config = StockfishConfig(stockfish_path="/explicit/path")
            assert config.stockfish_path == "/explicit/path"


@pytest.mark.unit
class TestStockfishAnalysis:
    """Test StockfishAnalysis dataclass."""

    def test_basic_creation(self) -> None:
        analysis = StockfishAnalysis(
            best_move="e2e4",
            evaluation_cp=50,
            evaluation_mate=None,
            depth=20,
            nodes=100000,
            time_ms=500.0,
            pv=["e2e4", "e7e5"],
        )
        assert analysis.best_move == "e2e4"
        assert analysis.evaluation_cp == 50
        assert analysis.evaluation_mate is None
        assert analysis.depth == 20

    def test_evaluation_score_positive_mate(self) -> None:
        analysis = StockfishAnalysis(
            best_move="d1h5",
            evaluation_cp=10000,
            evaluation_mate=3,
            depth=20,
            nodes=50000,
            time_ms=100.0,
            pv=["d1h5"],
        )
        assert analysis.evaluation_score == 1.0

    def test_evaluation_score_negative_mate(self) -> None:
        analysis = StockfishAnalysis(
            best_move=None,
            evaluation_cp=-10000,
            evaluation_mate=-2,
            depth=20,
            nodes=50000,
            time_ms=100.0,
            pv=[],
        )
        assert analysis.evaluation_score == -1.0

    def test_evaluation_score_centipawns_positive(self) -> None:
        analysis = StockfishAnalysis(
            best_move="e2e4",
            evaluation_cp=200,
            evaluation_mate=None,
            depth=20,
            nodes=100000,
            time_ms=500.0,
            pv=["e2e4"],
        )
        score = analysis.evaluation_score
        # Sigmoid-like: should be positive for positive cp
        assert 0.0 < score < 1.0
        # Check the formula: 2 / (1 + exp(-0.4 * 2.0)) - 1
        expected = 2.0 / (1.0 + math.exp(-0.4 * 2.0)) - 1.0
        assert abs(score - expected) < 0.001

    def test_evaluation_score_centipawns_zero(self) -> None:
        analysis = StockfishAnalysis(
            best_move="e2e4",
            evaluation_cp=0,
            evaluation_mate=None,
            depth=20,
            nodes=100000,
            time_ms=500.0,
            pv=["e2e4"],
        )
        assert analysis.evaluation_score == 0.0

    def test_to_dict(self) -> None:
        analysis = StockfishAnalysis(
            best_move="e2e4",
            evaluation_cp=50,
            evaluation_mate=None,
            depth=20,
            nodes=100000,
            time_ms=500.0,
            pv=["e2e4", "e7e5"],
        )
        d = analysis.to_dict()
        assert d["best_move"] == "e2e4"
        assert d["evaluation_cp"] == 50
        assert d["evaluation_mate"] is None
        assert d["depth"] == 20
        assert d["nodes"] == 100000
        assert d["time_ms"] == 500.0
        assert d["pv"] == ["e2e4", "e7e5"]
        assert "evaluation_score" in d

    def test_extra_info_default(self) -> None:
        analysis = StockfishAnalysis(
            best_move="e2e4",
            evaluation_cp=0,
            evaluation_mate=None,
            depth=10,
            nodes=1000,
            time_ms=50.0,
            pv=[],
        )
        assert analysis.extra_info == {}


@pytest.mark.unit
class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_agent_win_rate(self) -> None:
        result = EvaluationResult(
            total_games=10,
            agent_wins=3,
            stockfish_wins=5,
            draws=2,
            avg_move_diff_cp=50.0,
            agreement_rate=0.4,
        )
        assert result.agent_win_rate == 0.3

    def test_agent_win_rate_zero_games(self) -> None:
        result = EvaluationResult(
            total_games=0,
            agent_wins=0,
            stockfish_wins=0,
            draws=0,
            avg_move_diff_cp=0.0,
            agreement_rate=0.0,
        )
        assert result.agent_win_rate == 0.0

    def test_games_list_default(self) -> None:
        result = EvaluationResult(
            total_games=5,
            agent_wins=1,
            stockfish_wins=3,
            draws=1,
            avg_move_diff_cp=30.0,
            agreement_rate=0.5,
        )
        assert result.games == []


@pytest.mark.unit
class TestStockfishAdapter:
    """Test StockfishAdapter initialization and properties."""

    def test_default_init(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            assert adapter.config is not None
            assert adapter.config.stockfish_path is None

    def test_custom_config(self) -> None:
        config = StockfishConfig(stockfish_path="/usr/bin/stockfish", threads=4)
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter(config=config)
            assert adapter.config.stockfish_path == "/usr/bin/stockfish"
            assert adapter.config.threads == 4

    def test_is_available_false_when_not_found(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            with patch.object(adapter, "_find_stockfish", return_value=None):
                assert adapter.is_available is False

    def test_is_available_true_when_found(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            with patch.object(adapter, "_find_stockfish", return_value="/usr/bin/stockfish"):
                assert adapter.is_available is True

    def test_is_available_cached(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            adapter._available = True
            # Should return cached value without calling _find_stockfish
            assert adapter.is_available is True

    def test_find_stockfish_configured_path(self) -> None:
        config = StockfishConfig(stockfish_path="/usr/bin/stockfish")
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter(config=config)
            with patch("os.path.isfile", return_value=True):
                result = adapter._find_stockfish()
                assert result == "/usr/bin/stockfish"

    def test_find_stockfish_configured_path_not_found(self) -> None:
        config = StockfishConfig(stockfish_path="/nonexistent/stockfish")
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter(config=config)
            with (
                patch("os.path.isfile", return_value=False),
                patch("shutil.which", return_value=None),
                patch("src.config.settings.get_settings", side_effect=RuntimeError("no settings")),
            ):
                result = adapter._find_stockfish()
                assert result is None

    def test_find_stockfish_via_which(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            with (
                patch("shutil.which", side_effect=lambda name: "/usr/bin/stockfish" if name == "stockfish" else None),
                patch("src.config.settings.get_settings", side_effect=RuntimeError("no settings")),
            ):
                result = adapter._find_stockfish()
                assert result == "/usr/bin/stockfish"

    @pytest.mark.asyncio
    async def test_ensure_engine_returns_true_if_already_initialized(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            adapter._engine = MagicMock()
            assert await adapter._ensure_engine() is True

    @pytest.mark.asyncio
    async def test_ensure_engine_returns_false_when_not_found(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            with patch.object(adapter, "_find_stockfish", return_value=None):
                assert await adapter._ensure_engine() is False

    @pytest.mark.asyncio
    async def test_close_with_engine(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            mock_engine = AsyncMock()
            adapter._engine = mock_engine
            await adapter.close()
            mock_engine.quit.assert_awaited_once()
            assert adapter._engine is None

    @pytest.mark.asyncio
    async def test_close_without_engine(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            await adapter.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = StockfishAdapter()
            with patch.object(adapter, "_ensure_engine", new_callable=AsyncMock, return_value=True):
                with patch.object(adapter, "close", new_callable=AsyncMock):
                    async with adapter as ctx:
                        assert ctx is adapter
                    adapter.close.assert_awaited_once()


@pytest.mark.unit
class TestCreateStockfishAdapter:
    """Test factory function."""

    def test_create_with_defaults(self) -> None:
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = create_stockfish_adapter()
            assert isinstance(adapter, StockfishAdapter)

    def test_create_with_config(self) -> None:
        config = StockfishConfig(threads=8)
        with patch("src.games.chess.engines.stockfish_adapter.get_structured_logger"):
            adapter = create_stockfish_adapter(config=config)
            assert adapter.config.threads == 8
