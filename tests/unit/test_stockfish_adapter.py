"""Unit tests for src/games/chess/engines/stockfish_adapter.py."""

from __future__ import annotations

from unittest.mock import patch

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
    def test_defaults(self):
        config = StockfishConfig()
        assert config.hash_size_mb == 128
        assert config.threads == 1
        assert config.default_depth == 20
        assert config.comparison_depth == 15

    def test_custom(self):
        config = StockfishConfig(hash_size_mb=256, threads=4, default_depth=30)
        assert config.hash_size_mb == 256
        assert config.threads == 4

    def test_env_override(self):
        with patch.dict("os.environ", {"STOCKFISH_PATH": "/usr/bin/stockfish"}):
            config = StockfishConfig()
            assert config.stockfish_path == "/usr/bin/stockfish"

    def test_explicit_path_overrides_env(self):
        config = StockfishConfig(stockfish_path="/custom/path")
        assert config.stockfish_path == "/custom/path"


@pytest.mark.unit
class TestStockfishAnalysis:
    def test_creation(self):
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

    def test_evaluation_score_positive_cp(self):
        analysis = StockfishAnalysis(
            best_move="e2e4", evaluation_cp=100, evaluation_mate=None,
            depth=20, nodes=0, time_ms=0.0, pv=[],
        )
        score = analysis.evaluation_score
        assert 0 < score < 1.0

    def test_evaluation_score_negative_cp(self):
        analysis = StockfishAnalysis(
            best_move="e7e5", evaluation_cp=-200, evaluation_mate=None,
            depth=20, nodes=0, time_ms=0.0, pv=[],
        )
        score = analysis.evaluation_score
        assert -1.0 < score < 0

    def test_evaluation_score_zero_cp(self):
        analysis = StockfishAnalysis(
            best_move="e2e4", evaluation_cp=0, evaluation_mate=None,
            depth=20, nodes=0, time_ms=0.0, pv=[],
        )
        assert analysis.evaluation_score == pytest.approx(0.0)

    def test_evaluation_score_mate_white(self):
        analysis = StockfishAnalysis(
            best_move="Qf7", evaluation_cp=10000, evaluation_mate=3,
            depth=20, nodes=0, time_ms=0.0, pv=[],
        )
        assert analysis.evaluation_score == 1.0

    def test_evaluation_score_mate_black(self):
        analysis = StockfishAnalysis(
            best_move="Qf2", evaluation_cp=-10000, evaluation_mate=-2,
            depth=20, nodes=0, time_ms=0.0, pv=[],
        )
        assert analysis.evaluation_score == -1.0

    def test_to_dict(self):
        analysis = StockfishAnalysis(
            best_move="e2e4", evaluation_cp=50, evaluation_mate=None,
            depth=20, nodes=100, time_ms=500.0, pv=["e2e4"],
        )
        d = analysis.to_dict()
        assert d["best_move"] == "e2e4"
        assert d["evaluation_cp"] == 50
        assert d["depth"] == 20
        assert "evaluation_score" in d

    def test_extra_info(self):
        analysis = StockfishAnalysis(
            best_move="e2e4", evaluation_cp=50, evaluation_mate=None,
            depth=20, nodes=100, time_ms=500.0, pv=[],
            extra_info={"nps": 200000},
        )
        assert analysis.extra_info["nps"] == 200000


@pytest.mark.unit
class TestEvaluationResult:
    def test_creation(self):
        result = EvaluationResult(
            total_games=10,
            agent_wins=3,
            stockfish_wins=5,
            draws=2,
            avg_move_diff_cp=50.0,
            agreement_rate=0.6,
        )
        assert result.total_games == 10
        assert result.agent_win_rate == 0.3

    def test_win_rate_zero_games(self):
        result = EvaluationResult(
            total_games=0, agent_wins=0, stockfish_wins=0, draws=0,
            avg_move_diff_cp=0.0, agreement_rate=0.0,
        )
        assert result.agent_win_rate == 0.0

    def test_games_list(self):
        result = EvaluationResult(
            total_games=1, agent_wins=1, stockfish_wins=0, draws=0,
            avg_move_diff_cp=10.0, agreement_rate=0.8,
            games=[{"game_num": 0, "result": "agent_win"}],
        )
        assert len(result.games) == 1


@pytest.mark.unit
class TestStockfishAdapter:
    def test_init_default(self):
        adapter = StockfishAdapter()
        assert adapter.config.default_depth == 20
        assert adapter._engine is None

    def test_init_custom_config(self):
        config = StockfishConfig(default_depth=30)
        adapter = StockfishAdapter(config)
        assert adapter.config.default_depth == 30

    def test_find_stockfish_configured_path(self):
        config = StockfishConfig(stockfish_path="/usr/bin/stockfish")
        adapter = StockfishAdapter(config)
        with patch("os.path.isfile", return_value=True):
            result = adapter._find_stockfish()
            assert result == "/usr/bin/stockfish"

    def test_find_stockfish_not_found(self):
        config = StockfishConfig(stockfish_path=None)
        adapter = StockfishAdapter(config)
        with patch("shutil.which", return_value=None), \
             patch("src.games.chess.engines.stockfish_adapter.get_stockfish_executables", return_value=["stockfish"]):
            adapter._config.stockfish_path = None
            result = adapter._find_stockfish()
            assert result is None

    def test_is_available_cached(self):
        adapter = StockfishAdapter()
        adapter._available = True
        assert adapter.is_available is True

    def test_is_available_false(self):
        adapter = StockfishAdapter()
        with patch.object(adapter, "_find_stockfish", return_value=None):
            assert adapter.is_available is False


@pytest.mark.unit
class TestCreateStockfishAdapter:
    def test_factory(self):
        adapter = create_stockfish_adapter()
        assert isinstance(adapter, StockfishAdapter)

    def test_factory_with_config(self):
        config = StockfishConfig(threads=8)
        adapter = create_stockfish_adapter(config)
        assert adapter.config.threads == 8
