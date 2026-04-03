"""Unit tests for src/games/chess/training.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.games.chess.training import (
    ChessDataAugmentation,
    ChessOpeningBook,
    ChessTrainingMetrics,
    SelfPlayGame,
    create_chess_orchestrator,
)


@pytest.mark.unit
class TestChessTrainingMetrics:
    def test_creation(self):
        m = ChessTrainingMetrics(
            iteration=1,
            policy_loss=0.5,
            value_loss=0.3,
            total_loss=0.8,
            games_played=10,
            average_game_length=40.0,
            win_rate_white=0.4,
            win_rate_black=0.3,
            draw_rate=0.3,
            average_value_accuracy=0.7,
            learning_rate=0.001,
            elapsed_time_seconds=120.0,
        )
        assert m.iteration == 1
        assert m.total_loss == 0.8
        assert m.games_played == 10


@pytest.mark.unit
class TestSelfPlayGame:
    def test_creation(self):
        game = SelfPlayGame(
            positions=[torch.randn(12, 8, 8)],
            policies=[np.zeros(100)],
            values=[0.5],
            outcome=1.0,
            moves=["e2e4"],
            game_length=1,
        )
        assert game.outcome == 1.0
        assert game.game_length == 1


@pytest.mark.unit
class TestChessOpeningBook:
    def test_init(self):
        book = ChessOpeningBook()
        assert book._loaded is False
        assert book.book_path is None

    def test_load(self):
        book = ChessOpeningBook()
        book.load()
        assert book._loaded is True

    def test_load_idempotent(self):
        book = ChessOpeningBook()
        book.load()
        book.load()  # Should not raise
        assert book._loaded is True

    def test_get_book_move_starting_position(self):
        book = ChessOpeningBook()
        state = MagicMock()
        state.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move = book.get_book_move(state, temperature=1.0)
        assert move in ("e2e4", "d2d4", "c2c4", "g1f3")

    def test_get_book_move_unknown_position(self):
        book = ChessOpeningBook()
        state = MagicMock()
        state.fen = "8/8/8/8/8/8/8/8 w - - 0 1"  # Not in book
        move = book.get_book_move(state, temperature=1.0)
        assert move is None

    def test_get_book_move_zero_temperature(self):
        book = ChessOpeningBook()
        state = MagicMock()
        state.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move = book.get_book_move(state, temperature=0.0)
        # With temp=0, all weights stay the same (no power applied), still uniform
        assert move in ("e2e4", "d2d4", "c2c4", "g1f3")


@pytest.mark.unit
class TestChessDataAugmentation:
    def _make_config(self, use_board_flip=True):
        config = MagicMock()
        config.training.use_board_flip = use_board_flip
        return config

    def test_no_flip(self):
        aug = ChessDataAugmentation(self._make_config(use_board_flip=False))
        tensor = torch.randn(14, 8, 8)
        policy = np.random.rand(4672)
        results = aug.augment(tensor, policy)
        assert len(results) == 1

    def test_with_flip(self):
        aug = ChessDataAugmentation(self._make_config(use_board_flip=True))
        tensor = torch.randn(14, 8, 8)
        policy = np.random.rand(4672)
        results = aug.augment(tensor, policy)
        assert len(results) == 2
        # Flipped tensor should be different
        orig, flipped = results[0][0], results[1][0]
        assert not torch.equal(orig, flipped)

    def test_flip_policy(self):
        aug = ChessDataAugmentation(self._make_config())
        policy = np.zeros(4672)
        policy[0] = 1.0  # Set one action
        flipped = aug._flip_policy(policy)
        assert flipped.shape == policy.shape
        assert flipped.sum() > 0  # Something should be set


@pytest.mark.unit
class TestCreateChessOrchestrator:
    def test_creates_orchestrator(self):
        orch = create_chess_orchestrator(preset="small", device="cpu")
        assert orch.config is not None
        assert orch.current_iteration == 0

    def test_evaluate_vs_stockfish_disabled(self):
        orch = create_chess_orchestrator(preset="small", device="cpu")
        orch.config.training.evaluate_vs_stockfish = False

        import asyncio
        result = asyncio.run(orch.evaluate_vs_stockfish())
        assert "error" in result
