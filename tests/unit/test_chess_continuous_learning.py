"""Unit tests for src/games/chess/continuous_learning.py."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.games.chess.continuous_learning import (
    ContinuousLearningConfig,
    ContinuousLearningSession,
    GameRecord,
    GameResult,
    OnlineLearner,
    ScoreCard,
    create_learning_session,
)


@pytest.mark.unit
class TestGameResult:
    def test_values(self):
        assert GameResult.WHITE_WIN.value == "white_win"
        assert GameResult.BLACK_WIN.value == "black_win"
        assert GameResult.DRAW.value == "draw"
        assert GameResult.IN_PROGRESS.value == "in_progress"
        assert GameResult.TIMEOUT.value == "timeout"


@pytest.mark.unit
class TestGameRecord:
    def test_creation(self):
        now = datetime.now()
        record = GameRecord(
            game_id="game_1",
            white_agent="agent_a",
            black_agent="agent_b",
            result=GameResult.WHITE_WIN,
            moves=["e2e4", "e7e5"],
            positions=["fen1", "fen2", "fen3"],
            move_times_ms=[100.0, 150.0],
            total_time_ms=250.0,
            start_time=now,
            end_time=now + timedelta(seconds=1),
            final_fen="fen3",
            termination_reason="checkmate",
        )
        assert record.game_id == "game_1"
        assert record.result == GameResult.WHITE_WIN
        assert len(record.moves) == 2


@pytest.mark.unit
class TestScoreCard:
    def test_defaults(self):
        sc = ScoreCard()
        assert sc.total_games == 0
        assert sc.elo_estimate == 1500.0
        assert sc.win_rate == 0.0
        assert sc.draw_rate == 0.0

    def test_record_white_win(self):
        sc = ScoreCard()
        sc.record_game(GameResult.WHITE_WIN, 40, 5000.0, "white")
        assert sc.total_games == 1
        assert sc.white_wins == 1
        assert sc.win_streak == 1
        assert sc.elo_estimate > 1500.0

    def test_record_black_win(self):
        sc = ScoreCard()
        sc.record_game(GameResult.BLACK_WIN, 30, 4000.0, "black")
        assert sc.total_games == 1
        assert sc.black_wins == 1
        assert sc.elo_estimate > 1500.0

    def test_record_loss(self):
        sc = ScoreCard()
        sc.record_game(GameResult.BLACK_WIN, 30, 4000.0, "white")
        assert sc.total_games == 1
        assert sc.elo_estimate < 1500.0
        assert sc.loss_streak == 1

    def test_record_draw(self):
        sc = ScoreCard()
        sc.record_game(GameResult.DRAW, 50, 6000.0)
        assert sc.draws == 1
        assert sc.elo_estimate == 1500.0
        assert sc.current_streak == 0

    def test_win_streak(self):
        sc = ScoreCard()
        for _ in range(3):
            sc.record_game(GameResult.WHITE_WIN, 30, 3000.0, "white")
        assert sc.win_streak == 3
        assert sc.current_streak == 3

    def test_loss_streak_reset(self):
        sc = ScoreCard()
        sc.record_game(GameResult.BLACK_WIN, 30, 3000.0, "white")
        sc.record_game(GameResult.BLACK_WIN, 30, 3000.0, "white")
        assert sc.loss_streak == 2
        sc.record_game(GameResult.WHITE_WIN, 30, 3000.0, "white")
        assert sc.streak_type == "win"
        assert sc.current_streak == 1

    def test_win_rate(self):
        sc = ScoreCard()
        sc.record_game(GameResult.WHITE_WIN, 30, 3000.0, "white")
        sc.record_game(GameResult.BLACK_WIN, 30, 3000.0, "white")
        sc.record_game(GameResult.DRAW, 30, 3000.0)
        # win_rate = (white_wins + black_wins) / total = (1+1)/3
        assert sc.win_rate == pytest.approx(2 / 3)

    def test_white_win_rate(self):
        sc = ScoreCard()
        sc.record_game(GameResult.WHITE_WIN, 30, 3000.0)
        sc.record_game(GameResult.DRAW, 30, 3000.0)
        assert sc.white_win_rate == 0.5

    def test_draw_rate(self):
        sc = ScoreCard()
        sc.record_game(GameResult.DRAW, 30, 3000.0)
        sc.record_game(GameResult.DRAW, 30, 3000.0)
        assert sc.draw_rate == 1.0

    def test_avg_game_length(self):
        sc = ScoreCard()
        sc.record_game(GameResult.WHITE_WIN, 40, 3000.0)
        sc.record_game(GameResult.BLACK_WIN, 60, 3000.0)
        assert sc.avg_game_length == 50.0

    def test_to_dict(self):
        sc = ScoreCard()
        sc.record_game(GameResult.WHITE_WIN, 30, 3000.0)
        d = sc.to_dict()
        assert d["white_wins"] == 1
        assert d["total_games"] == 1
        assert "win_rate" in d
        assert "elo_estimate" in d

    def test_reset(self):
        sc = ScoreCard()
        sc.record_game(GameResult.WHITE_WIN, 30, 3000.0)
        sc.reset()
        assert sc.total_games == 0
        assert sc.elo_estimate == 1500.0
        assert sc.win_streak == 0


@pytest.mark.unit
class TestContinuousLearningConfig:
    def test_defaults(self):
        cfg = ContinuousLearningConfig()
        assert cfg.max_session_minutes == 60
        assert cfg.max_games == 100
        assert cfg.temperature_schedule == "linear_decay"
        assert cfg.initial_temperature == 1.0
        assert cfg.final_temperature == 0.1

    def test_custom(self):
        cfg = ContinuousLearningConfig(max_games=50, learning_rate=0.01)
        assert cfg.max_games == 50
        assert cfg.learning_rate == 0.01


@pytest.mark.unit
class TestOnlineLearner:
    def _make_learner(self):
        config = MagicMock()
        config.training.learning_rate = 0.001
        config.training.weight_decay = 0.0001
        return OnlineLearner(config, device="cpu")

    def test_init(self):
        learner = self._make_learner()
        assert learner.experience_buffer == []
        assert learner.network is None
        assert learner.optimizer is None

    def test_add_experience(self):
        learner = self._make_learner()
        state = torch.randn(12, 8, 8)
        policy = np.zeros(100)
        learner.add_experience(state, policy, 1.0)
        assert len(learner.experience_buffer) == 1

    def test_buffer_trimming(self):
        learner = self._make_learner()
        learner.max_buffer_size = 5
        for _i in range(10):
            learner.add_experience(torch.randn(12, 8, 8), np.zeros(100), 0.0)
        assert len(learner.experience_buffer) == 5

    def test_add_game_experience(self):
        learner = self._make_learner()
        positions = [torch.randn(12, 8, 8) for _ in range(4)]
        policies = [np.zeros(100) for _ in range(4)]
        learner.add_game_experience(positions, policies, 1.0)
        assert len(learner.experience_buffer) == 4

    def test_learn_no_network(self):
        learner = self._make_learner()
        loss = learner.learn()
        assert loss == 0.0

    def test_learn_insufficient_buffer(self):
        learner = self._make_learner()
        # Set up a real small network so optimizer can be created
        net = torch.nn.Linear(10, 10)
        learner.set_network(net)
        learner.add_experience(torch.randn(12, 8, 8), np.zeros(100), 1.0)
        loss = learner.learn(batch_size=256)
        assert loss == 0.0

    def test_get_buffer_size(self):
        learner = self._make_learner()
        assert learner.get_buffer_size() == 0
        learner.add_experience(torch.randn(12, 8, 8), np.zeros(100), 1.0)
        assert learner.get_buffer_size() == 1

    def test_clear_buffer(self):
        learner = self._make_learner()
        learner.add_experience(torch.randn(12, 8, 8), np.zeros(100), 1.0)
        learner.clear_buffer()
        assert learner.get_buffer_size() == 0

    def test_set_network(self):
        learner = self._make_learner()
        net = torch.nn.Linear(10, 10)
        learner.set_network(net)
        assert learner.network is net
        assert learner.optimizer is not None


@pytest.mark.unit
class TestContinuousLearningSession:
    def test_init_defaults(self):
        with patch("src.games.chess.continuous_learning.get_chess_small_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(device="cpu")
            session = ContinuousLearningSession()
            assert session.is_running is False
            assert session.is_paused is False
            assert session.current_game_id == 0

    def test_get_temperature_constant(self):
        config = MagicMock(device="cpu")
        learning_config = ContinuousLearningConfig(temperature_schedule="constant", initial_temperature=0.5)
        session = ContinuousLearningSession(config, learning_config)
        assert session.get_temperature(0) == 0.5
        assert session.get_temperature(100) == 0.5

    def test_get_temperature_linear_decay(self):
        config = MagicMock(device="cpu")
        learning_config = ContinuousLearningConfig(
            temperature_schedule="linear_decay",
            initial_temperature=1.0,
            final_temperature=0.1,
            temperature_decay_games=50,
        )
        session = ContinuousLearningSession(config, learning_config)
        assert session.get_temperature(0) == 1.0
        assert session.get_temperature(50) == pytest.approx(0.1)
        assert session.get_temperature(25) == pytest.approx(0.55)

    def test_get_temperature_step(self):
        config = MagicMock(device="cpu")
        learning_config = ContinuousLearningConfig(
            temperature_schedule="step",
            initial_temperature=1.0,
            final_temperature=0.1,
            temperature_decay_games=50,
        )
        session = ContinuousLearningSession(config, learning_config)
        assert session.get_temperature(0) == 1.0
        assert session.get_temperature(24) == 1.0
        assert session.get_temperature(25) == pytest.approx(0.55)
        assert session.get_temperature(50) == 0.1

    def test_pause_resume_stop(self):
        config = MagicMock(device="cpu")
        session = ContinuousLearningSession(config)
        session.pause()
        assert session.is_paused is True
        session.resume()
        assert session.is_paused is False
        session.stop()
        assert session.is_running is False

    def test_reset_scorecard(self):
        config = MagicMock(device="cpu")
        session = ContinuousLearningSession(config)
        session.scorecard.record_game(GameResult.WHITE_WIN, 30, 3000.0)
        session.reset_scorecard()
        assert session.scorecard.total_games == 0

    def test_get_session_duration_not_started(self):
        config = MagicMock(device="cpu")
        session = ContinuousLearningSession(config)
        assert session.get_session_duration() == timedelta(0)

    def test_get_remaining_time_not_started(self):
        config = MagicMock(device="cpu")
        session = ContinuousLearningSession(config)
        remaining = session.get_remaining_time(30)
        assert remaining == timedelta(minutes=30)

    def test_get_remaining_time_started(self):
        config = MagicMock(device="cpu")
        session = ContinuousLearningSession(config)
        session.session_start_time = datetime.now() - timedelta(minutes=10)
        remaining = session.get_remaining_time(30)
        assert remaining.total_seconds() < 20 * 60 + 5  # ~20 min remaining


@pytest.mark.unit
class TestCreateLearningSession:
    def test_creates_session(self):
        session = create_learning_session(preset="small", max_minutes=10, max_games=5)
        assert isinstance(session, ContinuousLearningSession)
        assert session.learning_config.max_session_minutes == 10
        assert session.learning_config.max_games == 5
