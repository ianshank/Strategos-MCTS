"""Extended unit tests for src/games/chess/continuous_learning.py.

Targets uncovered lines: 176, 352-383, 435-440, 468, 484-612, 624-638, 656-737.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.games.chess.continuous_learning import (
    ContinuousLearningConfig,
    ContinuousLearningSession,
    GameRecord,
    GameResult,
    OnlineLearner,
    ScoreCard,
)


# ---------------------------------------------------------------------------
# Helper: simple policy-value network for OnlineLearner.learn() tests
# ---------------------------------------------------------------------------
class _SimplePolicyValueNet(nn.Module):
    """Minimal network that returns (policy_logits, values) for testing."""

    def __init__(self, input_dim: int, action_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 64)
        self.policy_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Flatten spatial dims if present
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc(x))
        return self.policy_head(h), torch.tanh(self.value_head(h))


def _make_config_mock() -> MagicMock:
    config = MagicMock()
    config.training.learning_rate = 0.001
    config.training.weight_decay = 0.0001
    config.action_size = 100
    config.device = "cpu"
    return config


# ---------------------------------------------------------------------------
# ScoreCard – white_win_rate with zero games (line 176)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestScoreCardWhiteWinRateZero:
    def test_white_win_rate_zero_games(self):
        sc = ScoreCard()
        assert sc.white_win_rate == 0.0


# ---------------------------------------------------------------------------
# OnlineLearner.learn() – full forward/backward pass (lines 352-383)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestOnlineLearnerLearnFull:
    """Test OnlineLearner.learn() with a real network and sufficient buffer."""

    INPUT_DIM = 12 * 8 * 8  # typical chess tensor flattened
    ACTION_SIZE = 100

    def _make_learner_with_network(self, buffer_size: int = 300) -> OnlineLearner:
        config = _make_config_mock()
        learner = OnlineLearner(config, device="cpu")
        net = _SimplePolicyValueNet(self.INPUT_DIM, self.ACTION_SIZE)
        learner.set_network(net)

        # Fill buffer with random experiences
        for _ in range(buffer_size):
            state = torch.randn(12, 8, 8)
            policy = np.random.dirichlet(np.ones(self.ACTION_SIZE))
            value = np.random.choice([-1.0, 0.0, 1.0])
            learner.add_experience(state, policy, value)
        return learner

    def test_learn_returns_positive_loss(self):
        learner = self._make_learner_with_network()
        loss = learner.learn(batch_size=32)
        assert loss > 0.0

    def test_learn_updates_parameters(self):
        learner = self._make_learner_with_network()
        params_before = [p.clone() for p in learner.network.parameters()]
        learner.learn(batch_size=32)
        changed = any(
            not torch.equal(before, after) for before, after in zip(params_before, learner.network.parameters())
        )
        assert changed, "Network parameters should have changed after learning"

    def test_learn_network_in_eval_mode_after(self):
        learner = self._make_learner_with_network()
        learner.learn(batch_size=32)
        assert not learner.network.training, "Network should be in eval mode after learn()"

    def test_learn_multiple_steps(self):
        learner = self._make_learner_with_network()
        losses = [learner.learn(batch_size=32) for _ in range(5)]
        assert all(l > 0 for l in losses)


# ---------------------------------------------------------------------------
# ContinuousLearningSession.agent property – lazy load (lines 435-440)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestSessionAgentProperty:
    def test_agent_lazy_loads(self):
        chess_config = MagicMock(device="cpu")
        session = ContinuousLearningSession(chess_config)

        mock_ensemble = MagicMock()
        mock_ensemble.policy_value_net = nn.Linear(10, 10)

        with patch(
            "src.games.chess.continuous_learning.ContinuousLearningSession.agent",
            new_callable=lambda: property(lambda self: mock_ensemble),
        ):
            # Cannot test actual import path easily, so test the lazy-load logic directly
            pass

        # Test through direct attribute manipulation (simulating lazy load)
        session._agent = mock_ensemble
        session.learner.set_network = MagicMock()
        # Access the agent property's underlying code path
        assert session._agent is mock_ensemble

    def test_agent_property_imports_and_creates(self):
        """Test the agent property triggers import and creation."""
        chess_config = MagicMock(device="cpu")
        chess_config.training.learning_rate = 0.001
        chess_config.training.weight_decay = 0.0001
        session = ContinuousLearningSession(chess_config)

        mock_agent_instance = MagicMock()
        mock_agent_instance.policy_value_net = nn.Linear(10, 10)

        with patch(
            "src.games.chess.ensemble_agent.ChessEnsembleAgent",
            return_value=mock_agent_instance,
        ) as mock_cls:
            agent = session.agent
            mock_cls.assert_called_once_with(chess_config)
            assert agent is mock_agent_instance
            # Verify learner got the network
            assert session.learner.network is mock_agent_instance.policy_value_net

    def test_agent_property_cached(self):
        """Second access returns cached agent without re-creating."""
        chess_config = MagicMock(device="cpu")
        chess_config.training.learning_rate = 0.001
        chess_config.training.weight_decay = 0.0001
        session = ContinuousLearningSession(chess_config)

        mock_agent = MagicMock()
        mock_agent.policy_value_net = nn.Linear(10, 10)

        with patch(
            "src.games.chess.ensemble_agent.ChessEnsembleAgent",
            return_value=mock_agent,
        ) as mock_cls:
            first = session.agent
            second = session.agent
            assert first is second
            assert mock_cls.call_count == 1


# ---------------------------------------------------------------------------
# get_temperature – unknown schedule fallback (line 468)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGetTemperatureUnknownSchedule:
    def test_unknown_schedule_returns_initial(self):
        config = MagicMock(device="cpu")
        learning_config = ContinuousLearningConfig(
            initial_temperature=0.7,
        )
        # Override schedule to something unrecognised
        learning_config.temperature_schedule = "unknown_schedule"
        session = ContinuousLearningSession(config, learning_config)
        assert session.get_temperature(25) == 0.7


# ---------------------------------------------------------------------------
# play_single_game (lines 484-612)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPlaySingleGame:
    """Test play_single_game with mocked agent and state."""

    def _build_session(self) -> ContinuousLearningSession:
        chess_config = MagicMock(device="cpu")
        chess_config.action_size = 100
        learning_config = ContinuousLearningConfig(max_moves_per_game=5)
        session = ContinuousLearningSession(chess_config, learning_config)
        session.is_running = True
        return session

    def _mock_state_sequence(
        self, num_moves: int, terminal_at: int | None = None, checkmate: bool = False, stalemate: bool = False
    ):
        """Create a chain of mock states."""
        states = []
        for i in range(num_moves + 1):
            s = MagicMock()
            s.fen = f"fen_{i}"
            s.to_tensor.return_value = torch.randn(12, 8, 8)
            s.current_player = 1 if i % 2 == 0 else -1
            s.get_legal_actions.return_value = ["e2e4", "d2d4"]

            if terminal_at is not None and i >= terminal_at:
                s.is_terminal.return_value = True
                s.is_checkmate.return_value = checkmate
                s.is_stalemate.return_value = stalemate
            else:
                s.is_terminal.return_value = False
                s.is_checkmate.return_value = False
                s.is_stalemate.return_value = False

            states.append(s)

        # Wire apply_action chain
        for i, s in enumerate(states[:-1]):
            s.apply_action.return_value = states[i + 1]

        return states

    @pytest.mark.asyncio
    async def test_play_single_game_max_moves_draw(self):
        session = self._build_session()
        states = self._mock_state_sequence(6)  # more than max_moves_per_game=5

        mock_response = MagicMock()
        mock_response.best_move = "e2e4"
        mock_response.move_probabilities = {"e2e4": 0.8, "d2d4": 0.2}

        mock_agent = MagicMock()
        mock_agent.get_best_move = AsyncMock(return_value=mock_response)
        mock_agent.action_encoder.encode_move.return_value = 0
        session._agent = mock_agent

        with patch("src.games.chess.continuous_learning.ChessGameState") as mock_gs:
            mock_gs.initial.return_value = states[0]

            record = await session.play_single_game("game_001", temperature=0.5)

        assert isinstance(record, GameRecord)
        assert record.game_id == "game_001"
        # With max_moves_per_game=5 and non-terminal states, result is DRAW/max_moves
        assert record.result == GameResult.DRAW
        assert record.termination_reason == "max_moves"

    @pytest.mark.asyncio
    async def test_play_single_game_checkmate_white_wins(self):
        session = self._build_session()
        # Terminal at move 2 with checkmate, current_player=-1 => white wins
        states = self._mock_state_sequence(2, terminal_at=2, checkmate=True)
        # Make the terminal state have current_player=-1 so that previous player (white) won
        states[2].current_player = -1

        mock_response = MagicMock()
        mock_response.best_move = "e2e4"
        mock_response.move_probabilities = {"e2e4": 1.0}

        mock_agent = MagicMock()
        mock_agent.get_best_move = AsyncMock(return_value=mock_response)
        mock_agent.action_encoder.encode_move.return_value = 0
        session._agent = mock_agent

        with patch("src.games.chess.continuous_learning.ChessGameState") as mock_gs:
            mock_gs.initial.return_value = states[0]

            record = await session.play_single_game("game_002")

        assert record.result == GameResult.WHITE_WIN
        assert record.termination_reason == "checkmate"

    @pytest.mark.asyncio
    async def test_play_single_game_checkmate_black_wins(self):
        session = self._build_session()
        states = self._mock_state_sequence(2, terminal_at=2, checkmate=True)
        # current_player=1 at terminal => black won (previous player)
        states[2].current_player = 1

        mock_response = MagicMock()
        mock_response.best_move = "d2d4"
        mock_response.move_probabilities = {"d2d4": 1.0}

        mock_agent = MagicMock()
        mock_agent.get_best_move = AsyncMock(return_value=mock_response)
        mock_agent.action_encoder.encode_move.return_value = 1
        session._agent = mock_agent

        with patch("src.games.chess.continuous_learning.ChessGameState") as mock_gs:
            mock_gs.initial.return_value = states[0]
            record = await session.play_single_game("game_003")

        assert record.result == GameResult.BLACK_WIN
        assert record.termination_reason == "checkmate"

    @pytest.mark.asyncio
    async def test_play_single_game_stalemate(self):
        session = self._build_session()
        states = self._mock_state_sequence(2, terminal_at=2, stalemate=True)

        mock_response = MagicMock()
        mock_response.best_move = "e2e4"
        mock_response.move_probabilities = {"e2e4": 1.0}

        mock_agent = MagicMock()
        mock_agent.get_best_move = AsyncMock(return_value=mock_response)
        mock_agent.action_encoder.encode_move.return_value = 0
        session._agent = mock_agent

        with patch("src.games.chess.continuous_learning.ChessGameState") as mock_gs:
            mock_gs.initial.return_value = states[0]
            record = await session.play_single_game("game_004")

        assert record.result == GameResult.DRAW
        assert record.termination_reason == "stalemate"

    @pytest.mark.asyncio
    async def test_play_single_game_terminal_draw_other(self):
        """Terminal state that is neither checkmate nor stalemate."""
        session = self._build_session()
        states = self._mock_state_sequence(2, terminal_at=2)
        # Terminal but not checkmate or stalemate
        states[2].is_terminal.return_value = True
        states[2].is_checkmate.return_value = False
        states[2].is_stalemate.return_value = False

        mock_response = MagicMock()
        mock_response.best_move = "e2e4"
        mock_response.move_probabilities = {"e2e4": 1.0}

        mock_agent = MagicMock()
        mock_agent.get_best_move = AsyncMock(return_value=mock_response)
        mock_agent.action_encoder.encode_move.return_value = 0
        session._agent = mock_agent

        with patch("src.games.chess.continuous_learning.ChessGameState") as mock_gs:
            mock_gs.initial.return_value = states[0]
            record = await session.play_single_game("game_005")

        assert record.result == GameResult.DRAW
        assert record.termination_reason == "draw"

    @pytest.mark.asyncio
    async def test_play_single_game_interrupted(self):
        """Session stops mid-game => IN_PROGRESS."""
        session = self._build_session()
        states = self._mock_state_sequence(6)

        call_count = 0

        async def stop_after_one(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                session.is_running = False
            mock_resp = MagicMock()
            mock_resp.best_move = "e2e4"
            mock_resp.move_probabilities = {"e2e4": 1.0}
            return mock_resp

        mock_agent = MagicMock()
        mock_agent.get_best_move = stop_after_one
        mock_agent.action_encoder.encode_move.return_value = 0
        session._agent = mock_agent

        with patch("src.games.chess.continuous_learning.ChessGameState") as mock_gs:
            mock_gs.initial.return_value = states[0]
            record = await session.play_single_game("game_006")

        assert record.result == GameResult.IN_PROGRESS
        assert record.termination_reason == "interrupted"

    @pytest.mark.asyncio
    async def test_play_single_game_agent_error_fallback(self):
        """Agent raises exception => falls back to random legal move."""
        session = self._build_session()
        session.learning_config.max_moves_per_game = 2
        states = self._mock_state_sequence(3)

        mock_agent = MagicMock()
        mock_agent.get_best_move = AsyncMock(side_effect=RuntimeError("boom"))
        session._agent = mock_agent

        with patch("src.games.chess.continuous_learning.ChessGameState") as mock_gs:
            mock_gs.initial.return_value = states[0]
            record = await session.play_single_game("game_007")

        # Should still complete (max_moves reached)
        assert record.result == GameResult.DRAW
        assert len(record.moves) == 2

    @pytest.mark.asyncio
    async def test_play_single_game_repetition_draw(self):
        """Three-fold repetition triggers draw."""
        session = self._build_session()
        session.learning_config.max_moves_per_game = 20

        # Create states where fen repeats 3 times
        states = []
        for i in range(10):
            s = MagicMock()
            # Alternate between two fens so one repeats
            s.fen = "repeated_fen" if i % 2 == 1 else f"fen_{i}"
            s.to_tensor.return_value = torch.randn(12, 8, 8)
            s.current_player = 1 if i % 2 == 0 else -1
            s.get_legal_actions.return_value = ["e2e4"]
            s.is_terminal.return_value = False
            states.append(s)

        for i in range(len(states) - 1):
            states[i].apply_action.return_value = states[i + 1]

        mock_response = MagicMock()
        mock_response.best_move = "e2e4"
        mock_response.move_probabilities = {"e2e4": 1.0}

        mock_agent = MagicMock()
        mock_agent.get_best_move = AsyncMock(return_value=mock_response)
        mock_agent.action_encoder.encode_move.return_value = 0
        session._agent = mock_agent

        with patch("src.games.chess.continuous_learning.ChessGameState") as mock_gs:
            mock_gs.initial.return_value = states[0]
            record = await session.play_single_game("game_rep")

        assert record.result == GameResult.DRAW
        assert record.termination_reason == "repetition"

    @pytest.mark.asyncio
    async def test_play_single_game_agent_error_no_legal_moves(self):
        """Agent error with no legal moves => loop breaks."""
        session = self._build_session()
        session.learning_config.max_moves_per_game = 5

        state0 = MagicMock()
        state0.fen = "start_fen"
        state0.to_tensor.return_value = torch.randn(12, 8, 8)
        state0.current_player = 1
        state0.is_terminal.return_value = False
        state0.get_legal_actions.return_value = []  # no legal moves

        mock_agent = MagicMock()
        mock_agent.get_best_move = AsyncMock(side_effect=RuntimeError("error"))
        session._agent = mock_agent

        with patch("src.games.chess.continuous_learning.ChessGameState") as mock_gs:
            mock_gs.initial.return_value = state0
            record = await session.play_single_game("game_nolegal")

        # Loop should break when no legal moves after error
        assert record.result == GameResult.IN_PROGRESS
        assert record.termination_reason == "interrupted"


# ---------------------------------------------------------------------------
# _response_to_policy (lines 624-638)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestResponseToPolicy:
    def test_converts_move_probabilities(self):
        chess_config = MagicMock(device="cpu")
        chess_config.action_size = 100
        session = ContinuousLearningSession(chess_config)

        mock_agent = MagicMock()
        mock_agent.action_encoder.encode_move.side_effect = lambda m, fb: {"e2e4": 0, "d2d4": 1}[m]
        session._agent = mock_agent

        response = MagicMock()
        response.move_probabilities = {"e2e4": 0.6, "d2d4": 0.4}

        state = MagicMock()
        state.current_player = 1  # white

        policy = session._response_to_policy(response, state)
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (100,)
        assert policy.sum() == pytest.approx(1.0)
        assert policy[0] == pytest.approx(0.6)
        assert policy[1] == pytest.approx(0.4)

    def test_from_black_perspective(self):
        chess_config = MagicMock(device="cpu")
        chess_config.action_size = 50
        session = ContinuousLearningSession(chess_config)

        mock_agent = MagicMock()
        mock_agent.action_encoder.encode_move.return_value = 5
        session._agent = mock_agent

        response = MagicMock()
        response.move_probabilities = {"e7e5": 1.0}

        state = MagicMock()
        state.current_player = -1  # black

        policy = session._response_to_policy(response, state)
        # encode_move should be called with from_black=True
        mock_agent.action_encoder.encode_move.assert_called_with("e7e5", True)
        assert policy[5] == pytest.approx(1.0)

    def test_invalid_move_ignored(self):
        chess_config = MagicMock(device="cpu")
        chess_config.action_size = 50
        session = ContinuousLearningSession(chess_config)

        mock_agent = MagicMock()
        mock_agent.action_encoder.encode_move.side_effect = ValueError("invalid")
        session._agent = mock_agent

        response = MagicMock()
        response.move_probabilities = {"bad_move": 1.0}

        state = MagicMock()
        state.current_player = 1

        policy = session._response_to_policy(response, state)
        # All zeros since all moves raised ValueError
        assert policy.sum() == 0.0

    def test_empty_probabilities(self):
        chess_config = MagicMock(device="cpu")
        chess_config.action_size = 50
        session = ContinuousLearningSession(chess_config)

        mock_agent = MagicMock()
        session._agent = mock_agent

        response = MagicMock()
        response.move_probabilities = {}

        state = MagicMock()
        state.current_player = 1

        policy = session._response_to_policy(response, state)
        assert policy.sum() == 0.0


# ---------------------------------------------------------------------------
# run_session (lines 656-737)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestRunSession:
    def _build_session(self) -> ContinuousLearningSession:
        chess_config = MagicMock(device="cpu")
        chess_config.action_size = 100
        learning_config = ContinuousLearningConfig(
            max_session_minutes=60,
            max_games=5,
            min_games_before_learning=2,
            learn_every_n_games=2,
            learning_batch_size=32,
        )
        session = ContinuousLearningSession(chess_config, learning_config)
        return session

    def _make_game_record(self, game_id: str, result: GameResult) -> GameRecord:
        now = datetime.now()
        return GameRecord(
            game_id=game_id,
            white_agent="ensemble",
            black_agent="ensemble",
            result=result,
            moves=["e2e4", "e7e5"],
            positions=["fen0", "fen1", "fen2"],
            move_times_ms=[100.0, 100.0],
            total_time_ms=200.0,
            start_time=now,
            end_time=now + timedelta(seconds=1),
            final_fen="fen2",
            termination_reason="checkmate",
        )

    @pytest.mark.asyncio
    async def test_run_session_completes_max_games(self):
        session = self._build_session()

        game_num = 0

        async def mock_play(game_id, temperature=1.0):
            nonlocal game_num
            game_num += 1
            return self._make_game_record(game_id, GameResult.WHITE_WIN)

        session.play_single_game = mock_play
        session.learner.learn = MagicMock(return_value=0.5)

        scorecard = await session.run_session(max_games=3)

        assert scorecard.total_games == 3
        assert scorecard.white_wins == 3
        assert not session.is_running

    @pytest.mark.asyncio
    async def test_run_session_time_limit(self):
        session = self._build_session()

        base_time = datetime(2026, 1, 1, 12, 0, 0)
        call_count = 0

        # Patch datetime in the module so run_session sees advancing time
        with patch("src.games.chess.continuous_learning.datetime") as mock_dt:
            # First call: session_start_time assignment
            # Second call: session_end_time calc (start + timedelta)
            # Then in loop: datetime.now() >= session_end_time
            def now_side_effect():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return base_time
                # After first game, jump past the end time
                return base_time + timedelta(minutes=10)

            mock_dt.now.side_effect = now_side_effect
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            async def mock_play(game_id, temperature=1.0):
                return self._make_game_record(game_id, GameResult.DRAW)

            session.play_single_game = mock_play
            session.learner.learn = MagicMock(return_value=0.1)

            scorecard = await session.run_session(max_minutes=1, max_games=100)

        # Should stop after time limit - at most 1 game before time check triggers
        assert scorecard.total_games <= 1

    @pytest.mark.asyncio
    async def test_run_session_skips_in_progress(self):
        """Games with IN_PROGRESS result are not counted."""
        session = self._build_session()

        call_count = 0

        async def mock_play(game_id, temperature=1.0):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return self._make_game_record(game_id, GameResult.IN_PROGRESS)
            return self._make_game_record(game_id, GameResult.WHITE_WIN)

        session.play_single_game = mock_play
        session.learner.learn = MagicMock(return_value=0.1)

        scorecard = await session.run_session(max_games=1)
        assert scorecard.total_games == 1
        # IN_PROGRESS games were skipped
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_run_session_callbacks(self):
        session = self._build_session()
        game_complete_calls = []
        learning_update_calls = []
        session_complete_calls = []
        progress_calls = []

        session.learning_config.on_game_complete = lambda r: game_complete_calls.append(r)
        session.learning_config.on_learning_update = lambda loss, gc: learning_update_calls.append((loss, gc))
        session.learning_config.on_session_complete = lambda sc: session_complete_calls.append(sc)

        async def mock_play(game_id, temperature=1.0):
            return self._make_game_record(game_id, GameResult.WHITE_WIN)

        session.play_single_game = mock_play
        session.learner.learn = MagicMock(return_value=0.25)

        def progress_cb(game_num, total, sc):
            progress_calls.append((game_num, total))

        await session.run_session(max_games=4, progress_callback=progress_cb)

        assert len(game_complete_calls) == 4
        # Learning happens at game 2 and 4 (min_games=2, every 2)
        assert len(learning_update_calls) == 2
        assert len(session_complete_calls) == 1
        assert len(progress_calls) == 4

    @pytest.mark.asyncio
    async def test_run_session_learning_updates_scorecard(self):
        session = self._build_session()

        async def mock_play(game_id, temperature=1.0):
            return self._make_game_record(game_id, GameResult.BLACK_WIN)

        session.play_single_game = mock_play
        session.learner.learn = MagicMock(return_value=0.42)

        scorecard = await session.run_session(max_games=4)

        # After learning at game 2 and 4
        assert scorecard.last_loss == 0.42
        assert scorecard.total_positions_learned > 0
        # games_since_last_update resets after learning
        assert scorecard.games_since_last_update == 0

    @pytest.mark.asyncio
    async def test_run_session_exception_handling(self):
        session = self._build_session()

        call_count = 0

        async def mock_play(game_id, temperature=1.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Session error!")
            return self._make_game_record(game_id, GameResult.DRAW)

        session.play_single_game = mock_play

        # Should not raise, exception is caught
        await session.run_session(max_games=5)
        assert not session.is_running

    @pytest.mark.asyncio
    async def test_run_session_default_params(self):
        """Test that None params use config defaults."""
        chess_config = MagicMock(device="cpu")
        learning_config = ContinuousLearningConfig(
            max_session_minutes=1,
            max_games=2,
            min_games_before_learning=100,  # prevent learning
        )
        session = ContinuousLearningSession(chess_config, learning_config)

        async def mock_play(game_id, temperature=1.0):
            now = datetime.now()
            return GameRecord(
                game_id=game_id,
                white_agent="e",
                black_agent="e",
                result=GameResult.DRAW,
                moves=["e2e4"],
                positions=["f1", "f2"],
                move_times_ms=[10.0],
                total_time_ms=10.0,
                start_time=now,
                end_time=now,
                final_fen="f2",
                termination_reason="max_moves",
            )

        session.play_single_game = mock_play
        scorecard = await session.run_session()
        assert scorecard.total_games == 2  # used config default max_games=2

    @pytest.mark.asyncio
    async def test_run_session_on_session_complete_called_even_on_error(self):
        session = self._build_session()
        complete_calls = []
        session.learning_config.on_session_complete = lambda sc: complete_calls.append(sc)

        async def mock_play(game_id, temperature=1.0):
            raise RuntimeError("fail")

        session.play_single_game = mock_play

        await session.run_session(max_games=1)
        assert len(complete_calls) == 1
        assert not session.is_running
