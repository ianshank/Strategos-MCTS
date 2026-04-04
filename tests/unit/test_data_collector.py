"""Unit tests for src/training/data_collector.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from src.training.data_collector import (
    DataCollector,
    Experience,
    ExperienceBuffer,
    GameTrajectory,
    LLMDataCollector,
)


def _make_experience(value: float = 0.5, reward: float = 0.1, action: int = 0, done: bool = False) -> Experience:
    """Helper to create an Experience with default tensors."""
    return Experience(
        state=torch.randn(4),
        action=action,
        value=value,
        reward=reward,
        next_state=torch.randn(4) if not done else None,
        done=done,
    )


@pytest.mark.unit
class TestExperience:
    """Tests for Experience dataclass."""

    def test_creation(self):
        state = torch.zeros(4)
        exp = Experience(state=state, action=1, value=0.5)
        assert torch.equal(exp.state, state)
        assert exp.action == 1
        assert exp.value == 0.5
        assert exp.reward == 0.0
        assert exp.next_state is None
        assert exp.done is False
        assert exp.metadata == {}

    def test_creation_with_all_fields(self):
        exp = Experience(
            state=torch.zeros(4),
            action=2,
            value=1.0,
            policy=torch.ones(3),
            reward=0.5,
            next_state=torch.ones(4),
            done=True,
            metadata={"key": "val"},
        )
        assert exp.reward == 0.5
        assert exp.done is True
        assert exp.metadata == {"key": "val"}


@pytest.mark.unit
class TestGameTrajectory:
    """Tests for GameTrajectory dataclass."""

    def test_creation(self):
        exps = [_make_experience(), _make_experience()]
        traj = GameTrajectory(experiences=exps, outcome=1.0, game_id=0)
        assert len(traj.experiences) == 2
        assert traj.outcome == 1.0
        assert traj.game_id == 0
        assert traj.metadata == {}


@pytest.mark.unit
class TestExperienceBuffer:
    """Tests for ExperienceBuffer class."""

    def test_init_defaults(self):
        buf = ExperienceBuffer()
        assert len(buf) == 0
        assert buf.max_size == 100000
        assert buf.save_dir is None

    def test_init_with_save_dir(self, tmp_path):
        save_dir = str(tmp_path / "buffer")
        buf = ExperienceBuffer(save_dir=save_dir)
        assert buf.save_dir.exists()

    def test_add_single(self):
        buf = ExperienceBuffer()
        buf.add(_make_experience())
        assert len(buf) == 1

    def test_add_batch(self):
        buf = ExperienceBuffer()
        buf.add_batch([_make_experience() for _ in range(5)])
        assert len(buf) == 5

    def test_circular_buffer_evicts_oldest(self):
        buf = ExperienceBuffer(max_size=3)
        for i in range(5):
            buf.add(_make_experience(value=float(i)))
        assert len(buf) == 3
        # Oldest (0, 1) evicted; should have 2, 3, 4
        values = [exp.value for exp in buf.get_all()]
        assert values == [2.0, 3.0, 4.0]

    def test_add_trajectory(self):
        exps = [_make_experience(value=0.0), _make_experience(value=0.0)]
        traj = GameTrajectory(experiences=exps, outcome=1.0, game_id=0, metadata={"src": "test"})
        buf = ExperienceBuffer()
        buf.add_trajectory(traj)
        assert len(buf) == 2
        # value=0 should be updated to outcome
        assert buf.get_all()[0].value == 1.0
        assert buf.get_all()[0].metadata["src"] == "test"

    def test_add_trajectory_preserves_nonzero_value(self):
        exps = [_make_experience(value=0.8)]
        traj = GameTrajectory(experiences=exps, outcome=1.0, game_id=0)
        buf = ExperienceBuffer()
        buf.add_trajectory(traj)
        # value was already set to 0.8, should not be overwritten
        assert buf.get_all()[0].value == 0.8

    def test_sample_without_replacement(self):
        buf = ExperienceBuffer()
        buf.add_batch([_make_experience(value=float(i)) for i in range(10)])
        sample = buf.sample(5)
        assert len(sample) == 5

    def test_sample_with_replacement(self):
        buf = ExperienceBuffer()
        buf.add_batch([_make_experience() for _ in range(3)])
        sample = buf.sample(10, replace=True)
        assert len(sample) == 10

    def test_sample_clamps_to_buffer_size(self):
        buf = ExperienceBuffer()
        buf.add_batch([_make_experience() for _ in range(3)])
        sample = buf.sample(10, replace=False)
        assert len(sample) == 3

    def test_get_recent(self):
        buf = ExperienceBuffer()
        for i in range(5):
            buf.add(_make_experience(value=float(i)))
        recent = buf.get_recent(2)
        assert len(recent) == 2
        assert recent[0].value == 3.0
        assert recent[1].value == 4.0

    def test_get_all(self):
        buf = ExperienceBuffer()
        buf.add_batch([_make_experience() for _ in range(3)])
        assert len(buf.get_all()) == 3

    def test_clear(self):
        buf = ExperienceBuffer()
        buf.add_batch([_make_experience() for _ in range(3)])
        buf.clear()
        assert len(buf) == 0

    def test_save_and_load(self, tmp_path):
        buf = ExperienceBuffer(save_dir=str(tmp_path))
        buf.add_batch([_make_experience(value=float(i)) for i in range(3)])
        buf.save("test.pkl")

        buf2 = ExperienceBuffer(save_dir=str(tmp_path))
        buf2.load("test.pkl")
        assert len(buf2) == 3

    def test_save_without_save_dir_raises(self):
        buf = ExperienceBuffer()
        with pytest.raises(ValueError, match="save_dir not specified"):
            buf.save("test.pkl")

    def test_load_without_save_dir_raises(self):
        buf = ExperienceBuffer()
        with pytest.raises(ValueError, match="save_dir not specified"):
            buf.load("test.pkl")

    def test_load_nonexistent_file_raises(self, tmp_path):
        buf = ExperienceBuffer(save_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            buf.load("nonexistent.pkl")

    def test_statistics_empty(self):
        buf = ExperienceBuffer()
        stats = buf.statistics()
        assert stats == {"size": 0}

    def test_statistics_with_data(self):
        buf = ExperienceBuffer(max_size=100)
        buf.add_batch([_make_experience(value=1.0, reward=0.5) for _ in range(5)])
        stats = buf.statistics()
        assert stats["size"] == 5
        assert stats["capacity"] == 100
        assert stats["utilization"] == pytest.approx(0.05)
        assert stats["avg_value"] == pytest.approx(1.0)
        assert stats["avg_reward"] == pytest.approx(0.5)


@pytest.mark.unit
class TestDataCollector:
    """Tests for DataCollector class."""

    def test_init(self):
        buf = ExperienceBuffer()
        dc = DataCollector(buffer=buf)
        assert dc.buffer is buf
        assert dc.state_encoder is None
        assert dc.games_played == 0
        assert dc.total_experiences == 0

    def test_encode_state_with_tensor(self):
        dc = DataCollector(buffer=ExperienceBuffer())
        t = torch.randn(4)
        result = dc.encode_state(t)
        assert torch.equal(result, t)

    def test_encode_state_with_encoder(self):
        encoder = MagicMock(return_value=torch.zeros(4))
        dc = DataCollector(buffer=ExperienceBuffer(), state_encoder=encoder)
        result = dc.encode_state("raw_state")
        encoder.assert_called_once_with("raw_state")
        assert torch.equal(result, torch.zeros(4))

    def test_encode_state_no_encoder_non_tensor_raises(self):
        dc = DataCollector(buffer=ExperienceBuffer())
        with pytest.raises(ValueError, match="state_encoder required"):
            dc.encode_state("raw_state")

    async def test_collect_mcts_game(self):
        buf = ExperienceBuffer()
        encoder = MagicMock(return_value=torch.zeros(4))
        dc = DataCollector(buffer=buf, state_encoder=encoder)

        # Create a mock MCTS engine
        mcts = MagicMock()
        mcts.reset.return_value = "state0"
        mcts.is_terminal = MagicMock(side_effect=[False, False, True])
        mcts.search = AsyncMock(return_value={"policy": [0.7, 0.3], "value": 0.5})
        mcts.select_action = MagicMock(return_value=0)
        mcts.apply_action = MagicMock(return_value="state1")
        mcts.get_outcome = MagicMock(return_value=1.0)

        traj = await dc.collect_mcts_game(mcts, num_simulations=10)
        assert isinstance(traj, GameTrajectory)
        assert len(traj.experiences) == 2
        assert traj.outcome == 1.0
        assert dc.games_played == 1
        assert dc.total_experiences == 2

    async def test_collect_self_play_game(self):
        buf = ExperienceBuffer()
        encoder = MagicMock(return_value=torch.zeros(4))
        dc = DataCollector(buffer=buf, state_encoder=encoder)

        # Mock environment
        env = MagicMock()
        env.reset.return_value = "state0"
        env.is_done = MagicMock(side_effect=[False, True])
        env.step = MagicMock(return_value=("state1", 0.5, True))
        env.get_outcome = MagicMock(return_value=1.0)

        # Mock networks
        policy_net = MagicMock()
        policy_net.select_action = MagicMock(return_value=MagicMock(action=0, confidence=0.9))
        value_net = MagicMock()
        value_net.evaluate = MagicMock(return_value=0.5)

        traj = await dc.collect_self_play_game(policy_net, value_net, env)
        assert isinstance(traj, GameTrajectory)
        assert len(traj.experiences) == 1
        assert traj.outcome == 1.0
        assert dc.games_played == 1

    def test_create_policy_dataset_hard_targets(self):
        buf = ExperienceBuffer()
        buf.add_batch(
            [
                Experience(state=torch.randn(4), action=0, value=0.5),
                Experience(state=torch.randn(4), action=1, value=0.8),
            ]
        )
        dc = DataCollector(buffer=buf)
        states, targets, values = dc.create_policy_dataset(use_visit_counts=False)
        assert states.shape == (2, 4)
        assert targets.shape == (2,)
        assert values.shape == (2,)
        assert targets.dtype == torch.long

    def test_create_policy_dataset_soft_targets(self):
        buf = ExperienceBuffer()
        buf.add_batch(
            [
                Experience(state=torch.randn(4), action=0, value=0.5, policy=torch.tensor([0.7, 0.3])),
                Experience(state=torch.randn(4), action=1, value=0.8, policy=torch.tensor([0.2, 0.8])),
            ]
        )
        dc = DataCollector(buffer=buf)
        states, targets, values = dc.create_policy_dataset(use_visit_counts=True)
        assert targets.shape == (2, 2)

    def test_create_value_dataset(self):
        buf = ExperienceBuffer()
        buf.add_batch(
            [
                Experience(state=torch.randn(4), action=0, value=0.5),
                Experience(state=torch.randn(4), action=1, value=0.8),
            ]
        )
        dc = DataCollector(buffer=buf)
        states, values = dc.create_value_dataset()
        assert states.shape == (2, 4)
        assert values.shape == (2,)
        assert values[0] == pytest.approx(0.5)

    def test_create_td_dataset(self):
        buf = ExperienceBuffer()
        buf.add_batch(
            [
                Experience(
                    state=torch.randn(4), action=0, value=0.5, reward=0.1, next_state=torch.randn(4), done=False
                ),
                Experience(state=torch.randn(4), action=1, value=0.8, reward=0.2, next_state=None, done=True),
            ]
        )
        dc = DataCollector(buffer=buf)
        states, rewards, next_states, dones = dc.create_td_dataset()
        # Only 1 experience has next_state
        assert states.shape == (1, 4)
        assert rewards.shape == (1,)
        assert next_states.shape == (1, 4)
        assert dones.shape == (1,)

    def test_get_statistics(self):
        buf = ExperienceBuffer()
        dc = DataCollector(buffer=buf)
        stats = dc.get_statistics()
        assert stats["games_played"] == 0
        assert stats["total_experiences"] == 0
        assert "buffer_stats" in stats


@pytest.mark.unit
class TestLLMDataCollector:
    """Tests for LLMDataCollector class."""

    def test_init(self):
        buf = ExperienceBuffer()
        llm = MagicMock()
        dc = LLMDataCollector(buffer=buf, llm_client=llm)
        assert dc.llm_client is llm
        assert dc.llm_calls == 0
        assert dc.llm_cost == 0.0

    async def test_collect_llm_game(self):
        buf = ExperienceBuffer()
        encoder = MagicMock(return_value=torch.zeros(4))
        llm = MagicMock()
        llm.select_action = AsyncMock(return_value={"action": 0, "cost": 0.01, "reasoning": "test"})

        dc = LLMDataCollector(buffer=buf, llm_client=llm, state_encoder=encoder)

        env = MagicMock()
        env.reset.return_value = "state0"
        env.is_done = MagicMock(side_effect=[False, True])
        env.step = MagicMock(return_value=("state1", 0.5, True))
        env.get_outcome = MagicMock(return_value=1.0)

        traj = await dc.collect_llm_game(env)
        assert isinstance(traj, GameTrajectory)
        assert dc.llm_calls == 1
        assert dc.llm_cost == pytest.approx(0.01)
        assert traj.experiences[0].metadata["source"] == "llm"

    async def test_collect_llm_game_no_cost_tracking(self):
        buf = ExperienceBuffer()
        encoder = MagicMock(return_value=torch.zeros(4))
        llm = MagicMock()
        llm.select_action = AsyncMock(return_value={"action": 0, "cost": 0.05})

        dc = LLMDataCollector(buffer=buf, llm_client=llm, state_encoder=encoder)

        env = MagicMock()
        env.reset.return_value = "s0"
        env.is_done = MagicMock(side_effect=[False, True])
        env.step = MagicMock(return_value=("s1", 0.0, True))
        env.get_outcome = MagicMock(return_value=0.0)

        await dc.collect_llm_game(env, track_cost=False)
        assert dc.llm_cost == 0.0

    def test_get_llm_statistics(self):
        buf = ExperienceBuffer()
        llm = MagicMock()
        dc = LLMDataCollector(buffer=buf, llm_client=llm)
        dc.llm_calls = 10
        dc.llm_cost = 0.5

        stats = dc.get_llm_statistics()
        assert stats["llm_calls"] == 10
        assert stats["llm_cost"] == 0.5
        assert stats["avg_cost_per_call"] == pytest.approx(0.05)

    def test_get_llm_statistics_no_calls(self):
        buf = ExperienceBuffer()
        llm = MagicMock()
        dc = LLMDataCollector(buffer=buf, llm_client=llm)
        stats = dc.get_llm_statistics()
        assert stats["avg_cost_per_call"] == 0.0
