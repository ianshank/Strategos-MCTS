"""
Tests for data collector module.

Tests ExperienceBuffer, DataCollector, and LLMDataCollector
including persistence, statistics, trajectory handling, and dataset creation.
"""

from unittest.mock import MagicMock

import pytest
import torch

from src.training.data_collector import (
    DataCollector,
    Experience,
    ExperienceBuffer,
    GameTrajectory,
    LLMDataCollector,
)


def _make_experience(value: float = 0.5, action: int = 0, reward: float = 0.0) -> Experience:
    """Create a test experience."""
    return Experience(
        state=torch.randn(4),
        action=action,
        value=value,
        policy=torch.softmax(torch.randn(8), dim=0),
        reward=reward,
        next_state=torch.randn(4),
        done=False,
        metadata={"test": True},
    )


def _make_trajectory(num_moves: int = 3, outcome: float = 1.0, game_id: int = 0) -> GameTrajectory:
    """Create a test game trajectory."""
    experiences = [_make_experience(value=0.0, action=i) for i in range(num_moves)]
    return GameTrajectory(
        experiences=experiences,
        outcome=outcome,
        game_id=game_id,
        metadata={"num_moves": num_moves},
    )


@pytest.mark.unit
class TestExperienceBuffer:
    """Tests for ExperienceBuffer."""

    def test_init_without_save_dir(self):
        """Test buffer initializes without save directory."""
        buf = ExperienceBuffer(max_size=100)
        assert len(buf) == 0
        assert buf.max_size == 100
        assert buf.save_dir is None

    def test_init_with_save_dir(self, tmp_path):
        """Test buffer creates save directory."""
        save_dir = str(tmp_path / "buffer_data")
        buf = ExperienceBuffer(max_size=100, save_dir=save_dir)
        assert buf.save_dir.exists()

    def test_add_single(self):
        """Test adding a single experience."""
        buf = ExperienceBuffer(max_size=100)
        buf.add(_make_experience())
        assert len(buf) == 1

    def test_add_batch(self):
        """Test adding multiple experiences."""
        buf = ExperienceBuffer(max_size=100)
        batch = [_make_experience(value=i / 5) for i in range(5)]
        buf.add_batch(batch)
        assert len(buf) == 5

    def test_add_trajectory(self):
        """Test adding a game trajectory."""
        buf = ExperienceBuffer(max_size=100)
        trajectory = _make_trajectory(num_moves=3, outcome=1.0)
        buf.add_trajectory(trajectory)
        assert len(buf) == 3

    def test_add_trajectory_updates_values(self):
        """Test trajectory outcome is propagated to experiences with value=0."""
        buf = ExperienceBuffer(max_size=100)
        trajectory = _make_trajectory(num_moves=3, outcome=0.75)
        buf.add_trajectory(trajectory)

        for exp in buf.get_all():
            assert exp.value == 0.75

    def test_circular_eviction(self):
        """Test oldest items are evicted when buffer is full."""
        buf = ExperienceBuffer(max_size=3)
        for i in range(5):
            buf.add(_make_experience(value=float(i)))
        assert len(buf) == 3

    def test_sample_without_replacement(self):
        """Test sampling without replacement."""
        buf = ExperienceBuffer(max_size=100)
        for _ in range(20):
            buf.add(_make_experience())

        sample = buf.sample(batch_size=5, replace=False)
        assert len(sample) == 5

    def test_sample_with_replacement(self):
        """Test sampling with replacement."""
        buf = ExperienceBuffer(max_size=100)
        for _ in range(5):
            buf.add(_make_experience())

        sample = buf.sample(batch_size=10, replace=True)
        assert len(sample) == 10

    def test_sample_smaller_than_batch(self):
        """Test sample when buffer smaller than batch."""
        buf = ExperienceBuffer(max_size=100)
        for _ in range(3):
            buf.add(_make_experience())

        sample = buf.sample(batch_size=10, replace=False)
        assert len(sample) == 3

    def test_get_recent(self):
        """Test getting most recent experiences."""
        buf = ExperienceBuffer(max_size=100)
        for i in range(10):
            buf.add(_make_experience(value=float(i)))

        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].value == 9.0

    def test_get_all(self):
        """Test getting all experiences."""
        buf = ExperienceBuffer(max_size=100)
        for _ in range(5):
            buf.add(_make_experience())

        all_exps = buf.get_all()
        assert len(all_exps) == 5

    def test_clear(self):
        """Test clearing buffer."""
        buf = ExperienceBuffer(max_size=100)
        for _ in range(5):
            buf.add(_make_experience())
        buf.clear()
        assert len(buf) == 0

    def test_save_and_load(self, tmp_path):
        """Test saving and loading buffer."""
        save_dir = str(tmp_path / "buffer_data")
        buf = ExperienceBuffer(max_size=100, save_dir=save_dir)

        for i in range(5):
            buf.add(_make_experience(value=float(i)))

        buf.save("test_buffer.pkl")
        assert (tmp_path / "buffer_data" / "test_buffer.pkl").exists()

        # Load into new buffer
        buf2 = ExperienceBuffer(max_size=100, save_dir=save_dir)
        buf2.load("test_buffer.pkl")
        assert len(buf2) == 5

    def test_save_without_dir_raises(self):
        """Test saving raises when no save_dir."""
        buf = ExperienceBuffer(max_size=100)
        with pytest.raises(ValueError, match="save_dir not specified"):
            buf.save("test.pkl")

    def test_load_without_dir_raises(self):
        """Test loading raises when no save_dir."""
        buf = ExperienceBuffer(max_size=100)
        with pytest.raises(ValueError, match="save_dir not specified"):
            buf.load("test.pkl")

    def test_statistics_empty(self):
        """Test statistics on empty buffer."""
        buf = ExperienceBuffer(max_size=100)
        stats = buf.statistics()
        assert stats["size"] == 0

    def test_statistics_filled(self):
        """Test statistics on populated buffer."""
        buf = ExperienceBuffer(max_size=100)
        for i in range(10):
            buf.add(_make_experience(value=float(i) / 10, reward=0.1))

        stats = buf.statistics()
        assert stats["size"] == 10
        assert stats["capacity"] == 100
        assert 0 < stats["utilization"] <= 1.0
        assert "avg_value" in stats
        assert "avg_reward" in stats


@pytest.mark.unit
class TestDataCollector:
    """Tests for DataCollector."""

    def test_init(self):
        """Test collector initializes correctly."""
        buf = ExperienceBuffer(max_size=100)
        collector = DataCollector(buffer=buf)
        assert collector.games_played == 0
        assert collector.total_experiences == 0

    def test_encode_state_tensor(self):
        """Test encoding a tensor state returns it unchanged."""
        buf = ExperienceBuffer(max_size=100)
        collector = DataCollector(buffer=buf)

        state = torch.randn(4)
        encoded = collector.encode_state(state)
        assert torch.equal(state, encoded)

    def test_encode_state_with_encoder(self):
        """Test encoding with custom encoder."""
        buf = ExperienceBuffer(max_size=100)
        encoder = MagicMock(return_value=torch.randn(8))
        collector = DataCollector(buffer=buf, state_encoder=encoder)

        raw_state = {"board": [[1, 0], [0, 1]]}
        result = collector.encode_state(raw_state)
        encoder.assert_called_once_with(raw_state)
        assert result.shape == (8,)

    def test_encode_state_no_encoder_non_tensor_raises(self):
        """Test encoding non-tensor without encoder raises."""
        buf = ExperienceBuffer(max_size=100)
        collector = DataCollector(buffer=buf)

        with pytest.raises(ValueError, match="state_encoder required"):
            collector.encode_state({"raw": "data"})

    def test_create_value_dataset(self):
        """Test creating value network dataset."""
        buf = ExperienceBuffer(max_size=100)
        for i in range(5):
            buf.add(_make_experience(value=float(i) / 5))

        collector = DataCollector(buffer=buf)
        states, values = collector.create_value_dataset()

        assert states.shape == (5, 4)
        assert values.shape == (5,)

    def test_create_policy_dataset_with_visit_counts(self):
        """Test creating policy dataset with MCTS visit counts."""
        buf = ExperienceBuffer(max_size=100)
        for _ in range(5):
            buf.add(_make_experience())

        collector = DataCollector(buffer=buf)
        states, targets, values = collector.create_policy_dataset(use_visit_counts=True)

        assert states.shape[0] == 5
        assert targets.shape[0] == 5
        assert values.shape[0] == 5

    def test_create_policy_dataset_with_actions(self):
        """Test creating policy dataset with action targets."""
        buf = ExperienceBuffer(max_size=100)
        for i in range(5):
            exp = _make_experience(action=i)
            exp.policy = None  # No MCTS policy
            buf.add(exp)

        collector = DataCollector(buffer=buf)
        states, targets, values = collector.create_policy_dataset(use_visit_counts=False)

        assert states.shape[0] == 5
        assert targets.shape == (5,)
        assert targets.dtype == torch.long

    def test_create_td_dataset(self):
        """Test creating temporal difference dataset."""
        buf = ExperienceBuffer(max_size=100)
        for _ in range(5):
            exp = _make_experience(reward=0.1)
            exp.next_state = torch.randn(4)
            buf.add(exp)

        collector = DataCollector(buffer=buf)
        states, rewards, next_states, dones = collector.create_td_dataset()

        assert states.shape[0] == 5
        assert rewards.shape[0] == 5
        assert next_states.shape[0] == 5
        assert dones.shape[0] == 5

    def test_get_statistics(self):
        """Test getting collection statistics."""
        buf = ExperienceBuffer(max_size=100)
        collector = DataCollector(buffer=buf)

        stats = collector.get_statistics()
        assert stats["games_played"] == 0
        assert stats["total_experiences"] == 0
        assert "buffer_stats" in stats


@pytest.mark.unit
class TestLLMDataCollector:
    """Tests for LLMDataCollector."""

    def test_init(self):
        """Test LLM collector initializes correctly."""
        buf = ExperienceBuffer(max_size=100)
        llm_client = MagicMock()
        collector = LLMDataCollector(buffer=buf, llm_client=llm_client)
        assert collector.llm_calls == 0
        assert collector.llm_cost == 0.0

    def test_get_llm_statistics(self):
        """Test getting LLM-specific statistics."""
        buf = ExperienceBuffer(max_size=100)
        llm_client = MagicMock()
        collector = LLMDataCollector(buffer=buf, llm_client=llm_client)
        collector.llm_calls = 10
        collector.llm_cost = 0.5

        stats = collector.get_llm_statistics()
        assert stats["llm_calls"] == 10
        assert stats["llm_cost"] == 0.5
        assert stats["avg_cost_per_call"] == pytest.approx(0.05)

    def test_get_llm_statistics_zero_calls(self):
        """Test LLM statistics with zero calls doesn't divide by zero."""
        buf = ExperienceBuffer(max_size=100)
        llm_client = MagicMock()
        collector = LLMDataCollector(buffer=buf, llm_client=llm_client)

        stats = collector.get_llm_statistics()
        assert stats["avg_cost_per_call"] == 0.0


@pytest.mark.unit
class TestExperienceDataclass:
    """Tests for Experience and GameTrajectory dataclasses."""

    def test_experience_defaults(self):
        """Test Experience has correct defaults."""
        exp = Experience(state=torch.randn(4), action=1, value=0.5)
        assert exp.policy is None
        assert exp.reward == 0.0
        assert exp.next_state is None
        assert exp.done is False
        assert exp.metadata == {}

    def test_game_trajectory(self):
        """Test GameTrajectory structure."""
        experiences = [_make_experience() for _ in range(3)]
        traj = GameTrajectory(experiences=experiences, outcome=1.0, game_id=42)
        assert len(traj.experiences) == 3
        assert traj.outcome == 1.0
        assert traj.game_id == 42
        assert traj.metadata == {}


@pytest.mark.unit
@pytest.mark.security
class TestExperienceBufferPersistenceSafety:
    """The buffer must persist via safe torch (weights_only) and never auto-load a pickle."""

    def test_save_uses_safe_versioned_torch_format(self, tmp_path):
        from src.config.constants import EXPERIENCE_BUFFER_FORMAT_VERSION

        buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        buf.add(_make_experience(value=1.0))
        buf.save("buf.pkl")

        # Loadable with weights_only=True (no arbitrary globals) and carries version header.
        payload = torch.load(tmp_path / "buf.pkl", map_location="cpu", weights_only=True)
        assert payload["format_version"] == EXPERIENCE_BUFFER_FORMAT_VERSION
        assert len(payload["experiences"]) == 1

    def test_round_trip_restores_experiences(self, tmp_path):
        buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        for i in range(4):
            buf.add(_make_experience(value=float(i)))
        buf.save("buf.pkl")

        restored = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        restored.load("buf.pkl")
        assert len(restored) == 4
        assert restored.get_all()[0].metadata == {"test": True}

    def test_legacy_pickle_rejected_by_default(self, tmp_path):
        # Write a legacy raw pickle of Experience objects.
        import pickle

        path = tmp_path / "legacy.pkl"
        with open(path, "wb") as f:
            pickle.dump([_make_experience()], f)

        buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="TRAINING_TRUST_LEGACY_PICKLE"):
            buf.load("legacy.pkl")

    def test_legacy_pickle_migrated_when_trusted(self, tmp_path):
        import pickle

        from src.config.constants import EXPERIENCE_BUFFER_FORMAT_VERSION

        path = tmp_path / "legacy.pkl"
        with open(path, "wb") as f:
            pickle.dump([_make_experience(value=7.0), _make_experience(value=8.0)], f)

        buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path), trust_legacy_pickle=True)
        buf.load("legacy.pkl")
        assert len(buf) == 2

        # File migrated in place to the safe format: a default (untrusting) buffer can now load it.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["format_version"] == EXPERIENCE_BUFFER_FORMAT_VERSION
        safe_buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        safe_buf.load("legacy.pkl")
        assert len(safe_buf) == 2

    def test_non_primitive_metadata_is_sanitized(self, tmp_path):
        # A non-primitive metadata value must not break the safe loader.
        exp = _make_experience()
        exp.metadata = {"obj": object(), "nested": {"n": 1}, "items": [1, object()]}
        buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        buf.add(exp)
        buf.save("buf.pkl")

        restored = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        restored.load("buf.pkl")  # would raise if metadata held arbitrary objects
        meta = restored.get_all()[0].metadata
        assert isinstance(meta["obj"], str)
        assert meta["nested"] == {"n": 1}
        # Primitive inside list preserved; non-primitive inside list coerced.
        assert meta["items"][0] == 1
        assert isinstance(meta["items"][1], str)

    def test_nested_structures_in_lists_preserved(self, tmp_path):
        # Structured metadata (dict/list inside a list) must survive, not be str-coerced.
        exp = _make_experience()
        exp.metadata = {"history": [{"step": 1}, {"step": 2}], "pairs": [[1, 2], [3, 4]]}
        buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        buf.add(exp)
        buf.save("buf.pkl")

        restored = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        restored.load("buf.pkl")
        meta = restored.get_all()[0].metadata
        assert meta["history"] == [{"step": 1}, {"step": 2}]
        assert meta["pairs"] == [[1, 2], [3, 4]]

    def test_load_missing_file_raises_filenotfound(self, tmp_path):
        buf = ExperienceBuffer(max_size=100, save_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Buffer file not found"):
            buf.load("does_not_exist.pkl")
